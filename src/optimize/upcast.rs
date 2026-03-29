//! # Output loop upcasting
//!
//! Splits the trailing output (`LOOP`) axis into an outer `GLOBAL` loop and
//! an inner `UPCAST` loop of width [`LOOP_UPCAST_FACTOR`]. The
//! [`crate::expand`] pass later fully unrolls the inner strip so loads/stores
//! hit consecutive addresses and the C backend can auto-vectorize into SIMD.
//!
//! ```text
//!   // before                       // after
//!   for (int idx0 = 0;              for (int idx0 = 0;
//!        idx0 < 8; idx0++) {             idx0 < 2; idx0++) {
//!     *out[idx0] = …;                *out[idx0*4+0] = …;
//!   }                                *out[idx0*4+1] = …;
//!                                    *out[idx0*4+2] = …;
//!                                    *out[idx0*4+3] = …;
//!                                  }
//! ```
//!
//! Heuristics: pick the trailing output axis (e.g. matmul’s `N` for contiguous
//! RHS access), skip kernels that already have an upcast axis, and stay
//! conservative when post-reduce ALU sits between the reduction and the store.
//!
//! Runs **first** in the optimization pipeline, before unroll and symbolic.

use std::collections::HashMap;

use crate::dtype::DType;
use crate::rewrite::{graph_rewrite, substitute_with_map};
use crate::uop::{Arg, AxisKind, Op, UOp};

use super::const_i32;

/// Inner strip width for output upcasting. The expand pass fully unrolls
/// this many iterations so the backend can vectorize consecutive accesses.
const LOOP_UPCAST_FACTOR: i64 = 4;

/// Collect output ranges sorted by axis id so the heuristic can pick the
/// trailing one.
fn ordered_output_ranges(root: &UOp) -> Vec<UOp> {
    let mut ranges: Vec<UOp> = root
        .toposort()
        .into_iter()
        .filter(|node| {
            matches!(
                node.arg(),
                Arg::Range(
                    _,
                    AxisKind::Loop | AxisKind::Global | AxisKind::Local | AxisKind::Thread
                )
            )
        })
        .collect();
    ranges.sort_by_key(|range| {
        let Arg::Range(axis, _) = range.arg() else {
            unreachable!("ordered_output_ranges should only see range nodes");
        };
        *axis
    });
    ranges.dedup();
    ranges
}

/// Detect stores where ALU sits between a reduction and the store.
///
/// Our expand pass cannot yet keep post-reduce output lanes correctly paired
/// through extra ALU, so upcast stays conservative in this case.
fn has_post_reduce_store_alu(root: &UOp) -> bool {
    root.toposort()
        .iter()
        .filter(|node| node.op() == Op::Store && node.srcs().len() == 2)
        .any(|store| {
            let value = &store.srcs()[1];
            value.op() != Op::Reduce && value.toposort().iter().any(|node| node.op() == Op::Reduce)
        })
}

/// Pick the trailing `LOOP` axis whose bound is divisible by the upcast factor,
/// or [`None`] if the kernel already has an upcast axis or has post-reduce ALU.
fn select_upcast_range(root: &UOp) -> Option<UOp> {
    if root
        .toposort()
        .iter()
        .any(|node| matches!(node.arg(), Arg::Range(_, AxisKind::Upcast)))
    {
        return None;
    }
    if has_post_reduce_store_alu(root) {
        return None;
    }

    let range = ordered_output_ranges(root).into_iter().last()?;
    let Arg::Range(_, AxisKind::Loop) = range.arg() else {
        return None;
    };
    let [bound] = range.srcs() else {
        return None;
    };
    let size = const_i32(bound)?;
    if size <= LOOP_UPCAST_FACTOR || size % LOOP_UPCAST_FACTOR != 0 {
        return None;
    }
    Some(range)
}

/// Rewrite an `End` node: replace its `LOOP` range with
/// `outer * FACTOR + inner`, where `outer` becomes `GLOBAL` and `inner`
/// becomes `UPCAST`.
fn split_loop_upcast(end: &UOp, target_range: &UOp) -> Option<UOp> {
    if end.op() != Op::End || end.srcs().len() < 2 {
        return None;
    }

    let range = &end.srcs()[0];
    if range != target_range {
        return None;
    }
    let Arg::Range(axis, AxisKind::Loop) = range.arg() else {
        return None;
    };

    let [bound] = range.srcs() else {
        return None;
    };
    let size = const_i32(bound)?;
    if size <= LOOP_UPCAST_FACTOR || size % LOOP_UPCAST_FACTOR != 0 {
        return None;
    }

    let device = range.device();
    let outer_range = UOp::new(
        Op::Range,
        DType::I32,
        vec![UOp::const_int(
            size / LOOP_UPCAST_FACTOR,
            DType::I32,
            device,
        )],
        Arg::Range(*axis, AxisKind::Global),
    );
    let inner_range = UOp::new(
        Op::Range,
        DType::I32,
        vec![UOp::const_int(LOOP_UPCAST_FACTOR, DType::I32, device)],
        Arg::Range(*axis, AxisKind::Upcast),
    );
    let expanded_index = UOp::add(
        UOp::mul(
            outer_range.clone(),
            UOp::const_int(LOOP_UPCAST_FACTOR, DType::I32, device),
        ),
        inner_range.clone(),
    );
    let replacements = HashMap::from([(range.clone(), expanded_index)]);
    let mut srcs = vec![inner_range];
    srcs.extend(
        end.srcs()[1..]
            .iter()
            .map(|body| substitute_with_map(body, &replacements)),
    );
    let upcasted = UOp::new(Op::End, DType::Void, srcs, Arg::None);

    Some(UOp::new(
        Op::End,
        DType::Void,
        vec![outer_range, upcasted],
        Arg::None,
    ))
}

/// Split the selected output loop into an outer `GLOBAL` plus inner `UPCAST`.
#[must_use]
pub(crate) fn upcast_loops(root: &UOp) -> UOp {
    let Some(target_range) = select_upcast_range(root) else {
        return root.clone();
    };
    graph_rewrite(root, &mut |node| split_loop_upcast(node, &target_range))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceId;
    use crate::dtype::DType;
    use crate::schedule::rangeify::rangeify;
    use crate::shape::Shape;
    use crate::uop::{Arg, AxisKind, Op, UOp};

    fn range_bound(range: &UOp) -> i64 {
        let [bound] = range.srcs() else {
            panic!("range should carry one bound");
        };
        let Arg::Int(value) = bound.arg() else {
            panic!("range bound should be a const int");
        };
        *value
    }

    fn build_add_graph(n: i64) -> UOp {
        let device = DeviceId::Cpu;
        let numel = usize::try_from(n).expect("n should fit usize");
        let out_ptr = UOp::param_buffer(0, DType::F32, numel, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, numel, device);
        let bound = UOp::const_int(n, DType::I32, device);
        let idx = UOp::new(
            Op::Range,
            DType::I32,
            vec![bound],
            Arg::Range(0, AxisKind::Loop),
        );
        let a_idx = UOp::new(
            Op::Index,
            a_ptr.dtype(),
            vec![a_ptr, idx.clone()],
            Arg::None,
        );
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let one = UOp::const_float(1.0, DType::F32, device);
        let sum = UOp::add(a_val, one);
        let out_idx = UOp::new(
            Op::Index,
            out_ptr.dtype(),
            vec![out_ptr, idx.clone()],
            Arg::None,
        );
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, sum], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store], Arg::None);
        UOp::sink(vec![end])
    }

    fn build_matmul_like_kernel(m: i64, n: i64, k: i64) -> UOp {
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(
            0,
            DType::F32,
            usize::try_from(m * n).expect("m*n should fit usize"),
            device,
        );
        let lhs = UOp::param_buffer(
            1,
            DType::F32,
            usize::try_from(m * k).expect("m*k should fit usize"),
            device,
        );
        let rhs = UOp::param_buffer(
            2,
            DType::F32,
            usize::try_from(n * k).expect("n*k should fit usize"),
            device,
        );

        let m_range = UOp::new(
            Op::Range,
            DType::I32,
            vec![UOp::const_int(m, DType::I32, device)],
            Arg::Range(0, AxisKind::Loop),
        );
        let n_range = UOp::new(
            Op::Range,
            DType::I32,
            vec![UOp::const_int(n, DType::I32, device)],
            Arg::Range(1, AxisKind::Loop),
        );
        let k_range = UOp::new(
            Op::Range,
            DType::I32,
            vec![UOp::const_int(k, DType::I32, device)],
            Arg::Range(2, AxisKind::Reduce),
        );

        let out_idx = UOp::new(
            Op::Index,
            out.dtype(),
            vec![
                out.clone(),
                UOp::add(
                    UOp::mul(m_range.clone(), UOp::const_int(n, DType::I32, device)),
                    n_range.clone(),
                ),
            ],
            Arg::None,
        );
        let lhs_idx = UOp::new(
            Op::Index,
            lhs.dtype(),
            vec![
                lhs.clone(),
                UOp::add(
                    UOp::mul(m_range.clone(), UOp::const_int(k, DType::I32, device)),
                    k_range.clone(),
                ),
            ],
            Arg::None,
        );
        let rhs_idx = UOp::new(
            Op::Index,
            rhs.dtype(),
            vec![
                rhs.clone(),
                UOp::add(
                    UOp::mul(n_range.clone(), UOp::const_int(k, DType::I32, device)),
                    k_range.clone(),
                ),
            ],
            Arg::None,
        );

        let value = UOp::mul(
            UOp::new(Op::Load, DType::F32, vec![lhs_idx], Arg::None),
            UOp::new(Op::Load, DType::F32, vec![rhs_idx], Arg::None),
        );
        let reduce = UOp::new(
            Op::Reduce,
            DType::F32,
            vec![value, k_range],
            Arg::Reduce(Op::Add, Box::default()),
        );
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, reduce], Arg::None);
        let end_n = UOp::new(Op::End, DType::Void, vec![n_range, store], Arg::None);
        let end_m = UOp::new(Op::End, DType::Void, vec![m_range, end_n], Arg::None);
        UOp::sink(vec![end_m])
    }

    fn build_scaled_plane_reduce_graph(depth: usize, rows: usize, cols: usize) -> UOp {
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, rows * cols, device);
        let base = UOp::param_buffer(1, DType::F32, rows * cols, device);
        let input = UOp::param_buffer(2, DType::F32, depth * rows * cols, device);
        let scale = UOp::param_buffer(3, DType::F32, 1, device);

        let reduced = UOp::reduce_axis(
            UOp::reshape(input, Shape::from([depth, rows, cols])),
            Op::Add,
            &[0],
        );
        let reduced = UOp::reshape(reduced, Shape::from([rows, cols]));
        let base = UOp::reshape(base, Shape::from([rows, cols]));
        let scale = UOp::expand(
            UOp::reshape(scale, Shape::from([1, 1])),
            Shape::from([rows, cols]),
        );
        let updated = UOp::add(base, UOp::neg(UOp::mul(reduced, scale)));

        UOp::sink(vec![UOp::store(out, updated)])
    }

    #[test]
    fn splits_innermost_loop_by_four() {
        // Arrange
        let sink = build_add_graph(8);

        // Act
        let optimized = upcast_loops(&sink);
        let order = optimized.toposort();

        // Assert
        let ranges: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Range)
            .cloned()
            .collect();
        let stores = order.iter().filter(|node| node.op() == Op::Store).count();

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].arg(), &Arg::Range(0, AxisKind::Global));
        assert_eq!(range_bound(&ranges[0]), 2);
        assert_eq!(ranges[1].arg(), &Arg::Range(0, AxisKind::Upcast));
        assert_eq!(range_bound(&ranges[1]), 4);
        assert_eq!(stores, 1);
    }

    #[test]
    fn skips_reduction_bodies() {
        // Arrange
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 1, device);
        let input = UOp::param_buffer(1, DType::F32, 8, device);
        let reduced = UOp::reduce_axis(UOp::reshape(input, Shape::from([2, 4])), Op::Add, &[1]);
        let sink = UOp::sink(vec![UOp::store(out, reduced)]);
        let rangeified = rangeify(&sink);

        // Act
        let optimized = upcast_loops(&rangeified);

        // Assert
        assert_eq!(optimized, rangeified);
    }

    #[test]
    fn targets_trailing_matmul_output_axis() {
        // Arrange
        let sink = build_matmul_like_kernel(8, 8, 8);

        // Act
        let optimized = upcast_loops(&sink);
        let order = optimized.toposort();

        // Assert
        assert!(order
            .iter()
            .any(|node| matches!(node.arg(), Arg::Range(1, AxisKind::Upcast))));
        assert!(order
            .iter()
            .any(|node| matches!(node.arg(), Arg::Range(0, AxisKind::Loop))));
        assert!(!order
            .iter()
            .any(|node| matches!(node.arg(), Arg::Range(0, AxisKind::Upcast))));
    }

    #[test]
    fn skips_post_reduce_store_alu() {
        // Arrange
        let sink = build_scaled_plane_reduce_graph(8, 8, 8);
        let rangeified = rangeify(&sink);

        // Act
        let optimized = upcast_loops(&rangeified);

        // Assert
        assert_eq!(optimized, rangeified);
    }
}
