//! # Reduction loop unrolling (tinygrad `OptOps.UNROLL`)
//!
//! Splits a `Reduce` range into an outer `REDUCE` loop and a small inner
//! `UNROLL` loop of width [`REDUCE_UNROLL_FACTOR`]. The [`crate::expand`]
//! pass later fully unrolls the inner strip, yielding several loads/adds per
//! outer iteration so the CPU can exploit instruction-level parallelism.
//!
//! ```text
//!   // before                       // after (horizontal sum per iteration)
//!   float acc = 0.0f;               float acc = 0.0f;
//!   for (int ridx0 = 0;            for (int ridx0 = 0;
//!        ridx0 < 8; ridx0++) {          ridx0 < 2; ridx0++) {
//!     acc = acc + in[ridx0];          acc = acc + in[ridx0*4+0]
//!   }                                         + in[ridx0*4+1]
//!                                             + in[ridx0*4+2]
//!                                             + in[ridx0*4+3];
//!                                   }
//! ```
//!
//! When combined with upcast (e.g. row-reduce), each upcasted output lane gets
//! its own accumulator, so the accumulators are independent and expose true ILP.
//!
//! Runs **second** in the optimization pipeline, after upcast and before
//! symbolic simplification.

use std::collections::HashMap;

use crate::dtype::DType;
use crate::rewrite::{graph_rewrite, substitute_with_map};
use crate::uop::{Arg, AxisKind, Op, UOp};

use super::const_i32;

/// Inner strip width for reduction unrolling. The expand pass fully unrolls
/// this many iterations, exposing ILP without growing register pressure.
const REDUCE_UNROLL_FACTOR: i64 = 4;

/// Rewrite a `Reduce` node: replace its innermost `REDUCE` range with
/// `outer * FACTOR + inner`, where `outer` keeps `AxisKind::Reduce` and
/// `inner` becomes `AxisKind::Unroll`.
fn split_reduce_unroll(reduce: &UOp) -> Option<UOp> {
    if reduce.op() != Op::Reduce || reduce.srcs().len() < 2 {
        return None;
    }
    if reduce.srcs()[1..]
        .iter()
        .any(|src| matches!(src.arg(), Arg::Range(_, AxisKind::Unroll)))
    {
        return None;
    }

    let range_idx = reduce.srcs()[1..]
        .iter()
        .rposition(|src| matches!(src.arg(), Arg::Range(_, AxisKind::Reduce)))?
        + 1;
    let range = &reduce.srcs()[range_idx];
    let Arg::Range(axis, AxisKind::Reduce) = range.arg() else {
        unreachable!("range_idx must point at a reduce range");
    };
    let [bound] = range.srcs() else {
        return None;
    };
    let size = const_i32(bound)?;
    if size <= REDUCE_UNROLL_FACTOR || size % REDUCE_UNROLL_FACTOR != 0 {
        return None;
    }

    let device = range.device();
    let outer_range = UOp::new(
        Op::Range,
        DType::I32,
        vec![UOp::const_int(
            size / REDUCE_UNROLL_FACTOR,
            DType::I32,
            device,
        )],
        Arg::Range(*axis, AxisKind::Reduce),
    );
    let inner_range = UOp::new(
        Op::Range,
        DType::I32,
        vec![UOp::const_int(REDUCE_UNROLL_FACTOR, DType::I32, device)],
        Arg::Range(*axis, AxisKind::Unroll),
    );
    let scaled_outer = UOp::mul(
        outer_range.clone(),
        UOp::const_int(REDUCE_UNROLL_FACTOR, DType::I32, device),
    );
    let expanded_index = UOp::add(scaled_outer, inner_range.clone());

    let rewritten_value = substitute_with_map(
        &reduce.srcs()[0],
        &HashMap::from([(range.clone(), expanded_index)]),
    );

    let mut srcs = Vec::with_capacity(reduce.srcs().len() + 1);
    srcs.push(rewritten_value);
    for (idx, src) in reduce.srcs().iter().enumerate().skip(1) {
        if idx == range_idx {
            srcs.push(outer_range.clone());
            srcs.push(inner_range.clone());
            continue;
        }
        srcs.push(src.clone());
    }

    Some(UOp::new(
        Op::Reduce,
        reduce.dtype(),
        srcs,
        reduce.arg().clone(),
    ))
}

/// Split constant-size reduction loops into outer `REDUCE` plus inner `UNROLL`.
#[must_use]
pub(crate) fn unroll_reduce_loops(root: &UOp) -> UOp {
    graph_rewrite(root, &mut split_reduce_unroll)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceId;
    use crate::dtype::DType;
    use crate::optimize::upcast_loops;
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

    fn build_row_reduce_graph(rows: usize, cols: usize) -> UOp {
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, rows, device);
        let input = UOp::param_buffer(1, DType::F32, rows * cols, device);
        let reduced = UOp::reduce_axis(
            UOp::reshape(input, Shape::from([rows, cols])),
            Op::Add,
            &[1],
        );
        UOp::sink(vec![UOp::store(out, reduced)])
    }

    #[test]
    fn splits_constant_reduce_by_four() {
        // Arrange
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 1, device);
        let input = UOp::param_buffer(1, DType::F32, 8, device);
        let reduced_sum = UOp::reduce_axis(UOp::reshape(input, Shape::from([8])), Op::Add, &[0]);
        let sink = UOp::sink(vec![UOp::store(out, reduced_sum)]);
        let rangeified = rangeify(&sink);

        // Act
        let optimized = unroll_reduce_loops(&rangeified);
        let order = optimized.toposort();

        // Assert
        let ranges: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Range)
            .cloned()
            .collect();
        let reduce_nodes: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Reduce)
            .cloned()
            .collect();

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].arg(), &Arg::Range(0, AxisKind::Reduce));
        assert_eq!(range_bound(&ranges[0]), 2);
        assert_eq!(ranges[1].arg(), &Arg::Range(0, AxisKind::Unroll));
        assert_eq!(range_bound(&ranges[1]), 4);
        assert_eq!(reduce_nodes.len(), 1);
        assert_eq!(reduce_nodes[0].srcs()[1], ranges[0]);
        assert_eq!(reduce_nodes[0].srcs()[2], ranges[1]);
    }

    #[test]
    fn skips_non_divisible_reduce() {
        // Arrange
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 1, device);
        let input = UOp::param_buffer(1, DType::F32, 6, device);
        let reduced_sum = UOp::reduce_axis(UOp::reshape(input, Shape::from([6])), Op::Add, &[0]);
        let sink = UOp::sink(vec![UOp::store(out, reduced_sum)]);
        let rangeified = rangeify(&sink);

        // Act
        let optimized = unroll_reduce_loops(&rangeified);
        let order = optimized.toposort();

        // Assert
        let ranges: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Range)
            .cloned()
            .collect();
        let reduce_nodes: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Reduce)
            .cloned()
            .collect();

        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].arg(), &Arg::Range(0, AxisKind::Reduce));
        assert_eq!(range_bound(&ranges[0]), 6);
        assert_eq!(reduce_nodes.len(), 1);
    }

    #[test]
    fn does_not_resplit_outer_reduce() {
        // Arrange
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 1, device);
        let input = UOp::param_buffer(1, DType::F32, 64, device);
        let reduced_sum = UOp::reduce_axis(UOp::reshape(input, Shape::from([64])), Op::Add, &[0]);
        let sink = UOp::sink(vec![UOp::store(out, reduced_sum)]);
        let rangeified = rangeify(&sink);

        // Act
        let optimized = unroll_reduce_loops(&rangeified);
        let order = optimized.toposort();

        // Assert
        let ranges: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Range)
            .cloned()
            .collect();
        let reduce_nodes: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Reduce)
            .cloned()
            .collect();

        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].arg(), &Arg::Range(0, AxisKind::Reduce));
        assert_eq!(range_bound(&ranges[0]), 16);
        assert_eq!(ranges[1].arg(), &Arg::Range(0, AxisKind::Unroll));
        assert_eq!(range_bound(&ranges[1]), 4);
        assert_eq!(reduce_nodes.len(), 1);
    }

    #[test]
    fn composes_with_upcast() {
        // Arrange
        let rangeified = rangeify(&build_row_reduce_graph(8, 8));
        let upcasted = upcast_loops(&rangeified);

        // Act
        let optimized = unroll_reduce_loops(&upcasted);
        let order = optimized.toposort();

        // Assert
        assert!(order
            .iter()
            .any(|node| matches!(node.arg(), Arg::Range(_, AxisKind::Unroll))));
    }
}
