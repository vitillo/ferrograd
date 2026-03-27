//! # Kernel optimization — loop splitting and symbolic simplification
//!
//! After rangeify produces loop-and-load kernel IR, this pass reshapes the
//! loop structure to expose parallelism and eliminate redundant arithmetic.
//! The renderer stays a generic printer; all intelligence lives here.
//!
//! ## Passes
//!
//! **Upcast** and **unroll** are mechanically the same transformation:
//! split a loop into an outer loop plus a small inner loop (width 4), then
//! let [`crate::expand`] fully unroll the inner portion into explicit
//! scalar lanes. The difference is which axis type they target, which
//! changes what happens downstream:
//!
//! **Upcast** targets output (`LOOP`) axes. The inner lanes become
//! independent stores to consecutive memory, which clang can
//! auto-vectorize into SIMD instructions:
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
//! **Unroll** targets reduction (`REDUCE`) axes. The inner lanes become a
//! horizontal sum feeding a single accumulator — four consecutive loads
//! per iteration instead of one:
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
//! When both are applied (e.g. row-reduce), each upcasted output lane
//! gets its own accumulator — then the accumulators are independent and
//! expose true ILP. The names come from tinygrad (`OptOps.UPCAST` and
//! `OptOps.UNROLL`); despite the different names, the underlying
//! mechanism is identical.
//!
//! **Symbolic** folds constant arithmetic (`2+3 → 5`) and eliminates
//! identity operations (`x+0 → x`, `x*1 → x`, `x*0 → 0`) in index
//! expressions. These patterns appear frequently after rangeify generates
//! stride arithmetic with lots of literal zeros and ones.
//!
//! ## Configuration
//!
//! The `OPT` environment variable controls which passes run:
//! - unset / `all` / `1`: all passes (default)
//! - `none` / `off` / `0`: no passes (useful for debugging raw IR)
//! - comma-separated names: `symbolic,upcast,unroll`
//!
//! ## Tinygrad reference
//!
//! `tinygrad/codegen/kernel.py` — `Kernel.apply_opt` applies `UPCAST`,
//! `UNROLL`, `LOCAL`, `GROUP` and other scheduling actions to reshape the
//! loop structure before the expander materializes them.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::dtype::DType;
use crate::rewrite::{graph_rewrite, substitute_with_map};
use crate::uop::{Arg, AxisKind, Op, UOp};

/// How many iterations to peel off the innermost reduce loop into an `UNROLL`
/// range. Mirrors tinygrad's unroll factor: the compiler fully unrolls this
/// inner portion, exposing ILP without increasing register pressure too much.
const REDUCE_UNROLL_FACTOR: i64 = 4;

/// How many iterations to peel off an output loop into an `UPCAST` range.
/// The inner upcast portion gets fully unrolled by codegen, letting the backend
/// vectorize loads/stores across consecutive elements.
const LOOP_UPCAST_FACTOR: i64 = 4;

/// Controls which optimization passes run on the kernel IR.
///
/// Each flag corresponds to a rewrite pass: `symbolic` folds constants and
/// eliminates identity ops, `upcast` splits output loops for vectorization,
/// and `unroll` splits reduce loops to expose instruction-level parallelism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OptimizeConfig {
    symbolic: bool,
    upcast: bool,
    unroll: bool,
}

impl OptimizeConfig {
    pub(crate) const fn all() -> Self {
        Self {
            symbolic: true,
            upcast: true,
            unroll: true,
        }
    }

    pub(crate) const fn none() -> Self {
        Self {
            symbolic: false,
            upcast: false,
            unroll: false,
        }
    }
}

/// Parse the `OPT` environment variable into an [`OptimizeConfig`].
///
/// Accepts `"0"` / `"off"` / `"none"` to disable everything, `"1"` / `"on"` /
/// `"all"` (or unset) to enable everything, or a comma-separated list of pass
/// names like `"symbolic,unroll"` for selective control.
fn parse_optimize_config(raw: Option<&str>) -> OptimizeConfig {
    let Some(raw) = raw.map(str::trim) else {
        return OptimizeConfig::all();
    };
    if raw.is_empty() {
        return OptimizeConfig::all();
    }

    match raw {
        "0" | "off" | "none" => return OptimizeConfig::none(),
        "1" | "on" | "all" => return OptimizeConfig::all(),
        _ => {}
    }

    let mut config = OptimizeConfig::none();
    for pass in raw.split(',').map(str::trim) {
        if pass == "symbolic" {
            config.symbolic = true;
        }
        if pass == "upcast" {
            config.upcast = true;
        }
        if pass == "unroll" {
            config.unroll = true;
        }
    }
    config
}

static OPTIMIZE: LazyLock<OptimizeConfig> =
    LazyLock::new(|| parse_optimize_config(std::env::var("OPT").ok().as_deref()));

/// Run enabled kernel IR optimization passes in a fixed order.
///
/// The `OPT` environment variable controls which passes run:
/// - unset / empty / `all` / `on` / `1`: enable all passes
/// - `none` / `off` / `0`: disable all passes
/// - comma-separated pass names, e.g. `symbolic,unroll`
#[must_use]
pub(crate) fn optimize(kernel: &UOp) -> UOp {
    optimize_with_config(kernel, *OPTIMIZE)
}

/// Run kernel IR optimization passes using an explicit config.
#[must_use]
pub(crate) fn optimize_with_config(kernel: &UOp, config: OptimizeConfig) -> UOp {
    let mut current = kernel.clone();

    if config.upcast {
        current = upcast_loops(&current);
    }

    if config.unroll {
        current = unroll_reduce_loops(&current);
    }

    if config.symbolic {
        current = graph_rewrite(&current, &mut symbolic_simple);
    }

    current
}

/// Evaluate a binary op on two constant `Arg` values at compile time.
/// Returns `None` if the types or op are unsupported for folding.
fn fold_binary(op: Op, a: &Arg, b: &Arg) -> Option<Arg> {
    match (op, a, b) {
        (Op::Add, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x + y)),
        (Op::Add, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x + y)),
        (Op::Mul, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x * y)),
        (Op::Mul, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x * y)),
        _ => None,
    }
}

/// Reconstruct a `Const` `UOp` from a folded `Arg`, inheriting dtype/device from `node`.
fn const_from_arg(node: &UOp, arg: &Arg) -> UOp {
    match arg {
        Arg::Float(value) => UOp::const_float(*value, node.dtype(), node.device()),
        Arg::Int(value) => UOp::const_int(*value, node.dtype(), node.device()),
        Arg::Bool(value) => UOp::const_bool(*value, node.dtype(), node.device()),
        _ => panic!("constant folding produced non-literal arg"),
    }
}

/// Algebraic simplification rules for index arithmetic.
///
/// These run as a separate pass after rangeify to clean up redundant
/// operations in the generated index expressions.
#[must_use]
pub(crate) fn symbolic_simple(node: &UOp) -> Option<UOp> {
    match (node.op(), node.srcs()) {
        (Op::Add, [a, b]) if a.is_const() && b.is_const() => fold_binary(Op::Add, a.arg(), b.arg())
            .as_ref()
            .map(|arg| const_from_arg(a, arg)),
        (Op::Mul, [a, b]) if a.is_const() && b.is_const() => fold_binary(Op::Mul, a.arg(), b.arg())
            .as_ref()
            .map(|arg| const_from_arg(a, arg)),
        (Op::Add, [x, y]) if y.is_zero() => Some(x.clone()),
        (Op::Add, [x, y]) if x.is_zero() => Some(y.clone()),
        (Op::Mul, [x, y]) if y.is_one() => Some(x.clone()),
        (Op::Mul, [x, y]) if x.is_one() => Some(y.clone()),
        (Op::Mul, [_, y]) if y.is_zero() => Some(y.clone()),
        (Op::Mul, [x, _]) if x.is_zero() => Some(x.clone()),
        _ => None,
    }
}

/// Extract the integer value from a `Const` node if it has `I32` dtype.
fn const_i32(node: &UOp) -> Option<i64> {
    let Arg::Int(value) = node.arg() else {
        return None;
    };
    (node.dtype() == DType::I32).then_some(*value)
}

/// Return output ranges in logical axis order, mirroring tinygrad's scheduler.
///
/// Tinygrad chooses `UPCAST` candidates from an ordered range list instead of
/// scanning one loop body for other range references. We keep the same spirit
/// here: collect output axes, sort them by axis id, and let the heuristic pick
/// from that list.
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

/// Returns `true` when a kernel stores the result of a reduction only after
/// additional scalar ALU.
///
/// Tinygrad can keep those post-reduce lanes bundled through later lowering.
/// Our current late pipeline cannot do that yet without mispairing output lanes,
/// so `UPCAST` must stay conservative until that part is generalized.
fn has_post_reduce_store_alu(root: &UOp) -> bool {
    root.toposort()
        .iter()
        .filter(|node| node.op() == Op::Store && node.srcs().len() == 2)
        .any(|store| {
            let value = &store.srcs()[1];
            value.op() != Op::Reduce && value.toposort().iter().any(|node| node.op() == Op::Reduce)
        })
}

/// Tinygrad's fallback CPU heuristic upcasts the last upcastable output axis
/// when nothing better has already been chosen.
///
/// That matches matmul well: for `M x N`, the trailing `N` axis is the one with
/// contiguous RHS access after packing, so it is the best target for a small
/// output tile.
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

/// Rewrite rule: split a `Loop` range on an `End` node into an outer `Global`
/// range and an inner `Upcast` range of size [`LOOP_UPCAST_FACTOR`].
///
/// The original index `i` becomes `outer * FACTOR + inner`. Codegen fully
/// unrolls the inner `Upcast` range, which lets the C backend emit consecutive
/// memory accesses that auto-vectorize. This mirrors tinygrad's `UPCAST` axis.
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

/// Rewrite rule: split a `Reduce` range into an outer `Reduce` range and an
/// inner `Unroll` range of size [`REDUCE_UNROLL_FACTOR`].
///
/// The original reduction index `r` becomes `outer * FACTOR + inner`. Codegen
/// fully unrolls the inner portion, giving the CPU multiple independent
/// add/mul chains per outer iteration (instruction-level parallelism). This is
/// tinygrad's `UNROLL` axis applied to reduction dimensions.
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
    use crate::schedule::rangeify::rangeify;
    use crate::shape::Shape;

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

    fn range_bound(range: &UOp) -> i64 {
        let [bound] = range.srcs() else {
            panic!("range should carry one bound");
        };
        let Arg::Int(value) = bound.arg() else {
            panic!("range bound should be a const int");
        };
        *value
    }

    #[test]
    fn test_unroll_reduce_loops_splits_constant_reduce_by_four() {
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
    fn test_unroll_reduce_loops_skips_non_divisible_reduce() {
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
    fn test_unroll_reduce_loops_does_not_resplit_outer_reduce() {
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
    fn test_unroll_reduce_loops_composes_with_upcast() {
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

    #[test]
    fn test_parse_optimize_config_defaults_to_all_passes() {
        assert_eq!(parse_optimize_config(None), OptimizeConfig::all());
        assert_eq!(parse_optimize_config(Some("")), OptimizeConfig::all());
        assert_eq!(parse_optimize_config(Some("all")), OptimizeConfig::all());
    }

    #[test]
    fn test_parse_optimize_config_can_disable_all_passes() {
        assert_eq!(parse_optimize_config(Some("0")), OptimizeConfig::none());
        assert_eq!(parse_optimize_config(Some("off")), OptimizeConfig::none());
        assert_eq!(parse_optimize_config(Some("none")), OptimizeConfig::none());
    }

    #[test]
    fn test_parse_optimize_config_enables_named_passes() {
        assert_eq!(
            parse_optimize_config(Some("symbolic")),
            OptimizeConfig {
                symbolic: true,
                upcast: false,
                unroll: false,
            }
        );
        assert_eq!(
            parse_optimize_config(Some("symbolic,upcast,unroll,unknown")),
            OptimizeConfig {
                symbolic: true,
                upcast: true,
                unroll: true,
            }
        );
    }

    #[test]
    fn test_upcast_loops_splits_innermost_loop_by_four() {
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
    fn test_upcast_loops_skips_reduction_bodies() {
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
    fn test_upcast_loops_targets_trailing_matmul_output_axis() {
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
    fn test_upcast_loops_skips_post_reduce_store_alu() {
        // Arrange
        let sink = build_scaled_plane_reduce_graph(8, 8, 8);
        let rangeified = rangeify(&sink);

        // Act
        let optimized = upcast_loops(&rangeified);

        // Assert
        assert_eq!(optimized, rangeified);
    }

    #[test]
    fn test_symbolic_add_zero_eliminated() {
        let x = UOp::const_float(5.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(x, zero);
        let result = graph_rewrite(&sum, &mut symbolic_simple);
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_symbolic_zero_plus_x_eliminated() {
        let x = UOp::const_float(5.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(zero, x);
        let result = graph_rewrite(&sum, &mut symbolic_simple);
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_symbolic_constant_folding_add() {
        let two = UOp::const_float(2.0, DType::F32, DeviceId::Cpu);
        let three = UOp::const_float(3.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(two, three);
        let result = graph_rewrite(&sum, &mut symbolic_simple);
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_symbolic_fixed_point() {
        let x = UOp::const_float(7.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let one = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(x, zero);
        let prod = UOp::mul(sum, one);
        let result = graph_rewrite(&prod, &mut symbolic_simple);
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(7.0));
    }

    #[test]
    fn test_symbolic_mul_zero() {
        let x = UOp::const_float(42.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let prod = UOp::mul(x, zero);
        let result = graph_rewrite(&prod, &mut symbolic_simple);
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(0.0));
    }
}
