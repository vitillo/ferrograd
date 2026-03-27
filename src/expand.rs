//! # Late expansion — materializing scheduled lane axes
//!
//! After optimization splits loops into outer `GLOBAL` + inner `UPCAST` (or
//! `REDUCE` + inner `UNROLL`), those inner ranges still look like ordinary
//! control-flow loops in the IR. This pass replaces them with explicit
//! *lane values*: instead of a `for uidx0 in 0..4` loop, we get four
//! concrete scalar constants `[0, 1, 2, 3]` packed into an
//! `Unroll(Vectorize(..))` wrapper. Every expression that depends on the
//! old range variable gets duplicated once per lane.
//!
//! ## Why not expand immediately during optimization?
//!
//! Tinygrad keeps `REDUCE` structure alive through expansion so that later
//! passes can still reason about reduction semantics. If we eagerly
//! unrolled everything, we'd lose the information that four values belong
//! to the same reduction and should share an accumulator. By expanding
//! lanes first and lowering reductions separately (in [`crate::devectorize`]),
//! each pass stays simple.
//!
//! ## What the pass does
//!
//! 1. **Upcast End elimination**: `End` nodes whose `Range` is `UPCAST`
//!    are replaced by their body with the range variable substituted by
//!    a lane pack of constants `[0, 1, …, N-1]`.
//! 2. **Remaining scheduled ranges**: any `UPCAST`/`UNROLL` `Range` still
//!    alive becomes an `Unroll(Vectorize(0, 1, …))` lane pack.
//! 3. **Reduce unroll → Contract**: if a `Reduce` node references an
//!    unrolled range, the unrolled lanes are wrapped in a `Contract` node
//!    so devectorize knows which lanes to fold together.
//! 4. **Lane propagation**: ALU ops, `Index`, and `Load` that consume
//!    lane-valued inputs are duplicated per-lane. Scalar inputs are
//!    broadcast (repeated for every lane).
//! 5. **End merging**: multiple `End` nodes closing the same `Range` are
//!    merged into one, collecting all their body effects.
//!
//! The output is still lane-aware (`Unroll`/`Vectorize`/`Contract` nodes
//! remain). The next stage, [`crate::devectorize`], lowers those to
//! scalar accumulators and stores.
//!
//! ## Worked example: row-reduce `[8, 8].sum(axis=1)` with upcast + unroll
//!
//! After optimize splits the output loop (GLOBAL×4 + UPCAST×4) and the
//! reduce loop (REDUCE×2 + UNROLL×4), the IR entering expand looks like:
//!
//! ```text
//!   Range(0:GLOBAL, bound=2)     // outer output loop
//!     Range(0:UPCAST, bound=4)   // inner output (to be expanded)
//!       Reduce(value, Range(1:REDUCE, bound=2), Range(1:UNROLL, bound=4))
//!       Store(out[global*4 + upcast], reduced)
//!     End(upcast)
//!   End(global)
//! ```
//!
//! After expand:
//!
//! ```text
//!   Range(0:GLOBAL, bound=2)
//!     // UPCAST End is gone — its range became 4 lane constants
//!     // Each Store, Index, Load is duplicated ×4 (one per output lane)
//!     // Each reduce Load is duplicated ×4 output × ×4 unroll = ×16
//!     Reduce(
//!       Contract(Unroll(Vectorize(load₀..load₁₅)), axes=[(1,4)]),
//!       Range(1:REDUCE, bound=2)    // outer reduce survives
//!     )
//!     Store(out[global*4+0], lane₀)   // 4 scalar stores
//!     Store(out[global*4+1], lane₁)
//!     Store(out[global*4+2], lane₂)
//!     Store(out[global*4+3], lane₃)
//!   End(global)
//! ```
//!
//! The `Reduce` is still alive — it now wraps a `Contract` that marks which
//! lanes should be horizontally summed. Devectorize handles the rest.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/codegen/late/expander.py` — `do_expand` and the expander rewrite
//! rules that materialize `UPCAST`/`UNROLL` axes into explicit lane values.

use std::collections::{HashMap, HashSet};

use crate::dtype::DType;
use crate::rewrite::{
    flatten_end_bodies, flatten_nested_sinks, graph_rewrite, substitute_with_map,
};
use crate::uop::{Arg, AxisKind, Op, UOp};

/// Converts a scheduled `Range` node (Upcast or Unroll kind) into a lane pack
/// of constant indices `[0, 1, .., N-1]`. This is how the optimizer's
/// scheduling decisions become concrete: a `Range(4, Upcast)` turns into
/// `Unroll(Vectorize(0, 1, 2, 3))`, mirroring tinygrad's expander.
fn lane_pack_for_range(range: &UOp) -> Option<UOp> {
    let Arg::Range(axis, kind) = range.arg() else {
        return None;
    };
    if !matches!(kind, AxisKind::Upcast | AxisKind::Unroll) {
        return None;
    }

    let [bound] = range.srcs() else {
        return None;
    };
    let Arg::Int(size) = bound.arg() else {
        return None;
    };
    let lane_count = usize::try_from(*size).ok()?;
    let vcount = u16::try_from(lane_count).expect("lane count should fit u16");
    let lanes: Vec<UOp> = (0..lane_count)
        .map(|idx| {
            let idx = i64::try_from(idx).expect("lane index should fit i64");
            UOp::const_int(idx, DType::I32, range.device())
        })
        .collect();
    let vector = UOp::new(Op::Vectorize, DType::I32.vec(vcount), lanes, Arg::None);
    Some(UOp::new(
        Op::Unroll,
        DType::I32,
        vec![vector],
        Arg::Lanes(vec![(*axis, lane_count)].into_boxed_slice()),
    ))
}

/// Eliminates upcast `End` nodes by substituting their `Range` with a lane
/// pack of constant indices, then inlining the body effects directly into
/// the parent graph. This is the first expansion step: it removes the
/// control-flow wrapper that the scheduler created for upcast axes.
fn expand_upcast_end(end: &UOp) -> Option<UOp> {
    if end.op() != Op::End || end.srcs().len() < 2 {
        return None;
    }
    let range = &end.srcs()[0];
    let Arg::Range(_, AxisKind::Upcast) = range.arg() else {
        return None;
    };
    let lane_value = lane_pack_for_range(range)?;
    let replacements = HashMap::from([(range.clone(), lane_value)]);
    let effects = end.srcs()[1..]
        .iter()
        .map(|body| substitute_with_map(body, &replacements))
        .collect();
    Some(UOp::sink(effects))
}

/// Converts any remaining scheduled Upcast/Unroll `Range` nodes into lane
/// packs. After `expand_upcast_end` handles the End wrappers, some Ranges
/// may still exist as standalone references (e.g. inside reduce sources).
fn rewrite_scheduled_range(node: &UOp) -> Option<UOp> {
    if node.op() != Op::Range {
        return None;
    }
    lane_pack_for_range(node)
}

/// Rewrites `Reduce` nodes whose range sources have become `Unroll` lane
/// packs into `Contract` nodes. In tinygrad, `CONTRACT` collapses
/// lane-parallel values back to scalars via reduction -- this is how
/// unrolled reduce axes are finalized after expansion.
fn fix_reduce_unroll(node: &UOp) -> Option<UOp> {
    if node.op() != Op::Reduce || node.srcs().len() < 2 {
        return None;
    }

    let mut contract_axes: Vec<(usize, usize)> = Vec::new();
    let mut reduce_ranges = Vec::new();

    for src in &node.srcs()[1..] {
        match src.op() {
            Op::Unroll => {
                let Arg::Lanes(lanes) = src.arg() else {
                    return None;
                };
                contract_axes.extend(lanes.iter().copied());
            }
            Op::Range => reduce_ranges.push(src.clone()),
            _ => return None,
        }
    }

    if contract_axes.is_empty() {
        return None;
    }

    let contracted = UOp::new(
        Op::Contract,
        node.srcs()[0].dtype(),
        vec![node.srcs()[0].clone()],
        Arg::Lanes(contract_axes.into_boxed_slice()),
    );
    let mut srcs = vec![contracted];
    srcs.extend(reduce_ranges);
    Some(UOp::new(Op::Reduce, node.dtype(), srcs, node.arg().clone()))
}

/// Compute swizzle indices for broadcasting a source's lanes to the merged
/// metadata shape. If the source has axes `[(0,4)]` and the merged shape is
/// `[(0,4),(1,4)]` (16 lanes), this returns `[0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]`
/// — each source lane repeated along the missing axis.
fn swizzle_indices(src_meta: &[(usize, usize)], merged: &[(usize, usize)]) -> Vec<usize> {
    let merged_total: usize = merged.iter().map(|(_, s)| *s).product();
    if src_meta == merged {
        return (0..merged_total).collect();
    }

    // For each position in the merged grid, project to the source's axes
    // and compute the row-major index into the source lanes.
    let src_strides: Vec<usize> = {
        let mut strides = vec![1_usize; src_meta.len()];
        for i in (0..src_meta.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * src_meta[i + 1].1;
        }
        strides
    };
    let merged_strides: Vec<usize> = {
        let mut strides = vec![1_usize; merged.len()];
        for i in (0..merged.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * merged[i + 1].1;
        }
        strides
    };
    // Map src axis → position in merged
    let axis_to_merged: HashMap<usize, usize> = merged
        .iter()
        .enumerate()
        .map(|(i, (a, _))| (*a, i))
        .collect();

    (0..merged_total)
        .map(|flat| {
            // Decompose flat index into merged coordinates
            let mut src_flat = 0;
            for (src_pos, (axis, _)) in src_meta.iter().enumerate() {
                let merged_pos = axis_to_merged[axis];
                let coord = (flat / merged_strides[merged_pos]) % merged[merged_pos].1;
                src_flat += coord * src_strides[src_pos];
            }
            src_flat
        })
        .collect()
}

/// Expand ALU/Load/Index ops that have lane-valued (`Unroll`) operands by
/// creating one scalar op per lane in the merged axis space. Sources with
/// fewer axes are swizzled (lanes repeated along missing axes) via `gep`.
/// Scalar sources are broadcast for every lane.
fn expand_lane_expr(node: &UOp) -> Option<UOp> {
    if !matches!(
        node.op(),
        Op::Add
            | Op::Mul
            | Op::Max
            | Op::CmpLt
            | Op::Where
            | Op::Neg
            | Op::Exp2
            | Op::Log2
            | Op::Sqrt
            | Op::Reciprocal
            | Op::Index
            | Op::Load
    ) {
        return None;
    }

    // Collect Unroll metadata from all lane-valued sources.
    let mut all_metas: Vec<&[(usize, usize)]> = Vec::new();
    for src in node.srcs() {
        if src.op() == Op::Unroll {
            if let Arg::Lanes(meta) = src.arg() {
                all_metas.push(meta);
            }
        }
    }
    if all_metas.is_empty() {
        return None;
    }

    // Merge all axis metadata into a superset (sorted, deduplicated).
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for meta in &all_metas {
        for &(axis, size) in *meta {
            if !merged.iter().any(|(a, _)| *a == axis) {
                merged.push((axis, size));
            }
        }
    }
    merged.sort_by_key(|(a, _)| *a);
    let lane_count: usize = merged.iter().map(|(_, s)| *s).product();
    let vcount = u16::try_from(lane_count).expect("lane count should fit u16");

    // Build per-lane scalar ops using gep to extract from each source.
    let mut result_lanes = Vec::with_capacity(lane_count);
    for lane_idx in 0..lane_count {
        let scalar_srcs: Vec<UOp> = node
            .srcs()
            .iter()
            .map(|src| {
                if src.op() == Op::Unroll {
                    let Arg::Lanes(src_meta) = src.arg() else {
                        return src.clone();
                    };
                    let inner = &src.srcs()[0]; // the Vectorize
                    let swizzle = swizzle_indices(src_meta, &merged);
                    inner.gep(swizzle[lane_idx])
                } else {
                    // Scalar — use as-is for every lane.
                    src.clone()
                }
            })
            .collect();
        result_lanes.push(UOp::new(
            node.op(),
            node.dtype(),
            scalar_srcs,
            node.arg().clone(),
        ));
    }

    let vector = UOp::new(
        Op::Vectorize,
        node.dtype().vec(vcount),
        result_lanes,
        Arg::None,
    );
    Some(UOp::new(
        Op::Unroll,
        node.dtype(),
        vec![vector],
        Arg::Lanes(merged.into_boxed_slice()),
    ))
}

/// Merges multiple `End` nodes that close the same `Range` into a single
/// `End` with all body effects combined. After upcast expansion inlines
/// effects, the same loop range may end up with separate End nodes from
/// different store chains. Merging them restores the one-End-per-Range
/// invariant that later lowering expects.
fn merge_shared_range_ends(root: &UOp) -> UOp {
    let order = root.toposort();
    let mut ends_by_range: HashMap<UOp, Vec<UOp>> = HashMap::new();
    for node in &order {
        if node.op() != Op::End || node.srcs().is_empty() {
            continue;
        }
        ends_by_range
            .entry(node.srcs()[0].clone())
            .or_default()
            .push(node.clone());
    }

    let mut replacements = HashMap::new();
    for (range, ends) in ends_by_range {
        if ends.len() < 2 {
            continue;
        }
        let mut merged_srcs = vec![range.clone()];
        let mut seen_bodies = HashSet::new();
        for end in &ends {
            for body in &end.srcs()[1..] {
                let bodies = if body.op() == Op::Sink {
                    body.srcs().to_vec()
                } else {
                    vec![body.clone()]
                };
                for inner in bodies {
                    if seen_bodies.insert(inner.clone()) {
                        merged_srcs.push(inner);
                    }
                }
            }
        }
        let merged_end = UOp::new(Op::End, DType::Void, merged_srcs, Arg::None);
        for end in ends {
            replacements.insert(end, merged_end.clone());
        }
    }

    if replacements.is_empty() {
        return root.clone();
    }
    substitute_with_map(root, &replacements)
}

/// Materialize scheduled `UPCAST`/`UNROLL` decisions into lane-carrying IR.
///
/// This is the ferrograd equivalent of tinygrad's expander pre-pass plus
/// generic lane propagation. It intentionally keeps `Reduce` alive so the next
/// stage can still lower reductions with lane structure intact.
#[must_use]
pub(crate) fn late_expand(root: &UOp) -> UOp {
    let expanded_upcast = graph_rewrite(root, &mut expand_upcast_end);
    let rewritten_ranges = graph_rewrite(&expanded_upcast, &mut rewrite_scheduled_range);
    let fixed_reduces = graph_rewrite(&rewritten_ranges, &mut fix_reduce_unroll);
    let expanded_lanes = graph_rewrite(&fixed_reduces, &mut expand_lane_expr);
    let flattened_ends = graph_rewrite(&expanded_lanes, &mut flatten_end_bodies);
    let merged_ends = merge_shared_range_ends(&flattened_ends);
    graph_rewrite(&merged_ends, &mut flatten_nested_sinks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::{unroll_reduce_loops, upcast_loops};
    use crate::shape::Shape;

    fn build_add_graph(n: i64) -> UOp {
        let device = crate::device::DeviceId::Cpu;
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

    fn build_reduce_graph(n: i64) -> UOp {
        let device = crate::device::DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 1, device);
        let numel = usize::try_from(n).expect("n should fit usize");
        let input_ptr = UOp::param_buffer(1, DType::F32, numel, device);
        let reduced =
            UOp::reduce_axis(UOp::reshape(input_ptr, Shape::from([numel])), Op::Add, &[0]);
        UOp::sink(vec![UOp::store(out_ptr, reduced)])
    }

    fn build_row_reduce_graph(rows: usize, cols: usize) -> UOp {
        let device = crate::device::DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, rows, device);
        let input_ptr = UOp::param_buffer(1, DType::F32, rows * cols, device);
        let reduced = UOp::reduce_axis(
            UOp::reshape(input_ptr, Shape::from([rows, cols])),
            Op::Add,
            &[1],
        );
        UOp::sink(vec![UOp::store(out_ptr, reduced)])
    }

    #[test]
    fn test_late_expand_materializes_upcast_lanes() {
        let split = upcast_loops(&build_add_graph(8));
        let expanded = late_expand(&split);
        let order = expanded.toposort();

        assert!(!order
            .iter()
            .any(|node| matches!(node.arg(), Arg::Range(_, AxisKind::Upcast))));
        assert!(order.iter().any(|node| node.op() == Op::Unroll));
    }

    #[test]
    fn test_late_expand_rewrites_reduce_unroll_to_contract() {
        let rangeified = crate::schedule::rangeify::rangeify(&build_reduce_graph(8));
        let split = unroll_reduce_loops(&rangeified);
        let expanded = late_expand(&split);
        let order = expanded.toposort();

        assert!(order.iter().any(|node| node.op() == Op::Contract));
        assert!(!order
            .iter()
            .any(|node| matches!(node.arg(), Arg::Range(_, AxisKind::Unroll))));
    }

    #[test]
    fn test_late_expand_keeps_reduce_lane_aware_under_upcast() {
        let rangeified = crate::schedule::rangeify::rangeify(&build_row_reduce_graph(8, 8));
        let optimized = unroll_reduce_loops(&upcast_loops(&rangeified));
        let expanded = late_expand(&optimized);
        let order = expanded.toposort();

        assert!(order.iter().any(|node| node.op() == Op::Reduce));
        assert!(order.iter().any(|node| node.op() == Op::Contract));
        assert!(order.iter().any(|node| node.op() == Op::Unroll));
    }
}
