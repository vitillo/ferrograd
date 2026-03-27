//! # Devectorization — from lane-aware IR to scalar control flow
//!
//! After [`crate::expand`] has materialized scheduled axes into explicit
//! lane packs (`Unroll(Vectorize(..))`), the IR still contains high-level
//! abstractions that the renderer doesn't understand: `Reduce` nodes,
//! `Contract` nodes, and multi-lane `Store`s. This pass lowers all of them
//! into plain scalar control flow that the linearizer and renderer can
//! handle directly.
//!
//! ## Why is this separate from expansion?
//!
//! Expansion creates lane structure; devectorization consumes it. Keeping
//! them separate means each pass does one thing:
//! - Expand decides *which* values get lanes (based on scheduling decisions).
//! - Devectorize decides *how* those lanes become concrete accumulators and
//!   stores (based on reduction semantics).
//!
//! Tinygrad uses the same split: the expander keeps `REDUCE` alive, and a
//! later devectorizer pass does the final lowering.
//!
//! ## What the pass does
//!
//! 1. **Reduce lowering**: each `Reduce` node becomes:
//!    - One `DefineAcc` per output lane (initialized to the identity
//!      element: 0 for sum, -∞ for max)
//!    - An `Assign` that updates each accumulator inside the reduction loop
//!    - `End` nodes that close the reduction ranges
//!    - `After` nodes that sequence the accumulator read after the loop
//!
//!    If the reduce value carries lanes (from a `Contract`), each group of
//!    contracted lanes is horizontally folded into a single accumulator
//!    update.
//!
//! 2. **Store scalarization**: a `Store` whose index or value carries
//!    multiple lanes is split into N independent scalar stores — one per
//!    lane coordinate.
//!
//! 3. **After lifting**: `After` barriers buried inside a store's value
//!    subtree are lifted to the store's direct inputs so the linearizer
//!    can sequence them correctly.
//!
//! 4. **Structural cleanup**: nested `Sink`s and `End` bodies containing
//!    `Sink` wrappers are flattened.
//!
//! After this pass, no `Reduce`, `Contract`, `Vectorize`, or `Unroll` nodes
//! remain. The IR is ready for [`crate::linearize`].
//!
//! ## Worked example: `[2, 3].sum(axis=0)` — simple row-reduce
//!
//! The input from expand (no upcast/unroll here — too small):
//!
//! ```text
//!   Range(1:LOOP, bound=3)          // output axis
//!     Reduce(Load(input[ridx*3 + idx]), Range(0:REDUCE, bound=2))
//!     Store(out[idx], reduced)
//!   End(loop)
//! ```
//!
//! After devectorize — `Reduce` is gone, replaced by explicit control flow:
//!
//! ```text
//!   Range(1:LOOP, bound=3)
//!     DefineAcc(0.0, Range(1:LOOP))   // accumulator scoped to output loop
//!     Range(0:REDUCE, bound=2)        // reduce loop
//!       Assign(acc, acc + Load(input[ridx*3 + idx]))
//!     End(reduce)
//!     Store(out[idx], After(acc, End(reduce)))
//!   End(loop)
//! ```
//!
//! Which renders to C as:
//!
//! ```text
//!   for (int idx1 = 0; idx1 < 3; idx1++) {
//!     float acc0 = 0.0f;
//!     for (int ridx0 = 0; ridx0 < 2; ridx0++) {
//!       acc0 = acc0 + *(data1 + ridx0*3 + idx1);
//!     }
//!     *(data0 + idx1) = acc0;
//!   }
//! ```
//!
//! ## Tinygrad reference
//!
//! `tinygrad/codegen/late/devectorizer.py` — the rewrite rules that lower
//! `REDUCE` to `DEFINE_ACC` + `ASSIGN` and scalarize lane-valued stores.

use std::collections::{HashMap, HashSet};

use crate::dtype::DType;
use crate::lane::{
    combine_lane_sources, coord_product, coord_to_index, lane_value, make_lane_pack, LaneMeta,
};
use crate::rewrite::graph_rewrite;
use crate::uop::{Arg, Op, UOp};

/// Flattens `End(range, Sink(body0, body1))` into `End(range, body0, body1)`.
/// Devectorization can scalarize one logical write into several stores, which
/// temporarily introduces nested sink groupings.
fn flatten_end_bodies(node: &UOp) -> Option<UOp> {
    if node.op() != Op::End || node.srcs().len() < 2 {
        return None;
    }

    let mut changed = false;
    let mut srcs = vec![node.srcs()[0].clone()];
    for body in &node.srcs()[1..] {
        if body.op() == Op::Sink {
            changed = true;
            srcs.extend(body.srcs().iter().cloned());
            continue;
        }
        srcs.push(body.clone());
    }
    changed.then(|| UOp::new(Op::End, DType::Void, srcs, Arg::None))
}

/// Flattens `Sink(Sink(..), ..)` into one sink after scalarizing lane-valued
/// stores into multiple effects.
fn flatten_nested_sinks(node: &UOp) -> Option<UOp> {
    if node.op() != Op::Sink {
        return None;
    }

    let mut changed = false;
    let mut srcs = Vec::new();
    for src in node.srcs() {
        if src.op() == Op::Sink {
            changed = true;
            srcs.extend(src.srcs().iter().cloned());
            continue;
        }
        srcs.push(src.clone());
    }
    changed.then(|| UOp::sink(srcs))
}

/// Propagates lane-valued results introduced by reduce lowering through simple
/// scalar expressions until the final store can be scalarized.
fn expand_lane_expr(node: &UOp) -> Option<UOp> {
    let (src_lanes, meta) = combine_lane_sources(node)?;
    let lane_count = meta.iter().map(|(_, size)| *size).product::<usize>();

    match node.op() {
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
        | Op::After => {
            let mut lanes = Vec::with_capacity(lane_count);
            for lane_idx in 0..lane_count {
                let scalar_srcs = src_lanes
                    .iter()
                    .map(|lanes| {
                        if lanes.len() == 1 {
                            lanes[0].clone()
                        } else {
                            lanes[lane_idx].clone()
                        }
                    })
                    .collect();
                lanes.push(UOp::new(
                    node.op(),
                    node.dtype(),
                    scalar_srcs,
                    node.arg().clone(),
                ));
            }
            Some(make_lane_pack(Op::Unroll, node.dtype(), lanes, meta))
        }
        _ => None,
    }
}

/// Splits a `Contract(Unroll(Vectorize(...)))` into groups of lanes that share
/// the same output position. Each group contains the lanes that differ only
/// along contracted axes -- these will be horizontally reduced together. The
/// returned metadata describes the remaining (non-contracted) axes.
fn contract_lane_groups(node: &UOp) -> Option<(Vec<Vec<UOp>>, LaneMeta)> {
    if node.op() != Op::Contract || node.srcs().len() != 1 {
        return None;
    }
    let Arg::Lanes(contract_axes) = node.arg() else {
        return None;
    };
    let (lanes, full_meta) = lane_value(&node.srcs()[0])?;
    let contracted_axes: HashSet<usize> = contract_axes.iter().map(|(axis, _)| *axis).collect();
    let remaining_meta: Vec<(usize, usize)> = full_meta
        .iter()
        .copied()
        .filter(|(axis, _)| !contracted_axes.contains(axis))
        .collect();
    let contract_meta: Vec<(usize, usize)> = full_meta
        .iter()
        .copied()
        .filter(|(axis, _)| contracted_axes.contains(axis))
        .collect();

    let mut full_positions = HashMap::new();
    for (idx, (axis, _)) in full_meta.iter().enumerate() {
        full_positions.insert(*axis, idx);
    }

    let remaining_coords = coord_product(&remaining_meta);
    let contract_coords = coord_product(&contract_meta);
    let mut groups = Vec::with_capacity(remaining_coords.len());

    for remaining in &remaining_coords {
        let mut group = Vec::with_capacity(contract_coords.len().max(1));
        for contract in &contract_coords {
            let mut full = vec![0_usize; full_meta.len()];
            for ((axis, _), coord) in remaining_meta.iter().zip(remaining) {
                full[*full_positions
                    .get(axis)
                    .expect("remaining axis should exist in full lane metadata")] = *coord;
            }
            for ((axis, _), coord) in contract_meta.iter().zip(contract) {
                full[*full_positions
                    .get(axis)
                    .expect("contract axis should exist in full lane metadata")] = *coord;
            }
            group.push(lanes[coord_to_index(&full_meta, &full)].clone());
        }
        groups.push(group);
    }

    Some((groups, remaining_meta.into_boxed_slice()))
}

/// Prepares a reduce operand for per-lane reduction. If the value is a plain
/// lane pack, each lane becomes its own single-element group. If it is a
/// Contract, lanes are grouped by contracted axes. Either way the result is
/// ready for `horizontal_reduce` within each group.
fn reduce_lane_groups(value: &UOp) -> Option<(Vec<Vec<UOp>>, LaneMeta)> {
    if let Some((lanes, meta)) = lane_value(value) {
        let groups = lanes.into_iter().map(|lane| vec![lane]).collect();
        return Some((groups, meta));
    }
    contract_lane_groups(value)
}

/// Folds multiple lane values into one using a left-associative chain of
/// `reduce_op`. This is the unrolled "horizontal" reduction within a single
/// output lane, equivalent to tinygrad's cross-lane contraction.
fn horizontal_reduce(values: &[UOp], reduce_op: Op) -> UOp {
    let mut iter = values.iter().cloned();
    let Some(first) = iter.next() else {
        panic!("horizontal reduction requires at least one value");
    };
    iter.fold(first, |acc, value| {
        UOp::new(reduce_op, acc.dtype(), vec![acc, value], Arg::None)
    })
}

/// Returns the identity element for the given reduction operator (0 for Add,
/// -inf for Max). This initializes the accumulator so the first loop iteration
/// produces the correct result regardless of input values.
fn init_value(dtype: DType, device: crate::device::DeviceId, reduce_op: Op) -> UOp {
    match reduce_op {
        Op::Add => UOp::const_float(0.0, dtype, device),
        Op::Max => UOp::const_float(f64::NEG_INFINITY, dtype, device),
        _ => panic!("unsupported reduce op: {reduce_op:?}"),
    }
}

/// Builds the nested `End` chain that closes the reduction loop(s). When a
/// reduce has multiple range axes, each range gets its own End node, nested
/// innermost-first. The innermost End carries the accumulator assignments;
/// outer Ends close the remaining loops.
fn chain_reduce_ends(ranges: &[UOp], inner_effects: Vec<UOp>) -> UOp {
    if ranges.is_empty() {
        return UOp::sink(inner_effects);
    }

    let mut current = UOp::new(
        Op::End,
        DType::Void,
        std::iter::once(ranges[ranges.len() - 1].clone())
            .chain(inner_effects)
            .collect(),
        Arg::None,
    );
    for range in ranges[..ranges.len() - 1].iter().rev() {
        current = UOp::new(
            Op::End,
            DType::Void,
            vec![range.clone(), current],
            Arg::None,
        );
    }
    current
}

/// Collect the non-reduce Range nodes reachable from a Reduce's value subtree.
/// These are the enclosing output loops that the accumulator must be scoped
/// within — without this, the accumulator would be declared once globally
/// instead of once per outer loop iteration.
fn collect_outer_ranges(value: &UOp, reduce_ranges: &[UOp]) -> Vec<UOp> {
    let reduce_set: HashSet<&UOp> = reduce_ranges.iter().collect();
    value
        .toposort()
        .into_iter()
        .filter(|n| n.op() == Op::Range && !reduce_set.contains(n))
        .collect()
}

/// Lowers a high-level `Reduce` node into explicit accumulator-based control
/// flow: `DefineAcc` + loop body with `Assign` + `End` + `After`.
///
/// Ordering is expressed directly in the graph via `After` edges on each
/// `DefineAcc`, connecting it to enclosing non-reduce ranges. This lets the
/// linearizer place accumulators correctly without accumulator-specific logic.
fn lower_reduce(node: &UOp) -> Option<UOp> {
    if node.op() != Op::Reduce || node.srcs().len() < 2 {
        return None;
    }

    let Arg::Reduce(reduce_op, _) = node.arg() else {
        return None;
    };
    let value = &node.srcs()[0];
    let reduce_ranges: Vec<UOp> = node.srcs()[1..].to_vec();
    let outer_ranges = collect_outer_ranges(value, &reduce_ranges);
    let lane_groups = reduce_lane_groups(value);
    let (groups, meta) = lane_groups.unwrap_or_else(|| (vec![vec![value.clone()]], Box::default()));
    let init = init_value(node.dtype(), node.device(), *reduce_op);

    let mut accs = Vec::with_capacity(groups.len());
    let mut assigns = Vec::with_capacity(groups.len());
    for (idx, group) in groups.iter().enumerate() {
        let tag = u64::try_from(idx).expect("lane index should fit u64") + 1;
        // DefineAcc takes the init value plus any enclosing non-reduce ranges
        // as sources. This bakes scope into the node itself so the linearizer
        // places the declaration inside the right loop — no accumulator-specific
        // edge injection needed.
        let mut acc_srcs = vec![init.clone()];
        acc_srcs.extend(outer_ranges.iter().cloned());
        let acc = UOp::new_tagged(
            Op::DefineAcc,
            node.dtype(),
            acc_srcs,
            Arg::None,
            tag << 32,
        );
        let reduced_group = horizontal_reduce(group, *reduce_op);
        let updated = UOp::new(
            *reduce_op,
            node.dtype(),
            vec![acc.clone(), reduced_group],
            Arg::None,
        );
        let assign = UOp::new_tagged(
            Op::Assign,
            node.dtype(),
            vec![acc.clone(), updated],
            Arg::None,
            (tag << 32) | 1,
        );
        accs.push(acc);
        assigns.push(assign);
    }

    let reduce_end = chain_reduce_ends(&reduce_ranges, assigns);
    let afters: Vec<UOp> = accs
        .into_iter()
        .map(|acc| UOp::after(acc, reduce_end.clone()))
        .collect();

    if meta.is_empty() {
        return Some(
            afters
                .into_iter()
                .next()
                .expect("scalar reduction should produce one accumulator"),
        );
    }

    Some(make_lane_pack(Op::Unroll, node.dtype(), afters, meta))
}

/// Splits a store whose index or value carries lanes into one scalar store per
/// lane. After expansion, a single Store may write multiple output positions
/// (one per unrolled lane). This pass makes each write explicit so the
/// linearizer only sees simple scalar stores.
fn scalarize_lane_store(node: &UOp) -> Option<UOp> {
    if node.op() != Op::Store || node.srcs().len() != 2 {
        return None;
    }

    let idx_lanes = lane_value(&node.srcs()[0]);
    let value_lanes = lane_value(&node.srcs()[1]);
    if idx_lanes.is_none() && value_lanes.is_none() {
        return None;
    }

    let meta = idx_lanes
        .as_ref()
        .map(|(_, meta)| meta.clone())
        .or_else(|| value_lanes.as_ref().map(|(_, meta)| meta.clone()))
        .expect("at least one store input should carry lanes");
    if let (Some((_, idx_meta)), Some((_, value_meta))) = (&idx_lanes, &value_lanes) {
        if idx_meta.as_ref() != value_meta.as_ref() {
            return None;
        }
    }
    let lane_count = meta.iter().map(|(_, size)| *size).product::<usize>();

    let idx_values = idx_lanes.map_or_else(|| vec![node.srcs()[0].clone()], |(lanes, _)| lanes);
    let value_values = value_lanes.map_or_else(|| vec![node.srcs()[1].clone()], |(lanes, _)| lanes);

    let stores = (0..lane_count)
        .map(|lane_idx| {
            let idx = if idx_values.len() == 1 {
                idx_values[0].clone()
            } else {
                idx_values[lane_idx].clone()
            };
            let value = if value_values.len() == 1 {
                value_values[0].clone()
            } else {
                value_values[lane_idx].clone()
            };
            UOp::new(Op::Store, DType::Void, vec![idx, value], Arg::None)
        })
        .collect();
    Some(UOp::sink(stores))
}

/// Checks whether all `After` nodes in a subgraph point to the same effect
/// (End node). If so, returns that shared effect. This is a precondition for
/// lifting: we can only hoist After to the store level when there is a single
/// consistent barrier.
fn single_after_effect(root: &UOp) -> Option<UOp> {
    let mut effect = None;
    for node in root.toposort() {
        if node.op() != Op::After || node.srcs().len() != 2 {
            continue;
        }
        let end = node.srcs()[1].clone();
        match &effect {
            Some(existing) if existing != &end => return None,
            Some(_) => {}
            None => effect = Some(end),
        }
    }
    effect
}

/// Removes all `After` nodes referencing a specific effect from a subgraph,
/// replacing each `After(value, effect)` with just `value`. This is the
/// inverse of wrapping -- it strips the barrier so it can be re-attached at
/// a higher level by `lift_after_in_store`.
fn strip_after_effect(root: &UOp, effect: &UOp) -> UOp {
    let order = root.toposort();
    let mut rewritten = HashMap::new();

    for node in &order {
        let new_node =
            if node.op() == Op::After && node.srcs().len() == 2 && &node.srcs()[1] == effect {
                rewritten
                    .get(&node.srcs()[0])
                    .cloned()
                    .unwrap_or_else(|| node.srcs()[0].clone())
            } else {
                let new_srcs: Vec<UOp> = node
                    .srcs()
                    .iter()
                    .map(|src| rewritten.get(src).cloned().unwrap_or_else(|| src.clone()))
                    .collect();
                if new_srcs == node.srcs() {
                    node.clone()
                } else {
                    UOp::new(node.op(), node.dtype(), new_srcs, node.arg().clone())
                }
            };
        rewritten.insert(node.clone(), new_node);
    }

    rewritten.get(root).cloned().unwrap_or_else(|| root.clone())
}

/// Lifts `After` barriers from deep inside a store's value subgraph up to the
/// store's direct inputs. Reduce lowering inserts After on each accumulator
/// read, but the linearizer needs the barrier at the store level to correctly
/// sequence the loop end before the final write.
fn lift_after_in_store(node: &UOp) -> Option<UOp> {
    if node.op() != Op::Store || node.srcs().len() != 2 {
        return None;
    }
    let idx = &node.srcs()[0];
    let value = &node.srcs()[1];
    let effect = single_after_effect(value)?;
    let stripped = strip_after_effect(value, &effect);
    let lifted_idx = if idx.op() == Op::After && idx.srcs().len() == 2 && idx.srcs()[1] == effect {
        idx.clone()
    } else {
        UOp::after(idx.clone(), effect.clone())
    };
    let lifted_value = UOp::after(stripped, effect);
    if &lifted_idx == idx && &lifted_value == value {
        return None;
    }
    Some(UOp::new(
        Op::Store,
        DType::Void,
        vec![lifted_idx, lifted_value],
        Arg::None,
    ))
}

/// Lower late lane-carrying IR back to scalar control flow for codegen.
#[must_use]
pub(crate) fn devectorize(root: &UOp) -> UOp {
    let reduced = graph_rewrite(root, &mut lower_reduce);
    let expanded_lanes = graph_rewrite(&reduced, &mut expand_lane_expr);
    let scalarized_stores = graph_rewrite(&expanded_lanes, &mut scalarize_lane_store);
    let flattened_ends = graph_rewrite(&scalarized_stores, &mut flatten_end_bodies);
    let lifted_afters = graph_rewrite(&flattened_ends, &mut lift_after_in_store);
    graph_rewrite(&lifted_afters, &mut flatten_nested_sinks)
}

#[cfg(test)]
mod tests {
    use super::devectorize;
    use crate::expand::late_expand;
    use crate::optimize::{unroll_reduce_loops, upcast_loops};
    use crate::schedule::rangeify::rangeify;
    use crate::shape::Shape;
    use crate::uop::{Arg, AxisKind, Op, UOp};

    fn build_reduce_graph(n: i64) -> UOp {
        let device = crate::device::DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, crate::dtype::DType::F32, 1, device);
        let numel = usize::try_from(n).expect("n should fit usize");
        let input_ptr = UOp::param_buffer(1, crate::dtype::DType::F32, numel, device);
        let reduced =
            UOp::reduce_axis(UOp::reshape(input_ptr, Shape::from([numel])), Op::Add, &[0]);
        UOp::sink(vec![UOp::store(out_ptr, reduced)])
    }

    fn build_row_reduce_graph(rows: usize, cols: usize) -> UOp {
        let device = crate::device::DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, crate::dtype::DType::F32, rows, device);
        let input_ptr = UOp::param_buffer(1, crate::dtype::DType::F32, rows * cols, device);
        let reduced = UOp::reduce_axis(
            UOp::reshape(input_ptr, Shape::from([rows, cols])),
            Op::Add,
            &[1],
        );
        UOp::sink(vec![UOp::store(out_ptr, reduced)])
    }

    #[test]
    fn test_devectorize_lowers_scalar_reduce_to_accumulator_loop() {
        let rangeified = rangeify(&build_reduce_graph(8));
        let expanded = late_expand(&rangeified);
        let lowered = devectorize(&expanded);
        let order = lowered.toposort();

        assert!(order.iter().any(|node| node.op() == Op::DefineAcc));
        assert!(order.iter().any(|node| node.op() == Op::Assign));
        assert!(!order.iter().any(|node| node.op() == Op::Reduce));
    }

    #[test]
    fn test_devectorize_lowers_upcast_unroll_reduce_to_multiple_accumulators() {
        let rangeified = rangeify(&build_row_reduce_graph(8, 8));
        let optimized = unroll_reduce_loops(&upcast_loops(&rangeified));
        let expanded = late_expand(&optimized);
        let lowered = devectorize(&expanded);
        let order = lowered.toposort();

        assert!(!order
            .iter()
            .any(|node| matches!(node.op(), Op::Reduce | Op::Contract)));
        assert!(!order
            .iter()
            .any(|node| matches!(node.arg(), Arg::Range(_, AxisKind::Unroll))));
        assert_eq!(
            order
                .iter()
                .filter(|node| node.op() == Op::DefineAcc)
                .count(),
            4
        );
    }
}
