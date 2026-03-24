//! # Rangeify — adds ranges (loops) to a scheduled kernel graph
//!
//! Takes a `Sink(Store(Param, expr))` from the scheduler (where all Buffers
//! have already been converted to Params) and adds Range loops, pushes Index
//! down through the graph, and expands Reduce nodes.
//!
//! All transformations are rewrite rules in a single `graph_rewrite` pass.
//!
//! Mirrors tinygrad's `schedule/rangeify.py`.

use crate::dtype::DType;
use crate::rewrite::{graph_rewrite, Captures, PatternMatcher, RewriteFn, UPat};
use crate::uop::{Arg, Op, UOp};

use super::indexing::{
    contiguous_strides, flat_index, index_wrap, rewrite_index_alu, rewrite_index_movement,
    rewrite_index_param, rewrite_index_reduce,
};

// ── Store → ranged Store (like tinygrad's add_ranges_to_store) ────────────

/// Rule: `Store(Param, expr)` → create ranges, index both sides, add End nodes.
fn rewrite_store_add_ranges(store: &UOp) -> Option<UOp> {
    if store.srcs().len() != 2 {
        return None;
    }
    let out_param = &store.srcs()[0];
    let expr = &store.srcs()[1];
    // Only match un-rangeified stores: expr still has a tensor-level shape.
    let shape = expr.shape()?;

    let ranges: Vec<UOp> = shape
        .iter()
        .enumerate()
        .map(|(axis, &size)| {
            #[allow(clippy::cast_possible_wrap)]
            UOp::range(axis, UOp::const_int(size as i64, DType::I32))
        })
        .collect();

    let indexed_expr = index_wrap(expr, &ranges);

    // Flat output index (skip size-1 dims from reductions).
    let out_ranges: Vec<UOp> = ranges
        .iter()
        .zip(&shape)
        .filter(|(_, &s)| s != 1)
        .map(|(r, _)| r.clone())
        .collect();
    let squeezed: Vec<usize> = shape.iter().copied().filter(|&s| s != 1).collect();
    let out_flat = if out_ranges.is_empty() {
        UOp::const_int(0, DType::I32)
    } else {
        flat_index(&out_ranges, &contiguous_strides(&squeezed))
    };
    // Kernel-level Index for output (tagged to prevent rewrite_index_param from converting to Load).
    let out_idx = UOp::new(Op::Index, out_param.dtype(), vec![out_param.clone(), out_flat], Arg::Index(0));
    let new_store = UOp::store(out_idx, indexed_expr);

    // Chain End nodes (innermost first).
    let mut ends: Vec<UOp> = Vec::new();
    for r in out_ranges.iter().rev() {
        let mut end_srcs = vec![r.clone()];
        end_srcs.push(ends.last().unwrap_or(&new_store).clone());
        ends.push(UOp::new(Op::End, DType::Void, end_srcs, Arg::None));
    }

    Some(ends.last().unwrap_or(&new_store).clone())
}

// ── Pattern matcher ───────────────────────────────────────────────────────

/// Build all rangeify rewrite rules.
fn build_rules() -> PatternMatcher {
    PatternMatcher::new(vec![
        // Rule 1: Store → ranged Store with loops.
        (
            UPat {
                op: Some(vec![Op::Store]),
                name: Some("store".into()),
                arg: None,
                src: None,
                commutative: false,
            },
            Box::new(|caps: &Captures| {
                let store = caps.get("store");
                rewrite_store_add_ranges(&store)
            }) as RewriteFn,
        ),
        // Rule 2: Push Index down through the graph.
        (
            UPat {
                op: Some(vec![Op::Index]),
                name: Some("idx".into()),
                arg: None,
                src: None,
                commutative: false,
            },
            Box::new(|caps: &Captures| {
                let idx_node = caps.get("idx");
                // Skip kernel-level Index nodes (tagged by rewrite_index_param).
                if *idx_node.arg() != Arg::None {
                    return None;
                }
                let inner = idx_node.srcs()[0].clone();
                let idxs: Vec<UOp> = idx_node.srcs()[1..].to_vec();

                None.or_else(|| rewrite_index_alu(&inner, &idxs))
                    .or_else(|| rewrite_index_movement(&inner, &idxs))
                    .or_else(|| rewrite_index_reduce(&inner, &idxs))
                    .or_else(|| rewrite_index_param(&inner, &idxs))
            }) as RewriteFn,
        ),
        // Rule 3: Expand Reduce → After(DefineAcc, End(Range, Assign)).
        (
            UPat {
                op: Some(vec![Op::Reduce]),
                name: Some("r".into()),
                arg: None,
                src: None,
                commutative: false,
            },
            Box::new(|caps: &Captures| {
                let r = caps.get("r");
                Some(expand_reduce(&r))
            }) as RewriteFn,
        ),
    ])
}

/// Expand a single Reduce node into accumulator + loop + After.
fn expand_reduce(reduce: &UOp) -> UOp {
    let Arg::Reduce(reduce_op, _) = reduce.arg() else {
        panic!("Reduce must have Arg::Reduce");
    };
    let value = &reduce.srcs()[0];
    let reduce_ranges: Vec<UOp> = reduce.srcs()[1..].to_vec();
    let dtype = reduce.dtype();

    let init_val = match reduce_op {
        Op::Add => UOp::const_float(0.0, dtype),
        Op::Max => UOp::const_float(f64::NEG_INFINITY, dtype),
        _ => panic!("unsupported reduce op: {reduce_op:?}"),
    };

    let define_acc = UOp::new(Op::DefineAcc, dtype, vec![init_val], Arg::None);
    let accumulated = UOp::new(
        *reduce_op,
        dtype,
        vec![define_acc.clone(), value.clone()],
        Arg::None,
    );
    let assign = UOp::new(
        Op::Assign,
        dtype,
        vec![define_acc.clone(), accumulated],
        Arg::None,
    );

    let mut last_end: Option<UOp> = None;
    for range in reduce_ranges.iter().rev() {
        let mut end_srcs = vec![range.clone()];
        end_srcs.push(last_end.as_ref().unwrap_or(&assign).clone());
        last_end = Some(UOp::new(Op::End, DType::Void, end_srcs, Arg::None));
    }

    UOp::new(
        Op::After,
        dtype,
        vec![define_acc, last_end.expect("need at least one range")],
        Arg::None,
    )
}

// ── Entry point ───────────────────────────────────────────────────────────

/// Apply rangeify rewrites to a scheduled kernel graph.
///
/// Input: `Sink(Store(Param, expr))` where all Buffers are already Params.
/// Output: kernel-level Sink with Ranges, Loads, Stores.
#[must_use]
pub fn rangeify(sink: &UOp) -> UOp {
    let pm = build_rules();
    graph_rewrite(sink, &pm, "rangeify")
}
