//! # Rangeify — from tensor expressions to executable loops
//!
//! This is where abstract tensor math becomes concrete loops over memory.
//! The input is a `Sink(Store(Param, expr))` — the scheduled proto-kernel
//! where all Buffers have been converted to Params. The output is kernel-level
//! IR with Range loops, Loads, Stores, and accumulator patterns.
//!
//! All transformations are rewrite rules in a single `graph_rewrite` pass.
//! This is the same design as tinygrad's `schedule/rangeify.py`: rather than
//! a procedural lowering function, we define declarative rules and let the
//! fixed-point engine apply them until the graph stabilizes.
//!
//! ## The three rules
//!
//! 1. **Store → ranged Store**: creates output loops and injects Index
//! 2. **Index pushing**: delegates to [`super::indexing`] rules to push
//!    Index down through ALU/movement/reduce/param nodes
//! 3. **Reduce expansion**: converts Reduce into an accumulator loop pattern

use crate::dtype::DType;
use crate::rewrite::graph_rewrite;
use crate::uop::{Arg, Op, UOp};

use super::indexing::{
    contiguous_strides, flat_index, index_wrap, rewrite_index_alu, rewrite_index_const,
    rewrite_index_movement, rewrite_index_param, rewrite_index_reduce,
};

// ── Helpers ───────────────────────────────────────────────────────────────

/// Wrap `body` in nested End nodes, one per range (innermost range first).
///
/// Returns `body` unchanged if `ranges` is empty.
fn chain_ends(ranges: &[UOp], body: &UOp) -> UOp {
    let mut current = body.clone();
    for range in ranges.iter().rev() {
        current = UOp::new(Op::End, DType::Void, vec![range.clone(), current], Arg::None);
    }
    current
}

/// Compute the output `Index` node for a store destination that may be wrapped
/// in movement ops (e.g. `Shrink(Buffer, ...)`). Recursively pushes indices
/// through movements until it reaches the underlying `ParamBuffer` or `Buffer`,
/// then builds the flat `Index` for the store target. This handles in-place
/// assignment into a sub-region of an existing buffer.
fn store_output_index(dest: &UOp, idxs: &[UOp]) -> Option<UOp> {
    match dest.op() {
        Op::ParamBuffer => {
            assert_eq!(
                idxs.len(),
                1,
                "store destination must be flattened before reaching ParamBuffer"
            );
            Some(UOp::new(
                Op::Index,
                dest.dtype(),
                vec![dest.clone(), idxs[0].clone()],
                Arg::Index(0),
            ))
        }
        Op::Buffer => {
            assert_eq!(
                idxs.len(),
                1,
                "store destination must be flattened before reaching Buffer"
            );
            Some(UOp::new(
                Op::Index,
                dest.dtype(),
                vec![dest.clone(), idxs[0].clone()],
                Arg::Index(0),
            ))
        }
        op if op.is_movement() => {
            let moved = rewrite_index_movement(dest, idxs)?;
            store_output_index(&moved.srcs()[0], &moved.srcs()[1..])
        }
        _ => None,
    }
}

// ── Rewrite rules ─────────────────────────────────────────────────────────

/// **Store rule**: `Store(Param, expr)` → ranged Store with loops.
///
/// This is the entry point for lowering: it sees a Store whose source is still
/// a tensor-level expression (with a shape), and creates the loop nest that
/// will iterate over every element of the output.
///
/// Steps:
/// 1. Create a Range node for each dimension of the expression's shape.
///    These become the `for` loops in the generated C code.
/// 2. Wrap the expression in `Index(expr, ranges...)` to start the index
///    pushing process. Later rules will push this Index all the way down.
/// 3. Compute a flat output offset from the ranges (skipping size-1 dims,
///    which come from reductions and don't contribute to output size).
/// 4. Wrap everything in End nodes to close each Range loop.
///
/// The guard `expr.shape()?` ensures this rule only fires once: after the
/// Store is rangeified, its source is an Index (no shape), so the rule
/// won't match again.
fn rewrite_store_add_ranges(store: &UOp) -> Option<UOp> {
    if store.srcs().len() != 2 {
        return None;
    }
    let out_param = &store.srcs()[0];
    let expr = &store.srcs()[1];
    let shape = expr.shape()?;
    let device = expr.device();

    let axis_indices: Vec<UOp> = shape
        .iter()
        .enumerate()
        .map(|(axis, &size)| {
            if size == 1 {
                return UOp::const_int(0, DType::I32, device);
            }
            #[allow(clippy::cast_possible_wrap)]
            UOp::new(
                Op::Range,
                DType::I32,
                vec![UOp::const_int(size as i64, DType::I32, device)],
                Arg::Index(axis),
            )
        })
        .collect();

    let indexed_expr = index_wrap(expr, &axis_indices);

    // Size-1 dims come from reductions (e.g. sum(axis=0) on [3,4] → [1,4]).
    // They don't contribute to the output buffer size, so skip them for the
    // output flat index. The Range still exists to carry the full shape into
    // Index pushing.
    let out_ranges: Vec<UOp> = axis_indices
        .iter()
        .filter(|idx| idx.op() == Op::Range)
        .cloned()
        .collect();
    let squeezed: Vec<usize> = shape.iter().copied().filter(|&s| s != 1).collect();
    let out_flat = if out_ranges.is_empty() {
        UOp::const_int(0, DType::I32, device)
    } else {
        flat_index(&out_ranges, &contiguous_strides(&squeezed))
    };
    let out_idx = if out_param.op() == Op::ParamBuffer {
        UOp::new(
            Op::Index,
            out_param.dtype(),
            vec![out_param.clone(), out_flat],
            Arg::Index(0),
        )
    } else {
        store_output_index(out_param, &axis_indices)?
    };
    let new_store = UOp::new(Op::Store, DType::Void, vec![out_idx, indexed_expr], Arg::None);

    Some(chain_ends(&out_ranges, &new_store))
}

/// **Reduce expansion rule**: `Reduce(value, ranges...)` → accumulator loop.
///
/// A Reduce node (created by `rewrite_index_reduce`) says "sum/max this value
/// over these ranges", but codegen can't render it directly — it needs an
/// explicit accumulator variable and assignment. This rule lowers Reduce into:
///
/// ```text
/// After(
///   DefineAcc(init),           // declare accumulator (e.g. acc = 0.0)
///   End(range,                 // for r in 0..N:
///     Assign(acc, acc + value) //   acc = acc + value
///   )
/// )
/// ```
///
/// The After node is an ordering barrier: it tells the toposorter "the value
/// of this node is `DefineAcc` (the accumulator), but don't use it until the
/// End (the loop) has completed." Without After, the Store could read the
/// accumulator before the reduce loop finishes.
fn expand_reduce(reduce: &UOp) -> UOp {
    let Arg::Reduce(reduce_op, _) = reduce.arg() else {
        panic!("Reduce must have Arg::Reduce");
    };
    let value = &reduce.srcs()[0];
    let reduce_ranges: Vec<UOp> = reduce.srcs()[1..].to_vec();
    let dtype = reduce.dtype();
    let device = reduce.device();

    let init_val = match reduce_op {
        Op::Add => UOp::const_float(0.0, dtype, device),
        Op::Max => UOp::const_float(f64::NEG_INFINITY, dtype, device),
        _ => panic!("unsupported reduce op: {reduce_op:?}"),
    };

    let define_acc = UOp::new(Op::DefineAcc, dtype, vec![init_val], Arg::None);
    let accumulated = UOp::new(*reduce_op, dtype, vec![define_acc.clone(), value.clone()], Arg::None);
    let assign = UOp::new(Op::Assign, dtype, vec![define_acc.clone(), accumulated], Arg::None);

    UOp::new(
        Op::After,
        dtype,
        vec![define_acc, chain_ends(&reduce_ranges, &assign)],
        Arg::None,
    )
}

// ── Combined rule dispatch ───────────────────────────────────────────────

/// Single dispatch function for all rangeify rewrites. The graph rewriter calls
/// this on every node until no more rules fire (fixed-point). The three cases:
/// `Store` creates the loop nest, `Index` pushes indexing toward leaves, and
/// `Reduce` expands into an accumulator pattern.
fn rangeify_rule(node: &UOp) -> Option<UOp> {
    match node.op() {
        Op::Store => rewrite_store_add_ranges(node),
        // Index pushing: each arm handles one kind of inner node.
        // Skips kernel-level Index (tagged with Arg::Index) to avoid infinite loops.
        Op::Index if *node.arg() == Arg::None => {
            let inner = &node.srcs()[0];
            let idxs = &node.srcs()[1..];
            match inner.op() {
                op if op.is_alu() => rewrite_index_alu(inner, idxs),
                Op::Expand | Op::Permute | Op::Reshape | Op::Shrink => {
                    rewrite_index_movement(inner, idxs)
                }
                Op::ReduceAxis => rewrite_index_reduce(inner, idxs),
                Op::Const | Op::ParamScalar => rewrite_index_const(inner, idxs),
                Op::ParamBuffer => rewrite_index_param(inner, idxs),
                _ => None,
            }
        }
        Op::Reduce => Some(expand_reduce(node)),
        _ => None,
    }
}

// ── Entry point ───────────────────────────────────────────────────────────

/// Apply rangeify rewrites to a scheduled kernel graph.
///
/// Input: `Sink(Store(Param, expr))` where all Buffers are already Params.
/// Output: kernel-level Sink with Ranges, Loads, Stores, and accumulator loops.
#[must_use]
pub fn rangeify(sink: &UOp) -> UOp {
    graph_rewrite(sink, &rangeify_rule, "rangeify")
}
