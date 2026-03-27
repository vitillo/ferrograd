//! # Rangeify — from tensor expressions to executable loops
//!
//! This is where abstract tensor math becomes concrete loops over memory.
//! The input is a `Sink(Store(Param, expr))` — the scheduled proto-kernel
//! where all Buffers have been converted to Params. The output is kernel-level
//! IR with Range loops, Loads, Stores, and explicit kernel `Reduce` nodes.
//!
//! All transformations are rewrite rules in a single `graph_rewrite` pass.
//! This is the same design as tinygrad's `schedule/rangeify.py`: rather than
//! a procedural lowering function, we define declarative rules and let the
//! fixed-point engine apply them until the graph stabilizes.
//!
//! ## The two rules
//!
//! 1. **Store → ranged Store**: creates output loops and injects Index
//! 2. **Index pushing**: delegates to [`super::indexing`] rules to push
//!    Index down through ALU/movement/reduce/param nodes
//!
//! Tensor reductions are not lowered to accumulators here anymore. Like
//! tinygrad, rangeify only introduces kernel-level `Reduce` nodes; later late
//! passes decide how to expand lanes and only then lower reductions to
//! accumulators.

use crate::device::DeviceId;
use crate::dtype::DType;
use crate::rewrite::graph_rewrite;
use crate::uop::{Arg, AxisKind, Op, UOp};

use super::indexing::{
    contiguous_strides, flat_index, index_wrap, rewrite_index_alu, rewrite_index_const,
    rewrite_index_movement, rewrite_index_param, rewrite_index_reduce,
};

// ── Helpers ───────────────────────────────────────────────────────────────

/// Create a loop index for one axis, mirroring tinygrad's `new_range()`.
///
/// Size-1 dimensions (from reductions like `sum(axis=0)` on shape `[3,4]` →
/// `[1,4]`) always index at 0 — there's nothing to iterate. Rather than
/// emitting a trivial `Range(0..1)` loop, we return `const(0)` directly.
/// This avoids a useless loop and lets downstream code distinguish real
/// loops (Range ops) from collapsed dims when building End nodes.
#[allow(clippy::cast_possible_wrap)]
fn new_range(axis: usize, size: usize, device: DeviceId) -> UOp {
    if size == 1 {
        return UOp::const_int(0, DType::I32, device);
    }
    UOp::new(
        Op::Range,
        DType::I32,
        vec![UOp::const_int(size as i64, DType::I32, device)],
        Arg::Range(axis, AxisKind::Loop),
    )
}

/// Wrap `body` in nested End nodes, one per range (innermost range first).
///
/// Returns `body` unchanged if `ranges` is empty.
fn chain_ends(ranges: &[UOp], body: &UOp) -> UOp {
    let mut current = body.clone();
    for range in ranges.iter().rev() {
        current = UOp::new(
            Op::End,
            DType::Void,
            vec![range.clone(), current],
            Arg::None,
        );
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
        Op::ParamBuffer | Op::Buffer => {
            assert_eq!(
                idxs.len(),
                1,
                "store destination must be flattened before reaching ParamBuffer/Buffer"
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
/// 3. Compute a flat output offset from the full-rank indices.
///    Singleton axes are already `0`, so stride math naturally drops them.
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
        .map(|(axis, &size)| new_range(axis, size, device))
        .collect();

    let indexed_expr = index_wrap(expr, &axis_indices);

    // Flatten the full-rank output coordinate. Singleton axes are constants,
    // so symbolic cleanup will fold terms like `0 * stride` away.
    let out_flat = flat_index(&axis_indices, &contiguous_strides(shape.as_slice()));
    let out_ranges: Vec<UOp> = axis_indices
        .iter()
        .filter(|idx| idx.op() == Op::Range)
        .cloned()
        .collect();
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
    let new_store = UOp::new(
        Op::Store,
        DType::Void,
        vec![out_idx, indexed_expr],
        Arg::None,
    );

    Some(chain_ends(&out_ranges, &new_store))
}

// ── Combined rule dispatch ───────────────────────────────────────────────

/// Single dispatch function for all rangeify rewrites. The graph rewriter calls
/// this on every node until no more rules fire (fixed-point). The two cases:
/// `Store` creates the output loop nest, and `Index` pushes indexing toward
/// leaves while introducing kernel-level `Reduce` nodes for tensor reductions.
///
/// Tinygrad also keeps kernel-level `REDUCE` nodes alive after rangeify so
/// later scheduling and expansion passes can still reason about reduction
/// structure before lowering to accumulators.
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
                Op::Expand | Op::Permute | Op::Reshape | Op::Shrink | Op::Contiguous => {
                    rewrite_index_movement(inner, idxs)
                }
                Op::ReduceAxis => rewrite_index_reduce(inner, idxs),
                Op::Const | Op::ParamScalar => rewrite_index_const(inner, idxs),
                Op::ParamBuffer => rewrite_index_param(inner, idxs),
                _ => None,
            }
        }
        _ => None,
    }
}

// ── Entry point ───────────────────────────────────────────────────────────

/// Apply rangeify rewrites to a scheduled kernel graph.
///
/// Input: `Sink(Store(Param, expr))` where all Buffers are already Params.
/// Output: kernel-level Sink with Ranges, Loads, Stores, and kernel `Reduce`
/// nodes ready for later optimization and late lowering.
#[must_use]
pub fn rangeify(sink: &UOp) -> UOp {
    graph_rewrite(sink, &mut rangeify_rule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceId;
    use crate::optimize::symbolic_simple;
    use crate::shape::Shape;

    #[test]
    fn test_singleton_output_axis_flattens_from_full_rank_indices() {
        // Arrange
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 3, device);
        let input = UOp::param_buffer(1, DType::F32, 6, device);
        let reshaped = UOp::reshape(input, Shape::from([2, 3]));
        let narrowed = UOp::new(
            Op::Shrink,
            DType::F32,
            vec![
                reshaped,
                UOp::const_int(1, DType::I32, device),
                UOp::const_int(0, DType::I32, device),
            ],
            Arg::Bounds(Box::from([1, 3])),
        );
        let store = UOp::new(Op::Store, DType::Void, vec![out, narrowed], Arg::None);
        let sink = UOp::sink(vec![store]);

        // Act
        let rangeified = rangeify(&sink);
        let simplified = graph_rewrite(&rangeified, &mut symbolic_simple);
        let order = simplified.toposort();

        // Assert
        let ranges: Vec<UOp> = order
            .iter()
            .filter(|node| node.op() == Op::Range)
            .cloned()
            .collect();
        assert_eq!(
            ranges.len(),
            1,
            "singleton output axis should not emit a loop"
        );

        let store = order
            .iter()
            .find(|node| node.op() == Op::Store)
            .expect("rangeified graph should contain a store");
        let out_idx = &store.srcs()[0];
        assert_eq!(out_idx.op(), Op::Index);
        assert_eq!(
            out_idx.srcs()[1],
            ranges[0],
            "output address should simplify to the live loop index",
        );
    }

    #[test]
    fn test_rangeify_marks_output_and_reduce_axes() {
        // Arrange
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 2, device);
        let input = UOp::param_buffer(1, DType::F32, 6, device);
        let matrix = UOp::reshape(input, Shape::from([2, 3]));
        let reduced = UOp::reduce_axis(matrix, Op::Add, &[1]);
        let store = UOp::new(Op::Store, DType::Void, vec![out, reduced], Arg::None);
        let sink = UOp::sink(vec![store]);

        // Act
        let rangeified = rangeify(&sink);
        let ranges: Vec<UOp> = rangeified
            .toposort()
            .iter()
            .filter(|node| node.op() == Op::Range)
            .cloned()
            .collect();

        // Assert
        assert_eq!(ranges.len(), 2);
        assert!(ranges
            .iter()
            .any(|range| range.arg() == &Arg::Range(0, AxisKind::Loop)));
        assert!(ranges
            .iter()
            .any(|range| range.arg() == &Arg::Range(1, AxisKind::Reduce)));
    }
}
