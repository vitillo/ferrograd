//! # Indexing — core index transformation logic
//!
//! This module implements the heart of how tensor operations become memory
//! accesses. The central idea: an `Index` node carries per-dimension index
//! expressions (one per axis of the tensor). It gets "pushed down" through
//! the expression tree by rewrite rules, and each op it passes through
//! transforms the index expressions to reflect how that op rearranges data.
//!
//! When Index finally reaches a `Param` (a kernel buffer pointer), all the
//! movement ops above have been consumed and the indices have been collapsed
//! into a single flat memory offset. This is the same approach tinygrad uses
//! in `schedule/indexing.py`.
//!
//! ## Why push Index down?
//!
//! Tensor ops like Reshape and Permute don't move data — they just change
//! how we *interpret* the indices into the underlying buffer. By pushing
//! Index down, each movement op gets a chance to adjust the indices, and
//! then disappears from the graph. What remains is just loads from flat
//! memory offsets — exactly what the C codegen needs.
//!
//! ## The four rules
//!
//! Each rule handles one kind of node that Index can land on:
//!
//! - [`rewrite_index_alu`] — math ops: push Index into each operand
//! - [`rewrite_index_movement`] — Reshape/Permute/Expand: transform the indices
//! - [`rewrite_index_reduce`] — `ReduceAxis`: create inner loop ranges for reduced axes
//! - [`rewrite_index_param`] — Param (leaf): convert to a flat Load — we're done

use crate::dtype::DType;
use crate::uop::{Arg, Op, UOp};

/// Offset added to reduce range IDs to avoid collision with output loop range IDs.
/// Output ranges use axis indices `0..ndim`, reduce ranges use `REDUCE_RANGE_OFFSET + axis`.
const REDUCE_RANGE_OFFSET: usize = 100;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Contiguous (row-major) strides for a shape.
///
/// For shape `[2, 3, 4]` returns `[12, 4, 1]` — the number of elements to
/// skip in a flat buffer when advancing one step along each dimension.
/// Used to convert multi-dimensional indices into a flat memory offset.
#[must_use]
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if !shape.is_empty() {
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Build a flat index `UOp` from per-dimension index `UOps` and strides.
///
/// Computes `sum(idx[i] * stride[i])` — the standard row-major offset formula.
/// Skips stride-0 dimensions (broadcast dims) and avoids `* 1` for the
/// innermost dimension.
///
/// # Panics
///
/// Panics if `idxs` and `strides` have different lengths.
#[must_use]
pub fn flat_index(idxs: &[UOp], strides: &[usize]) -> UOp {
    assert_eq!(idxs.len(), strides.len());
    let mut terms: Vec<UOp> = Vec::new();
    for (idx, &stride) in idxs.iter().zip(strides) {
        if stride == 0 {
            continue;
        }
        let term = if stride == 1 {
            idx.clone()
        } else {
            #[allow(clippy::cast_possible_wrap)]
            let s = UOp::const_int(stride as i64, DType::I32);
            UOp::new(Op::Mul, DType::I32, vec![idx.clone(), s], Arg::None)
        };
        terms.push(term);
    }
    if terms.is_empty() {
        return UOp::const_int(0, DType::I32);
    }
    let mut result = terms.remove(0);
    for term in terms {
        result = UOp::new(Op::Add, DType::I32, vec![result, term], Arg::None);
    }
    result
}

/// Wrap a source in an `Index` node with the given per-dimension indices.
#[must_use]
pub fn index_wrap(src: &UOp, idxs: &[UOp]) -> UOp {
    let mut srcs = vec![src.clone()];
    srcs.extend_from_slice(idxs);
    UOp::new(Op::Index, src.dtype(), srcs, Arg::None)
}

// ── Rewrite rules ─────────────────────────────────────────────────────────

/// **ALU rule**: `Index(ALU(a, b), [i, j])` → `ALU(Index(a, [i, j]), Index(b, [i, j]))`
///
/// Math ops like Add and Mul are element-wise — they don't change how data
/// is laid out in memory, so the same indices apply to every operand.
/// We just distribute Index into each source and let it keep sinking.
#[must_use]
pub fn rewrite_index_alu(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    if !inner.op().is_alu() {
        return None;
    }
    let new_srcs: Vec<UOp> = inner.srcs().iter().map(|s| index_wrap(s, idxs)).collect();
    Some(UOp::new(inner.op(), inner.dtype(), new_srcs, inner.arg().clone()))
}

/// **Movement rule**: `Index(Movement(src), [i, j])` → `Index(src, [transformed...])`
///
/// Movement ops don't move data — they reinterpret index→element mapping.
/// This rule absorbs each movement op by adjusting the index expressions,
/// then continues pushing Index into the source. After this rule fires,
/// the movement op is gone from the graph.
///
/// **Expand**: broadcasts a size-1 dim to a larger size. Since all elements
/// along a broadcast dim read from the same position, we replace that dim's
/// index with 0. Example: `Index(Expand([1,3]→[4,3]), [i,j])` → `Index(src, [0,j])`
///
/// **Permute**: reorders dimensions. We invert the permutation so the source
/// gets indices in its original axis order. Example: `Permute([1,0])` swaps
/// axes, so `[i,j]` becomes `[j,i]` for the source.
///
/// **Reshape**: changes the number/size of dimensions without moving data.
/// Two cases:
/// - *Simple* (only inserting/removing size-1 dims): map non-1 indices through.
///   Example: `[6]→[1,6]` just inserts a 0-index for the new dim.
/// - *General* (actually merging/splitting dims): flatten the new indices into
///   a single offset and let the source's shape decompose it. This is correct
///   because both shapes describe the same contiguous memory layout.
#[must_use]
pub fn rewrite_index_movement(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    let Arg::Dims(ref arg) = inner.arg() else { return None };
    let src = &inner.srcs()[0];

    let new_idxs = match inner.op() {
        Op::Expand => {
            let src_shape = src.shape()?;
            idxs.iter()
                .enumerate()
                .map(|(i, idx)| {
                    if src_shape[i] == 1 && arg[i] > 1 {
                        UOp::const_int(0, DType::I32)
                    } else {
                        idx.clone()
                    }
                })
                .collect()
        }
        Op::Permute => {
            let mut inv = vec![UOp::const_int(0, DType::I32); arg.len()];
            for (new_pos, &old_pos) in arg.iter().enumerate() {
                inv[old_pos] = idxs[new_pos].clone();
            }
            inv
        }
        Op::Reshape => {
            let src_shape = src.shape()?;
            let src_non1: Vec<usize> = src_shape.iter().copied().filter(|&s| s != 1).collect();
            let new_non1: Vec<usize> = arg.iter().copied().filter(|&s| s != 1).collect();

            if src_non1 == new_non1 {
                let non1_idxs: Vec<UOp> = idxs.iter().zip(arg)
                    .filter(|(_, &s)| s != 1)
                    .map(|(idx, _)| idx.clone())
                    .collect();
                let mut result = Vec::with_capacity(src_shape.len());
                let mut j = 0;
                for &s in &src_shape {
                    if s == 1 {
                        result.push(UOp::const_int(0, DType::I32));
                    } else {
                        result.push(non1_idxs[j].clone());
                        j += 1;
                    }
                }
                result
            } else {
                vec![flat_index(idxs, &contiguous_strides(arg))]
            }
        }
        _ => return None,
    };

    Some(index_wrap(src, &new_idxs))
}

/// **Reduce rule**: `Index(ReduceAxis(src, axes), [i, j])` → `Reduce(Index(src, [i, r0]), [r0])`
///
/// A reduction like `sum(axis=1)` on a `[2,3]` tensor produces a `[2,1]` result.
/// The output needs one loop variable (`i` over dim 0), but the *input* also
/// needs a loop over the reduced dim (`r0` over dim 1). This rule creates
/// fresh Range nodes for the reduced axes and splices them into the index
/// expressions, so the source gets its full set of indices.
///
/// The result is a `Reduce` node whose sources are: the indexed expression,
/// followed by the reduce Range nodes. A later rule (`expand_reduce`) will
/// lower this into DefineAcc/Assign/End — the actual accumulator loop.
///
/// Reduce ranges use IDs offset by [`REDUCE_RANGE_OFFSET`] to avoid
/// colliding with the output loop ranges created by `rewrite_store_add_ranges`.
///
/// # Panics
///
/// Panics if `idxs` length doesn't match the source shape.
#[must_use]
pub fn rewrite_index_reduce(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    if inner.op() != Op::ReduceAxis { return None; }
    let Arg::Reduce(reduce_op, axes) = inner.arg() else { return None };
    let src = &inner.srcs()[0];
    let src_shape = src.shape()?;

    assert_eq!(idxs.len(), src_shape.len(), "idxs should match source ndim");
    let mut full_idxs = Vec::with_capacity(src_shape.len());
    let mut reduce_ranges = Vec::new();
    for (i, &size) in src_shape.iter().enumerate() {
        if axes.contains(&i) {
            #[allow(clippy::cast_possible_wrap)]
            let range = UOp::range(REDUCE_RANGE_OFFSET + i, UOp::const_int(size as i64, DType::I32));
            full_idxs.push(range.clone());
            reduce_ranges.push(range);
        } else {
            full_idxs.push(idxs[i].clone());
        }
    }

    let indexed_src = index_wrap(src, &full_idxs);
    let mut reduce_srcs = vec![indexed_src];
    reduce_srcs.extend(reduce_ranges);
    Some(UOp::new(Op::Reduce, inner.dtype(), reduce_srcs, Arg::Reduce(*reduce_op, vec![])))
}

/// **Param rule** (terminal): `Index(Param, [flat_idx])` → `Load(Index(Param, flat_idx))`
///
/// This is where Index pushing bottoms out. By the time Index reaches a Param,
/// all movement ops above have been consumed and the indices have been collapsed
/// to a single flat offset (by Reshape's general case). We emit a Load — the
/// kernel-level "read from memory at this offset" instruction.
///
/// The inner Index is tagged with `Arg::Index(0)` to mark it as kernel-level
/// (pointer arithmetic), which prevents the rangeify rewriter from trying to
/// push it down again. Without this tag, `graph_rewrite` would loop forever:
/// it would see an Index node, try to apply rules, create a new Index, and repeat.
///
/// # Panics
///
/// Panics if the index list has more than 1 element — by this point all
/// multi-dimensional indices should have been flattened by movement rules.
#[must_use]
pub fn rewrite_index_param(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    if inner.op() != Op::Param { return None; }
    assert_eq!(idxs.len(), 1, "Param should have 1 (flat) index");
    let dtype = inner.dtype();
    let kernel_idx = UOp::new(
        Op::Index,
        dtype,
        vec![inner.clone(), idxs[0].clone()],
        Arg::Index(0),
    );
    Some(UOp::load(kernel_idx, dtype))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[3]), vec![1]);
    }
}
