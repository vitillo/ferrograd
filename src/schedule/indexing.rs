//! # Indexing — core index transformation logic
//!
//! Provides the rewrite rules that push `Index` down through the expression
//! tree, transforming index expressions as they pass through movement ops.
//! Mirrors tinygrad's `schedule/indexing.py`.
//!
//! - [`rewrite_index_alu`] — distribute Index into ALU sources
//! - [`rewrite_index_movement`] — transform indices through Expand/Permute/Reshape
//! - [`rewrite_index_reduce`] — `ReduceAxis` → Reduce with inner ranges
//! - [`rewrite_index_param`] — Param → Load(Index(Param, `flat_idx`))

use crate::dtype::DType;
use crate::uop::{Arg, Op, UOp};

// ── Helpers ───────────────────────────────────────────────────────────────

/// Contiguous (row-major) strides for a shape.
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

/// Build flat index `UOp` from per-dimension index `UOps` and strides.
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

fn is_alu(op: Op) -> bool {
    matches!(
        op,
        Op::Add | Op::Mul | Op::Max | Op::CmpLt | Op::Neg
            | Op::Exp2 | Op::Log2 | Op::Sqrt | Op::Reciprocal | Op::Where
    )
}

/// Wrap a source in Index with given indices.
#[must_use] 
pub fn index_wrap(src: &UOp, idxs: &[UOp]) -> UOp {
    let mut srcs = vec![src.clone()];
    srcs.extend_from_slice(idxs);
    UOp::new(Op::Index, src.dtype(), srcs, Arg::None)
}

// ── Rewrite rules ─────────────────────────────────────────────────────────

/// Index(ALU(srcs...), idxs) → ALU(Index(src0, idxs), ...)
#[must_use] 
pub fn rewrite_index_alu(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    if !is_alu(inner.op()) {
        return None;
    }
    let new_srcs: Vec<UOp> = inner.srcs().iter().map(|s| index_wrap(s, idxs)).collect();
    Some(UOp::new(inner.op(), inner.dtype(), new_srcs, inner.arg().clone()))
}

/// Index(MovementOp(src, arg), idxs) → Index(src, `transformed_idxs`)
///
/// Like tinygrad's `apply_movement_op`: transforms index expressions
/// as they pass through Expand, Permute, and Reshape.
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
                // Only inserting/removing size-1 dims.
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

/// Index(ReduceAxis(src), idxs) → Reduce(Index(src, `full_idxs`), `reduce_ranges`)
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
            let range = UOp::range(100 + i, UOp::const_int(size as i64, DType::I32));
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

/// Index(Param, [`flat_idx`]) → Load(Index(Param, `flat_idx`))
///
/// Terminal case: Param is already a kernel-level buffer reference
/// (created by the scheduler from Buffer nodes). Wraps in Load for
/// memory access. Uses `Arg::Index(0)` on the kernel-level Index to
/// prevent re-matching on subsequent `graph_rewrite` iterations.
///
/// # Panics
///
/// Panics if Param has more than 1 index (should be flat by this point).
#[must_use]
pub fn rewrite_index_param(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    if inner.op() != Op::Param { return None; }
    assert_eq!(idxs.len(), 1, "Param should have 1 (flat) index");
    let dtype = inner.dtype();
    // Tag the kernel-level Index so we don't match it again.
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
