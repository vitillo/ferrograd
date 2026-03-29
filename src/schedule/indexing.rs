//! # Indexing — pushing indices from output to leaves
//!
//! During rangeify, `Index` nodes are created at the kernel output and then
//! rewrite rules push those indices *down* through the graph -- through
//! elementwise ops, movement ops (reshape, permute, expand, shrink), and
//! reductions -- until they reach leaf nodes (kernel parameters and constants).
//! At each op the index is transformed to account for that op's semantics: a
//! permute reorders index components, a reshape remaps them via stride
//! arithmetic, an expand zeroes out broadcast dimensions, and a reduce replaces
//! collapsed axes with new `Range` iterators.
//!
//! By the time every `Index` has been pushed to a leaf, the abstract tensor
//! graph has been lowered into concrete loop-and-load kernel IR. This module
//! contains the per-op rewrite functions that `rangeify` dispatches to.

use crate::dtype::DType;
use crate::shape::Shape;
use crate::uop::{Arg, AxisKind, Op, UOp};

/// Row-major contiguous strides for a shape.
#[must_use]
pub(super) fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides: Vec<usize> = shape
        .iter()
        .rev()
        .scan(1, |acc, &dim| {
            let stride = *acc;
            *acc *= dim;
            Some(stride)
        })
        .collect();
    strides.reverse();
    strides
}

/// Build `sum(idx[i] * stride[i])`.
///
/// # Panics
///
/// Panics if `idxs` and `strides` have different lengths or `idxs` is empty.
#[must_use]
pub(super) fn flat_index(idxs: &[UOp], strides: &[usize]) -> UOp {
    assert_eq!(idxs.len(), strides.len());
    let device = idxs
        .first()
        .map(UOp::device)
        .expect("flat_index requires at least one index");

    idxs.iter()
        .zip(strides)
        .filter(|(_, &stride)| stride != 0)
        .map(|(idx, &stride)| {
            if stride == 1 {
                idx.clone()
            } else {
                let s = UOp::const_int(i64::try_from(stride).expect("stride exceeds i64 range"), DType::I32, idx.device());
                UOp::mul(idx.clone(), s)
            }
        })
        .reduce(UOp::add)
        .unwrap_or_else(|| UOp::const_int(0, DType::I32, device))
}

/// Wrap a source in a tensor-level `Index` with per-dimension indices.
#[must_use]
pub(super) fn index_wrap(src: &UOp, idxs: &[UOp]) -> UOp {
    let mut srcs = vec![src.clone()];
    srcs.extend_from_slice(idxs);
    UOp::new(Op::Index, src.dtype(), srcs, Arg::None)
}

/// Check whether two shapes differ only in the placement of size-1 dimensions.
/// When true, a reshape between them is a pure squeeze/unsqueeze and indices
/// can be mapped 1:1 (skipping the 1-dims) instead of falling back to flat
/// index arithmetic, which avoids unnecessary mul/div/mod in the generated IR.
fn same_squeezed_dims(src_shape: &Shape, dst_shape: &Shape) -> bool {
    let src_non1: Vec<usize> = src_shape.iter().copied().filter(|&dim| dim != 1).collect();
    let dst_non1: Vec<usize> = dst_shape.iter().copied().filter(|&dim| dim != 1).collect();
    src_non1 == dst_non1
}

/// Push `Index` through elementwise ops.
#[must_use]
#[allow(clippy::unnecessary_wraps)]
pub(super) fn rewrite_index_alu(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    let new_srcs: Vec<UOp> = inner
        .srcs()
        .iter()
        .map(|src| index_wrap(src, idxs))
        .collect();
    Some(UOp::new(
        inner.op(),
        inner.dtype(),
        new_srcs,
        inner.arg().clone(),
    ))
}

/// Push `Index` through movement ops.
#[must_use]
#[allow(clippy::missing_panics_doc)]
pub(super) fn rewrite_index_movement(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    let src = &inner.srcs()[0];

    let new_idxs = match inner.op() {
        Op::Shrink => {
            let Arg::Bounds(lengths) = inner.arg() else {
                panic!("Shrink must have Arg::Bounds");
            };
            idxs.iter()
                .zip(inner.srcs()[1..].iter().zip(lengths.iter()))
                .map(|(idx, (start, _))| {
                    if start.is_zero() {
                        return idx.clone();
                    }
                    UOp::add(idx.clone(), start.clone())
                })
                .collect()
        }
        Op::Expand => {
            let src_shape = src.shape()?;
            let Arg::Shape(expanded) = inner.arg() else {
                panic!("Expand must have Arg::Shape");
            };
            idxs.iter()
                .enumerate()
                .map(|(axis, idx)| {
                    if src_shape[axis] == 1 && expanded[axis] > 1 {
                        UOp::const_int(0, DType::I32, idx.device())
                    } else {
                        idx.clone()
                    }
                })
                .collect()
        }
        Op::Permute => {
            let Arg::Axes(order) = inner.arg() else {
                panic!("Permute must have Arg::Axes");
            };
            let inverse = Shape::invert_permutation(order);
            inverse.iter().map(|&axis| idxs[axis].clone()).collect()
        }
        Op::Reshape => {
            let src_shape = src.shape()?;
            let Arg::Shape(shape) = inner.arg() else {
                panic!("Reshape must have Arg::Shape");
            };
            if same_squeezed_dims(&src_shape, shape) {
                let non1_idxs: Vec<UOp> = idxs
                    .iter()
                    .zip(shape.iter())
                    .filter(|(_, &dim)| dim != 1)
                    .map(|(idx, _)| idx.clone())
                    .collect();
                let mut result = Vec::with_capacity(src_shape.len());
                let mut next_non1 = 0;
                for &dim in src_shape.as_slice() {
                    if dim == 1 {
                        result.push(UOp::const_int(0, DType::I32, src.device()));
                    } else {
                        result.push(non1_idxs[next_non1].clone());
                        next_non1 += 1;
                    }
                }
                result
            } else {
                vec![flat_index(idxs, &contiguous_strides(shape.as_slice()))]
            }
        }
        Op::Contiguous => idxs.to_vec(),
        _ => unreachable!("rangeify only sends movement ops here"),
    };

    Some(index_wrap(src, &new_idxs))
}

/// Push `Index` through a tensor reduction by introducing reduce ranges.
///
/// # Panics
///
/// Panics if the reduction node does not carry `Arg::Reduce` or if `idxs`
/// does not match the source rank.
#[must_use]
pub(super) fn rewrite_index_reduce(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    let Arg::Reduce(reduce_op, axes) = inner.arg() else {
        panic!("ReduceAxis must have Arg::Reduce");
    };
    let src = &inner.srcs()[0];
    let src_shape = src.shape()?;
    assert_eq!(idxs.len(), src_shape.len(), "idxs should match source ndim");
    let mut full_idxs = Vec::with_capacity(src_shape.len());
    let mut reduce_ranges = Vec::new();
    for (axis, &size) in src_shape.iter().enumerate() {
        if axes.contains(&axis) {
            let bound = UOp::const_int(i64::try_from(size).expect("dimension size exceeds i64 range"), DType::I32, src.device());
            let range = UOp::new(
                Op::Range,
                DType::I32,
                vec![bound],
                Arg::Range(axis, AxisKind::Reduce),
            );
            full_idxs.push(range.clone());
            reduce_ranges.push(range);
        } else {
            full_idxs.push(idxs[axis].clone());
        }
    }

    let mut srcs = vec![index_wrap(src, &full_idxs)];
    srcs.extend(reduce_ranges);
    Some(UOp::new(
        Op::Reduce,
        inner.dtype(),
        srcs,
        Arg::Reduce(*reduce_op, Box::default()),
    ))
}

/// Constants are scalars, so indexing does nothing.
#[must_use]
#[allow(clippy::unnecessary_wraps)]
pub(super) fn rewrite_index_const(inner: &UOp, _idxs: &[UOp]) -> Option<UOp> {
    Some(inner.clone())
}

/// Turn tensor-level indexing on a kernel parameter into a kernel `Load`.
///
/// # Panics
///
/// Panics if the index list has not already been flattened to one dimension.
#[must_use]
#[allow(clippy::unnecessary_wraps)]
pub(super) fn rewrite_index_param(inner: &UOp, idxs: &[UOp]) -> Option<UOp> {
    assert_eq!(idxs.len(), 1, "Param should have exactly one flat index");
    let index = UOp::new(
        Op::Index,
        inner.dtype(),
        vec![inner.clone(), idxs[0].clone()],
        Arg::Index(0),
    );
    Some(UOp::new(Op::Load, inner.dtype(), vec![index], Arg::None))
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
