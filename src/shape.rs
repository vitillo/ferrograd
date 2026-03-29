//! # Shape — Immutable Tensor Dimensions
//!
//! A [`Shape`] is a thin wrapper around a boxed slice of dimension sizes. It's
//! separate from tensors because shape metadata is needed in many places that
//! don't involve actual data: broadcasting rules, stride calculation, axis
//! validation, and IR construction all operate purely on shapes.
//!
//! ## Tinygrad equivalent
//!
//! Tinygrad doesn't have a dedicated `Shape` type — it passes around plain
//! Python tuples of ints. We use a named type instead because Rust benefits
//! from the type safety (you can't accidentally pass an axis list where a
//! shape is expected), and because it gives us a natural place to hang
//! shape-related operations.
//!
//! ## Key operations
//!
//! - **Broadcasting** ([`Shape::broadcast_with`]): numpy-style broadcast, used
//!   to align operand shapes before elementwise ops.
//! - **Left-padding** ([`Shape::pad_left`]): prepend size-1 dimensions so two
//!   shapes have the same rank — the first step of broadcasting.
//! - **Permutation inversion** ([`Shape::invert_permutation`]): given an axis
//!   reordering `[2,0,1]`, compute the inverse `[1,2,0]`. Needed to undo
//!   transposes during gradient backpropagation.

use std::fmt;
use std::ops::Deref;

/// Tensor shape stored as fixed-size immutable dimensions.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(Box<[usize]>);

impl Shape {
    /// Create a shape from owned dimensions.
    #[must_use]
    pub fn new(dims: Vec<usize>) -> Self {
        Self(dims.into_boxed_slice())
    }

    /// Create a one-dimensional shape with `numel` elements.
    #[must_use]
    pub fn flat(numel: usize) -> Self {
        Self::new(vec![numel])
    }

    /// Borrow the dimensions.
    #[must_use]
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    /// Left-pad with ones to `ndim`.
    #[must_use]
    pub fn pad_left(&self, ndim: usize) -> Self {
        let mut padded = vec![1; ndim - self.ndim()];
        padded.extend_from_slice(self.as_slice());
        Self::new(padded)
    }

    /// Compute a numpy-style broadcast target shape.
    #[must_use]
    pub fn broadcast_with(&self, other: &Self) -> Option<Self> {
        let ndim = self.ndim().max(other.ndim());
        let left = self.pad_left(ndim);
        let right = other.pad_left(ndim);

        left.iter()
            .zip(right.iter())
            .map(|(&lhs, &rhs)| match (lhs, rhs) {
                (x, y) if x == y => Some(x),
                (1, y) => Some(y),
                (x, 1) => Some(x),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .map(Self::new)
    }

    /// Axes reduced when backpropagating through an expand from `self` to `expanded`.
    #[must_use]
    pub fn expand_gradient_axes(&self, expanded: &Self) -> Box<[usize]> {
        self.iter()
            .zip(expanded.iter())
            .enumerate()
            .filter(|(_, (&src, &dst))| src == 1 && dst > 1)
            .map(|(axis, _)| axis)
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    /// Invert a permutation.
    #[must_use]
    pub fn invert_permutation(order: &[usize]) -> Box<[usize]> {
        let mut inverted = vec![0; order.len()];
        for (i, &axis) in order.iter().enumerate() {
            inverted[axis] = i;
        }
        inverted.into_boxed_slice()
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl fmt::Display for Shape {
    /// Prints as `[d0, d1, ...]`, matching the intuitive notation used in
    /// error messages and debug output throughout the codebase.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_slice(), f)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self::new(value)
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Self::new(value.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self::new(value.to_vec())
    }
}

impl<const N: usize> PartialEq<[usize; N]> for Shape {
    fn eq(&self, other: &[usize; N]) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<const N: usize> PartialEq<&[usize; N]> for Shape {
    fn eq(&self, other: &&[usize; N]) -> bool {
        self.as_slice() == other.as_slice()
    }
}
