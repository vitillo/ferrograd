//! # ferrograd
//!
//! A from-scratch tensor compiler in Rust, inspired by tinygrad.
//! Built for learning how tensor compilers work, one milestone at a time.
//!
//! ## Architecture
//!
//! ```text
//! Tensor API → Lazy UOp Graph → Scheduling → Rangeify → Codegen → Compilation → Execution
//! ```
//!
//! The design follows tinygrad's core split:
//!
//! - tensor semantics live in the lazy [`uop`] graph
//! - device identity is part of that graph
//! - concrete buffers and compiled kernels live in internal device-scoped
//!   state
//! - `UOp` interning lives with the graph itself
//!
//! That keeps [`tensor`] intentionally thin. Methods like reshape, expand,
//! narrow, reduction, and matmul mostly build `UOp`s; [`schedule`] and
//! [`codegen`] turn those graph nodes into executable kernels only when a value
//! is realized.
//!
//! ## Module overview
//!
//! - [`tensor`] exposes the educational user-facing API: lazy tensors,
//!   explicit realization, `narrow`, and opt-in gradient tracking.
//! - [`uop`] defines the shared IR used by tensor graphs, scheduling, and
//!   code generation.
//! - [`schedule`] materializes lazy tensor graphs into proto-kernels, and its
//!   `rangeify` pass turns tensor indexing into explicit loop IR.
//! - [`rewrite`] applies small fixed-point graph simplifications, mainly to the
//!   symbolic index arithmetic generated during lowering.
//! - [`codegen`] renders kernel-level `UOp` graphs to C for the CPU backend.
//! - [`gradient`] builds reverse-mode graphs on top of the same lazy IR.
//! - [`nn`] provides lightweight layer structs such as `Linear`, staying close
//!   to tinygrad's small `nn` surface without introducing a heavy module base
//!   class.
//! - [`optim`] applies parameter updates using lazy tensor assignments.
//! - [`shape`] centralizes shape metadata and shape-only transformations.
//! - [`dataset`] provides pre-packaged datasets (MNIST) with automatic
//!   download and caching.
//! - [`dtype`] and [`device`] bridge Rust values, IR types, buffers, and
//!   backend execution.
//!
//! Internal device state is kept private on purpose. It stores concrete
//! buffers and compiled-kernel caches, but it is not part of the educational
//! surface area.
#![allow(clippy::mutable_key_type)]
// `UOp` keys intentionally contain buffers with interior mutability.
// Hash/Eq use stable identity (`UOp` pointer identity, `Buffer` id), so
// mutating buffer contents does not invalidate map/set keys.

pub mod codegen;
pub mod dataset;
mod devectorize;
pub mod device;
pub mod dtype;
mod expand;
pub mod gradient;
mod lane;
mod linearize;
pub mod nn;
pub mod optim;
mod optimize;
pub mod rewrite;
pub mod schedule;
pub mod shape;
pub mod tensor;
pub mod uop;
