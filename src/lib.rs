//! # ferrograd
//!
//! A from-scratch tensor compiler in Rust, inspired by tinygrad.
//! Built for learning how tensor compilers work, one milestone at a time.
//!
//! ## Architecture (same as tinygrad, simplified)
//!
//! ```text
//! Tensor API → Lazy UOp Graph → Scheduling → Codegen → Compilation → Execution
//! ```
//!
//! ## Module overview
//!
//! - [`dtype`] -- Data types that bridge Rust, IR, and C worlds (M2)
//! - [`device`] -- Device trait, Buffer, Storage, plus backend implementations (M2)
//! - [`uop`] -- Rc-based computation graph nodes (M3)
//! - [`codegen`] -- Code generation: `UOp` IR to C source (M4)
//! - [`rewrite`] -- Graph rewriting: pattern matching and algebraic simplification (M5)
//! - [`tensor`] -- Lazy tensor API with kernel fusion (M6)
//! - [`lower`] -- Lowering: lazy tensor ops to kernel-level `UOp`s (M6)

pub mod codegen;
pub mod device;
pub mod dtype;
pub mod lower;
pub mod rewrite;
pub mod tensor;
pub mod uop;
