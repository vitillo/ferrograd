//! # Code Generation — IR to target language
//!
//! Defines the [`Renderer`] trait and backend implementations.
//! A renderer walks a toposorted `UOp` graph and emits source code.

pub mod clang;

pub use clang::ClangRenderer;

use crate::uop::UOp;

/// Converts a `UOp` graph into source code for a specific backend.
pub trait Renderer {
    /// Render a `UOp` graph into a complete function as a string.
    fn render(&self, root: &UOp, name: &str) -> String;
}
