//! # Code Generation — IR to target language
//!
//! The last stage before execution: after scheduling and rangeify have
//! lowered tensor ops into scalar loops, codegen walks the final `UOp`
//! graph and emits source code for a specific backend.
//!
//! Defines the [`Renderer`] trait and backend implementations (currently
//! just [`ClangRenderer`]). Each renderer translates the same IR into a
//! different target language — tinygrad's `renderer.py` serves the same role.

pub mod clang;

pub use clang::ClangRenderer;

use crate::uop::UOp;

/// Converts a `UOp` graph into source code for a specific backend.
pub trait Renderer {
    /// Render a `UOp` graph into a complete function as a string.
    fn render(&self, root: &UOp, name: &str) -> String;
}
