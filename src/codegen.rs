//! # Code Generation — IR to target language
//!
//! The last stage before execution: after scheduling and rangeify have
//! lowered tensor ops into scalar loops, codegen walks the final `UOp`
//! graph and emits source code for a specific backend.
//!
//! Defines the [`Renderer`] trait and backend implementations (currently
//! just [`ClangRenderer`]). Each renderer translates the same IR into a
//! different target language — tinygrad's renderer serves the same role, but
//! consumes a late-linearized list instead of recovering order from the DAG.

pub mod clang;

pub use clang::ClangRenderer;

use crate::uop::UOp;

/// Converts a `UOp` graph into source code for a specific backend.
pub trait Renderer {
    /// Render a linearized kernel into a complete function as a string.
    fn render(&self, uops: &[UOp], name: &str) -> String;
}
