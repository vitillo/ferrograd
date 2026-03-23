//! # Code Generation — IR to target language
//!
//! Defines the [`Renderer`] trait and backend implementations.
//! A renderer walks a toposorted `UOp` graph and emits source code in the
//! target language. This is the "lowering" step: abstract IR becomes
//! concrete, compilable text.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/renderer/__init__.py` — `Renderer` base class.
//! Backend renderers live in `tinygrad/renderer/cstyle.py`, etc.

pub mod clang;

pub use clang::ClangRenderer;

use crate::uop::{UOpGraph, UOpId};

/// Converts a `UOp` graph into source code for a specific backend.
///
/// Each backend (Clang, CUDA, etc.) implements this trait to emit code
/// in its target language. Maps to tinygrad's `Renderer` base class.
pub trait Renderer {
    /// Render a `UOp` graph into a complete function as a string.
    ///
    /// `name` is the function name. `root` is the Sink node to render from.
    fn render(&self, graph: &UOpGraph, name: &str, root: UOpId) -> String;
}
