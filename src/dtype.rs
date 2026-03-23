//! # Data Types
//!
//! Every tensor compiler needs a type system that maps between three worlds:
//!
//! 1. **Rust types** (`f32`, `i32`, `bool`) -- what the user works with
//! 2. **IR types** -- what our compiler reasons about (size, alignment, promotion rules)
//! 3. **C types** (`float`, `int`, `_Bool`) -- what we emit in generated code
//!
//! Tinygrad's `DType` (in `tinygrad/dtype.py`) is a frozen dataclass with priority,
//! bitsize, name, and format string. Ours is simpler: an enum with methods that
//! bridge the three worlds.
//!
//! We start with just `F32`, `I32`, and `Bool` -- enough for an MLP. More types
//! (f16, i8, etc.) can be added later without changing the architecture.

/// The data types our compiler understands.
///
/// Each variant knows its byte size and how to represent itself in generated C code.
/// This is the single source of truth for type information throughout the compiler:
/// buffers use it for allocation sizing, the codegen uses it for C type names,
/// and the IR uses it for type checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit IEEE 754 floating point. The workhorse type for neural networks.
    F32,
    /// 32-bit signed integer. Used for indices, shapes, and loop bounds.
    I32,
    /// Boolean. Used for masks and comparison results.
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    ///
    /// Used by [`Buffer`](crate::buffer::Buffer) to compute allocation sizes:
    /// `total_bytes = dtype.size_bytes() * numel`.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::Bool => 1,
        }
    }

    /// The C type name emitted in generated kernel code.
    ///
    /// This is what appears in function signatures and variable declarations
    /// when our codegen (M4) renders the IR to C source.
    #[must_use]
    pub const fn c_type(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::I32 => "int",
            Self::Bool => "_Bool",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::I32 => write!(f, "i32"),
            Self::Bool => write!(f, "bool"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bytes() {
        // Arrange / Act / Assert
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_c_type_names() {
        // Arrange / Act / Assert
        assert_eq!(DType::F32.c_type(), "float");
        assert_eq!(DType::I32.c_type(), "int");
        assert_eq!(DType::Bool.c_type(), "_Bool");
    }
}
