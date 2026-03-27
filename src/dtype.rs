//! # Data Types
//!
//! Every tensor compiler needs a type system that maps between three worlds:
//!
//! 1. **Rust types** (`f32`, `i32`, `bool`) -- what the user works with
//! 2. **IR types** -- what our compiler reasons about (size, alignment, vector width)
//! 3. **C types** (`float`, `int`, `_Bool`) -- what we emit in generated code
//!
//! Tinygrad's `DType` carries a `count` field for vectorized types (e.g.
//! `dtypes.float.vec(4)` for a 4-wide SIMD lane). We do the same: `DType`
//! is a struct pairing a scalar kind with a vector count, so the late
//! expander can widen types without creating explicit per-lane IR nodes.
//!
//! We start with just `F32`, `I32`, and `Bool` -- enough for an MLP. More types
//! (f16, i8, etc.) can be added later without changing the architecture.

/// The scalar element types our compiler understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeKind {
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 32-bit signed integer.
    I32,
    /// Boolean.
    Bool,
    /// No value (side-effecting IR nodes).
    Void,
}

/// A data type with optional vector width.
///
/// Scalars have `vcount == 1`. During late expansion, types are widened
/// (e.g. `DType::F32.vec(4)`) to represent multiple lanes computed together.
/// The devectorizer scalarizes everything back to `vcount == 1` before codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DType {
    kind: DTypeKind,
    vcount: u16,
}

impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}

/// Scalar type constants — these are the primary constructors used everywhere.
/// Names match the old enum variants for backwards compatibility.
#[allow(non_upper_case_globals)]
impl DType {
    /// 32-bit float, scalar.
    pub const F32: Self = Self { kind: DTypeKind::F32, vcount: 1 };
    /// 32-bit signed integer, scalar.
    pub const I32: Self = Self { kind: DTypeKind::I32, vcount: 1 };
    /// Boolean, scalar.
    pub const Bool: Self = Self { kind: DTypeKind::Bool, vcount: 1 };
    /// Void (side-effects only).
    pub const Void: Self = Self { kind: DTypeKind::Void, vcount: 1 };

    /// Create a vectorized version of this type with `n` lanes.
    #[must_use]
    pub const fn vec(self, n: u16) -> Self {
        Self { kind: self.kind, vcount: n }
    }

    /// Return the scalar (single-lane) version of this type.
    #[must_use]
    pub const fn scalar(self) -> Self {
        Self { kind: self.kind, vcount: 1 }
    }

    /// The scalar element kind.
    #[must_use]
    pub const fn kind(self) -> DTypeKind {
        self.kind
    }

    /// Number of lanes (1 for scalars).
    #[must_use]
    pub const fn vcount(self) -> u16 {
        self.vcount
    }

    /// Whether this is a scalar (single-lane) type.
    #[must_use]
    pub const fn is_scalar(self) -> bool {
        self.vcount == 1
    }

    /// Size of one scalar element in bytes.
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self.kind {
            DTypeKind::F32 | DTypeKind::I32 => 4,
            DTypeKind::Bool => 1,
            DTypeKind::Void => 0,
        }
    }

    /// The C type name for the scalar element.
    #[must_use]
    pub const fn c_type(self) -> &'static str {
        match self.kind {
            DTypeKind::F32 => "float",
            DTypeKind::I32 => "int",
            DTypeKind::Bool => "_Bool",
            DTypeKind::Void => "void",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self.kind {
            DTypeKind::F32 => "f32",
            DTypeKind::I32 => "i32",
            DTypeKind::Bool => "bool",
            DTypeKind::Void => "void",
        };
        if self.vcount == 1 {
            write!(f, "{name}")
        } else {
            write!(f, "{name}x{}", self.vcount)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_bytes() {
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_c_type_names() {
        assert_eq!(DType::F32.c_type(), "float");
        assert_eq!(DType::I32.c_type(), "int");
        assert_eq!(DType::Bool.c_type(), "_Bool");
    }

    #[test]
    fn test_vec_and_scalar() {
        let v = DType::F32.vec(4);
        assert_eq!(v.vcount(), 4);
        assert!(!v.is_scalar());
        assert_eq!(v.scalar(), DType::F32);
        assert_eq!(v.kind(), DTypeKind::F32);
    }

    #[test]
    fn test_display_vectorized() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::F32.vec(4)), "f32x4");
    }
}
