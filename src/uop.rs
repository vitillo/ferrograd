//! # `UOp` — The Core Graph Node
//!
//! Every computation in the compiler is represented as a DAG of `UOp` nodes.
//! Each node holds its op, dtype, children (srcs), and an optional argument.
//! Nodes are reference-counted (`Rc`) for cheap sharing — this matches
//! tinygrad where `UOp`s are Python heap objects.
//!
//! ## Tinygrad reference
//!
//! - `Ops` enum: `tinygrad/uop/__init__.py`
//! - `UOp` class: `tinygrad/uop/ops.py`

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::rc::Rc;

pub use crate::device::Buffer;
use crate::dtype::DType;

// ── Op ──────────────────────────────────────────────────────────────────────

/// The operations our IR supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    // Tensor-level
    /// A realized tensor buffer. Arg is `Arg::Buffer(buf)`.
    Buffer,

    // Kernel structure
    /// A kernel buffer parameter. Arg is `Arg::Index(slot)`.
    Param,
    /// Loop counter. Arg is `Arg::Index(axis)`.
    Range,
    /// Closes a Range loop.
    End,
    /// Root of a kernel graph.
    Sink,

    // Memory
    /// Pointer arithmetic: Param + Range.
    Index,
    /// Read from an address.
    Load,
    /// Write to an address.
    Store,

    // Constants
    /// A compile-time constant. Arg is the value.
    Const,

    // Math: unary
    /// Negate: `-x`.
    Neg,
    /// Base-2 exponential: `2^x`.
    Exp2,
    /// Base-2 logarithm: `log2(x)`.
    Log2,
    /// Square root.
    Sqrt,
    /// Reciprocal: `1/x`.
    Reciprocal,

    // Math: binary
    /// Addition.
    Add,
    /// Multiplication.
    Mul,
    /// Maximum.
    Max,
    /// Less-than comparison, returns bool.
    CmpLt,

    // Math: ternary
    /// Conditional select: `where(cond, true_val, false_val)`.
    Where,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ── Arg ─────────────────────────────────────────────────────────────────────

/// Op-specific payload attached to a `UOp` node.
#[derive(Debug, Clone)]
pub enum Arg {
    /// No argument.
    None,
    /// An integer index — buffer slot for Param, axis id for Range.
    Index(usize),
    /// A constant float value.
    Float(f64),
    /// A constant integer value.
    Int(i64),
    /// A constant boolean value.
    Bool(bool),
    /// A realized tensor buffer. Rc-wrapped for cheap cloning during rewrites.
    /// Identity is by Rc pointer, not by content.
    Buffer(Rc<Buffer>),
}

impl PartialEq for Arg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Index(a), Self::Index(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Buffer(a), Self::Buffer(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for Arg {}

impl std::hash::Hash for Arg {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::None => {}
            Self::Index(i) => i.hash(state),
            Self::Float(f) => f.to_bits().hash(state),
            Self::Int(i) => i.hash(state),
            Self::Bool(b) => b.hash(state),
            Self::Buffer(rc) => Rc::as_ptr(rc).hash(state),
        }
    }
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, ""),
            Self::Index(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Buffer(buf) => write!(f, "buf({})", buf.numel()),
        }
    }
}

// ── UOp ─────────────────────────────────────────────────────────────────────

struct UOpInner {
    op: Op,
    dtype: DType,
    srcs: Vec<UOp>,
    arg: Arg,
}

/// A node in the computation graph.
///
/// Reference-counted for cheap cloning and sharing.
#[derive(Clone)]
pub struct UOp(Rc<UOpInner>);

impl UOp {
    /// Create a new node.
    #[must_use]
    pub fn new(op: Op, dtype: DType, srcs: Vec<Self>, arg: Arg) -> Self {
        Self(Rc::new(UOpInner {
            op,
            dtype,
            srcs,
            arg,
        }))
    }

    /// The operation.
    #[must_use]
    pub fn op(&self) -> Op {
        self.0.op
    }

    /// The result type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    /// Input edges.
    #[must_use]
    pub fn srcs(&self) -> &[UOp] {
        &self.0.srcs
    }

    /// Op-specific payload.
    #[must_use]
    pub fn arg(&self) -> &Arg {
        &self.0.arg
    }

    /// Stable pointer identity, used by the `Hash` impl.
    fn ptr_id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    // ── Builder methods ─────────────────────────────────────────────────

    /// Kernel buffer parameter at `slot`.
    #[must_use]
    pub fn param(slot: usize, dtype: DType) -> Self {
        Self::new(Op::Param, dtype, vec![], Arg::Index(slot))
    }

    /// Scalar float constant.
    #[must_use]
    pub fn const_float(value: f64, dtype: DType) -> Self {
        Self::new(Op::Const, dtype, vec![], Arg::Float(value))
    }

    /// Scalar integer constant.
    #[must_use]
    pub fn const_int(value: i64, dtype: DType) -> Self {
        Self::new(Op::Const, dtype, vec![], Arg::Int(value))
    }

    /// Loop range with `axis` id and upper `bound`.
    #[must_use]
    pub fn range(axis: usize, bound: Self) -> Self {
        Self::new(Op::Range, DType::I32, vec![bound], Arg::Index(axis))
    }

    /// Close a loop.
    #[must_use]
    pub fn end(range: Self) -> Self {
        Self::new(Op::End, DType::Void, vec![range], Arg::None)
    }

    /// Pointer arithmetic.
    #[must_use]
    pub fn index(ptr: Self, offset: Self) -> Self {
        let dtype = ptr.dtype();
        Self::new(Op::Index, dtype, vec![ptr, offset], Arg::None)
    }

    /// Load from an address.
    #[must_use]
    pub fn load(index: Self, dtype: DType) -> Self {
        Self::new(Op::Load, dtype, vec![index], Arg::None)
    }

    /// Store a value to an address.
    #[must_use]
    pub fn store(index: Self, value: Self) -> Self {
        Self::new(Op::Store, DType::Void, vec![index, value], Arg::None)
    }

    /// Kernel root.
    #[must_use]
    pub fn sink(stores: Vec<Self>) -> Self {
        Self::new(Op::Sink, DType::Void, stores, Arg::None)
    }

    // ── Toposort ────────────────────────────────────────────────────────

    /// Iterative post-order DFS. Returns nodes in dependency order
    /// (sources before consumers).
    #[must_use]
    pub fn toposort(&self) -> Vec<UOp> {
        let mut result = Vec::new();
        let mut visited: HashSet<UOp> = HashSet::new();
        let mut stack: Vec<(UOp, bool)> = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            if visited.contains(&node) {
                continue;
            }
            if processed {
                visited.insert(node.clone());
                result.push(node);
            } else {
                stack.push((node.clone(), true));
                for src in node.srcs().iter().rev() {
                    if !visited.contains(src) {
                        stack.push((src.clone(), false));
                    }
                }
            }
        }
        result
    }

    /// Debug dump of the toposorted graph.
    #[must_use]
    pub fn dump(&self) -> String {
        use std::fmt::Write;
        let order = self.toposort();
        let id_map: HashMap<&UOp, usize> = order
            .iter()
            .enumerate()
            .map(|(i, n)| (n, i))
            .collect();

        let mut out = String::new();
        for node in &order {
            let idx = id_map[&node];
            let srcs: Vec<String> = node
                .srcs()
                .iter()
                .map(|s| format!("%{}", id_map[&s]))
                .collect();
            let src_str = if srcs.is_empty() {
                String::new()
            } else {
                format!(" ({})", srcs.join(", "))
            };
            let arg_str = match node.arg() {
                Arg::None => String::new(),
                a => format!("  arg={a}"),
            };
            let _ = writeln!(
                out,
                "  %{idx} = {} {}{src_str}{arg_str}",
                node.op(),
                node.dtype(),
            );
        }
        out
    }
}

impl PartialEq for UOp {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for UOp {}

impl std::hash::Hash for UOp {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr_id().hash(state);
    }
}

impl fmt::Debug for UOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UOp({:?}, {:?})", self.0.op, self.0.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_elementwise_add_kernel() {
        // Arrange/Act — build: out[i] = a[i] + b[i]
        let out_ptr = UOp::param(0, DType::F32);
        let a_ptr = UOp::param(1, DType::F32);
        let b_ptr = UOp::param(2, DType::F32);
        let n = UOp::const_int(1024, DType::I32);
        let idx = UOp::range(0, n);
        let a_val = UOp::load(UOp::index(a_ptr, idx.clone()), DType::F32);
        let b_val = UOp::load(UOp::index(b_ptr, idx.clone()), DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![a_val, b_val], Arg::None);
        let store = UOp::store(UOp::index(out_ptr, idx.clone()), sum);
        let end = UOp::end(idx);
        let sink = UOp::sink(vec![store, end]);

        // Assert
        assert_eq!(sink.op(), Op::Sink);
        assert_eq!(sink.srcs().len(), 2);
    }

    #[test]
    fn test_toposort_sources_before_consumers() {
        // Arrange
        let a = UOp::const_float(1.0, DType::F32);
        let b = UOp::const_float(2.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![a.clone(), b.clone()], Arg::None);

        // Act
        let order = sum.toposort();

        // Assert — a and b appear before sum
        let pos = |u: &UOp| order.iter().position(|n| n == u).unwrap();
        assert!(pos(&a) < pos(&sum));
        assert!(pos(&b) < pos(&sum));
    }

    #[test]
    fn test_shared_node_appears_once_in_toposort() {
        // Arrange — a + a shares one node
        let a = UOp::const_float(1.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![a.clone(), a.clone()], Arg::None);

        // Act
        let order = sum.toposort();

        // Assert — a appears exactly once
        let count = order.iter().filter(|n| *n == &a).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_dump_shows_toposorted_graph() {
        // Arrange
        let a = UOp::const_float(1.0, DType::F32);
        let b = UOp::const_float(2.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![a, b], Arg::None);

        // Act
        let dump = sum.dump();

        // Assert
        assert!(dump.contains("Const"));
        assert!(dump.contains("Add"));
        assert!(dump.contains("%0"));
    }

    #[test]
    fn test_pointer_identity() {
        // Arrange
        let a = UOp::const_float(1.0, DType::F32);
        let b = UOp::const_float(1.0, DType::F32);
        let a_clone = a.clone();

        // Assert — clone shares identity, separate creation does not
        assert_eq!(a, a_clone);
        assert_ne!(a, b);
    }
}
