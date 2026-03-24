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
///
/// The same `Op` enum is used at both the tensor level (lazy graph built by
/// the user) and the kernel level (executable loops + loads + stores produced
/// by rangeify). Tensor-level ops are lowered away before codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    // ── Tensor-level ──────────────────────────────────────────────────
    /// A realized data buffer.
    /// srcs: none. arg: `Arg::Buffer(data)`.
    Buffer,
    /// Change shape without moving data.
    /// srcs: `[source]`. arg: `Arg::Dims(new_shape)`.
    Reshape,
    /// Reorder dimensions.
    /// srcs: `[source]`. arg: `Arg::Dims(axis_order)`.
    Permute,
    /// Broadcast dimensions of size 1 to a larger size.
    /// srcs: `[source]`. arg: `Arg::Dims(new_shape)`.
    Expand,
    /// Reduce over axes (e.g. sum, max). Tensor-level, lowered to `Reduce`.
    /// srcs: `[source]`. arg: `Arg::Reduce(op, axes)`.
    ReduceAxis,

    // ── Kernel structure ──────────────────────────────────────────────
    /// A kernel buffer parameter (pointer to device memory).
    /// srcs: none. arg: `Arg::Param(slot, numel)` where slot 0 is the output.
    Param,
    /// Loop from 0 to bound. Opens a `for` loop in codegen.
    /// srcs: `[bound]` + optional ordering deps. arg: `Arg::Index(axis_id)`.
    Range,
    /// Closes a `Range` loop.
    /// srcs: `[range]` + ordering deps (e.g. the Store or Assign inside).
    End,
    /// Root of a completed kernel graph.
    /// srcs: all top-level nodes (End, Store).
    Sink,

    // ── Memory / Indexing ─────────────────────────────────────────────
    /// At the tensor level: `srcs: [expr, range0, range1, ...]` — index an
    /// expression at loop positions. At the kernel level: `srcs: [param,
    /// flat_offset]` — pointer arithmetic for memory access.
    Index,
    /// Read a value from memory.
    /// srcs: `[index]` (an Index node).
    Load,
    /// Write a value to memory.
    /// srcs: `[index, value]`.
    Store,

    // ── Constants ─────────────────────────────────────────────────────
    /// A compile-time constant.
    /// srcs: none. arg: `Arg::Float`, `Arg::Int`, or `Arg::Bool`.
    Const,

    // ── Math: unary ───────────────────────────────────────────────────
    /// `-x`. srcs: `[x]`.
    Neg,
    /// `2^x`. srcs: `[x]`.
    Exp2,
    /// `log2(x)`. srcs: `[x]`.
    Log2,
    /// `sqrt(x)`. srcs: `[x]`.
    Sqrt,
    /// `1/x`. srcs: `[x]`.
    Reciprocal,

    // ── Math: binary ──────────────────────────────────────────────────
    /// `x + y`. srcs: `[x, y]`.
    Add,
    /// `x * y`. srcs: `[x, y]`.
    Mul,
    /// `max(x, y)`. srcs: `[x, y]`.
    Max,
    /// `x < y`, returns bool. srcs: `[x, y]`.
    CmpLt,

    // ── Math: ternary ─────────────────────────────────────────────────
    /// `if cond then true_val else false_val`. srcs: `[cond, true_val, false_val]`.
    Where,

    // ── Kernel-level reduction ────────────────────────────────────────
    /// Reduce a value over loop ranges (e.g. sum over a loop).
    /// srcs: `[value, range0, range1, ...]`. arg: `Arg::Reduce(op, _)`.
    /// Expanded to DefineAcc/Assign/End/After before codegen.
    Reduce,
    /// Ordering barrier: makes `value` depend on `barrier` in the toposort.
    /// srcs: `[value, barrier]`. Codegen passes through `value`.
    After,
    /// Declares a mutable accumulator variable.
    /// srcs: `[initial_value]` + optional ordering deps.
    DefineAcc,
    /// Updates an accumulator: `acc = new_value`.
    /// srcs: `[acc, new_value]`.
    Assign,
}

impl Op {
    /// Whether this op is an element-wise ALU operation (math on scalars).
    #[must_use]
    pub fn is_alu(self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Max | Self::CmpLt | Self::Where
                | Self::Neg | Self::Exp2 | Self::Log2 | Self::Sqrt | Self::Reciprocal
        )
    }
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
    /// An integer index — axis id for Range.
    Index(usize),
    /// Kernel buffer parameter: (slot, numel). Slot 0 is the output.
    Param(usize, usize),
    /// A constant float value.
    Float(f64),
    /// A constant integer value.
    Int(i64),
    /// A constant boolean value.
    Bool(bool),
    /// A realized tensor buffer. Rc-wrapped for cheap cloning during rewrites.
    /// Identity is by Rc pointer, not by content.
    Buffer(Rc<Buffer>),
    /// Dimension list — shape for Reshape/Expand, axis order for Permute.
    Dims(Vec<usize>),
    /// Reduction: (`reduce_op`, axes).
    Reduce(Op, Vec<usize>),
}

impl PartialEq for Arg {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Index(a), Self::Index(b)) => a == b,
            (Self::Param(sa, na), Self::Param(sb, nb)) => sa == sb && na == nb,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Buffer(a), Self::Buffer(b)) => Rc::ptr_eq(a, b),
            (Self::Dims(a), Self::Dims(b)) => a == b,
            (Self::Reduce(op_a, ax_a), Self::Reduce(op_b, ax_b)) => op_a == op_b && ax_a == ax_b,
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
            Self::Param(s, n) => { s.hash(state); n.hash(state); }
            Self::Float(f) => f.to_bits().hash(state),
            Self::Int(i) => i.hash(state),
            Self::Bool(b) => b.hash(state),
            Self::Buffer(rc) => Rc::as_ptr(rc).hash(state),
            Self::Dims(d) => d.hash(state),
            Self::Reduce(op, axes) => {
                op.hash(state);
                axes.hash(state);
            }
        }
    }
}

impl fmt::Display for Arg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, ""),
            Self::Index(i) => write!(f, "{i}"),
            Self::Param(slot, numel) => write!(f, "slot={slot},n={numel}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Buffer(buf) => write!(f, "buf({})", buf.numel()),
            Self::Dims(d) => write!(f, "{d:?}"),
            Self::Reduce(op, axes) => write!(f, "{op:?}({axes:?})"),
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

    // ── Shape ─────────────────────────────────────────────────────────

    /// Compute shape from the graph structure (like tinygrad's `_shape`).
    ///
    /// Returns `None` for scalar/kernel-level nodes (Const, Param, Range, etc.).
    #[must_use]
    pub fn shape(&self) -> Option<Vec<usize>> {
        match self.op() {
            // Buffer and Param are always flat (1D).
            Op::Buffer => {
                if let Arg::Buffer(ref buf) = self.arg() {
                    return Some(vec![buf.numel()]);
                }
                None
            }
            Op::Param => {
                if let Arg::Param(_, numel) = self.arg() {
                    return Some(vec![*numel]);
                }
                None
            }
            // Movement ops derive shape from their arg or source.
            Op::Reshape | Op::Expand => {
                if let Arg::Dims(ref dims) = self.arg() {
                    return Some(dims.clone());
                }
                None
            }
            Op::Permute => {
                if let Arg::Dims(ref order) = self.arg() {
                    let src_shape = self.srcs()[0].shape()?;
                    return Some(order.iter().map(|&i| src_shape[i]).collect());
                }
                None
            }
            // Reduction: reduced axes become 1.
            Op::ReduceAxis => {
                if let Arg::Reduce(_, ref axes) = self.arg() {
                    let src_shape = self.srcs()[0].shape()?;
                    return Some(
                        src_shape
                            .iter()
                            .enumerate()
                            .map(|(i, &s)| if axes.contains(&i) { 1 } else { s })
                            .collect(),
                    );
                }
                None
            }
            op if op.is_alu() => self.srcs()[0].shape(),
            // Kernel-level / scalar nodes: no shape.
            _ => None,
        }
    }

    // ── Builder methods ─────────────────────────────────────────────────

    /// Kernel buffer parameter at `slot`.
    #[must_use]
    pub fn param(slot: usize, dtype: DType, numel: usize) -> Self {
        Self::new(Op::Param, dtype, vec![], Arg::Param(slot, numel))
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

    /// Build an Index node. At the tensor level: multi-dim indexing.
    /// At the kernel level: flat pointer arithmetic.
    #[must_use]
    pub fn index(src: Self, offset: Self) -> Self {
        let dtype = src.dtype();
        Self::new(Op::Index, dtype, vec![src, offset], Arg::None)
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
        let id_map: HashMap<&UOp, usize> = order.iter().enumerate().map(|(i, n)| (n, i)).collect();

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
        let out_ptr = UOp::param(0, DType::F32, 1024);
        let a_ptr = UOp::param(1, DType::F32, 1024);
        let b_ptr = UOp::param(2, DType::F32, 1024);
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
