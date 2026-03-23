//! # `UOp` IR — The Core Graph
//!
//! Every computation in the compiler is represented as a directed acyclic graph
//! (DAG) of `UOp` nodes. This is the heart of tinygrad's architecture and ours:
//! the tensor API builds this graph lazily, the scheduler groups it into kernels,
//! and the codegen walks it to emit C/CUDA code.
//!
//! ## Key concepts
//!
//! - **Arena storage**: all `UOp` nodes live in a flat `Vec<UOp>` inside a [`UOpGraph`].
//!   Nodes reference each other via [`UOpId`] indices, not heap pointers. This is
//!   cache-friendly and makes topological sorting trivial.
//!
//! - **Hash-consing (interning)**: before creating a new node, the graph checks
//!   whether an identical `(op, dtype, srcs, arg)` tuple already exists. If so,
//!   it returns the existing ID. This gives us automatic common subexpression
//!   elimination — `(a + b) * (a + b)` has one `add` node, not two.
//!
//! - **How a kernel looks**: an element-wise `C[i] = A[i] + B[i]` becomes:
//!
//!   ```text
//!   Param(slot=0)  ─── out ptr
//!   Param(slot=1)  ─── a ptr
//!   Param(slot=2)  ─── b ptr
//!   Const(1024)    ─── loop bound
//!   Range(axis=0)  ─── for (int idx0 = 0; idx0 < 1024; idx0++)
//!   Index(a, idx)  ─── a + idx0
//!   Load(index_a)  ─── *(a + idx0)
//!   Index(b, idx)  ─── b + idx0
//!   Load(index_b)  ─── *(b + idx0)
//!   Add(load_a, load_b)
//!   Index(out, idx) ── out + idx0
//!   Store(index_out, add_result)
//!   End(range)     ─── }
//!   Sink(store)    ─── kernel root
//!   ```
//!
//! ## Tinygrad reference
//!
//! - `Ops` enum: `tinygrad/uop/__init__.py`
//! - `UOp` class + hash-consing: `tinygrad/uop/ops.py` (`UOpMetaClass.__call__`)
//! - Toposort: `UOp.toposort()` in `ops.py`

use std::collections::HashMap;
use std::fmt;

use crate::dtype::DType;

// ── Op ──────────────────────────────────────────────────────────────────────

/// The operations our IR supports.
///
/// Ordered to match tinygrad's convention: defines/structure first, then
/// memory, then math, then constants. The ordering also determines the
/// natural toposort priority (lower variants come first in the emitted code).
///
/// This is a small subset of tinygrad's `Ops` — just enough for element-wise
/// and reduction kernels. More ops can be added without changing the graph
/// machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    // Structure: kernel inputs and control flow
    /// A kernel buffer parameter (function argument). Arg is the slot number.
    Param,
    /// Loop counter. Takes the loop bound as input. Arg is the axis id.
    Range,
    /// Closes a Range loop. Takes the Range it closes as input.
    End,
    /// Root of a kernel graph. Takes all stores as inputs.
    Sink,

    // Memory: pointer arithmetic and data movement
    /// Pointer arithmetic: `Param + Range`. Takes a Param (buffer) and a Range (loop variable).
    Index,
    /// Read a value at the address computed by an Index.
    Load,
    /// Write a value to the address computed by an Index. Takes an Index and the value to store.
    Store,

    // Constants
    /// A compile-time constant value. Arg is the scalar value.
    Const,

    // Math: unary
    /// Negate: `-x`.
    Neg,
    /// Base-2 exponential: `2^x`.
    Exp2,
    /// Base-2 logarithm: `log2(x)`.
    Log2,
    /// Square root: `sqrt(x)`.
    Sqrt,
    /// Reciprocal: `1/x`.
    Reciprocal,

    // Math: binary
    /// Addition.
    Add,
    /// Multiplication.
    Mul,
    /// Maximum of two values.
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
///
/// Tinygrad uses Python's `Any` for this field. We use a typed enum so the
/// compiler enforces that each op gets the right kind of argument.
#[derive(Debug, Clone, PartialEq)]
pub enum Arg {
    /// No argument (most ALU ops).
    None,
    /// An integer index — buffer slot for Param, axis id for Range.
    Index(usize),
    /// A constant float value.
    Float(f64),
    /// A constant integer value.
    Int(i64),
    /// A constant boolean value.
    Bool(bool),
}

// Custom Eq: f64 doesn't implement Eq, but we only use exact bit patterns
// for constants, so we delegate to total_cmp-based equality.
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
        }
    }
}

// ── UOpId ───────────────────────────────────────────────────────────────────

/// A handle to a `UOp` node in the graph, implemented as an index into the arena.
///
/// This is a lightweight, `Copy` type — passing `UOpId`s around is just passing
/// an integer. Two `UOpId`s from the same graph can be compared with `==`
/// to check if they refer to the same node (structural equality is guaranteed
/// by hash-consing).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct UOpId(u32);

impl UOpId {
    /// Returns this ID as a `usize` index, for use in parallel arrays.
    #[must_use]
    pub fn idx(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for UOpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl fmt::Display for UOpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

// ── UOp ─────────────────────────────────────────────────────────────────────

/// A single node in the computation graph.
///
/// Nodes are stored in a [`UOpGraph`] arena and referenced by [`UOpId`].
/// Each node knows its operation, result type, input edges, and an optional
/// argument. This maps directly to tinygrad's `UOp(op, dtype, src, arg)`.
#[derive(Debug, Clone)]
pub struct UOp {
    /// What operation this node performs.
    pub op: Op,
    /// The type of the value this node produces.
    pub dtype: DType,
    /// Input edges (indices of source nodes in the arena).
    pub srcs: Vec<UOpId>,
    /// Op-specific payload.
    pub arg: Arg,
}

// ── UOpGraph ────────────────────────────────────────────────────────────────

/// The interning arena that holds all [`UOp`] nodes for a kernel.
///
/// All node creation goes through the graph's methods, which check the
/// intern cache before allocating. This guarantees hash-consing: identical
/// subexpressions always share the same `UOpId`.
///
/// ## Builder pattern
///
/// ```ignore
/// let mut g = UOpGraph::new();
/// let a_ptr = g.param(0, DType::F32);
/// let n = g.const_int(1024, DType::I32);
/// let idx = g.range(0, n);
/// let a_idx = g.index(a_ptr, idx);
/// let a_val = g.load(a_idx, DType::F32);
/// // ... build more of the graph ...
/// ```
pub struct UOpGraph {
    /// The arena: all nodes, indexed by [`UOpId`].
    nodes: Vec<UOp>,
    /// Intern cache: maps `(op, dtype, srcs, arg)` → existing [`UOpId`].
    /// This is what gives us hash-consing.
    cache: HashMap<(Op, DType, Vec<UOpId>, Arg), UOpId>,
}

impl UOpGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            cache: HashMap::new(),
        }
    }

    /// How many nodes are in the graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look up a node by its ID.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of bounds.
    #[must_use]
    pub fn get(&self, id: UOpId) -> &UOp {
        &self.nodes[id.0 as usize]
    }

    // ── Core insertion (with interning) ─────────────────────────────────

    /// Insert a node, returning the existing ID if an identical node exists.
    ///
    /// This is the only way to create nodes — all builder methods below
    /// delegate here. The intern check is what makes hash-consing work.
    #[allow(clippy::cast_possible_truncation)]
    pub fn add(&mut self, op: Op, dtype: DType, srcs: Vec<UOpId>, arg: Arg) -> UOpId {
        let key = (op, dtype, srcs.clone(), arg.clone());
        if let Some(&existing) = self.cache.get(&key) {
            return existing;
        }
        let id = UOpId(self.nodes.len() as u32);
        self.nodes.push(UOp {
            op,
            dtype,
            srcs,
            arg,
        });
        self.cache.insert(key, id);
        id
    }

    // ── Builder methods ─────────────────────────────────────────────────
    //
    // These provide a convenient, typed API for building graphs. Each method
    // enforces the correct src/arg pattern for its op, preventing invalid
    // graphs at construction time.

    /// Create a kernel buffer parameter (function argument).
    ///
    /// `slot` is the argument position: 0 for the first buffer, 1 for the
    /// second, etc. This becomes `float* data0` in the generated C code.
    pub fn param(&mut self, slot: usize, dtype: DType) -> UOpId {
        self.add(Op::Param, dtype, vec![], Arg::Index(slot))
    }

    /// Create a scalar constant.
    pub fn const_float(&mut self, value: f64, dtype: DType) -> UOpId {
        self.add(Op::Const, dtype, vec![], Arg::Float(value))
    }

    /// Create an integer constant.
    pub fn const_int(&mut self, value: i64, dtype: DType) -> UOpId {
        self.add(Op::Const, dtype, vec![], Arg::Int(value))
    }

    /// Create a boolean constant.
    pub fn const_bool(&mut self, value: bool) -> UOpId {
        self.add(Op::Const, DType::Bool, vec![], Arg::Bool(value))
    }

    /// Create a loop range. `bound` is the upper limit (exclusive).
    ///
    /// `axis` is a unique identifier for this loop dimension — axis 0 is
    /// the outermost loop, axis 1 the next inner, etc.
    pub fn range(&mut self, axis: usize, bound: UOpId) -> UOpId {
        self.add(Op::Range, DType::I32, vec![bound], Arg::Index(axis))
    }

    /// Close a loop opened by `range`.
    pub fn end(&mut self, range: UOpId) -> UOpId {
        self.add(Op::End, DType::Void, vec![range], Arg::None)
    }

    /// Pointer arithmetic: `ptr + offset`.
    pub fn index(&mut self, ptr: UOpId, offset: UOpId) -> UOpId {
        let dtype = self.get(ptr).dtype;
        self.add(Op::Index, dtype, vec![ptr, offset], Arg::None)
    }

    /// Load a value from a pointer produced by [`index`](Self::index).
    pub fn load(&mut self, index: UOpId, dtype: DType) -> UOpId {
        self.add(Op::Load, dtype, vec![index], Arg::None)
    }

    /// Store a value through a pointer produced by [`index`](Self::index).
    pub fn store(&mut self, index: UOpId, value: UOpId) -> UOpId {
        self.add(Op::Store, DType::Void, vec![index, value], Arg::None)
    }

    /// The kernel root. All stores must be listed as sources.
    pub fn sink(&mut self, stores: Vec<UOpId>) -> UOpId {
        self.add(Op::Sink, DType::Void, stores, Arg::None)
    }

    // ── ALU helpers ─────────────────────────────────────────────────────

    /// Negate: `-x`.
    pub fn neg(&mut self, x: UOpId) -> UOpId {
        let dtype = self.get(x).dtype;
        self.add(Op::Neg, dtype, vec![x], Arg::None)
    }

    /// Base-2 exponential: `2^x`.
    pub fn exp2(&mut self, x: UOpId) -> UOpId {
        let dtype = self.get(x).dtype;
        self.add(Op::Exp2, dtype, vec![x], Arg::None)
    }

    /// Base-2 logarithm: `log2(x)`.
    pub fn log2(&mut self, x: UOpId) -> UOpId {
        let dtype = self.get(x).dtype;
        self.add(Op::Log2, dtype, vec![x], Arg::None)
    }

    /// Square root: `sqrt(x)`.
    pub fn sqrt(&mut self, x: UOpId) -> UOpId {
        let dtype = self.get(x).dtype;
        self.add(Op::Sqrt, dtype, vec![x], Arg::None)
    }

    /// Reciprocal: `1/x`.
    pub fn reciprocal(&mut self, x: UOpId) -> UOpId {
        let dtype = self.get(x).dtype;
        self.add(Op::Reciprocal, dtype, vec![x], Arg::None)
    }

    /// Addition: `a + b`.
    pub fn add_op(&mut self, a: UOpId, b: UOpId) -> UOpId {
        let dtype = self.get(a).dtype;
        self.add(Op::Add, dtype, vec![a, b], Arg::None)
    }

    /// Multiplication: `a * b`.
    pub fn mul(&mut self, a: UOpId, b: UOpId) -> UOpId {
        let dtype = self.get(a).dtype;
        self.add(Op::Mul, dtype, vec![a, b], Arg::None)
    }

    /// Maximum: `max(a, b)`.
    pub fn max(&mut self, a: UOpId, b: UOpId) -> UOpId {
        let dtype = self.get(a).dtype;
        self.add(Op::Max, dtype, vec![a, b], Arg::None)
    }

    /// Less-than comparison: `a < b`.
    pub fn cmplt(&mut self, a: UOpId, b: UOpId) -> UOpId {
        self.add(Op::CmpLt, DType::Bool, vec![a, b], Arg::None)
    }

    /// Conditional select: `cond ? true_val : false_val`.
    pub fn where_op(&mut self, cond: UOpId, true_val: UOpId, false_val: UOpId) -> UOpId {
        let dtype = self.get(true_val).dtype;
        self.add(Op::Where, dtype, vec![cond, true_val, false_val], Arg::None)
    }

    // ── Graph traversal ─────────────────────────────────────────────────

    /// Topological sort starting from `root`, returning nodes in dependency order.
    ///
    /// Uses iterative post-order DFS (same algorithm as tinygrad's `UOp.toposort()`).
    /// The result is ordered so that every node appears after all of its sources —
    /// exactly the order codegen needs to emit code.
    #[must_use]
    pub fn toposort(&self, root: UOpId) -> Vec<UOpId> {
        let mut visited = vec![false; self.nodes.len()];
        let mut result = Vec::new();
        let mut stack: Vec<(UOpId, bool)> = vec![(root, false)];

        while let Some((id, processed)) = stack.pop() {
            let idx = id.0 as usize;
            if visited[idx] {
                continue;
            }
            if processed {
                visited[idx] = true;
                result.push(id);
            } else {
                stack.push((id, true));
                // Push sources in reverse so they're visited in order.
                for &src in self.nodes[idx].srcs.iter().rev() {
                    if !visited[src.0 as usize] {
                        stack.push((src, false));
                    }
                }
            }
        }

        result
    }
}

impl Default for UOpGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl UOpGraph {
    /// Print the graph as a flat list, one line per node.
    ///
    /// Similar to tinygrad's `print_uops()`. Each line shows the node index,
    /// op, dtype, source references, and arg. This is the format you want
    /// when debugging codegen (the order matches what the renderer will walk).
    ///
    /// Example output:
    /// ```text
    ///   %0 = Param        f32    arg=0
    ///   %1 = Const        i32    arg=1024
    ///   %2 = Range        i32   (%1)  arg=0
    ///   %3 = Index        f32   (%0, %2)
    ///   %4 = Load         f32   (%3)
    /// ```
    pub fn dump(&self, root: UOpId) -> String {
        use std::fmt::Write;
        let order = self.toposort(root);
        let mut out = String::new();
        for &id in &order {
            let node = self.get(id);
            let _ = write!(out, "  {id} = {:<12} {:<5}", node.op, node.dtype);
            if !node.srcs.is_empty() {
                let srcs: Vec<String> = node.srcs.iter().map(ToString::to_string).collect();
                let _ = write!(out, " ({})", srcs.join(", "));
            }
            if node.arg != Arg::None {
                let _ = write!(out, "  arg={}", node.arg);
            }
            out.push('\n');
        }
        out
    }
}

impl fmt::Display for UOpGraph {
    /// Prints all nodes in arena order (not toposorted).
    /// For a toposorted view from a specific root, use [`dump`](Self::dump).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, node) in self.nodes.iter().enumerate() {
            write!(f, "  %{i} = {:<12} {:<5}", node.op, node.dtype)?;
            if !node.srcs.is_empty() {
                let srcs: Vec<String> = node.srcs.iter().map(ToString::to_string).collect();
                write!(f, " ({})", srcs.join(", "))?;
            }
            if node.arg != Arg::None {
                write!(f, "  arg={}", node.arg)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consing_identical_nodes_share_id() {
        // Arrange
        let mut g = UOpGraph::new();

        // Act — create two identical constants
        let a = g.const_float(42.0, DType::F32);
        let b = g.const_float(42.0, DType::F32);

        // Assert — same ID, only one node in the arena
        assert_eq!(a, b);
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_hash_consing_different_values_get_different_ids() {
        // Arrange
        let mut g = UOpGraph::new();

        // Act
        let a = g.const_float(1.0, DType::F32);
        let b = g.const_float(2.0, DType::F32);

        // Assert
        assert_ne!(a, b);
        assert_eq!(g.len(), 2);
    }

    #[test]
    fn test_hash_consing_shared_subexpression() {
        // Arrange — build (a + b) * (a + b)
        let mut g = UOpGraph::new();
        let a = g.const_float(1.0, DType::F32);
        let b = g.const_float(2.0, DType::F32);

        // Act — add(a, b) built twice should return the same ID
        let sum1 = g.add_op(a, b);
        let sum2 = g.add_op(a, b);
        let _product = g.mul(sum1, sum2);

        // Assert — the two sums share a node
        assert_eq!(sum1, sum2);
        // 4 nodes total: const(1.0), const(2.0), add, mul
        assert_eq!(g.len(), 4);
    }

    #[test]
    fn test_build_elementwise_add_kernel() {
        // Arrange / Act — build the UOp graph for: out[i] = a[i] + b[i]
        let mut g = UOpGraph::new();

        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let b_ptr = g.param(2, DType::F32);

        let n = g.const_int(1024, DType::I32);
        let idx = g.range(0, n);

        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let b_idx = g.index(b_ptr, idx);
        let b_val = g.load(b_idx, DType::F32);

        let sum = g.add_op(a_val, b_val);

        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, sum);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);

        // Assert — verify graph structure
        let sink_node = g.get(sink);
        assert_eq!(sink_node.op, Op::Sink);
        assert_eq!(sink_node.srcs.len(), 2); // store + end

        let store_node = g.get(store);
        assert_eq!(store_node.op, Op::Store);
        assert_eq!(store_node.srcs.len(), 2); // index + value

        // 14 nodes: 3 params + const + range + 3 index + 2 load + add + store + end + sink
        assert_eq!(g.len(), 14);
    }

    #[test]
    fn test_toposort_sources_before_consumers() {
        // Arrange — build a simple: out = a + b
        let mut g = UOpGraph::new();
        let a = g.const_float(1.0, DType::F32);
        let b = g.const_float(2.0, DType::F32);
        let sum = g.add_op(a, b);

        // Act
        let order = g.toposort(sum);

        // Assert — a and b appear before sum
        let pos_a = order.iter().position(|&id| id == a).unwrap();
        let pos_b = order.iter().position(|&id| id == b).unwrap();
        let pos_sum = order.iter().position(|&id| id == sum).unwrap();
        assert!(pos_a < pos_sum);
        assert!(pos_b < pos_sum);
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_toposort_full_kernel() {
        // Arrange — build out[i] = a[i] + b[i]
        let mut g = UOpGraph::new();
        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let b_ptr = g.param(2, DType::F32);
        let n = g.const_int(1024, DType::I32);
        let idx = g.range(0, n);
        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let b_idx = g.index(b_ptr, idx);
        let b_val = g.load(b_idx, DType::F32);
        let sum = g.add_op(a_val, b_val);
        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, sum);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);

        // Act
        let order = g.toposort(sink);

        // Assert — every node's sources come before it
        let pos_of = |id: UOpId| order.iter().position(|&x| x == id).unwrap();
        for &id in &order {
            for &src in &g.get(id).srcs {
                assert!(pos_of(src) < pos_of(id), "{src} should come before {id}");
            }
        }
        // All 14 nodes should be in the sort
        assert_eq!(order.len(), 14);
    }

    #[test]
    fn test_dump_shows_toposorted_graph() {
        // Arrange
        let mut g = UOpGraph::new();
        let a = g.const_float(1.0, DType::F32);
        let b = g.const_float(2.0, DType::F32);
        let sum = g.add_op(a, b);

        // Act
        let output = g.dump(sum);

        // Assert — should contain all three nodes in dependency order
        assert!(output.contains("Const"));
        assert!(output.contains("Add"));
        let pos_const = output.find("Const").unwrap();
        let pos_add = output.find("Add").unwrap();
        assert!(pos_const < pos_add);
    }

    #[test]
    fn test_dump_full_kernel() {
        // Arrange — out[i] = a[i] + b[i]
        let mut g = UOpGraph::new();
        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let b_ptr = g.param(2, DType::F32);
        let n = g.const_int(1024, DType::I32);
        let idx = g.range(0, n);
        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let b_idx = g.index(b_ptr, idx);
        let b_val = g.load(b_idx, DType::F32);
        let sum = g.add_op(a_val, b_val);
        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, sum);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);

        // Act
        let output = g.dump(sink);

        // Assert — print it so we can see the format, and verify structure
        println!("{output}");
        assert!(output.contains("Param"));
        assert!(output.contains("Range"));
        assert!(output.contains("Load"));
        assert!(output.contains("Add"));
        assert!(output.contains("Store"));
        assert!(output.contains("Sink"));
        // 14 lines (one per node)
        assert_eq!(output.lines().filter(|l| !l.is_empty()).count(), 14);
    }

    #[test]
    fn test_unary_ops_preserve_dtype() {
        // Arrange
        let mut g = UOpGraph::new();
        let x = g.const_float(2.0, DType::F32);

        // Act
        let neg = g.neg(x);
        let exp = g.exp2(x);
        let log = g.log2(x);
        let sqrt = g.sqrt(x);
        let recip = g.reciprocal(x);

        // Assert — all should inherit F32 from their input
        assert_eq!(g.get(neg).dtype, DType::F32);
        assert_eq!(g.get(exp).dtype, DType::F32);
        assert_eq!(g.get(log).dtype, DType::F32);
        assert_eq!(g.get(sqrt).dtype, DType::F32);
        assert_eq!(g.get(recip).dtype, DType::F32);
    }

    #[test]
    fn test_cmplt_returns_bool() {
        // Arrange
        let mut g = UOpGraph::new();
        let a = g.const_float(1.0, DType::F32);
        let b = g.const_float(2.0, DType::F32);

        // Act
        let cmp = g.cmplt(a, b);

        // Assert
        assert_eq!(g.get(cmp).dtype, DType::Bool);
    }

    #[test]
    fn test_where_op() {
        // Arrange — relu: max(x, 0) = where(x > 0, x, 0)
        let mut g = UOpGraph::new();
        let x = g.const_float(5.0, DType::F32);
        let zero = g.const_float(0.0, DType::F32);
        let cond = g.cmplt(zero, x); // 0 < x

        // Act
        let result = g.where_op(cond, x, zero);

        // Assert
        assert_eq!(g.get(result).op, Op::Where);
        assert_eq!(g.get(result).dtype, DType::F32);
        assert_eq!(g.get(result).srcs.len(), 3);
    }
}
