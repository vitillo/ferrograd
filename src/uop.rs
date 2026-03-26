//! # `UOp` — The Core Graph Node
//!
//! Every computation in the compiler is represented as a DAG of `UOp` nodes.
//! Each node holds its op, dtype, children, and op-specific payload.
//!
//! Tinygrad interns `UOp`s so structurally identical nodes are shared. We keep
//! the same idea, but move graph identity onto an explicit device node while
//! compiled kernels live in device-scoped state and the interner lives here.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
use std::rc::{Rc, Weak};

use crate::device::{Buffer, DeviceId};
use crate::dtype::DType;
use crate::shape::Shape;

/// The operations our IR supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// A device identity leaf.
    Device,
    /// A symbolic integer variable with a known min/max range.
    DefineVar,
    /// Bind a concrete value to a symbolic variable for one execution.
    Bind,
    /// A realized data buffer.
    Buffer,
    /// Narrow each dimension to a half-open range without copying.
    Shrink,
    /// Change shape without moving data.
    Reshape,
    /// Reorder dimensions.
    Permute,
    /// Broadcast dimensions of size 1 to a larger size.
    Expand,
    /// Reduce over tensor axes.
    ReduceAxis,
    /// A kernel buffer parameter.
    ParamBuffer,
    /// A kernel scalar parameter.
    ParamScalar,
    /// Loop from 0 to bound.
    Range,
    /// Close a `Range` loop.
    End,
    /// Root of a completed kernel graph.
    Sink,
    /// Tensor-level or kernel-level indexing.
    Index,
    /// Read a value from memory.
    Load,
    /// Write a value to memory.
    Store,
    /// A compile-time constant.
    Const,
    /// `-x`.
    Neg,
    /// `2^x`.
    Exp2,
    /// `log2(x)`.
    Log2,
    /// `sqrt(x)`.
    Sqrt,
    /// `1/x`.
    Reciprocal,
    /// `x + y`.
    Add,
    /// `x * y`.
    Mul,
    /// `max(x, y)`.
    Max,
    /// `x < y`.
    CmpLt,
    /// `if cond { t } else { f }`.
    Where,
    /// Kernel-level reduction placeholder.
    Reduce,
    /// Ordering barrier.
    After,
    /// Declare an accumulator.
    DefineAcc,
    /// Update an accumulator.
    Assign,
}

impl Op {
    /// Whether this op is a scalar ALU operation.
    #[must_use]
    pub fn is_alu(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Mul
                | Self::Max
                | Self::CmpLt
                | Self::Where
                | Self::Neg
                | Self::Exp2
                | Self::Log2
                | Self::Sqrt
                | Self::Reciprocal
        )
    }

    /// Whether this op is a zero-copy movement/view operation.
    #[must_use]
    pub fn is_movement(self) -> bool {
        matches!(
            self,
            Self::Shrink | Self::Reshape | Self::Permute | Self::Expand
        )
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Op-specific payload attached to a `UOp`.
#[derive(Debug, Clone, Default)]
pub enum Arg {
    /// No argument.
    #[default]
    None,
    /// Device identity for `Device`.
    Device(DeviceId),
    /// Symbolic variable metadata: name, min, max.
    Variable(String, i64, i64),
    /// Axis id for `Range`, or sentinel tag for kernel-level `Index`.
    Index(usize),
    /// Kernel buffer parameter slot and flattened element count.
    ParamBuffer(usize, usize),
    /// Kernel scalar parameter slot.
    ParamScalar(usize),
    /// Float literal.
    Float(f64),
    /// Integer literal.
    Int(i64),
    /// Boolean literal.
    Bool(bool),
    /// Realized device buffer.
    Buffer(Buffer),
    /// Per-dimension concrete lengths for `Shrink`.
    Bounds(Box<[usize]>),
    /// Shape payload for `Reshape` and `Expand`.
    Shape(Shape),
    /// Axis payload for `Permute`.
    Axes(Box<[usize]>),
    /// Reduction op and axes.
    Reduce(Op, Box<[usize]>),
}

impl PartialEq for Arg {
    #[allow(clippy::match_same_arms)]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Device(a), Self::Device(b)) => a == b,
            (Self::Variable(name_a, min_a, max_a), Self::Variable(name_b, min_b, max_b)) => {
                name_a == name_b && min_a == min_b && max_a == max_b
            }
            (Self::Index(a), Self::Index(b)) => a == b,
            (Self::ParamBuffer(sa, na), Self::ParamBuffer(sb, nb)) => sa == sb && na == nb,
            (Self::ParamScalar(a), Self::ParamScalar(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Buffer(handle_a), Self::Buffer(handle_b)) => handle_a == handle_b,
            (Self::Bounds(a), Self::Bounds(b)) => a == b,
            (Self::Shape(a), Self::Shape(b)) => a == b,
            (Self::Axes(a), Self::Axes(b)) => a == b,
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
            Self::Device(device) => device.hash(state),
            Self::Variable(name, min, max) => {
                name.hash(state);
                min.hash(state);
                max.hash(state);
            }
            Self::Index(i) => i.hash(state),
            Self::ParamBuffer(slot, numel) => {
                slot.hash(state);
                numel.hash(state);
            }
            Self::ParamScalar(slot) => slot.hash(state),
            Self::Float(value) => value.to_bits().hash(state),
            Self::Int(value) => value.hash(state),
            Self::Bool(value) => value.hash(state),
            Self::Buffer(handle) => {
                handle.hash(state);
            }
            Self::Bounds(lengths) => lengths.hash(state),
            Self::Shape(shape) => shape.hash(state),
            Self::Axes(axes) => axes.hash(state),
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
            Self::Device(device) => write!(f, "{device:?}"),
            Self::Variable(name, min, max) => write!(f, "{name}[{min}, {max}]"),
            Self::Index(i) => write!(f, "{i}"),
            Self::ParamBuffer(slot, numel) => write!(f, "buf_slot={slot},n={numel}"),
            Self::ParamScalar(slot) => write!(f, "scalar_slot={slot}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Buffer(handle) => write!(f, "buf#{}[{}]", handle.id(), handle.numel()),
            Self::Bounds(lengths) => write!(f, "{lengths:?}"),
            Self::Shape(shape) => write!(f, "{shape}"),
            Self::Axes(axes) => write!(f, "{axes:?}"),
            Self::Reduce(op, axes) => write!(f, "{op:?}({axes:?})"),
        }
    }
}

/// The heap-allocated interior of a [`UOp`].
///
/// Every `UOp` is an `Rc<UOpInner>`, so cloning a node is cheap (reference
/// count bump) and two nodes can be compared by pointer identity. The interner
/// guarantees that structurally identical inner values share the same allocation.
pub(crate) struct UOpInner {
    /// Which IR operation this node represents.
    pub(crate) op: Op,
    /// The result data type (e.g. `F32`, `Bool`, `Void` for side-effects).
    pub(crate) dtype: DType,
    /// Operand edges — ordering matters (e.g. `srcs[0]` is the data source
    /// for movement ops, `srcs[1..]` may carry index expressions).
    pub(crate) srcs: Vec<UOp>,
    /// Op-specific payload (shape, axis list, literal value, etc.).
    pub(crate) arg: Arg,
}

/// Value-based key used by the interner's `HashMap` to detect duplicate nodes.
///
/// Unlike [`UOp`] itself (which compares by pointer identity for speed),
/// `UOpKey` implements `Hash`/`Eq` structurally so the interner can find an
/// existing allocation for a `(op, dtype, srcs, arg)` combination.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct UOpKey {
    pub(crate) op: Op,
    pub(crate) dtype: DType,
    pub(crate) srcs: Vec<UOp>,
    pub(crate) arg: Arg,
}

/// A node in the computation graph.
#[derive(Clone)]
pub struct UOp(pub(crate) Rc<UOpInner>);

thread_local! {
    static UOP_INTERNER: RefCell<HashMap<UOpKey, Weak<UOpInner>>> = RefCell::new(HashMap::new());
}

fn intern_uop(op: Op, dtype: DType, srcs: Vec<UOp>, arg: Arg) -> UOp {
    let key = UOpKey {
        op,
        dtype,
        srcs,
        arg,
    };
    UOP_INTERNER.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(existing) = cache.get(&key).and_then(Weak::upgrade) {
            return UOp::from_inner(existing);
        }

        let inner = Rc::new(UOpInner {
            op: key.op,
            dtype: key.dtype,
            srcs: key.srcs.clone(),
            arg: key.arg.clone(),
        });
        cache.insert(key, Rc::downgrade(&inner));
        UOp::from_inner(inner)
    })
}

#[cfg(test)]
pub(crate) fn clear_for_tests() {
    UOP_INTERNER.with(|cache| cache.borrow_mut().clear());
}

impl UOp {
    pub(crate) fn from_inner(inner: Rc<UOpInner>) -> Self {
        Self(inner)
    }

    /// Central constructor — interns the node through the global `UOp` interner.
    ///
    /// All `UOp` creation funnels through here. The interner either
    /// returns an existing `Rc<UOpInner>` if an identical node already exists,
    /// or allocates a new one and caches it. This is how tinygrad achieves
    /// graph deduplication: structurally identical sub-expressions become the
    /// same object in memory, which makes equality checks O(1) and naturally
    /// deduplicates common sub-expressions in the graph.
    fn build(op: Op, dtype: DType, srcs: Vec<Self>, arg: Arg) -> Self {
        intern_uop(op, dtype, srcs, arg)
    }

    /// Derive the device for a non-leaf node from its sources.
    ///
    /// All sources must belong to the same device — cross-device ops are not
    /// supported (tinygrad enforces the same constraint). Leaf nodes like
    /// `Const` and `Buffer` carry an explicit `Device` source instead.
    fn device_from_srcs(srcs: &[Self]) -> DeviceId {
        let device = srcs
            .first()
            .map(Self::device)
            .expect("leaf UOps must be created with explicit device");
        assert!(
            srcs.iter().all(|src| src.device() == device),
            "all UOp sources must belong to the same device"
        );
        device
    }

    /// Create a non-leaf node by deriving the device from its sources.
    ///
    /// This is a raw IR constructor. Tensor-level API methods do user-facing
    /// shape validation; lowering and rewrite passes can build intermediate IR
    /// directly and rely on later stages to reject malformed graphs.
    #[must_use]
    pub(crate) fn new(op: Op, dtype: DType, srcs: Vec<Self>, arg: Arg) -> Self {
        Self::device_from_srcs(&srcs);
        Self::build(op, dtype, srcs, arg)
    }

    /// Create a device identity leaf.
    #[must_use]
    pub(crate) fn device_uop(device: DeviceId) -> Self {
        Self::build(Op::Device, DType::Void, vec![], Arg::Device(device))
    }

    /// Create a kernel buffer parameter leaf on `device`.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn param_buffer(slot: usize, dtype: DType, numel: usize, device: DeviceId) -> Self {
        Self::build(
            Op::ParamBuffer,
            dtype,
            vec![Self::device_uop(device)],
            Arg::ParamBuffer(slot, numel),
        )
    }

    /// Create a kernel scalar parameter leaf on `device`.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn param_scalar(slot: usize, dtype: DType, device: DeviceId) -> Self {
        Self::build(
            Op::ParamScalar,
            dtype,
            vec![Self::device_uop(device)],
            Arg::ParamScalar(slot),
        )
    }

    /// Create a buffer leaf on `device`.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn buffer(buffer: Buffer, dtype: DType, device: DeviceId) -> Self {
        Self::build(
            Op::Buffer,
            dtype,
            vec![Self::device_uop(device)],
            Arg::Buffer(buffer),
        )
    }

    /// Create a float constant on `device`.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn const_float(value: f64, dtype: DType, device: DeviceId) -> Self {
        Self::build(
            Op::Const,
            dtype,
            vec![Self::device_uop(device)],
            Arg::Float(value),
        )
    }

    /// Create an integer constant on `device`.
    #[must_use]
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn const_int(value: i64, dtype: DType, device: DeviceId) -> Self {
        Self::build(
            Op::Const,
            dtype,
            vec![Self::device_uop(device)],
            Arg::Int(value),
        )
    }

    /// Create a boolean constant on `device`.
    #[must_use]
    pub(crate) fn const_bool(value: bool, dtype: DType, device: DeviceId) -> Self {
        Self::build(
            Op::Const,
            dtype,
            vec![Self::device_uop(device)],
            Arg::Bool(value),
        )
    }

    /// Build a constant-filled tensor expression (e.g. all-zeros or all-ones).
    ///
    /// Creates `Expand(Reshape(Const(value), [1,…,1]), shape)` — the same
    /// pattern tinygrad uses for `full`. If every dimension is already 1, the
    /// expand is elided.
    #[must_use]
    pub(crate) fn full(shape: &Shape, dtype: DType, device: DeviceId, value: f64) -> Self {
        let base_shape = Shape::new(vec![1; shape.ndim()]);
        let scalar = Self::const_float(value, dtype, device);
        let base = Self::reshape(scalar, base_shape);
        if shape.iter().all(|&dim| dim == 1) {
            return base;
        }
        Self::expand(base, shape.clone())
    }

    /// Return the device this node belongs to.
    ///
    /// # Panics
    ///
    /// Panics if a malformed non-device node has no source from which to
    /// derive its device.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        match self.op() {
            Op::Device => match self.arg() {
                Arg::Device(device) => *device,
                _ => unreachable!("Device UOp must carry Arg::Device"),
            },
            _ => self
                .srcs()
                .first()
                .map(Self::device)
                .expect("non-device UOps must derive device from sources"),
        }
    }

    /// Return the op.
    #[must_use]
    pub fn op(&self) -> Op {
        self.0.op
    }

    /// Return the result dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    /// Return the source nodes.
    #[must_use]
    pub fn srcs(&self) -> &[UOp] {
        &self.0.srcs
    }

    /// Return the op-specific payload.
    #[must_use]
    pub fn arg(&self) -> &Arg {
        &self.0.arg
    }

    /// Cast the `Rc` pointer to a `usize` for use as a hash key.
    ///
    /// Because the interner guarantees structural uniqueness, pointer identity
    /// is equivalent to value equality — so we can hash/compare `UOp`s in O(1)
    /// instead of walking the entire sub-graph.
    fn ptr_id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// Compute the tensor shape carried by this node.
    #[must_use]
    pub fn shape(&self) -> Option<Shape> {
        match self.op() {
            Op::Buffer => {
                let Arg::Buffer(handle) = self.arg() else {
                    return None;
                };
                Some(Shape::flat(handle.numel()))
            }
            Op::ParamBuffer => {
                let Arg::ParamBuffer(_, numel) = self.arg() else {
                    return None;
                };
                Some(Shape::flat(*numel))
            }
            Op::Shrink => {
                let Arg::Bounds(lengths) = self.arg() else {
                    return None;
                };
                Some(Shape::new(lengths.to_vec()))
            }
            Op::Reshape | Op::Expand => {
                let Arg::Shape(shape) = self.arg() else {
                    return None;
                };
                Some(shape.clone())
            }
            Op::Permute => {
                let Arg::Axes(order) = self.arg() else {
                    return None;
                };
                let src_shape = self.srcs()[0].shape()?;
                Some(Shape::new(
                    order.iter().map(|&axis| src_shape[axis]).collect(),
                ))
            }
            Op::ReduceAxis => {
                let Arg::Reduce(_, axes) = self.arg() else {
                    return None;
                };
                let src_shape = self.srcs()[0].shape()?;
                Some(Shape::new(
                    src_shape
                        .iter()
                        .enumerate()
                        .map(|(axis, &dim)| if axes.contains(&axis) { 1 } else { dim })
                        .collect(),
                ))
            }
            Op::Const => Some(Shape::flat(1)),
            Op::After => self.srcs()[0].shape(),
            Op::ParamScalar | Op::Device | Op::DefineVar | Op::Bind => None,
            op if op.is_alu() => self.srcs()[0].shape(),
            _ => None,
        }
    }

    /// Whether this node has a concrete buffer identity under movement ops.
    #[must_use]
    pub fn has_buffer_identity(&self) -> bool {
        match self.op() {
            Op::Buffer => true,
            Op::After => self.srcs()[0].has_buffer_identity(),
            op if op.is_movement() => self.srcs()[0].has_buffer_identity(),
            _ => false,
        }
    }

    /// Whether this node is a literal constant.
    #[must_use]
    pub fn is_const(&self) -> bool {
        self.op() == Op::Const
    }

    /// Whether this node is a float constant with the given value.
    #[must_use]
    pub fn is_const_float(&self, val: f64) -> bool {
        self.op() == Op::Const && *self.arg() == Arg::Float(val)
    }

    /// Whether this node is an int constant with the given value.
    #[must_use]
    pub fn is_const_int(&self, val: i64) -> bool {
        self.op() == Op::Const && *self.arg() == Arg::Int(val)
    }

    /// Whether this constant is zero.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.is_const_float(0.0) || self.is_const_int(0)
    }

    /// Whether this constant is one.
    #[must_use]
    pub fn is_one(&self) -> bool {
        self.is_const_float(1.0) || self.is_const_int(1)
    }

    #[must_use]
    pub(crate) fn reshape(src: Self, shape: Shape) -> Self {
        let dtype = src.dtype();
        Self::new(Op::Reshape, dtype, vec![src], Arg::Shape(shape))
    }

    #[must_use]
    pub(crate) fn shrink(src: Self, starts: &[Self], lengths: &[usize]) -> Self {
        let dtype = src.dtype();
        let mut srcs = vec![src];
        srcs.extend_from_slice(starts);
        Self::new(
            Op::Shrink,
            dtype,
            srcs,
            Arg::Bounds(lengths.to_vec().into_boxed_slice()),
        )
    }

    #[must_use]
    pub(crate) fn permute(src: Self, axes: &[usize]) -> Self {
        let dtype = src.dtype();
        Self::new(
            Op::Permute,
            dtype,
            vec![src],
            Arg::Axes(axes.to_vec().into_boxed_slice()),
        )
    }

    #[must_use]
    pub(crate) fn expand(src: Self, shape: Shape) -> Self {
        let dtype = src.dtype();
        Self::new(Op::Expand, dtype, vec![src], Arg::Shape(shape))
    }

    #[must_use]
    pub(crate) fn reduce_axis(src: Self, reduce_op: Op, axes: &[usize]) -> Self {
        let dtype = src.dtype();
        Self::new(
            Op::ReduceAxis,
            dtype,
            vec![src],
            Arg::Reduce(reduce_op, axes.to_vec().into_boxed_slice()),
        )
    }

    #[must_use]
    pub(crate) fn add(a: Self, b: Self) -> Self {
        let dtype = a.dtype();
        Self::new(Op::Add, dtype, vec![a, b], Arg::None)
    }

    #[must_use]
    pub(crate) fn mul(a: Self, b: Self) -> Self {
        let dtype = a.dtype();
        Self::new(Op::Mul, dtype, vec![a, b], Arg::None)
    }

    #[must_use]
    pub(crate) fn neg(a: Self) -> Self {
        let dtype = a.dtype();
        Self::new(Op::Neg, dtype, vec![a], Arg::None)
    }

    #[must_use]
    pub(crate) fn reciprocal(a: Self) -> Self {
        let dtype = a.dtype();
        Self::new(Op::Reciprocal, dtype, vec![a], Arg::None)
    }

    #[must_use]
    pub(crate) fn cmplt(a: Self, b: Self) -> Self {
        Self::new(Op::CmpLt, DType::Bool, vec![a, b], Arg::None)
    }

    #[must_use]
    pub(crate) fn where_(cond: Self, t: Self, f: Self) -> Self {
        let dtype = t.dtype();
        Self::new(Op::Where, dtype, vec![cond, t, f], Arg::None)
    }

    #[must_use]
    pub(crate) fn store(dest: Self, value: Self) -> Self {
        Self::new(Op::Store, DType::Void, vec![dest, value], Arg::None)
    }

    #[must_use]
    pub(crate) fn after(value: Self, effect: Self) -> Self {
        Self::new(Op::After, value.dtype(), vec![value, effect], Arg::None)
    }

    #[must_use]
    pub(crate) fn sink(stores: Vec<Self>) -> Self {
        Self::build(Op::Sink, DType::Void, stores, Arg::None)
    }

    /// Iterative post-order DFS. Returns nodes in dependency order.
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
            .map(|(i, node)| (node, i))
            .collect();

        let mut out = String::new();
        for node in &order {
            let idx = id_map[&node];
            let srcs: Vec<String> = node
                .srcs()
                .iter()
                .map(|src| format!("%{}", id_map[&src]))
                .collect();
            let src_str = if srcs.is_empty() {
                String::new()
            } else {
                format!(" ({})", srcs.join(", "))
            };
            let arg_str = match node.arg() {
                Arg::None => String::new(),
                arg => format!("  arg={arg}"),
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

/// Build a map from each node to the nodes that consume it as a source.
///
/// This is a common graph analysis used by both scheduling (to detect
/// multi-consumer nodes that need materialization) and autograd (to propagate
/// gradients along consumer edges).
#[must_use]
pub fn build_consumer_map(order: &[UOp]) -> HashMap<UOp, Vec<UOp>> {
    let mut consumers: HashMap<UOp, Vec<UOp>> = HashMap::new();
    for node in order {
        for src in node.srcs() {
            consumers.entry(src.clone()).or_default().push(node.clone());
        }
    }
    consumers
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
    use crate::device::DeviceId;

    #[test]
    fn test_build_elementwise_add_kernel() {
        let device = DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 1024, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, 1024, device);
        let b_ptr = UOp::param_buffer(2, DType::F32, 1024, device);
        let n = UOp::const_int(1024, DType::I32, device);
        let idx = UOp::new(Op::Range, DType::I32, vec![n], Arg::Index(0));
        let a_idx = UOp::new(
            Op::Index,
            a_ptr.dtype(),
            vec![a_ptr, idx.clone()],
            Arg::None,
        );
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let b_idx = UOp::new(
            Op::Index,
            b_ptr.dtype(),
            vec![b_ptr, idx.clone()],
            Arg::None,
        );
        let b_val = UOp::new(Op::Load, DType::F32, vec![b_idx], Arg::None);
        let sum = UOp::add(a_val, b_val);
        let out_idx = UOp::new(
            Op::Index,
            out_ptr.dtype(),
            vec![out_ptr, idx.clone()],
            Arg::None,
        );
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, sum], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store.clone()], Arg::None);
        let sink = UOp::sink(vec![store, end]);

        assert_eq!(sink.op(), Op::Sink);
        assert_eq!(sink.srcs().len(), 2);
    }

    #[test]
    fn test_toposort_sources_before_consumers() {
        let a = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let b = UOp::const_float(2.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(a.clone(), b.clone());

        let order = sum.toposort();

        let pos = |u: &UOp| order.iter().position(|node| node == u).unwrap();
        assert!(pos(&a) < pos(&sum));
        assert!(pos(&b) < pos(&sum));
    }

    #[test]
    fn test_shared_node_appears_once_in_toposort() {
        let a = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(a.clone(), a.clone());

        let order = sum.toposort();

        let count = order.iter().filter(|node| *node == &a).count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_dump_shows_toposorted_graph() {
        let a = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let b = UOp::const_float(2.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(a, b);

        let dump = sum.dump();

        assert!(dump.contains("Const"));
        assert!(dump.contains("Add"));
        assert!(dump.contains("%0"));
    }

    #[test]
    fn test_pointer_identity() {
        let a = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let b = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let a_clone = a.clone();

        assert_eq!(a, a_clone);
        assert_eq!(a, b);
    }

    #[test]
    fn test_uop_new_interns_identical_nodes() {
        let device = DeviceId::Cpu;
        let backend = crate::device::get(device);
        let buffer = backend.allocate(DType::F32, 2);
        backend.copy_from_host(&buffer, bytemuck::cast_slice(&[1.0_f32, 2.0]));
        let left = UOp::buffer(buffer.clone(), DType::F32, device);
        let right = UOp::buffer(buffer, DType::F32, device);

        assert_eq!(left, right);

        let sum_a = UOp::add(left.clone(), left);
        let sum_b = UOp::add(right.clone(), right);
        assert_eq!(sum_a, sum_b);
    }
}
