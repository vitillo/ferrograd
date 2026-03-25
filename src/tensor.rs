//! # Tensor API — lazy evaluation, mutation, and training
//!
//! `Tensor` is a shared handle to lazy `UOp` graph state. Clones alias the same
//! tensor identity, so realization rewrites, in-place updates, and gradient
//! storage are visible through all handles.
//!
//! ## Lazy evaluation model
//!
//! Operations like `add`, `matmul`, and `reshape` do not compute anything
//! immediately. They append nodes to an internal `UOp` directed acyclic graph
//! (the "lazy graph"). Computation only happens when [`Tensor::realize`] or
//! [`Tensor::realize_many`] is called, which lowers the graph through a
//! multi-stage pipeline:
//!
//! 1. **Schedule** — partition the lazy graph into individual kernels and
//!    identify buffer inputs/outputs.
//! 2. **Rangeify** — lower high-level tensor ops (reshape, permute, reduce) into
//!    explicit index arithmetic with range loops, producing the kernel IR.
//! 3. **Symbolic simplification** — constant-fold and simplify the index math.
//! 4. **Codegen** — render the kernel IR into C source code.
//! 5. **Compile** — invoke the platform C compiler (via the `Device` trait).
//! 6. **Execute** — run the compiled kernel, writing results into device buffers.
//!
//! This mirrors tinygrad's `Tensor` class, where `.realize()` triggers the same
//! lazy-graph → schedule → lower → codegen → run pipeline.
//!
//! ## Shared handle pattern
//!
//! `Tensor` wraps `Rc<RefCell<TensorInner>>`, so cloning a tensor produces a
//! second handle to the *same* underlying state. This is essential because
//! realization must rewrite every handle's `uop` from the old lazy expression
//! to a realized buffer reference. Without shared identity, an optimizer holding
//! a parameter clone would go stale after the training loop realizes it. This
//! matches tinygrad, where `Tensor` objects are mutable Python references that
//! get rewritten in-place by the scheduler.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use crate::codegen::{ClangRenderer, Renderer};
use crate::device::{Buffer, DeviceId, KernelArg};
use crate::dtype::DType;
use crate::gradient;
use crate::runtime;
use crate::schedule::{self, rangeify::rangeify, ScheduleItem};
use crate::shape::Shape;
use crate::uop::{Arg, Op, UOp};

static DEBUG: LazyLock<u8> = LazyLock::new(|| {
    std::env::var("DEBUG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
});

/// Global monotonic counter for naming compiled kernels (`kernel_0`, `kernel_1`, …).
/// Also used in debug output to correlate log lines with specific kernel invocations.
static KERNEL_COUNT: AtomicUsize = AtomicUsize::new(0);

// Weak references to every live tensor on this thread.
//
// After realization, the scheduler produces a replacement map (old lazy UOp →
// new buffer UOp). We walk this list to patch every tensor that transitively
// referenced a realized subgraph, so all handles stay consistent. Weak refs
// let tensors be dropped normally; dead entries are pruned during traversal.
thread_local! {
    static LIVE_TENSORS: RefCell<Vec<Weak<RefCell<TensorInner>>>> = const { RefCell::new(Vec::new()) };
}

/// The mutable state behind every [`Tensor`] handle.
struct TensorInner {
    /// The current lazy graph expression (pre-realization) or realized buffer
    /// reference (post-realization). Rewritten in-place by `apply_map_to_tensors`.
    uop: UOp,
    /// Whether this tensor participates in autograd. Set at creation or via
    /// `with_requires_grad`; never toggled by the framework itself.
    requires_grad: bool,
    /// Accumulated gradient from `backward()`, if any.
    grad: Option<Tensor>,
}

/// A lazily-evaluated tensor bound to a specific device.
#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorInner>>);

impl Tensor {
    /// Create a tensor handle and register it in `LIVE_TENSORS` so
    /// post-realization graph rewrites can reach it.
    fn new(uop: UOp, requires_grad: bool) -> Self {
        let tensor = Self(Rc::new(RefCell::new(TensorInner {
            uop,
            requires_grad,
            grad: None,
        })));
        LIVE_TENSORS.with(|live| live.borrow_mut().push(Rc::downgrade(&tensor.0)));
        tensor
    }

    /// Read the current lazy graph node (or realized buffer reference).
    fn uop(&self) -> UOp {
        self.0.borrow().uop.clone()
    }

    /// Whether two tensor handles alias the same shared tensor identity.
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// Overwrite the graph node, visible through all cloned handles.
    fn set_uop(&self, uop: UOp) {
        self.0.borrow_mut().uop = uop;
    }

    fn grad_inner(&self) -> Option<Tensor> {
        self.0.borrow().grad.clone()
    }

    fn set_grad_inner(&self, grad: Option<Tensor>) {
        self.0.borrow_mut().grad = grad;
    }

    /// Return the "target" buffer identity, stripping any `After` wrapper.
    ///
    /// `assign` wraps a tensor as `After(base, Store(base, value))` to express
    /// an in-place write. When we need the underlying buffer identity (e.g. for
    /// a second assign), we strip the `After` to get back to `base`.
    fn target_uop(&self) -> UOp {
        let current = self.uop();
        if current.op() == Op::After {
            return current.srcs()[0].clone();
        }
        current
    }

    /// Collect all live tensor handles on this thread, pruning dead weak refs.
    ///
    /// Used after realization to apply the replacement map to every tensor
    /// whose graph references a now-realized subexpression.
    fn live_tensors() -> Vec<Self> {
        LIVE_TENSORS.with(|live| {
            let mut live = live.borrow_mut();
            let mut tensors = Vec::new();
            live.retain(|weak| {
                let Some(inner) = weak.upgrade() else {
                    return false;
                };
                tensors.push(Self(inner));
                true
            });
            tensors
        })
    }

    /// Rewrite every live tensor's graph using the scheduler's replacement map.
    ///
    /// After realization, each realized subexpression is replaced by a direct
    /// buffer reference. This walks all live tensors, finds those whose graphs
    /// transitively contain a replaced node, and substitutes in one pass per
    /// device (so shared subgraphs are rewritten only once).
    fn apply_map_to_tensors(replacements: &HashMap<UOp, UOp>) {
        if replacements.is_empty() {
            return;
        }
        let keys: HashSet<UOp> = replacements.keys().cloned().collect();
        let mut affected_by_device: HashMap<DeviceId, Vec<(Self, UOp)>> = HashMap::new();
        for tensor in Self::live_tensors() {
            let current = tensor.uop();
            let uses_replacement = current.toposort().iter().any(|node| keys.contains(node));
            if !uses_replacement {
                continue;
            }
            affected_by_device
                .entry(current.device())
                .or_default()
                .push((tensor, current));
        }

        for affected in affected_by_device.into_values() {
            // Rewrite all affected roots in one combined sink so shared subgraphs
            // are only substituted once, matching tinygrad's approach more closely.
            let roots: Vec<UOp> = affected.iter().map(|(_, root)| root.clone()).collect();
            let rewritten_roots = substitute_roots_with_map(&roots, replacements);

            for ((tensor, current), rewritten) in affected.into_iter().zip(rewritten_roots) {
                if rewritten != current {
                    tensor.set_uop(rewritten);
                }
            }
        }
    }

    /// Create a tensor from a float slice on the default CPU device.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not match `shape`.
    #[must_use]
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let shape = Shape::from(shape);
        assert_eq!(
            data.len(),
            shape.numel(),
            "data length {} doesn't match shape {shape:?}",
            data.len()
        );
        let state = runtime::state(DeviceId::Cpu);
        let buffer_id = state.store_buffer(Buffer::from_f32(data));
        let buffer = UOp::buffer(buffer_id, DType::F32, shape.numel(), DeviceId::Cpu);
        Self::new(UOp::reshape(buffer, shape), false)
    }

    /// Create a tensor filled with zeros on the default CPU device.
    #[must_use]
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let shape = Shape::from(shape);
        let state = runtime::state(DeviceId::Cpu);
        let buffer_id = state.store_buffer(state.device().allocate(dtype, shape.numel()));
        let buffer = UOp::buffer(buffer_id, dtype, shape.numel(), DeviceId::Cpu);
        Self::new(UOp::reshape(buffer, shape), false)
    }

    /// Create a tensor filled with ones on the default CPU device.
    #[must_use]
    pub fn ones(shape: &[usize]) -> Self {
        let numel = shape.iter().product();
        Self::from_slice(&vec![1.0_f32; numel], shape)
    }

    /// Create a scalar tensor of shape `[1]` on the default CPU device.
    #[must_use]
    pub fn scalar(value: f32) -> Self {
        Self::from_slice(&[value], &[1])
    }

    /// Return the owning device.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        self.uop().device()
    }

    /// Return the tensor dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.uop().dtype()
    }

    /// Return the tensor shape.
    ///
    /// # Panics
    ///
    /// Panics if the underlying graph node does not carry tensor shape
    /// information.
    #[must_use]
    pub fn shape(&self) -> Shape {
        self.uop().shape().expect("tensor must have shape")
    }

    /// Return the flat element count.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    /// Return the rank.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape().ndim()
    }

    /// Whether this tensor already points at a direct realized buffer.
    #[must_use]
    pub fn is_realized(&self) -> bool {
        let uop = self.uop();
        match uop.op() {
            Op::Buffer => true,
            Op::Reshape => uop.srcs()[0].op() == Op::Buffer,
            _ => false,
        }
    }

    /// Whether this tensor should participate in gradient computation.
    #[must_use]
    pub fn requires_grad(&self) -> bool {
        self.0.borrow().requires_grad
    }

    /// Return the currently stored gradient.
    #[must_use]
    pub fn grad(&self) -> Option<Self> {
        self.grad_inner()
    }

    pub(crate) fn clear_grad(&self) {
        self.set_grad_inner(None);
    }

    /// Mark this tensor as participating in gradient computation.
    ///
    /// This mutates the shared tensor handle and returns `self` for chaining.
    #[allow(clippy::must_use_candidate, clippy::return_self_not_must_use)]
    pub fn with_requires_grad(self, requires_grad: bool) -> Self {
        self.0.borrow_mut().requires_grad = requires_grad;
        self
    }

    /// Return a detached tensor handle with the same graph value.
    #[must_use]
    pub fn detach(&self) -> Self {
        Self::new(self.uop(), false)
    }

    /// Replace this tensor's graph with another tensor's graph.
    ///
    /// This mutates the shared tensor handle and returns a clone for chaining.
    ///
    /// # Panics
    ///
    /// Panics if the shapes do not match.
    #[allow(clippy::must_use_candidate, clippy::return_self_not_must_use)]
    pub fn replace(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "replace: shape mismatch {:?} != {:?}",
            self.shape(),
            other.shape()
        );
        self.set_uop(other.uop());
        self.clone()
    }

    /// Assign a new lazy value into this tensor.
    ///
    /// The assignment is represented in the graph as `After(base, Store(base,
    /// value))`, so realization can lower it as an in-place write. This mutates
    /// the shared tensor handle and returns a clone for chaining.
    ///
    /// # Panics
    ///
    /// Panics if shape, dtype, or device do not match.
    #[allow(clippy::must_use_candidate, clippy::return_self_not_must_use)]
    pub fn assign(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "assign: shape mismatch {:?} != {:?}",
            self.shape(),
            other.shape()
        );
        assert_eq!(self.dtype(), other.dtype(), "assign: dtype mismatch");
        assert_eq!(self.device(), other.device(), "assign: device mismatch");

        let target = self.target_uop();
        assert!(
            target.has_buffer_identity(),
            "assign requires a tensor with concrete buffer identity"
        );
        let effect = UOp::store(target.clone(), other.uop());
        self.set_uop(UOp::after(target, effect));
        self.clone()
    }

    fn unary(&self, op: Op) -> Self {
        Self::new(UOp::new(op, self.dtype(), vec![self.uop()], Arg::None), self.requires_grad())
    }

    fn binary(&self, other: &Self, op: Op, out_dtype: DType) -> Self {
        assert!(
            self.device() == other.device(),
            "binary ops require tensors on the same device"
        );
        assert_eq!(self.shape(), other.shape(), "shape mismatch for {op:?}");
        Self::new(
            UOp::new(op, out_dtype, vec![self.uop(), other.uop()], Arg::None),
            self.requires_grad() || other.requires_grad(),
        )
    }

    fn broadcasted(&self, other: &Self, op: Op, out_dtype: DType) -> Self {
        if self.shape() == other.shape() {
            return self.binary(other, op, out_dtype);
        }
        let (left, right) = broadcast_shapes(self, other);
        left.binary(&right, op, out_dtype)
    }

    /// Element-wise addition with broadcasting.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.broadcasted(other, Op::Add, self.dtype())
    }

    /// Element-wise multiplication with broadcasting.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        self.broadcasted(other, Op::Mul, self.dtype())
    }

    /// Element-wise negation.
    #[must_use]
    pub fn neg(&self) -> Self {
        self.unary(Op::Neg)
    }

    /// Element-wise subtraction with broadcasting.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Element-wise reciprocal.
    #[must_use]
    pub fn reciprocal(&self) -> Self {
        self.unary(Op::Reciprocal)
    }

    /// Element-wise base-2 exponential.
    #[must_use]
    pub fn exp2(&self) -> Self {
        self.unary(Op::Exp2)
    }

    /// Element-wise base-2 logarithm.
    #[must_use]
    pub fn log2(&self) -> Self {
        self.unary(Op::Log2)
    }

    /// Element-wise square root.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        self.unary(Op::Sqrt)
    }

    /// `max(0, x)` implemented with `Where`.
    #[must_use]
    pub fn relu(&self) -> Self {
        let zero = Self::zeros(self.shape().as_slice(), self.dtype());
        let cond = zero.binary(self, Op::CmpLt, DType::Bool);
        Self::new(
            UOp::new(
                Op::Where,
                self.dtype(),
                vec![cond.uop(), self.uop(), zero.uop()],
                Arg::None,
            ),
            self.requires_grad(),
        )
    }

    /// Reshape without moving data.
    ///
    /// # Panics
    ///
    /// Panics if `new_shape` changes the total element count.
    #[must_use]
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_shape = Shape::from(new_shape);
        if self.shape() == new_shape {
            return self.clone();
        }
        assert_eq!(
            self.numel(),
            new_shape.numel(),
            "reshape: numel mismatch for {:?} -> {:?}",
            self.shape(),
            new_shape
        );
        Self::new(UOp::reshape(self.uop(), new_shape), self.requires_grad())
    }

    /// Slice a single dimension without copying.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of range or if `start..start + len` is not a
    /// valid half-open interval for that dimension.
    #[must_use]
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        let start = bound_const(start, self.device());
        self.narrow_with_start(dim, &start, len)
    }

    fn narrow_with_start(&self, dim: usize, start: &UOp, len: usize) -> Self {
        let shape = self.shape();
        assert!(dim < shape.ndim(), "narrow: dim {dim} out of range");
        let start_value = bound_value(start);
        let end = start_value
            .checked_add(len)
            .expect("narrow: start + len overflowed usize");
        assert!(
            end <= shape[dim],
            "narrow: range [{start_value}, {end}) out of bounds for axis {dim} with size {}",
            shape[dim]
        );
        if start_value == 0 && len == shape[dim] {
            return self.clone();
        }

        let starts: Vec<UOp> = shape
            .iter()
            .enumerate()
            .map(|(axis, &_size)| {
                if axis == dim {
                    start.clone()
                } else {
                    UOp::const_int(0, DType::I32, self.device())
                }
            })
            .collect();
        let lengths: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(axis, &size)| if axis == dim { len } else { size })
            .collect();

        Self::new(
            UOp::shrink(self.uop(), &starts, &lengths),
            self.requires_grad(),
        )
    }

    /// Reorder dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `order` is not a permutation of `0..self.ndim()`.
    #[must_use]
    pub fn permute(&self, order: &[usize]) -> Self {
        let ndim = self.ndim();
        assert_eq!(order.len(), ndim, "permute: wrong number of axes");
        let mut seen = vec![false; ndim];
        for &axis in order {
            assert!(axis < ndim, "permute: axis {axis} out of range");
            assert!(!seen[axis], "permute: axis {axis} duplicated");
            seen[axis] = true;
        }
        Self::new(UOp::permute(self.uop(), order), self.requires_grad())
    }

    /// Broadcast size-1 dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `new_shape` is not a valid broadcast target.
    #[must_use]
    pub fn expand(&self, new_shape: &[usize]) -> Self {
        let new_shape = Shape::from(new_shape);
        if self.shape() == new_shape {
            return self.clone();
        }
        let src_shape = self.shape();
        assert_eq!(
            src_shape.ndim(),
            new_shape.ndim(),
            "expand: rank mismatch for {src_shape:?} -> {new_shape:?}"
        );
        assert!(
            src_shape
                .iter()
                .zip(new_shape.iter())
                .all(|(&src_dim, &dst_dim)| src_dim == dst_dim || src_dim == 1),
            "expand: incompatible source shape {src_shape:?} -> {new_shape:?}"
        );
        Self::new(UOp::expand(self.uop(), new_shape), self.requires_grad())
    }

    /// Sum over the given axes.
    ///
    /// # Panics
    ///
    /// Panics if any axis is out of range.
    #[must_use]
    pub fn sum(&self, axes: &[usize]) -> Self {
        let ndim = self.ndim();
        assert!(
            axes.iter().all(|&axis| axis < ndim),
            "sum: axes {axes:?} out of range for ndim {ndim}"
        );
        Self::new(
            UOp::reduce_axis(self.uop(), Op::Add, axes),
            self.requires_grad(),
        )
    }

    /// Max over the given axes.
    ///
    /// # Panics
    ///
    /// Panics if any axis is out of range.
    #[must_use]
    pub fn max(&self, axes: &[usize]) -> Self {
        let ndim = self.ndim();
        assert!(
            axes.iter().all(|&axis| axis < ndim),
            "max: axes {axes:?} out of range for ndim {ndim}"
        );
        Self::new(
            UOp::reduce_axis(self.uop(), Op::Max, axes),
            self.requires_grad(),
        )
    }

    /// Natural exponential implemented as `2^(x * log2(e))`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn exp(&self) -> Self {
        let log2e = Self::scalar(std::f64::consts::LOG2_E as f32);
        self.mul(&log2e).exp2()
    }

    /// Natural logarithm implemented as `log2(x) * ln(2)`.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn log(&self) -> Self {
        let ln2 = Self::scalar(std::f64::consts::LN_2 as f32);
        self.log2().mul(&ln2)
    }

    /// Log-softmax over one axis, computed in a numerically stable way.
    ///
    /// # Panics
    ///
    /// Panics if `axis` is out of range.
    #[must_use]
    pub fn log_softmax(&self, axis: usize) -> Self {
        assert!(axis < self.ndim(), "log_softmax: axis {axis} out of range");
        let max_x = self.max(&[axis]);
        let shifted = self.sub(&max_x);
        let sum_exp = shifted.exp().sum(&[axis]);
        shifted.sub(&sum_exp.log())
    }

    /// Cross-entropy loss between logits and dense target probabilities.
    ///
    /// This follows tinygrad's choice to keep losses on `Tensor` instead of in
    /// a separate `nn::loss` module. The current Rust version keeps the first
    /// implementation simple and expects `targets` to already have the same
    /// shape as `self`, for example one-hot labels.
    ///
    /// The class axis matches tinygrad's default: axis `0` for rank-1 logits
    /// and axis `1` otherwise. The returned loss is the mean over the
    /// non-class dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the target shape does not match the logits.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cross_entropy(&self, targets: &Self) -> Self {
        assert_eq!(
            self.shape(),
            targets.shape(),
            "cross_entropy: target shape {:?} must match logits shape {:?}",
            targets.shape(),
            self.shape()
        );
        let classes_axis = usize::from(self.ndim() != 1);
        let log_probs = self.log_softmax(classes_axis);
        let per_sample = targets.mul(&log_probs).sum(&[classes_axis]).neg();
        let reduction_axes: Vec<usize> = (0..per_sample.ndim()).collect();
        let scale = Self::scalar(1.0 / per_sample.numel() as f32);
        per_sample.sum(&reduction_axes).mul(&scale)
    }

    /// Matrix multiply `[M,K] @ [K,N] -> [M,N]`.
    ///
    /// # Panics
    ///
    /// Panics if either tensor is not rank-2, if the inner dimensions do not
    /// match, or if the tensors belong to different devices.
    #[must_use]
    #[allow(clippy::many_single_char_names)]
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2D");
        assert_eq!(other.ndim(), 2, "matmul: rhs must be 2D");
        let lhs_shape = self.shape();
        let rhs_shape = other.shape();
        let (m, k) = (lhs_shape[0], lhs_shape[1]);
        let (k2, n) = (rhs_shape[0], rhs_shape[1]);
        assert_eq!(k, k2, "matmul: inner dim mismatch ({k} vs {k2})");

        let a = self.reshape(&[m, 1, k]).expand(&[m, n, k]);
        let b = other
            .permute(&[1, 0])
            .reshape(&[1, n, k])
            .expand(&[m, n, k]);
        a.mul(&b).sum(&[2]).reshape(&[m, n])
    }

    /// Realize several tensors together, sharing one schedule for common work.
    pub fn realize_many(tensors: &[&Self]) {
        let mut roots = Vec::new();
        let mut seen = HashSet::new();
        for tensor in tensors {
            if tensor.is_realized() {
                continue;
            }
            let root = tensor.uop();
            if seen.insert(root.clone()) {
                roots.push(root);
            }
        }
        if roots.is_empty() {
            return;
        }

        let plan = schedule::schedule_many(&roots);
        for item in &plan.items {
            Self::execute_item(item);
        }
        Self::apply_map_to_tensors(&plan.replacements);
    }

    /// Lower the lazy graph to kernels, compile, and execute them.
    #[must_use]
    pub fn realize(&self) -> Self {
        Self::realize_many(&[self]);
        self.clone()
    }

    /// Run the full pipeline for a single scheduled kernel: rangeify → symbolic
    /// simplification → codegen → compile (or cache hit) → execute.
    fn execute_item(item: &ScheduleItem) {
        let state = runtime::state(item.sink.device());
        let debug = *DEBUG;
        let lowered = rangeify(&item.sink);
        let lowered =
            crate::rewrite::graph_rewrite(&lowered, &crate::rewrite::symbolic_simple, "symbolic");
        assert_codegen_ready(&lowered);

        let num_args = item
            .output_id
            .map_or(item.inputs.len(), |_| item.inputs.len() + 1);
        let program = if let Some(program) = state.cached_program(&lowered) {
            if debug >= 1 {
                let kid = KERNEL_COUNT.fetch_add(1, Ordering::Relaxed);
                eprintln!("*** CPU {kid:>4}  (cached)         arg {num_args:>2}");
            }
            program
        } else {
            let kid = KERNEL_COUNT.fetch_add(1, Ordering::Relaxed);
            let name = format!("kernel_{kid}");
            let code = ClangRenderer.render(&lowered, &name);

            if debug >= 4 {
                eprintln!("{code}");
            }
            if debug >= 3 {
                eprintln!("{}", lowered.dump());
            }

            let program = Rc::new(
                state
                    .device()
                    .compile(&code, &name, num_args)
                    .expect("compile failed"),
            );

            if debug >= 2 {
                eprintln!("*** CPU {kid:>4}  {name:<16} arg {num_args:>2}  (compiled)");
            } else if debug >= 1 {
                eprintln!("*** CPU {kid:>4}  {name:<16} arg {num_args:>2}");
            }

            state.insert_program(lowered.clone(), program.clone());
            program
        };

        let mut args: Vec<KernelArg> = Vec::with_capacity(num_args);
        if let Some(output_id) = item.output_id {
            let out = state.device().allocate(
                item.out_dtype.expect("allocated item must have dtype"),
                item.out_shape
                    .as_ref()
                    .expect("allocated item must have shape")
                    .numel(),
            );
            args.push(KernelArg::Buffer(out));
            execute_inputs(&state, &item.inputs, &mut args);

            let t0 = Instant::now();
            state
                .device()
                .execute(&program, &mut args)
                .expect("execution failed");
            if debug >= 2 {
                let elapsed = t0.elapsed();
                eprintln!(
                    "              exec time={:.3}ms",
                    elapsed.as_secs_f64() * 1000.0,
                );
            }

            let KernelArg::Buffer(out) = args.remove(0) else {
                panic!("output kernel arg must remain a buffer");
            };
            state
                .write_buffer(output_id, out)
                .expect("failed to store kernel output");
            return;
        }

        execute_inputs(&state, &item.inputs, &mut args);
        let t0 = Instant::now();
        state
            .device()
            .execute(&program, &mut args)
            .expect("execution failed");
        if debug >= 2 {
            let elapsed = t0.elapsed();
            eprintln!(
                "              exec time={:.3}ms",
                elapsed.as_secs_f64() * 1000.0,
            );
        }

        let dest_id = item.inputs.iter().find_map(|input| match input {
            schedule::KernelInput::Buffer(id) => Some(*id),
            schedule::KernelInput::I32(_)
            | schedule::KernelInput::F32(_)
            | schedule::KernelInput::Bool(_) => None,
        });
        let dest_buffer = args.iter().find_map(|arg| match arg {
            KernelArg::Buffer(buffer) => Some(buffer.clone()),
            KernelArg::I32(_) | KernelArg::F32(_) | KernelArg::Bool(_) => None,
        });
        if let (Some(dest_id), Some(dest_buffer)) = (dest_id, dest_buffer) {
            state
                .write_buffer(dest_id, dest_buffer)
                .expect("failed to store in-place kernel output");
        }
    }

    /// Realize and extract data as `Vec<f32>`.
    ///
    /// # Panics
    ///
    /// Panics if realization fails or the device state no longer holds the
    /// tensor's buffer.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        let _ = self.realize();
        let uop = self.uop();
        let buffer = match uop.op() {
            Op::Buffer => uop,
            Op::Reshape => uop.srcs()[0].clone(),
            _ => panic!("realized tensor must point at a buffer"),
        };
        let Arg::Buffer(id, _) = buffer.arg() else {
            panic!("realized tensor must point at Arg::Buffer");
        };
        runtime::state(buffer.device())
            .load_buffer(*id)
            .expect("realized buffer missing")
            .to_f32()
    }

    /// Compute gradients of `self` with respect to `targets`.
    ///
    /// # Panics
    ///
    /// Panics if any target lives on a different device.
    #[must_use]
    pub fn gradient(&self, targets: &[&Self]) -> Vec<Self> {
        assert!(
            targets.iter().all(|target| self.device() == target.device()),
            "gradient targets must share the same device"
        );

        let root_grad = full(self.shape(), DType::F32, self.device(), 1.0);
        let target_uops: Vec<UOp> = targets
            .iter()
            .filter(|target| target.requires_grad())
            .map(|target| target.uop())
            .collect();
        let grad_map = gradient::compute_gradient(&self.uop(), &root_grad, &target_uops);

        targets
            .iter()
            .map(|target| match grad_map.get(&target.uop()) {
                Some(grad) => Self::new(grad.clone(), false),
                None => Self::new(full(target.shape(), DType::F32, target.device(), 0.0), false),
            })
            .collect()
    }

    /// Run reverse-mode autograd and store gradients on reachable tensors.
    ///
    /// # Panics
    ///
    /// Panics if called on a non-scalar loss.
    pub fn backward(&self) {
        assert_eq!(self.numel(), 1, "backward requires a scalar loss");
        let all_uops: HashSet<UOp> = self.uop().toposort().into_iter().collect();
        let targets: Vec<Tensor> = Self::live_tensors()
            .into_iter()
            .filter(|tensor| tensor.requires_grad() && all_uops.contains(&tensor.uop()))
            .collect();
        if targets.is_empty() {
            return;
        }

        let target_uops: Vec<UOp> = targets.iter().map(Self::uop).collect();
        let root_grad = full(self.shape(), DType::F32, self.device(), 1.0);
        let grads = gradient::compute_gradient(&self.uop(), &root_grad, &target_uops);

        for target in targets {
            let grad = grads
                .get(&target.uop())
                .cloned()
                .unwrap_or_else(|| full(target.shape(), DType::F32, target.device(), 0.0));
            let grad_tensor = Self::new(grad, false);
            let accumulated = match target.grad_inner() {
                Some(existing) => existing.add(&grad_tensor),
                None => grad_tensor,
            };
            target.set_grad_inner(Some(accumulated));
        }
    }
}

/// Convert scheduled kernel inputs (buffer ids and scalar constants) into
/// concrete `KernelArg` values by loading buffers from device state.
fn execute_inputs(
    state: &runtime::DeviceState,
    inputs: &[schedule::KernelInput],
    args: &mut Vec<KernelArg>,
) {
    let input_args = inputs.iter().map(|input| match input {
        schedule::KernelInput::Buffer(id) => {
            KernelArg::Buffer(state.load_buffer(*id).expect("scheduled input buffer missing"))
        }
        schedule::KernelInput::I32(value) => KernelArg::I32(*value),
        schedule::KernelInput::F32(value) => KernelArg::F32(*value),
        schedule::KernelInput::Bool(value) => KernelArg::Bool(*value),
    });
    args.extend(input_args);
}

/// Build a `UOp` expression for a constant-filled tensor (e.g. all-ones for the
/// initial backward gradient, or all-zeros for missing gradients).
fn full(shape: Shape, dtype: DType, device: DeviceId, value: f64) -> UOp {
    let base_shape = Shape::new(vec![1; shape.ndim()]);
    let scalar = UOp::const_float(value, dtype, device);
    let base = UOp::reshape(scalar, base_shape);
    if shape.iter().all(|&dim| dim == 1) {
        return base;
    }
    UOp::expand(base, shape)
}

/// Walk `root` in topological order and replace any node found in
/// `replacements`. Children of replaced nodes are *not* traversed — the
/// replacement is taken as-is, which is correct because replaced subtrees are
/// fully realized and self-contained.
fn substitute_with_map(root: &UOp, replacements: &HashMap<UOp, UOp>) -> UOp {
    let order = root.toposort();
    let mut substituted: HashMap<UOp, UOp> = HashMap::new();

    for node in &order {
        if let Some(replacement) = replacements.get(node) {
            substituted.insert(node.clone(), replacement.clone());
            continue;
        }

        let new_srcs: Vec<UOp> = node
            .srcs()
            .iter()
            .map(|src| substituted.get(src).cloned().unwrap_or_else(|| src.clone()))
            .collect();
        let changed = node
            .srcs()
            .iter()
            .zip(&new_srcs)
            .any(|(old, new)| old != new);
        let rewritten = if changed {
            UOp::new(node.op(), node.dtype(), new_srcs, node.arg().clone())
        } else {
            node.clone()
        };
        substituted.insert(node.clone(), rewritten);
    }

    substituted
        .get(root)
        .cloned()
        .unwrap_or_else(|| root.clone())
}

/// Substitute multiple roots in a single pass by wrapping them in a temporary
/// `Sink` node, so shared subgraphs between roots are only rewritten once.
fn substitute_roots_with_map(roots: &[UOp], replacements: &HashMap<UOp, UOp>) -> Vec<UOp> {
    if roots.is_empty() {
        return Vec::new();
    }

    let sink = UOp::sink(roots.to_vec());
    let rewritten_sink = substitute_with_map(&sink, replacements);
    rewritten_sink.srcs().to_vec()
}

/// Wrap a compile-time `usize` as a `Const` i32 `UOp`, used for narrow offsets.
fn bound_const(value: usize, device: DeviceId) -> UOp {
    #[allow(clippy::cast_possible_wrap)]
    UOp::const_int(value as i64, DType::I32, device)
}

/// Extract the concrete `usize` from a narrow start `UOp` (either a plain
/// `Const` or a `Bind` whose second source carries the current value).
fn bound_value(start: &UOp) -> usize {
    match start.op() {
        Op::Const => {
            let Arg::Int(value) = start.arg() else {
                panic!("narrow start constant must be Arg::Int");
            };
            usize::try_from(*value).expect("narrow start must be non-negative")
        }
        Op::Bind => {
            let Arg::Int(value) = start.srcs()[1].arg() else {
                panic!("bound narrow start must carry an integer value");
            };
            usize::try_from(*value).expect("bound narrow start must be non-negative")
        }
        _ => panic!("narrow start must be a bound integer"),
    }
}

/// Numpy-style broadcasting: pad ranks to match, then expand size-1 dims.
fn broadcast_shapes(left: &Tensor, right: &Tensor) -> (Tensor, Tensor) {
    assert!(
        left.device() == right.device(),
        "broadcast requires tensors on the same device"
    );
    let target = left
        .shape()
        .broadcast_with(&right.shape())
        .unwrap_or_else(|| {
            panic!(
                "broadcast: incompatible dims for {:?} vs {:?}",
                left.shape(),
                right.shape()
            )
        });
    let left_shape = left.shape().pad_left(target.ndim());
    let right_shape = right.shape().pad_left(target.ndim());
    let left = left.reshape(left_shape.as_slice()).expand(target.as_slice());
    let right = right.reshape(right_shape.as_slice()).expand(target.as_slice());
    (left, right)
}

/// Validate that rangeify and symbolic passes have lowered all high-level ops.
/// Any surviving tensor-level op (`Reshape`, `Permute`, `ReduceAxis`, etc.) indicates
/// a bug in the lowering pipeline and would produce nonsense in codegen.
fn assert_codegen_ready(root: &UOp) {
    assert_eq!(root.op(), Op::Sink, "render: root must be a Sink node");
    for node in root.toposort() {
        assert!(
            !matches!(
                node.op(),
                Op::DefineVar
                    | Op::Bind
                    | Op::Buffer
                    | Op::Shrink
                    | Op::Reshape
                    | Op::Permute
                    | Op::Expand
                    | Op::ReduceAxis
                    | Op::Reduce
            ),
            "{:?} should be lowered before codegen",
            node.op()
        );
    }
}

/// Return the default CPU device.
#[must_use]
pub fn cpu() -> DeviceId {
    DeviceId::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_slice_roundtrip() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(t.shape(), [3]);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_realized());
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_is_lazy() {
        let a = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let b = Tensor::from_slice(&[3.0, 4.0], &[2]);
        let c = a.add(&b);
        assert!(!c.is_realized());
        assert_eq!(c.shape(), [2]);
    }

    #[test]
    fn test_add_realize() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        assert_eq!(a.add(&b).to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_realize() {
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3]);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], &[3]);
        assert_eq!(a.mul(&b).to_vec(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0], &[4]);
        assert_eq!(a.relu().to_vec(), vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_log_softmax_matches_known_values() {
        let logits = Tensor::from_slice(&[0.0, 1.0], &[1, 2]);
        let result = logits.log_softmax(1).realize().to_vec();
        let expected = [-1.313_261_6_f32, -0.313_261_66_f32];
        for (actual, target) in result.iter().zip(expected) {
            assert!((actual - target).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cross_entropy_matches_one_hot_mean_loss() {
        let logits = Tensor::from_slice(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);
        let targets = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

        let loss = logits.cross_entropy(&targets).realize().to_vec()[0];

        assert!((loss - 0.126_928_05).abs() < 1e-5);
    }

    #[test]
    fn test_narrow_rows() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let narrowed = x.narrow(0, 1, 1);
        assert_eq!(narrowed.shape(), [1, 3]);
        assert_eq!(narrowed.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_gradients_skip_untracked_narrow_inputs() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).narrow(0, 1, 1);
        let w = Tensor::from_slice(&[10.0, 20.0], &[2, 1]).with_requires_grad(true);
        let loss = x.matmul(&w).sum(&[0, 1]);
        let grads = loss.gradient(&[&w]);
        assert_eq!(grads[0].to_vec(), vec![3.0, 4.0]);
        assert!(!grads[0].requires_grad());
    }

    #[test]
    fn test_realize_idempotent() {
        let a = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let b = a.add(&Tensor::from_slice(&[3.0, 4.0], &[2]));
        assert_eq!(b.realize().realize().to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_clone_shares_updates() {
        let left = Tensor::zeros(&[2], DType::F32);
        let alias = left.clone();
        left.replace(&Tensor::from_slice(&[5.0, 7.0], &[2]));
        assert_eq!(alias.to_vec(), vec![5.0, 7.0]);
    }

    #[test]
    fn test_assign_realizes_in_place() {
        let tensor = Tensor::from_slice(&[1.0, 2.0], &[2]);
        let alias = tensor.clone();
        tensor.assign(&Tensor::from_slice(&[8.0, 9.0], &[2]));
        Tensor::realize_many(&[&tensor]);
        assert_eq!(tensor.to_vec(), vec![8.0, 9.0]);
        assert_eq!(alias.to_vec(), vec![8.0, 9.0]);
    }

    #[test]
    fn test_backward_stores_gradients() {
        let x = Tensor::from_slice(&[3.0], &[1]).with_requires_grad(true);
        let y = Tensor::from_slice(&[5.0], &[1]).with_requires_grad(true);
        let loss = x.mul(&y).sum(&[0]);
        loss.backward();
        assert_eq!(x.grad().expect("x grad").to_vec(), vec![5.0]);
        assert_eq!(y.grad().expect("y grad").to_vec(), vec![3.0]);
    }
}
