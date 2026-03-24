//! # Tensor API — lazy evaluation and kernel fusion
//!
//! Tensor ops build a lazy `UOp` graph. Calling `realize()` lowers it to a
//! fused kernel, compiles, and executes on the tensor's device.
//!
//! ## Debug output
//!
//! Set `DEBUG` env var (same as tinygrad):
//! - 1: kernel summary, 2: + timing, 3: + graph rewrite before/after, 4: + generated C

use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use crate::codegen::{ClangRenderer, Renderer};
use crate::device::{Buffer, CpuDevice, Device};
use crate::dtype::DType;
use crate::gradient;
use crate::schedule::{self, rangeify::rangeify};
use crate::uop::{Arg, Op, UOp};

static DEBUG: LazyLock<u8> = LazyLock::new(|| {
    std::env::var("DEBUG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
});

static KERNEL_COUNT: AtomicUsize = AtomicUsize::new(0);

// ── Tensor ──────────────────────────────────────────────────────────────────

/// A lazily-evaluated tensor bound to a specific device.
#[derive(Clone)]
pub struct Tensor {
    uop: UOp,
    device: Rc<dyn Device>,
}

impl Tensor {
    /// Create a tensor from a float slice on the given device.
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match the product of shape.
    #[must_use]
    pub fn from_slice(data: &[f32], shape: &[usize], device: &Rc<dyn Device>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "data length {} doesn't match shape {shape:?}",
            data.len()
        );
        let buf_uop = UOp::new(
            Op::Buffer,
            DType::F32,
            vec![],
            Arg::Buffer(Rc::new(Buffer::from_f32(data))),
        );
        Self {
            uop: UOp::new(
                Op::Reshape,
                DType::F32,
                vec![buf_uop],
                Arg::Dims(shape.to_vec()),
            ),
            device: device.clone(),
        }
    }

    /// Create a tensor filled with zeros on the given device.
    #[must_use]
    pub fn zeros(shape: &[usize], dtype: DType, device: &Rc<dyn Device>) -> Self {
        let numel = shape.iter().product();
        let buf = device.allocate(dtype, numel);
        let buf_uop = UOp::new(Op::Buffer, dtype, vec![], Arg::Buffer(Rc::new(buf)));
        Self {
            uop: UOp::new(Op::Reshape, dtype, vec![buf_uop], Arg::Dims(shape.to_vec())),
            device: device.clone(),
        }
    }

    /// The tensor's element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.uop.dtype()
    }

    /// Number of elements (product of shape).
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Shape derived from the `UOp` graph.
    ///
    /// # Panics
    ///
    /// Panics if the `UOp` graph doesn't have shape info.
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        self.uop.shape().expect("tensor must have shape")
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Whether this tensor's data has been computed. `false` means it's
    /// still a lazy graph that needs `realize()` to execute.
    #[must_use]
    pub fn is_realized(&self) -> bool {
        match self.uop.op() {
            Op::Buffer => true,
            Op::Reshape => self.uop.srcs()[0].op() == Op::Buffer,
            _ => false,
        }
    }

    /// Extract the Buffer from a realized tensor (handles Reshape wrapper).
    fn realized_buffer(&self) -> &Rc<Buffer> {
        let buf_uop = match self.uop.op() {
            Op::Buffer => &self.uop,
            Op::Reshape => &self.uop.srcs()[0],
            _ => panic!("not a realized tensor"),
        };
        match buf_uop.arg() {
            Arg::Buffer(rc) => rc,
            _ => panic!("realized tensor must have Arg::Buffer"),
        }
    }

    // ── Lazy ops ────────────────────────────────────────────────────────

    fn unary(&self, op: Op) -> Self {
        Self {
            uop: UOp::new(op, self.uop.dtype(), vec![self.uop.clone()], Arg::None),
            device: self.device.clone(),
        }
    }

    fn binary(&self, other: &Self, op: Op, out_dtype: DType) -> Self {
        assert_eq!(self.shape(), other.shape(), "shape mismatch for {op:?}");
        Self {
            uop: UOp::new(
                op,
                out_dtype,
                vec![self.uop.clone(), other.uop.clone()],
                Arg::None,
            ),
            device: self.device.clone(),
        }
    }

    /// Broadcast two tensors to a common shape, then apply a binary op.
    fn broadcasted(&self, other: &Self, op: Op, out_dtype: DType) -> Self {
        if self.shape() == other.shape() {
            return self.binary(other, op, out_dtype);
        }
        let (a, b) = broadcast_shapes(self, other);
        a.binary(&b, op, out_dtype)
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

    /// Relu: `where(0 < self, self, 0)`.
    #[must_use]
    pub fn relu(&self) -> Self {
        let zero = Self::zeros(&self.shape(), self.dtype(), &self.device);
        let cond = zero.binary(self, Op::CmpLt, DType::Bool);
        Self {
            uop: UOp::new(
                Op::Where,
                self.dtype(),
                vec![cond.uop, self.uop.clone(), zero.uop],
                Arg::None,
            ),
            device: self.device.clone(),
        }
    }

    /// Element-wise subtraction with broadcasting.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    /// Element-wise `1/x`.
    #[must_use]
    pub fn reciprocal(&self) -> Self {
        self.unary(Op::Reciprocal)
    }

    /// Element-wise `2^x`.
    #[must_use]
    pub fn exp2(&self) -> Self {
        self.unary(Op::Exp2)
    }

    /// Element-wise `log₂(x)`.
    #[must_use]
    pub fn log2(&self) -> Self {
        self.unary(Op::Log2)
    }

    /// Element-wise `√x`.
    #[must_use]
    pub fn sqrt(&self) -> Self {
        self.unary(Op::Sqrt)
    }

    // ── Movement ops ───────────────────────────────────────────────────

    /// Change the shape without moving data.
    ///
    /// # Panics
    ///
    /// Panics if the product of new shape differs from the current numel.
    #[must_use]
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_numel, "reshape: numel mismatch");
        if self.shape() == new_shape {
            return self.clone();
        }
        Self {
            uop: UOp::new(
                Op::Reshape,
                self.dtype(),
                vec![self.uop.clone()],
                Arg::Dims(new_shape.to_vec()),
            ),
            device: self.device.clone(),
        }
    }

    /// Reorder dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `order` isn't a valid permutation.
    #[must_use]
    pub fn permute(&self, order: &[usize]) -> Self {
        let shape = self.shape();
        assert_eq!(order.len(), shape.len(), "permute: wrong number of axes");
        let mut seen = vec![false; shape.len()];
        for &ax in order {
            assert!(ax < shape.len(), "permute: axis {ax} out of range");
            assert!(!seen[ax], "permute: duplicate axis {ax}");
            seen[ax] = true;
        }
        Self {
            uop: UOp::new(
                Op::Permute,
                self.dtype(),
                vec![self.uop.clone()],
                Arg::Dims(order.to_vec()),
            ),
            device: self.device.clone(),
        }
    }

    /// Broadcast dimensions of size 1 to a larger size.
    ///
    /// # Panics
    ///
    /// Panics if any non-1 dimension doesn't match.
    #[must_use]
    pub fn expand(&self, new_shape: &[usize]) -> Self {
        let shape = self.shape();
        assert_eq!(new_shape.len(), shape.len(), "expand: ndim mismatch");
        for (i, (&old, &new)) in shape.iter().zip(new_shape).enumerate() {
            assert!(
                old == new || old == 1,
                "expand: dim {i} is {old}, can only expand from 1"
            );
        }
        if shape == new_shape {
            return self.clone();
        }
        Self {
            uop: UOp::new(
                Op::Expand,
                self.dtype(),
                vec![self.uop.clone()],
                Arg::Dims(new_shape.to_vec()),
            ),
            device: self.device.clone(),
        }
    }

    // ── Reduction ─────────────────────────────────────────────────────

    /// Sum over the given axes. Reduced dims become size 1.
    ///
    /// # Panics
    ///
    /// Panics if any axis is out of range.
    #[must_use]
    pub fn sum(&self, axes: &[usize]) -> Self {
        for &ax in axes {
            assert!(ax < self.ndim(), "sum: axis {ax} out of range");
        }
        Self {
            uop: UOp::new(
                Op::ReduceAxis,
                self.dtype(),
                vec![self.uop.clone()],
                Arg::Reduce(Op::Add, axes.to_vec()),
            ),
            device: self.device.clone(),
        }
    }

    // ── Matmul ────────────────────────────────────────────────────────

    /// Matrix multiply: `[M,K] @ [K,N] → [M,N]`.
    #[allow(clippy::many_single_char_names)]
    ///
    /// # Panics
    ///
    /// Panics if inner dimensions don't match or tensors aren't 2D.
    #[must_use]
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.ndim(), 2, "matmul: lhs must be 2D");
        assert_eq!(other.ndim(), 2, "matmul: rhs must be 2D");
        let shape = self.shape();
        let (m, k) = (shape[0], shape[1]);
        let other_shape = other.shape();
        let (k2, n) = (other_shape[0], other_shape[1]);
        assert_eq!(k, k2, "matmul: inner dim mismatch ({k} vs {k2})");

        // a[M,K] → [M,1,K] → [M,N,K]
        let a = self.reshape(&[m, 1, k]).expand(&[m, n, k]);
        // b[K,N] → [N,K] → [1,N,K] → [M,N,K]
        let b = other
            .permute(&[1, 0])
            .reshape(&[1, n, k])
            .expand(&[m, n, k]);
        // element-wise multiply then sum over K axis
        a.mul(&b).sum(&[2]).reshape(&[m, n])
    }

    // ── Realize ─────────────────────────────────────────────────────────

    /// Lower the lazy graph to a fused kernel, compile, and execute.
    ///
    /// # Panics
    ///
    /// Panics if compilation or execution fails.
    #[must_use]
    pub fn realize(&self) -> Self {
        if self.is_realized() {
            return self.clone();
        }

        let debug = *DEBUG;
        let numel = self.numel();
        let dtype = self.dtype();

        let (scheduled, mut input_bufs) = schedule::schedule(&self.uop);
        let sink = rangeify(&scheduled);

        // Simplify index arithmetic (x+0→x, x*1→x, constant folding).
        let sink =
            crate::rewrite::graph_rewrite(&sink, &crate::rewrite::symbolic_simple, "symbolic");

        let kid = KERNEL_COUNT.fetch_add(1, Ordering::Relaxed);
        let name = format!("kernel_{kid}");
        // TODO: select renderer based on device (ClangRenderer for CPU, CudaRenderer for CUDA)
        let code = ClangRenderer.render(&sink, &name);

        if debug >= 4 {
            eprintln!("{code}");
        }
        if debug >= 3 {
            eprintln!("{}", sink.dump());
        }

        let dev = &*self.device;
        let num_bufs = input_bufs.len() + 1;
        let program = dev.compile(&code, &name, num_bufs).expect("compile failed");
        let mut out = dev.allocate(dtype, numel);

        let mut buf_refs: Vec<&mut Buffer> = Vec::with_capacity(num_bufs);
        buf_refs.push(&mut out);
        for buf in &mut input_bufs {
            buf_refs.push(buf);
        }

        let t0 = Instant::now();
        dev.execute(&program, &mut buf_refs)
            .expect("execution failed");
        let elapsed = t0.elapsed();

        if debug >= 2 {
            eprintln!(
                "*** CPU {kid:>4}  {name:<16} arg {num_bufs:>2}  time={:.3}ms",
                elapsed.as_secs_f64() * 1000.0,
            );
        } else if debug >= 1 {
            eprintln!("*** CPU {kid:>4}  {name:<16} arg {num_bufs:>2}");
        }

        let buf_uop = UOp::new(Op::Buffer, dtype, vec![], Arg::Buffer(Rc::new(out)));
        Self {
            uop: UOp::new(Op::Reshape, dtype, vec![buf_uop], Arg::Dims(self.shape())),
            device: self.device.clone(),
        }
    }

    /// Realize and extract data as `Vec<f32>`.
    ///
    /// # Panics
    ///
    /// Panics if dtype is not `F32`.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        let realized = self.realize();
        realized.realized_buffer().to_f32()
    }

    // ── Autograd ─────────────────────────────────────────────────────────

    /// Compute gradients of `self` with respect to each target tensor.
    ///
    /// Returns one gradient `Tensor` per target, in the same order.
    /// Each gradient is a lazy tensor — call `realize()` or `to_vec()`
    /// to execute the backward computation.
    ///
    /// `self` should be a scalar (e.g. a loss after `.sum()`). The initial
    /// gradient is implicitly 1.0.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let loss = x.mul(&w).sum(&[0]);
    /// let grads = loss.gradient(&[&w]);
    /// let w_grad = grads[0].to_vec();
    /// ```
    #[must_use]
    pub fn gradient(&self, targets: &[&Self]) -> Vec<Self> {
        let shape = self.shape();
        let numel: usize = shape.iter().product();

        // Initial gradient: ones with the same shape as self
        let ones_data = vec![1.0_f32; numel];
        let ones_buf = UOp::new(
            Op::Buffer,
            DType::F32,
            vec![],
            Arg::Buffer(Rc::new(Buffer::from_f32(&ones_data))),
        );
        let root_grad = UOp::new(Op::Reshape, DType::F32, vec![ones_buf], Arg::Dims(shape));

        let target_uops: Vec<UOp> = targets.iter().map(|t| t.uop.clone()).collect();
        let grad_map = gradient::compute_gradient(&self.uop, &root_grad, &target_uops);

        targets
            .iter()
            .map(|t| Self {
                uop: grad_map[&t.uop].clone(),
                device: t.device.clone(),
            })
            .collect()
    }
}

/// Broadcast two tensors to a common shape (numpy-style).
/// Pads with 1s on the left, then expands mismatched dims.
fn broadcast_shapes(a: &Tensor, b: &Tensor) -> (Tensor, Tensor) {
    let ndim = a.ndim().max(b.ndim());

    // Pad shapes with 1s on the left to match ndim.
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut v = vec![1; ndim - s.len()];
        v.extend_from_slice(s);
        v
    };
    let sa = pad(&a.shape());
    let sb = pad(&b.shape());

    let mut target = Vec::with_capacity(ndim);
    for (i, (&da, &db)) in sa.iter().zip(&sb).enumerate() {
        match (da, db) {
            (x, y) if x == y => target.push(x),
            (1, y) => target.push(y),
            (x, 1) => target.push(x),
            _ => panic!("broadcast: incompatible dims at axis {i}: {da} vs {db}"),
        }
    }

    let a = a.reshape(&sa).expand(&target);
    let b = b.reshape(&sb).expand(&target);
    (a, b)
}

/// Create a default CPU device for convenience.
#[must_use]
pub fn cpu() -> Rc<dyn Device> {
    Rc::new(CpuDevice)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Rc<dyn Device> {
        cpu()
    }

    #[test]
    fn test_from_slice_roundtrip() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &dev());
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_realized());
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_is_lazy() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0], &[2], &d);
        let b = Tensor::from_slice(&[3.0, 4.0], &[2], &d);
        let c = a.add(&b);
        assert!(!c.is_realized());
        assert_eq!(c.shape(), &[2]);
    }

    #[test]
    fn test_add_realize() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &d);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3], &d);
        assert_eq!(a.add(&b).to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_realize() {
        let d = dev();
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], &[3], &d);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], &[3], &d);
        assert_eq!(a.mul(&b).to_vec(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_fused_add_mul() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &d);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0], &[3], &d);
        let two = Tensor::from_slice(&[2.0, 2.0, 2.0], &[3], &d);
        assert_eq!(a.add(&b).mul(&two).to_vec(), vec![22.0, 44.0, 66.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0], &[3], &dev());
        assert_eq!(a.neg().to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0], &[4], &dev());
        assert_eq!(a.relu().to_vec(), vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_shared_input() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &dev());
        assert_eq!(a.add(&a).to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_chained_ops() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &dev());
        assert_eq!(a.neg().neg().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_realize_idempotent() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0], &[2], &d);
        let b = a.add(&Tensor::from_slice(&[3.0, 4.0], &[2], &d));
        assert_eq!(b.realize().realize().to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "broadcast: incompatible dims")]
    fn test_shape_mismatch() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0], &[2], &d);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &d);
        let _ = a.add(&b);
    }

    // ── Multi-dimensional tests ──────────────────────────────────────

    #[test]
    fn test_2d_add() {
        // Arrange — [2,3] + [2,3]
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &d);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3], &d);

        // Act
        let result = a.add(&b).to_vec();

        // Assert
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    fn test_reshape_lazy() {
        // Arrange
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &dev());

        // Act
        let b = a.reshape(&[2, 3]);

        // Assert — reshape is lazy, shape changes but data unchanged
        assert_eq!(b.shape(), vec![2, 3]);
        assert_eq!(b.numel(), 6);
    }

    #[test]
    fn test_2d_reshape_add() {
        // Arrange — reshape then add
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &d);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[6], &d);

        // Act — reshape both to [2,3] then add
        let result = a.reshape(&[2, 3]).add(&b.reshape(&[2, 3])).to_vec();

        // Assert
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]);
    }

    #[test]
    fn test_broadcast_add() {
        // Arrange — [2,3] + [1,3] broadcasts row
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &d);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0], &[1, 3], &d);

        // Act
        let result = a.add(&b).to_vec();

        // Assert — b is broadcast across rows
        assert_eq!(result, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_add_col() {
        // Arrange — [2,3] + [2,1] broadcasts column
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &d);
        let b = Tensor::from_slice(&[10.0, 20.0], &[2, 1], &d);

        // Act
        let result = a.add(&b).to_vec();

        // Assert — b is broadcast across columns
        assert_eq!(result, vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn test_sum_1d() {
        // Arrange
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3], &dev());

        // Act
        let result = a.sum(&[0]).to_vec();

        // Assert
        assert_eq!(result, vec![6.0]);
    }

    #[test]
    fn test_sum_2d_axis0() {
        // Arrange — sum over rows: [2,3] → [1,3]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev());

        // Act
        let result = a.sum(&[0]).to_vec();

        // Assert
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_2d_axis1() {
        // Arrange — sum over cols: [2,3] → [2,1]
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev());

        // Act
        let result = a.sum(&[1]).to_vec();

        // Assert
        assert_eq!(result, vec![6.0, 15.0]);
    }

    #[test]
    fn test_matmul_2x3_3x2() {
        // Arrange
        // a = [[1,2,3],[4,5,6]]  (2x3)
        // b = [[1,2],[3,4],[5,6]]  (3x2)
        // expected = [[22,28],[49,64]]
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &d);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &d);

        // Act
        let result = a.matmul(&b).to_vec();

        // Assert
        assert_eq!(result, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_identity() {
        // Arrange — a @ I = a
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &d);
        let eye = Tensor::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2], &d);

        // Act
        let result = a.matmul(&eye).to_vec();

        // Assert
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ── Autograd tests ──────────────────────────────────────────────────

    /// Finite-difference gradient check: (f(x+eps) - f(x-eps)) / 2eps.
    /// Perturbs each element of `input_data` and compares against
    /// the analytical gradient from `Tensor::gradient`.
    fn check_gradient(
        input_data: &[f32],
        shape: &[usize],
        build_loss: impl Fn(&Tensor) -> Tensor,
        eps: f32,
        tol: f32,
    ) {
        let d = dev();
        let x = Tensor::from_slice(input_data, shape, &d);
        let loss = build_loss(&x);
        let grads = loss.gradient(&[&x]);
        let analytical = grads[0].to_vec();

        let mut numerical = vec![0.0_f32; input_data.len()];
        for i in 0..input_data.len() {
            let mut plus = input_data.to_vec();
            let mut minus = input_data.to_vec();
            plus[i] += eps;
            minus[i] -= eps;
            let f_plus: f32 = build_loss(&Tensor::from_slice(&plus, shape, &d))
                .to_vec()
                .iter()
                .sum();
            let f_minus: f32 = build_loss(&Tensor::from_slice(&minus, shape, &d))
                .to_vec()
                .iter()
                .sum();
            numerical[i] = (f_plus - f_minus) / (2.0 * eps);
        }

        for (i, (&a, &n)) in analytical.iter().zip(&numerical).enumerate() {
            assert!(
                (a - n).abs() < tol,
                "gradient mismatch at [{i}]: analytical={a}, numerical={n}"
            );
        }
    }

    #[test]
    fn test_grad_add_sum() {
        // d/dx sum(x + y) = ones
        check_gradient(&[1.0, 2.0, 3.0], &[3], |x| {
            let y = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3], &x.device.clone());
            x.add(&y).sum(&[0])
        }, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_mul_sum() {
        // d/dx sum(x * y) = y
        check_gradient(&[1.0, 2.0, 3.0], &[3], |x| {
            let y = Tensor::from_slice(&[4.0, 5.0, 6.0], &[3], &x.device.clone());
            x.mul(&y).sum(&[0])
        }, 1e-3, 1e-2);
    }

    #[test]
    fn test_grad_neg_sum() {
        // d/dx sum(-x) = -1
        check_gradient(&[1.0, 2.0, 3.0], &[3], |x| {
            x.neg().sum(&[0])
        }, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_x_squared() {
        // d/dx sum(x * x) = 2x
        check_gradient(&[1.0, 2.0, 3.0], &[3], |x| {
            x.mul(x).sum(&[0])
        }, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_chain() {
        // d/dx sum(x*x + x) = 2x + 1
        check_gradient(&[1.0, 2.0, 3.0], &[3], |x| {
            x.mul(x).add(x).sum(&[0])
        }, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_broadcast_mul() {
        // x[2,3] * y[1,3] → sum. Tests Expand gradient (reduce over broadcast dim).
        check_gradient(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], |x| {
            let y = Tensor::from_slice(&[2.0, 3.0, 4.0], &[1, 3], &x.device.clone());
            x.mul(&y).sum(&[0, 1])
        }, 1e-3, 1e-2);
    }

    #[test]
    fn test_grad_reshape() {
        // reshape doesn't move data, gradient reshapes back
        check_gradient(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], |x| {
            x.reshape(&[2, 3]).sum(&[0, 1])
        }, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_relu() {
        // d/dx relu(x) = (x > 0) ? 1 : 0
        check_gradient(&[1.0, -2.0, 3.0, -4.0], &[4], |x| {
            x.relu().sum(&[0])
        }, 1e-3, 1e-3);
    }

    #[test]
    fn test_grad_matmul() {
        // loss = sum(a @ b), gradient w.r.t. a
        check_gradient(&[1.0, 2.0, 3.0, 4.0], &[2, 2], |a| {
            let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &a.device.clone());
            a.matmul(&b).sum(&[0, 1])
        }, 1e-3, 1e-2);
    }

    #[test]
    fn test_grad_linear_layer() {
        // loss = sum(x @ w + b), gradients for w and b
        let d = dev();
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &d);
        let w = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], &[2, 2], &d);
        let b = Tensor::from_slice(&[0.5, 0.6], &[1, 2], &d);

        let loss = x.matmul(&w).add(&b).sum(&[0, 1]);
        let grads = loss.gradient(&[&w, &b]);
        let w_grad = grads[0].to_vec();
        let b_grad = grads[1].to_vec();

        // Numerical check for w
        let eps = 1e-3;
        let w_data = [0.1_f32, 0.2, 0.3, 0.4];
        for i in 0..4 {
            let mut plus = w_data;
            let mut minus = w_data;
            plus[i] += eps;
            minus[i] -= eps;
            let f_plus: f32 = x
                .matmul(&Tensor::from_slice(&plus, &[2, 2], &d))
                .add(&b)
                .sum(&[0, 1])
                .to_vec()
                .iter()
                .sum();
            let f_minus: f32 = x
                .matmul(&Tensor::from_slice(&minus, &[2, 2], &d))
                .add(&b)
                .sum(&[0, 1])
                .to_vec()
                .iter()
                .sum();
            let numerical = (f_plus - f_minus) / (2.0 * eps);
            assert!(
                (w_grad[i] - numerical).abs() < 1e-2,
                "w_grad[{i}]: analytical={}, numerical={numerical}",
                w_grad[i]
            );
        }

        // b gradient: d/db sum(x@w + b) = [2, 2] (batch_size for each output)
        assert!((b_grad[0] - 2.0).abs() < 1e-3, "b_grad[0] = {}", b_grad[0]);
        assert!((b_grad[1] - 2.0).abs() < 1e-3, "b_grad[1] = {}", b_grad[1]);
    }
}
