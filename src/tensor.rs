//! # Tensor API — lazy evaluation and kernel fusion
//!
//! Tensor ops build a lazy `UOp` graph. Calling `realize()` lowers it to a
//! fused kernel, compiles, and executes on the tensor's device.
//!
//! ## Debug output
//!
//! Set `DEBUG` env var (same as tinygrad):
//! - 1: kernel summary, 2: + timing, 3: + `UOp` dump, 4: + generated C

use std::rc::Rc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::codegen::{ClangRenderer, Renderer};
use crate::device::{Buffer, CpuDevice, Device};
use crate::dtype::DType;
use crate::lower::lower_to_kernel;
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
    #[must_use]
    pub fn from_slice(data: &[f32], device: &Rc<dyn Device>) -> Self {
        Self {
            uop: UOp::new(
                Op::Buffer,
                DType::F32,
                vec![],
                Arg::Buffer(Rc::new(Buffer::from_f32(data))),
            ),
            device: device.clone(),
        }
    }

    /// Create a tensor filled with zeros on the given device.
    #[must_use]
    pub fn zeros(numel: usize, dtype: DType, device: &Rc<dyn Device>) -> Self {
        let buf = device.allocate(dtype, numel);
        Self {
            uop: UOp::new(Op::Buffer, dtype, vec![], Arg::Buffer(Rc::new(buf))),
            device: device.clone(),
        }
    }

    /// The tensor's element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.uop.dtype()
    }

    /// Number of elements (walks to a Buffer leaf).
    #[must_use]
    pub fn numel(&self) -> usize {
        Self::derive_numel(&self.uop)
    }

    /// The tensor's shape (1D for now).
    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        vec![self.numel()]
    }

    /// Whether this tensor's data has been computed. `false` means it's
    /// still a lazy graph that needs `realize()` to execute.
    #[must_use]
    pub fn is_realized(&self) -> bool {
        self.uop.op() == Op::Buffer
    }

    fn derive_numel(uop: &UOp) -> usize {
        if let Arg::Buffer(ref buf) = uop.arg() {
            return buf.numel();
        }
        assert!(!uop.srcs().is_empty(), "non-Buffer leaf has no sources");
        Self::derive_numel(&uop.srcs()[0])
    }

    // ── Lazy ops ────────────────────────────────────────────────────────

    fn unary(&self, op: Op) -> Self {
        Self {
            uop: UOp::new(op, self.uop.dtype(), vec![self.uop.clone()], Arg::None),
            device: self.device.clone(),
        }
    }

    fn binary(&self, other: &Self, op: Op, out_dtype: DType) -> Self {
        assert_eq!(self.numel(), other.numel(), "shape mismatch for {op:?}");
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

    /// Element-wise addition.
    ///
    /// # Panics
    ///
    /// Panics if shapes don't match.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        self.binary(other, Op::Add, self.dtype())
    }

    /// Element-wise multiplication.
    ///
    /// # Panics
    ///
    /// Panics if shapes don't match.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        self.binary(other, Op::Mul, self.dtype())
    }

    /// Element-wise negation.
    #[must_use]
    pub fn neg(&self) -> Self {
        self.unary(Op::Neg)
    }

    /// Relu: `where(0 < self, self, 0)`.
    #[must_use]
    pub fn relu(&self) -> Self {
        let zero = Self::zeros(self.numel(), self.dtype(), &self.device);
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

        let (sink, mut input_bufs) = lower_to_kernel(&self.uop, numel);
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

        Self {
            uop: UOp::new(Op::Buffer, dtype, vec![], Arg::Buffer(Rc::new(out))),
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
        match realized.uop.arg() {
            Arg::Buffer(buf) => buf.to_f32(),
            _ => panic!("realized tensor must have Arg::Buffer"),
        }
    }
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
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], &dev());
        assert_eq!(t.shape(), vec![3]);
        assert_eq!(t.dtype(), DType::F32);
        assert!(t.is_realized());
        assert_eq!(t.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_add_is_lazy() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0], &d);
        let b = Tensor::from_slice(&[3.0, 4.0], &d);
        let c = a.add(&b);
        assert!(!c.is_realized());
        assert_eq!(c.shape(), vec![2]);
    }

    #[test]
    fn test_add_realize() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &d);
        let b = Tensor::from_slice(&[4.0, 5.0, 6.0], &d);
        assert_eq!(a.add(&b).to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_realize() {
        let d = dev();
        let a = Tensor::from_slice(&[2.0, 3.0, 4.0], &d);
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0], &d);
        assert_eq!(a.mul(&b).to_vec(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_fused_add_mul() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &d);
        let b = Tensor::from_slice(&[10.0, 20.0, 30.0], &d);
        let two = Tensor::from_slice(&[2.0, 2.0, 2.0], &d);
        assert_eq!(a.add(&b).mul(&two).to_vec(), vec![22.0, 44.0, 66.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0], &dev());
        assert_eq!(a.neg().to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0], &dev());
        assert_eq!(a.relu().to_vec(), vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_shared_input() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &dev());
        assert_eq!(a.add(&a).to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_chained_ops() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &dev());
        assert_eq!(a.neg().neg().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_realize_idempotent() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0], &d);
        let b = a.add(&Tensor::from_slice(&[3.0, 4.0], &d));
        assert_eq!(b.realize().realize().to_vec(), vec![4.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_shape_mismatch() {
        let d = dev();
        let a = Tensor::from_slice(&[1.0, 2.0], &d);
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0], &d);
        let _ = a.add(&b);
    }
}
