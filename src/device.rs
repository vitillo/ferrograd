//! # Device Layer
//!
//! This module mirrors tinygrad's `device.py`: it defines the abstractions that
//! all backends implement, plus the `Buffer` and `Storage` types that represent
//! device memory.
//!
//! - [`Device`] trait -- the interface every backend (CPU, CUDA, ...) implements
//! - [`Buffer`] -- typed memory that lives on some device
//! - [`Storage`] -- opaque, device-specific memory (CPU = `Vec<u8>`, CUDA = device ptr)
//! - [`Program`] -- a compiled kernel, ready to execute on its device
//! - a device registry for resolving [`DeviceId`] to backend objects
//!
//! Backend implementations live in submodules:
//! - [`cpu`] -- compiles C with clang, runs via dlopen
//!
//! This module also owns the per-device backend registry, similar to
//! tinygrad's global `Device[...]` lookup in `device.py`.

pub mod cpu;

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::dtype::DType;
use crate::uop::UOp;

// Re-export CpuDevice for convenience.
pub use cpu::CpuDevice;

/// Stable identifier for a backend device.
///
/// Tinygrad threads device identity through the graph rather than storing a
/// runtime handle on each node. We mirror that split with a small value type
/// that is cheap to copy and hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceId {
    /// The CPU backend.
    Cpu,
}

/// Errors that can occur during device operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DeviceError {
    /// CPU backend: clang compilation or dlopen/dlsym failed.
    #[error("cpu: {0}")]
    Cpu(#[from] cpu::CpuError),
    // Future: Cuda(CudaError)
}

/// Opaque device memory. Each backend defines its own variant.
///
/// The compiler and tensor layers never inspect this directly -- they pass
/// it to the device which knows how to use it. This is what makes the same
/// `Buffer` type work across CPU, CUDA, and future backends.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Storage {
    /// Host memory, owned as a flat byte array. Used by `CpuDevice`.
    Cpu(Vec<u8>),
    // Future: Cuda { device_ptr: u64, len: usize }
}

/// Opaque identifier for debug output tied to a concrete buffer slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

impl BufferId {
    /// Return the raw numeric id, useful in debug output.
    #[must_use]
    pub fn raw(self) -> usize {
        self.0
    }
}

impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Shared device buffer object.
///
/// A `Buffer` carries stable identity, metadata, and optional concrete storage.
/// This lets one value represent both reserved outputs (no storage yet) and
/// realized device memory.
#[derive(Clone, Debug)]
pub struct Buffer(Rc<BufferInner>);

#[derive(Debug)]
struct BufferInner {
    id: BufferId,
    device: DeviceId,
    dtype: DType,
    numel: usize,
    storage: RefCell<Option<Storage>>,
}

impl Buffer {
    fn with_id(
        id: BufferId,
        device: DeviceId,
        dtype: DType,
        numel: usize,
        storage: Option<Storage>,
    ) -> Self {
        if let Some(Storage::Cpu(ref data)) = storage {
            debug_assert_eq!(
                data.len(),
                dtype.size_bytes() * numel,
                "storage size {} doesn't match dtype {:?} * numel {}",
                data.len(),
                dtype,
                numel
            );
        }
        Self(Rc::new(BufferInner {
            id,
            device,
            dtype,
            numel,
            storage: RefCell::new(storage),
        }))
    }

    /// Create a buffer wrapping device-provided storage.
    ///
    /// Called by [`Device::allocate`], not by user code directly.
    #[must_use]
    pub fn new(device: DeviceId, dtype: DType, numel: usize, storage: Storage) -> Self {
        Self::with_id(next_buffer_id(), device, dtype, numel, Some(storage))
    }

    /// Create an uninitialized output slot that will be filled after execution.
    #[must_use]
    pub fn reserved(device: DeviceId, dtype: DType, numel: usize) -> Self {
        Self::with_id(next_buffer_id(), device, dtype, numel, None)
    }

    /// Return whether this buffer already owns concrete device storage.
    #[must_use]
    pub fn is_realized(&self) -> bool {
        self.0.storage.borrow().is_some()
    }

    /// Return the stable debug id for this buffer slot.
    #[must_use]
    pub fn id(&self) -> BufferId {
        self.0.id
    }

    /// Return the owning device.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        self.0.device
    }

    /// Return the element type expected in this slot.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    /// Return the element count expected in this slot.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.0.numel
    }

    /// Total size in bytes.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.dtype().size_bytes() * self.numel()
    }

    /// Ensure this buffer owns concrete storage, allocating it through the
    /// owning device on first use.
    ///
    /// # Panics
    ///
    /// Panics if the backend returns a buffer with metadata that does not
    /// match this reservation.
    pub fn ensure_allocated(&self) {
        if self.is_realized() {
            return;
        }
        let allocated = get(self.device()).allocate(self.dtype(), self.numel());
        assert_eq!(
            allocated.dtype(),
            self.dtype(),
            "allocated buffer dtype must match reservation"
        );
        assert_eq!(
            allocated.numel(),
            self.numel(),
            "allocated buffer size must match reservation"
        );
        let storage = allocated
            .0
            .storage
            .borrow_mut()
            .take()
            .expect("allocated buffer must contain storage");
        *self.0.storage.borrow_mut() = Some(storage);
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for Buffer {}

impl std::hash::Hash for Buffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

static NEXT_BUFFER_ID: AtomicUsize = AtomicUsize::new(0);

fn next_buffer_id() -> BufferId {
    BufferId(NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed))
}

/// A runtime argument passed to a compiled kernel.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum KernelArg {
    /// Tensor storage passed by pointer.
    Buffer(Buffer),
    /// Scalar `i32` argument used in indexing and loop bounds.
    I32(i32),
    /// Scalar `f32` argument.
    F32(f32),
    /// Scalar boolean argument.
    Bool(bool),
}

/// A compiled program ready to be executed on a device.
///
/// Each backend has its own program representation. `CpuDevice` wraps a
/// [`CompiledKernel`](cpu::CompiledKernel) (clang + dlopen).
/// A future `CudaDevice` would wrap a `CUmodule`/`CUfunction`.
#[derive(Debug)]
#[non_exhaustive]
pub enum Program {
    /// CPU program: a compiled shared library loaded via dlopen.
    Cpu {
        /// The compiled shared library containing the kernel.
        kernel: cpu::CompiledKernel,
        /// Number of runtime arguments the kernel expects.
        num_args: usize,
    },
    // Future: Cuda { module: CudaModule, func: CudaFunction, ... }
}

/// The device trait -- abstracts over CPU, CUDA, and future backends.
///
/// All tensor operations eventually go through a device: allocate buffers,
/// compile generated code, and execute kernels. By programming against this
/// trait, the IR and codegen layers stay device-agnostic.
pub trait Device {
    /// Return this backend's stable identifier.
    fn id(&self) -> DeviceId;

    /// Create an unrealized buffer identity on this device.
    fn reserve_buffer(&self, dtype: DType, numel: usize) -> Buffer;

    /// Allocate a zero-initialized buffer on this device.
    fn allocate(&self, dtype: DType, numel: usize) -> Buffer;

    /// Copy raw host bytes into a device buffer.
    ///
    /// Backends own host transfer semantics, so generic [`Buffer`] stays
    /// device-agnostic.
    ///
    /// # Panics
    ///
    /// Panics if `buffer` belongs to a different device or `src` has the wrong
    /// byte length.
    fn copy_from_host(&self, buffer: &Buffer, src: &[u8]);

    /// Copy raw device bytes back to host memory.
    ///
    /// # Panics
    ///
    /// Panics if `buffer` belongs to a different device or cannot be copied
    /// back to host memory.
    fn copy_to_host(&self, buffer: &Buffer) -> Vec<u8>;

    /// Compile source code into a program that can be executed on this device.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError`] if compilation fails.
    fn compile(
        &self,
        source: &str,
        func_name: &str,
        num_args: usize,
    ) -> Result<Program, DeviceError>;

    /// Execute a compiled program with the given runtime arguments.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError`] if execution fails (e.g. symbol lookup).
    fn execute(&self, program: &Program, args: &mut [KernelArg]) -> Result<(), DeviceError>;

    /// Return a cached compiled program for `sink`, if any.
    fn cached_program(&self, sink: &UOp) -> Option<Rc<Program>>;

    /// Insert a compiled program into the backend cache.
    fn insert_program(&self, sink: UOp, program: Rc<Program>);

    #[cfg(test)]
    /// Clear backend-owned caches so tests start from a clean slate.
    fn clear_for_tests(&self);
}

thread_local! {
    static DEVICES: RefCell<std::collections::HashMap<DeviceId, Rc<dyn Device>>> = RefCell::new(std::collections::HashMap::new());
}

fn new_device(device: DeviceId) -> Rc<dyn Device> {
    match device {
        DeviceId::Cpu => Rc::new(CpuDevice::new()),
    }
}

/// Return the backend object for `device`.
#[must_use]
pub(crate) fn get(device: DeviceId) -> Rc<dyn Device> {
    DEVICES.with(|devices| {
        let mut devices = devices.borrow_mut();
        devices
            .entry(device)
            .or_insert_with(|| new_device(device))
            .clone()
    })
}

#[cfg(test)]
pub(crate) fn clear_for_tests(device: DeviceId) {
    get(device).clear_for_tests();
    crate::uop::clear_for_tests();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_zeroed() {
        // Arrange / Act
        let buf = Buffer::new(DeviceId::Cpu, DType::F32, 4, Storage::Cpu(vec![0u8; 16]));
        let dev = CpuDevice::new();

        // Assert
        assert_eq!(buf.numel(), 4);
        assert_eq!(buf.nbytes(), 16);
        assert_eq!(buf.dtype(), DType::F32);
        assert!(dev.copy_to_host(&buf).iter().all(|&b| b == 0));
    }

    #[test]
    fn test_reserve_buffer_keeps_metadata_unrealized() {
        // Arrange
        let dev = CpuDevice::new();

        // Act
        let buf = dev.reserve_buffer(DType::I32, 6);

        // Assert
        assert_eq!(buf.device(), DeviceId::Cpu);
        assert_eq!(buf.dtype(), DType::I32);
        assert_eq!(buf.numel(), 6);
        assert_eq!(buf.nbytes(), 24);
        assert!(!buf.is_realized());
    }

    #[test]
    fn test_ensure_allocated_realizes_reserved_buffer() {
        // Arrange
        let dev = CpuDevice::new();
        let buf = dev.reserve_buffer(DType::F32, 4);

        // Act
        buf.ensure_allocated();

        // Assert
        assert!(buf.is_realized());
        assert_eq!(buf.device(), DeviceId::Cpu);
        assert_eq!(buf.dtype(), DType::F32);
        assert_eq!(buf.numel(), 4);
        assert!(dev.copy_to_host(&buf).iter().all(|&b| b == 0));
    }

    #[test]
    fn test_buffer_identity_is_stable_per_allocation() {
        // Arrange
        let dev = CpuDevice::new();
        let left = dev.reserve_buffer(DType::F32, 2);
        let left_clone = left.clone();
        let right = dev.reserve_buffer(DType::F32, 2);

        // Act / Assert
        assert_eq!(left, left_clone);
        assert_eq!(left.id(), left_clone.id());
        assert_ne!(left, right);
        assert_ne!(left.id(), right.id());
    }

    #[test]
    fn test_f32_roundtrip() {
        // Arrange
        let dev = CpuDevice::new();
        let input = vec![1.0f32, 2.0, 3.0];

        // Act
        let buf = dev.allocate(DType::F32, input.len());
        dev.copy_from_host(&buf, bytemuck::cast_slice(&input));
        let output: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&dev.copy_to_host(&buf)).to_vec();

        // Assert
        assert_eq!(output, input);
    }

    #[test]
    fn test_copyin_copyout_raw() {
        // Arrange
        let dev = CpuDevice::new();
        let buf = Buffer::new(DeviceId::Cpu, DType::I32, 2, Storage::Cpu(vec![0u8; 8]));
        let src: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0]; // little-endian i32: 1, 2

        // Act
        dev.copy_from_host(&buf, &src);
        let out = dev.copy_to_host(&buf);

        // Assert
        assert_eq!(out, src);
    }

    #[test]
    #[should_panic(expected = "copy_from_host: expected 8 bytes, got 4")]
    fn test_copyin_wrong_size_panics() {
        // Arrange
        let dev = CpuDevice::new();
        let buf = Buffer::new(DeviceId::Cpu, DType::F32, 2, Storage::Cpu(vec![0u8; 8]));

        // Act -- should panic
        dev.copy_from_host(&buf, &[0u8; 4]);
    }

    #[test]
    #[should_panic(expected = "copy_to_host called on unrealized or non-CPU buffer")]
    fn test_copyout_unrealized_reserved_buffer_panics() {
        // Arrange
        let dev = CpuDevice::new();
        let buf = dev.reserve_buffer(DType::F32, 2);

        // Act -- should panic
        let _ = dev.copy_to_host(&buf);
    }
}
