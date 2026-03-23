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
//!
//! Backend implementations live in submodules:
//! - [`cpu`] -- compiles C with clang, runs via dlopen

pub mod cpu;

use crate::dtype::DType;

// Re-export CpuDevice for convenience.
pub use cpu::CpuDevice;

/// Errors that can occur during device operations.
#[derive(Debug, thiserror::Error)]
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
pub enum Storage {
    /// Host memory, owned as a flat byte array. Used by `CpuDevice`.
    Cpu(Vec<u8>),
    // Future: Cuda { device_ptr: u64, len: usize }
}

/// A typed memory buffer for kernel data.
///
/// Buffers are created by a [`Device`] and carry the device's opaque
/// [`Storage`]. The `dtype` and `numel` give the raw bytes meaning.
///
/// A device-allocated memory region holding tensor data.
#[derive(Debug, Clone)]
pub struct Buffer {
    /// What type each element is.
    dtype: DType,
    /// How many elements (not bytes).
    numel: usize,
    /// Device-managed memory.
    storage: Storage,
}

impl Buffer {
    /// Create a buffer wrapping device-provided storage.
    ///
    /// Called by [`Device::allocate`], not by user code directly.
    #[must_use]
    pub fn new(dtype: DType, numel: usize, storage: Storage) -> Self {
        let Storage::Cpu(ref data) = storage;
        debug_assert_eq!(
            data.len(),
            dtype.size_bytes() * numel,
            "storage size {} doesn't match dtype {:?} * numel {}",
            data.len(),
            dtype,
            numel
        );
        Self {
            dtype,
            numel,
            storage,
        }
    }

    /// Total size in bytes.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.dtype.size_bytes() * self.numel
    }

    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// The element type.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Borrow the underlying storage.
    #[must_use]
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Mutably borrow the underlying storage.
    #[must_use]
    pub fn storage_mut(&mut self) -> &mut Storage {
        &mut self.storage
    }

    /// Get a mutable raw pointer to the buffer's CPU memory.
    ///
    /// # Panics
    ///
    /// Panics if the storage isn't `Storage::Cpu`.
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        let Storage::Cpu(ref mut data) = self.storage;
        data.as_mut_ptr()
    }

    /// Copy raw bytes from the host into this buffer.
    ///
    /// # Panics
    ///
    /// Panics if `src` length doesn't match the buffer's byte size, or if
    /// the storage isn't CPU-backed.
    pub fn copyin(&mut self, src: &[u8]) {
        let Storage::Cpu(ref mut data) = self.storage;
        assert_eq!(
            src.len(),
            data.len(),
            "copyin: expected {} bytes, got {}",
            data.len(),
            src.len()
        );
        data.copy_from_slice(src);
    }

    /// Copy the buffer's raw bytes back to the host.
    ///
    /// # Panics
    ///
    /// Panics if the storage isn't CPU-backed.
    #[must_use]
    pub fn copyout(&self) -> Vec<u8> {
        let Storage::Cpu(ref data) = self.storage;
        data.clone()
    }

    /// Create a CPU buffer from a `&[f32]`, copying the data in.
    ///
    /// Convenience for tests and demos. In real usage, buffers are created
    /// through a [`Device`].
    #[must_use]
    pub fn from_f32(data: &[f32]) -> Self {
        let bytes = bytemuck::cast_slice(data).to_vec();
        Self {
            dtype: DType::F32,
            numel: data.len(),
            storage: Storage::Cpu(bytes),
        }
    }

    /// Read this buffer's contents as a `Vec<f32>`.
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not `F32` or storage isn't CPU-backed.
    #[must_use]
    pub fn to_f32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, DType::F32, "to_f32 called on {}", self.dtype);
        let Storage::Cpu(ref data) = self.storage;
        bytemuck::cast_slice(data).to_vec()
    }
}

/// A compiled program ready to be executed on a device.
///
/// Each backend has its own program representation. `CpuDevice` wraps a
/// [`CompiledKernel`](cpu::CompiledKernel) (clang + dlopen).
/// A future `CudaDevice` would wrap a `CUmodule`/`CUfunction`.
pub enum Program {
    /// CPU program: a compiled shared library loaded via dlopen.
    Cpu {
        /// The compiled shared library containing the kernel.
        kernel: cpu::CompiledKernel,
        /// Number of buffer arguments the kernel expects.
        num_bufs: usize,
    },
    // Future: Cuda { module: CudaModule, func: CudaFunction, ... }
}

/// The device trait -- abstracts over CPU, CUDA, and future backends.
///
/// All tensor operations eventually go through a device: allocate buffers,
/// compile generated code, and execute kernels. By programming against this
/// trait, the IR and codegen layers stay device-agnostic.
pub trait Device {
    /// Allocate a zero-initialized buffer on this device.
    fn allocate(&self, dtype: DType, numel: usize) -> Buffer;

    /// Compile source code into a program that can be executed on this device.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError`] if compilation fails.
    fn compile(
        &self,
        source: &str,
        func_name: &str,
        num_bufs: usize,
    ) -> Result<Program, DeviceError>;

    /// Execute a compiled program with the given buffer arguments.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError`] if execution fails (e.g. symbol lookup).
    fn execute(&self, program: &Program, bufs: &mut [&mut Buffer]) -> Result<(), DeviceError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_zeroed() {
        // Arrange / Act
        let buf = Buffer::new(DType::F32, 4, Storage::Cpu(vec![0u8; 16]));

        // Assert
        assert_eq!(buf.numel(), 4);
        assert_eq!(buf.nbytes(), 16);
        assert_eq!(buf.dtype(), DType::F32);
        assert!(buf.copyout().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_f32_roundtrip() {
        // Arrange
        let input = vec![1.0f32, 2.0, 3.0];

        // Act
        let buf = Buffer::from_f32(&input);
        let output = buf.to_f32();

        // Assert
        assert_eq!(output, input);
    }

    #[test]
    fn test_copyin_copyout_raw() {
        // Arrange
        let mut buf = Buffer::new(DType::I32, 2, Storage::Cpu(vec![0u8; 8]));
        let src: Vec<u8> = vec![1, 0, 0, 0, 2, 0, 0, 0]; // little-endian i32: 1, 2

        // Act
        buf.copyin(&src);
        let out = buf.copyout();

        // Assert
        assert_eq!(out, src);
    }

    #[test]
    #[should_panic(expected = "copyin: expected 8 bytes, got 4")]
    fn test_copyin_wrong_size_panics() {
        // Arrange
        let mut buf = Buffer::new(DType::F32, 2, Storage::Cpu(vec![0u8; 8]));

        // Act -- should panic
        buf.copyin(&[0u8; 4]);
    }
}
