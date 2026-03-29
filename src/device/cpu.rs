//! # CPU Backend
//!
//! The complete CPU backend: compiles C with clang, loads via dlopen, and
//! executes kernels. Bundles the compiler, allocator, and kernel dispatch
//! for the CPU target.
//!
//! ## Compilation pipeline
//!
//! ```text
//! C source string
//!       │
//!       ▼
//! ┌─────────────┐
//! │  Write to    │   tempfile crate creates a .c file
//! │  temp file   │
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │  clang       │   Compiles C → shared library (.dylib on macOS, .so on Linux)
//! │  -shared -O2 │   The -shared flag produces a dynamically loadable library
//! └──────┬──────┘
//!        │
//!        ▼
//! ┌─────────────┐
//! │  dlopen      │   libloading crate loads the .dylib/.so into our process
//! │  (libloading)│   This maps the compiled code into our address space
//! └──────┬──────┘
//!        │
//!        ▼
//!   Function pointer   Ready to call with our data buffers
//! ```
//!
//! ## Why clang + dlopen?
//!
//! This is the simplest JIT strategy: let an existing optimizing compiler (clang/LLVM)
//! do the hard work, then load the result. Tinygrad uses this exact approach for its
//! CPU backend. The alternative would be emitting machine code directly (like `LuaJIT`
//! or V8 do), but that's vastly more complex and not needed for a tensor compiler --
//! we're generating simple loop nests, and clang optimizes those well.
//!
//! ## Platform notes
//!
//! - **macOS**: clang produces `.dylib` files. Always available (ships with Xcode CLT).
//! - **Linux**: clang (or gcc) produces `.so` files. Install via `apt install clang`.

use std::io::Write;
use std::process::Command;
use std::{cell::RefCell, collections::HashMap, rc::Rc};

use crate::device::{Buffer, Device, DeviceError, DeviceId, KernelArg, Program, Storage};
use crate::dtype::DType;
use crate::uop::UOp;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors that can occur during CPU JIT compilation and loading.
#[derive(Debug, thiserror::Error)]
pub enum CpuError {
    /// Failed to create a temporary file for C source or compiled output.
    #[error("failed to create temp file: {0}")]
    TempFile(#[from] std::io::Error),

    /// Temp file path contained non-UTF8 characters (rare, but possible).
    #[error("non-UTF8 path: {path}")]
    NonUtf8Path {
        /// The lossy representation of the path that failed.
        path: String,
    },

    /// clang exited with a non-zero status. The stderr output is captured
    /// so you can see the actual compiler errors.
    #[error("clang compilation failed:\n{stderr}")]
    ClangFailed {
        /// The stderr output from clang.
        stderr: String,
    },

    /// dlopen failed to load the compiled shared library.
    #[error("failed to load shared library: {0}")]
    LibLoad(String),

    /// dlsym failed to find the requested function symbol.
    #[error("symbol '{symbol}' not found: {reason}")]
    SymbolNotFound {
        /// The symbol name we tried to look up.
        symbol: String,
        /// The underlying error message.
        reason: String,
    },
}

// ── CompiledKernel ──────────────────────────────────────────────────────────

/// A compiled C kernel loaded into memory, ready to be called.
///
/// Holds the loaded shared library and its backing file. The library stays loaded
/// (and the file stays on disk) as long as this struct is alive. Dropping it unloads
/// the library and invalidates any function pointers obtained from it.
pub struct CompiledKernel {
    /// The loaded shared library. Used by `get_func` to look up symbols.
    lib: libloading::Library,

    /// The function name inside the library (used for symbol lookup).
    func_name: String,
}

enum CallArg {
    Ptr(*mut u8),
    I32(i32),
    F32(f32),
    Bool(u8),
}

impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("func_name", &self.func_name)
            .finish_non_exhaustive()
    }
}

impl CompiledKernel {
    /// Return the CPU tuning flag for the current host.
    ///
    /// Tinygrad uses the same split in its clang JIT: x86 targets prefer
    /// `-march=native`, while ARM uses `-mcpu=native`.
    fn native_cpu_flag() -> &'static str {
        match std::env::consts::ARCH {
            "x86_64" => "-march=native",
            _ => "-mcpu=native",
        }
    }

    /// Compile C source code and load the resulting shared library.
    ///
    /// # Arguments
    /// * `source` - C source code as a string
    /// * `func_name` - The name of the function to look up after loading
    ///
    /// # Errors
    ///
    /// Returns [`CpuError`] if temp file creation, clang compilation, or
    /// library loading fails.
    ///
    /// # The compilation pipeline
    ///
    /// 1. Write `source` to a temporary .c file
    /// 2. Run `clang -shared -O3 <native-cpu-flag> -o output.dylib input.c`
    /// 3. Load the shared library with dlopen
    pub fn new(source: &str, func_name: &str) -> Result<Self, CpuError> {
        let src_file = tempfile::Builder::new().suffix(".c").tempfile()?;

        let lib_ext = if cfg!(target_os = "macos") {
            ".dylib"
        } else {
            ".so"
        };
        let so_file = tempfile::Builder::new().suffix(lib_ext).tempfile()?;

        // Keep so_temp_path alive -- we'll mem::forget it so the .dylib stays
        // on disk while the library is loaded (some OSes require this for dlopen).
        let so_temp_path = so_file.into_temp_path();
        let so_path_str = so_temp_path
            .to_str()
            .ok_or_else(|| CpuError::NonUtf8Path {
                path: so_temp_path.to_string_lossy().into_owned(),
            })?
            .to_string();

        let src_path = src_file.path().to_path_buf();
        {
            let mut file = src_file.as_file();
            file.write_all(source.as_bytes())?;
            file.flush()?;
        }

        let src_path_str = src_path.to_str().ok_or_else(|| CpuError::NonUtf8Path {
            path: src_path.to_string_lossy().into_owned(),
        })?;

        // -shared: produce a dynamically loadable library (not an executable)
        // -O3: let LLVM be more aggressive now that kernels are cached
        // -march/-mcpu=native: tune for the exact host CPU without changing semantics
        let output = Command::new("clang")
            .args(["-shared", "-O3", Self::native_cpu_flag(), "-o"])
            .arg(&so_path_str)
            .arg(src_path_str)
            .output()?;

        if !output.status.success() {
            return Err(CpuError::ClangFailed {
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        let lib = unsafe {
            libloading::Library::new(&*so_temp_path)
                .map_err(|e| CpuError::LibLoad(e.to_string()))?
        };

        std::mem::forget(so_temp_path);

        Ok(CompiledKernel {
            lib,
            func_name: func_name.to_string(),
        })
    }

    /// Look up the function symbol in the loaded library.
    ///
    /// The caller must ensure that the type parameter `F` matches the actual
    /// C function's signature. There is no runtime check for this -- getting
    /// it wrong is undefined behavior (crashes, corruption, etc.).
    ///
    /// # Errors
    ///
    /// Returns [`CpuError::SymbolNotFound`] if the symbol is not in the library.
    pub unsafe fn get_func<F>(&self) -> Result<libloading::Symbol<'_, F>, CpuError> {
        let func: libloading::Symbol<'_, F> =
            self.lib
                .get(self.func_name.as_bytes())
                .map_err(|e| CpuError::SymbolNotFound {
                    symbol: self.func_name.clone(),
                    reason: e.to_string(),
                })?;
        Ok(func)
    }
}

// ── CpuDevice ───────────────────────────────────────────────────────────────

/// The CPU device -- compiles C with clang and runs it via dlopen.
pub struct CpuDevice {
    kernels: RefCell<HashMap<UOp, Rc<Program>>>,
}

impl CpuDevice {
    /// Create a fresh CPU backend with an empty program cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernels: RefCell::new(HashMap::new()),
        }
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

fn buffer_mut_ptr(buffer: &Buffer) -> *mut u8 {
    assert_eq!(
        buffer.device(),
        DeviceId::Cpu,
        "CPU backend received non-CPU buffer {}",
        buffer.id()
    );
    let mut storage = buffer.0.storage.borrow_mut();
    let Some(Storage::Cpu(data)) = storage.as_mut() else {
        panic!("CPU backend received unrealized or non-CPU buffer");
    };
    data.as_mut_ptr()
}

impl Device for CpuDevice {
    fn id(&self) -> DeviceId {
        DeviceId::Cpu
    }

    fn reserve_buffer(&self, dtype: DType, numel: usize) -> Buffer {
        Buffer::reserved(DeviceId::Cpu, dtype, numel)
    }

    fn allocate(&self, dtype: DType, numel: usize) -> Buffer {
        let nbytes = dtype.size_bytes() * numel;
        Buffer::new(DeviceId::Cpu, dtype, numel, Storage::Cpu(vec![0u8; nbytes]))
    }

    fn copy_from_host(&self, buffer: &Buffer, src: &[u8]) {
        assert_eq!(
            buffer.device(),
            DeviceId::Cpu,
            "CPU backend received non-CPU buffer {}",
            buffer.id()
        );
        buffer.ensure_allocated();
        let mut storage = buffer.0.storage.borrow_mut();
        let Some(Storage::Cpu(data)) = storage.as_mut() else {
            panic!("copy_from_host called on non-CPU buffer");
        };
        assert_eq!(
            src.len(),
            data.len(),
            "copy_from_host: expected {} bytes, got {}",
            data.len(),
            src.len()
        );
        data.copy_from_slice(src);
    }

    fn copy_to_host(&self, buffer: &Buffer) -> Vec<u8> {
        assert_eq!(
            buffer.device(),
            DeviceId::Cpu,
            "CPU backend received non-CPU buffer {}",
            buffer.id()
        );
        let storage = buffer.0.storage.borrow();
        let Some(Storage::Cpu(data)) = storage.as_ref() else {
            panic!("copy_to_host called on unrealized or non-CPU buffer");
        };
        data.clone()
    }

    fn compile(
        &self,
        source: &str,
        func_name: &str,
        num_args: usize,
    ) -> Result<Program, DeviceError> {
        let kernel = CompiledKernel::new(source, func_name)?;
        Ok(Program::Cpu { kernel, num_args })
    }

    fn execute(&self, program: &Program, args: &mut [KernelArg]) -> Result<(), DeviceError> {
        let Program::Cpu { kernel, num_args } = program;

        assert_eq!(
            args.len(),
            *num_args,
            "expected {num_args} kernel args, got {}",
            args.len()
        );

        let call_args: Vec<CallArg> = args
            .iter_mut()
            .map(|arg| match arg {
                KernelArg::Buffer(buffer) => CallArg::Ptr(buffer_mut_ptr(buffer)),
                KernelArg::I32(value) => CallArg::I32(*value),
                KernelArg::F32(value) => CallArg::F32(*value),
                KernelArg::Bool(value) => CallArg::Bool(u8::from(*value)),
            })
            .collect();

        // Use libffi to call the kernel with a dynamic mix of pointer and scalar args.
        let cif = libffi::middle::Cif::new(
            call_args
                .iter()
                .map(|arg| match arg {
                    CallArg::Ptr(_) => libffi::middle::Type::pointer(),
                    CallArg::I32(_) => libffi::middle::Type::i32(),
                    CallArg::F32(_) => libffi::middle::Type::f32(),
                    CallArg::Bool(_) => libffi::middle::Type::u8(),
                })
                .collect::<Vec<_>>(),
            libffi::middle::Type::void(),
        );
        let ffi_args: Vec<libffi::middle::Arg> = call_args
            .iter()
            .map(|arg| match arg {
                CallArg::Ptr(value) => libffi::middle::arg(value),
                CallArg::I32(value) => libffi::middle::arg(value),
                CallArg::F32(value) => libffi::middle::arg(value),
                CallArg::Bool(value) => libffi::middle::arg(value),
            })
            .collect();

        // SAFETY: We trust that the compiled kernel's signature matches
        // the number and type of arguments we're passing.
        unsafe {
            let func: libloading::Symbol<'_, fn()> = kernel.get_func()?;
            let code_ptr = libffi::high::CodePtr(func.into_raw().as_raw_ptr().cast());
            cif.call::<()>(code_ptr, &ffi_args);
        }

        Ok(())
    }

    fn cached_program(&self, sink: &UOp) -> Option<Rc<Program>> {
        self.kernels.borrow().get(sink).cloned()
    }

    fn insert_program(&self, sink: UOp, program: Rc<Program>) {
        self.kernels.borrow_mut().insert(sink, program);
    }

    #[cfg(test)]
    fn clear_for_tests(&self) {
        self.kernels.borrow_mut().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_buffer(dev: &CpuDevice, data: &[f32]) -> Buffer {
        let buffer = dev.allocate(DType::F32, data.len());
        dev.copy_from_host(&buffer, bytemuck::cast_slice(data));
        buffer
    }

    fn read_f32(dev: &CpuDevice, buffer: &Buffer) -> Vec<f32> {
        bytemuck::cast_slice::<u8, f32>(&dev.copy_to_host(buffer)).to_vec()
    }

    // ── CompiledKernel tests ────────────────────────────────────────────

    /// Helper type alias to reduce noise in tests.
    type BinOpFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);

    #[test]
    fn test_compile_and_call_add() {
        // Arrange
        let source = r"
            void add(float* a, float* b, float* out, int n) {
                for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
            }
        ";
        let kernel = CompiledKernel::new(source, "add").expect("compile failed");
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = vec![0.0f32; 3];

        // Act
        unsafe {
            let f: libloading::Symbol<'_, BinOpFn> = kernel.get_func().unwrap();
            f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 3);
        }

        // Assert
        assert_eq!(out, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_compile_and_call_mul() {
        // Arrange
        let source = r"
            void mul(float* a, float* b, float* out, int n) {
                for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
            }
        ";
        let kernel = CompiledKernel::new(source, "mul").expect("compile failed");
        let a = [2.0f32, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0];
        let mut out = vec![0.0f32; 3];

        // Act
        unsafe {
            let f: libloading::Symbol<'_, BinOpFn> = kernel.get_func().unwrap();
            f(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 3);
        }

        // Assert
        assert_eq!(out, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_compile_and_call_scalar() {
        // Arrange
        let source = r"
            void scale(float* data, float* out, int n, float scalar) {
                for (int i = 0; i < n; i++) out[i] = data[i] * scalar;
            }
        ";
        let kernel = CompiledKernel::new(source, "scale").expect("compile failed");
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];

        // Act
        unsafe {
            let f: libloading::Symbol<'_, unsafe extern "C" fn(*const f32, *mut f32, i32, f32)> =
                kernel.get_func().unwrap();
            f(data.as_ptr(), out.as_mut_ptr(), 4, 3.0);
        }

        // Assert
        assert_eq!(out, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_invalid_c_source_returns_error() {
        // Arrange
        let bad_source = "this is not valid C!";

        // Act
        let result = CompiledKernel::new(bad_source, "nope");

        // Assert
        assert!(
            matches!(result, Err(CpuError::ClangFailed { .. })),
            "expected ClangFailed, got: {result:?}"
        );
    }

    #[test]
    fn test_wrong_symbol_returns_error() {
        // Arrange
        let source = r"
            void real_name(float* a, int n) {}
        ";
        let kernel = CompiledKernel::new(source, "wrong_name").unwrap();

        // Act
        let result: Result<libloading::Symbol<'_, unsafe extern "C" fn()>, _> =
            unsafe { kernel.get_func() };

        // Assert
        assert!(
            matches!(result, Err(CpuError::SymbolNotFound { .. })),
            "expected SymbolNotFound, got: {result:?}"
        );
    }

    // ── CpuDevice tests ────────────────────────────────────────────────

    #[test]
    fn test_allocate_through_device() {
        // Arrange
        let dev = CpuDevice::new();

        // Act
        let buf = dev.allocate(DType::F32, 4);

        // Assert
        assert_eq!(buf.numel(), 4);
        assert_eq!(buf.nbytes(), 16);
    }

    #[test]
    fn test_device_add() {
        // Arrange
        let dev = CpuDevice::new();
        let source = r"
            void add(float* out, float* a, float* b) {
                for (int i = 0; i < 3; i++) out[i] = a[i] + b[i];
            }
        ";
        let program = dev.compile(source, "add", 3).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 3)),
            KernelArg::Buffer(f32_buffer(&dev, &[1.0, 2.0, 3.0])),
            KernelArg::Buffer(f32_buffer(&dev, &[4.0, 5.0, 6.0])),
        ];

        // Act
        dev.execute(&program, &mut args).unwrap();

        // Assert
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, out), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_device_mul() {
        // Arrange
        let dev = CpuDevice::new();
        let source = r"
            void mul(float* out, float* a, float* b) {
                for (int i = 0; i < 3; i++) out[i] = a[i] * b[i];
            }
        ";
        let program = dev.compile(source, "mul", 3).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 3)),
            KernelArg::Buffer(f32_buffer(&dev, &[2.0, 3.0, 4.0])),
            KernelArg::Buffer(f32_buffer(&dev, &[5.0, 6.0, 7.0])),
        ];

        // Act
        dev.execute(&program, &mut args).unwrap();

        // Assert
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, out), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_device_unary() {
        // Arrange
        let dev = CpuDevice::new();
        let source = r"
            void negate(float* output, float* input) {
                for (int i = 0; i < 3; i++) output[i] = -input[i];
            }
        ";
        let program = dev.compile(source, "negate", 2).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 3)),
            KernelArg::Buffer(f32_buffer(&dev, &[1.0, -2.0, 3.0])),
        ];

        // Act
        dev.execute(&program, &mut args).unwrap();

        // Assert
        let KernelArg::Buffer(output) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, output), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_device_mixed_scalar_args() {
        // Arrange
        let dev = CpuDevice::new();
        let source = r"
            void add_offset(float* out, float* input, int offset, float scale) {
                for (int i = 0; i < 2; i++) out[i] = (input[offset + i] * scale);
            }
        ";
        let program = dev
            .compile(source, "add_offset", 4)
            .expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 2)),
            KernelArg::Buffer(f32_buffer(&dev, &[1.0, 2.0, 3.0, 4.0])),
            KernelArg::I32(1),
            KernelArg::F32(10.0),
        ];

        // Act
        dev.execute(&program, &mut args).unwrap();

        // Assert
        let KernelArg::Buffer(output) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, output), vec![20.0, 30.0]);
    }

    #[test]
    fn test_device_compile_bad_source() {
        // Arrange
        let dev = CpuDevice::new();

        // Act
        let result = dev.compile("not valid C!", "nope", 1);

        // Assert
        assert!(result.is_err());
    }
}
