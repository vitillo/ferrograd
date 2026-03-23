//! # CPU Backend
//!
//! The complete CPU backend: compiles C with clang, loads via dlopen, and
//! executes kernels. This is tinygrad's `ops_cpu.py` equivalent -- it bundles
//! the compiler, allocator, and kernel dispatch for the CPU target.
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

use crate::device::{Buffer, Device, DeviceError, Program, Storage};
use crate::dtype::DType;

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

impl std::fmt::Debug for CompiledKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledKernel")
            .field("func_name", &self.func_name)
            .finish_non_exhaustive()
    }
}

impl CompiledKernel {
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
    /// 2. Run `clang -shared -O2 -o output.dylib input.c`
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
        // -O2: optimize without slow compile times
        let output = Command::new("clang")
            .args(["-shared", "-O2", "-o", &so_path_str, src_path_str])
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
///
/// Tinygrad's equivalent is `CPUDevice` in `tinygrad/runtime/ops_cpu.py`,
/// which uses `ClangJITCompiler` + `CPUProgram` in the same way.
pub struct CpuDevice;

impl Device for CpuDevice {
    fn allocate(&self, dtype: DType, numel: usize) -> Buffer {
        let nbytes = dtype.size_bytes() * numel;
        Buffer::new(dtype, numel, Storage::Cpu(vec![0u8; nbytes]))
    }

    fn compile(
        &self,
        source: &str,
        func_name: &str,
        num_bufs: usize,
    ) -> Result<Program, DeviceError> {
        let kernel = CompiledKernel::new(source, func_name)?;
        Ok(Program::Cpu { kernel, num_bufs })
    }

    fn execute(&self, program: &Program, bufs: &mut [&mut Buffer]) -> Result<(), DeviceError> {
        let Program::Cpu { kernel, num_bufs } = program;

        assert_eq!(
            bufs.len(),
            *num_bufs,
            "expected {num_bufs} buffers, got {}",
            bufs.len()
        );

        let ptrs: Vec<*mut u8> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();

        // Use libffi to call the kernel with a dynamic number of pointer args.
        // The kernel signature is `void kernel(float* data0, float* data1, ...)`.
        let cif = libffi::middle::Cif::new(
            vec![libffi::middle::Type::pointer(); ptrs.len()],
            libffi::middle::Type::void(),
        );
        let args: Vec<libffi::middle::Arg> =
            ptrs.iter().map(|p| libffi::middle::arg(p)).collect();

        // SAFETY: We trust that the compiled kernel's signature matches
        // the number and type of pointers we're passing.
        unsafe {
            let func: libloading::Symbol<'_, fn()> = kernel.get_func()?;
            let code_ptr = libffi::high::CodePtr(func.into_raw().as_raw_ptr().cast());
            cif.call::<()>(code_ptr, &args);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let dev = CpuDevice;

        // Act
        let buf = dev.allocate(DType::F32, 4);

        // Assert
        assert_eq!(buf.numel(), 4);
        assert_eq!(buf.nbytes(), 16);
    }

    #[test]
    fn test_device_add() {
        // Arrange
        let dev = CpuDevice;
        let source = r"
            void add(float* out, float* a, float* b) {
                for (int i = 0; i < 3; i++) out[i] = a[i] + b[i];
            }
        ";
        let program = dev.compile(source, "add", 3).expect("compile failed");
        let mut a = Buffer::from_f32(&[1.0, 2.0, 3.0]);
        let mut b = Buffer::from_f32(&[4.0, 5.0, 6.0]);
        let mut out = dev.allocate(DType::F32, 3);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a, &mut b])
            .unwrap();

        // Assert
        assert_eq!(out.to_f32(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_device_mul() {
        // Arrange
        let dev = CpuDevice;
        let source = r"
            void mul(float* out, float* a, float* b) {
                for (int i = 0; i < 3; i++) out[i] = a[i] * b[i];
            }
        ";
        let program = dev.compile(source, "mul", 3).expect("compile failed");
        let mut a = Buffer::from_f32(&[2.0, 3.0, 4.0]);
        let mut b = Buffer::from_f32(&[5.0, 6.0, 7.0]);
        let mut out = dev.allocate(DType::F32, 3);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a, &mut b])
            .unwrap();

        // Assert
        assert_eq!(out.to_f32(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_device_unary() {
        // Arrange
        let dev = CpuDevice;
        let source = r"
            void negate(float* output, float* input) {
                for (int i = 0; i < 3; i++) output[i] = -input[i];
            }
        ";
        let program = dev.compile(source, "negate", 2).expect("compile failed");
        let mut input = Buffer::from_f32(&[1.0, -2.0, 3.0]);
        let mut output = dev.allocate(DType::F32, 3);

        // Act
        dev.execute(&program, &mut [&mut output, &mut input])
            .unwrap();

        // Assert
        assert_eq!(output.to_f32(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_device_compile_bad_source() {
        // Arrange
        let dev = CpuDevice;

        // Act
        let result = dev.compile("not valid C!", "nope", 1);

        // Assert
        assert!(result.is_err());
    }
}
