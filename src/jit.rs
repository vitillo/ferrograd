//! # JIT Compilation Module
//!
//! This module handles the "compile C source → load shared library → get function pointer"
//! pipeline. It's the runtime backbone that every later milestone builds on.
//!
//! ## How it works
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

/// Errors that can occur during JIT compilation and loading.
#[derive(Debug, thiserror::Error)]
pub enum JitError {
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
    /// Returns [`JitError`] if temp file creation, clang compilation, or
    /// library loading fails.
    ///
    /// # The compilation pipeline
    ///
    /// 1. Write `source` to a temporary .c file
    /// 2. Run `clang -shared -O2 -o output.dylib input.c`
    /// 3. Load the shared library with dlopen
    pub fn new(source: &str, func_name: &str) -> Result<Self, JitError> {
        // Create temp files for the C source and compiled output.
        // We use tempfile::Builder so we can control the suffix (.c and .dylib/.so).
        let src_file = tempfile::Builder::new()
            .suffix(".c")
            .tempfile()?;

        // Determine the shared library extension based on the platform.
        let lib_ext = if cfg!(target_os = "macos") {
            ".dylib"
        } else {
            ".so"
        };

        let so_file = tempfile::Builder::new()
            .suffix(lib_ext)
            .tempfile()?;

        // Convert to TempPath so the file stays on disk but we get the path.
        // We'll forget this later to prevent deletion while the library is loaded.
        let so_temp_path = so_file.into_temp_path();
        let so_path_str = so_temp_path
            .to_str()
            .ok_or_else(|| JitError::NonUtf8Path {
                path: so_temp_path.to_string_lossy().into_owned(),
            })?
            .to_string();

        // Step 1: Write the C source to the temp file
        let src_path = src_file.path().to_path_buf();
        {
            let mut file = src_file.as_file();
            file.write_all(source.as_bytes())?;
            file.flush()?;
        }

        // Step 2: Compile with clang
        //
        // Flags:
        //   -shared    → produce a shared library (not an executable)
        //   -O2        → optimize (makes the generated code fast without slow compile times)
        //   -o <path>  → output path
        //
        // On macOS, clang is always available via Xcode Command Line Tools.
        // On Linux, install with: apt install clang
        let src_path_str = src_path
            .to_str()
            .ok_or_else(|| JitError::NonUtf8Path {
                path: src_path.to_string_lossy().into_owned(),
            })?;

        let output = Command::new("clang")
            .args(["-shared", "-O2", "-o", &so_path_str, src_path_str])
            .output()?;

        if !output.status.success() {
            return Err(JitError::ClangFailed {
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        // Step 3: Load the shared library
        //
        // dlopen maps the compiled machine code into our process's address space.
        // After this, we can look up symbols (function names) and get callable
        // function pointers.
        let lib = unsafe {
            libloading::Library::new(&*so_temp_path)
                .map_err(|e| JitError::LibLoad(e.to_string()))?
        };

        // Leak the TempPath so the .dylib stays on disk while the library is loaded.
        // Some OSes need the file to exist for dlopen'd code to work.
        // In a production system we'd clean these up; for an educational JIT this is fine.
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
    /// For example, if the C function is:
    ///   `void add_arrays(float* a, float* b, float* out, int n)`
    ///
    /// Then F must be:
    ///   `unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32)`
    ///
    /// # Errors
    ///
    /// Returns [`JitError::SymbolNotFound`] if the symbol is not in the library.
    pub unsafe fn get_func<F>(&self) -> Result<libloading::Symbol<'_, F>, JitError> {
        let func: libloading::Symbol<'_, F> =
            self.lib.get(self.func_name.as_bytes()).map_err(|e| {
                JitError::SymbolNotFound {
                    symbol: self.func_name.clone(),
                    reason: e.to_string(),
                }
            })?;
        Ok(func)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper type alias to reduce noise in tests.
    type BinOpFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);

    #[test]
    fn test_add_kernel() {
        // Arrange
        let source = r#"
            void add(float* a, float* b, float* out, int n) {
                for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
            }
        "#;
        let kernel = CompiledKernel::new(source, "add").expect("compile failed");
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
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
    fn test_mul_kernel() {
        // Arrange
        let source = r#"
            void mul(float* a, float* b, float* out, int n) {
                for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
            }
        "#;
        let kernel = CompiledKernel::new(source, "mul").expect("compile failed");
        let a = vec![2.0f32, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0];
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
    fn test_scalar_kernel() {
        // Arrange
        let source = r#"
            void scale(float* data, float* out, int n, float scalar) {
                for (int i = 0; i < n; i++) out[i] = data[i] * scalar;
            }
        "#;
        let kernel = CompiledKernel::new(source, "scale").expect("compile failed");
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
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
    fn test_invalid_c_source_returns_clang_error() {
        // Arrange
        let bad_source = "this is not valid C!";

        // Act
        let result = CompiledKernel::new(bad_source, "nope");

        // Assert
        assert!(
            matches!(result, Err(JitError::ClangFailed { .. })),
            "expected ClangFailed, got: {result:?}"
        );
    }

    #[test]
    fn test_wrong_symbol_name_returns_symbol_error() {
        // Arrange
        let source = r#"
            void real_name(float* a, int n) {}
        "#;
        let kernel = CompiledKernel::new(source, "wrong_name").unwrap();

        // Act
        let result: Result<libloading::Symbol<'_, unsafe extern "C" fn()>, _> =
            unsafe { kernel.get_func() };

        // Assert
        assert!(
            matches!(result, Err(JitError::SymbolNotFound { .. })),
            "expected SymbolNotFound, got: {result:?}"
        );
    }
}
