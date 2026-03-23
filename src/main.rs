//! # rustgrad
//!
//! A from-scratch tensor compiler in Rust, inspired by tinygrad.
//! Built for learning how tensor compilers work, one milestone at a time.
//!
//! ## Architecture (same as tinygrad, simplified)
//!
//! ```text
//! Tensor API → Lazy UOp Graph → Scheduling → Codegen → Compilation → Execution
//! ```
//!
//! ## Current milestone: M1 -- JIT Compilation Hello World
//!
//! The foundational pattern behind every runtime code generator:
//!
//!   1. **Emit** source code as a string (here: C)
//!   2. **Compile** it to a shared library (here: clang → .dylib/.so)
//!   3. **Load** the compiled code at runtime (here: dlopen via `libloading`)
//!   4. **Run** the function pointer with our data
//!
//! This is exactly what tinygrad does on the CPU backend:
//!   - `ClangJITCompiler` (`tinygrad/runtime/support/compiler_cpu.py`) calls clang
//!   - `CPUProgram` (`tinygrad/runtime/ops_cpu.py`) loads the result into memory
//!
//! In later milestones, the "emit" step will be replaced by our IR → C codegen,
//! but the compile → load → run loop stays the same.

pub mod jit;

fn main() {
    println!("=== Milestone 1: JIT Compilation ===\n");

    // The C kernel we want to compile and run.
    // This is hardcoded now; in M4 we'll generate this from our IR.
    let c_source = r"
        void add_arrays(float* a, float* b, float* out, int n) {
            for (int i = 0; i < n; i++) {
                out[i] = a[i] + b[i];
            }
        }
    ";

    // Step 1-3: Compile the C source and load it as a callable function.
    let kernel = jit::CompiledKernel::new(c_source, "add_arrays")
        .expect("Failed to compile kernel");

    // Prepare input data
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];
    let mut out: Vec<f32> = vec![0.0; 3];
    let n: i32 = 3;

    // Step 4: Call the compiled function through its function pointer.
    //
    // SAFETY: We're calling a C function with the correct signature.
    // The function pointer type must exactly match what the C code declares.
    // This is inherently unsafe -- there's no way for Rust to verify the C
    // function's actual signature matches what we claim here.
    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32)> =
            kernel.get_func().expect("Failed to get function symbol");

        func(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n);
    }

    println!("  a   = {a:?}");
    println!("  b   = {b:?}");
    println!("  out = {out:?}");
    println!();

    // Verify correctness
    let expected: Vec<f32> = vec![5.0, 7.0, 9.0];
    assert_eq!(out, expected, "Kernel output doesn't match expected result!");
    println!("  ✓ [1,2,3] + [4,5,6] = [5,7,9] -- JIT compilation works!\n");

    // Bonus: demonstrate that we can compile and run different kernels
    println!("--- Bonus: multiply kernel ---\n");

    let mul_source = r"
        void mul_arrays(float* a, float* b, float* out, int n) {
            for (int i = 0; i < n; i++) {
                out[i] = a[i] * b[i];
            }
        }
    ";

    let mul_kernel = jit::CompiledKernel::new(mul_source, "mul_arrays")
        .expect("Failed to compile multiply kernel");

    let mut mul_out: Vec<f32> = vec![0.0; 3];

    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32)> =
            mul_kernel.get_func().expect("Failed to get function symbol");

        func(a.as_ptr(), b.as_ptr(), mul_out.as_mut_ptr(), n);
    }

    println!("  a   = {a:?}");
    println!("  b   = {b:?}");
    println!("  out = {mul_out:?}");

    let expected_mul: Vec<f32> = vec![4.0, 10.0, 18.0];
    assert_eq!(mul_out, expected_mul);
    println!("\n  ✓ [1,2,3] * [4,5,6] = [4,10,18] -- second kernel works too!\n");
}
