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
//! ## Module overview
//!
//! - [`dtype`] -- Data types that bridge Rust, IR, and C worlds (M2)
//! - [`device`] -- Device trait, Buffer, Storage, plus backend implementations (M2)
//! - [`uop`] -- DAG-based intermediate representation with hash-consing (M3)
//! - [`codegen`] -- Code generation: `UOp` IR to C source (M4)

pub mod codegen;
pub mod device;
pub mod dtype;
pub mod uop;

use codegen::{ClangRenderer, Renderer};
use device::{Buffer, CpuDevice, Device};
use dtype::DType;
use uop::UOpGraph;

fn main() {
    println!("=== Milestone 4: Code Generation (IR → C) ===\n");

    // Build a UOp graph: out[i] = (a[i] + b[i]) * 2.0
    let mut g = UOpGraph::new();
    let out_ptr = g.param(0, DType::F32);
    let a_ptr = g.param(1, DType::F32);
    let b_ptr = g.param(2, DType::F32);
    let n = g.const_int(3, DType::I32);
    let two = g.const_float(2.0, DType::F32);
    let idx = g.range(0, n);
    let a_idx = g.index(a_ptr, idx);
    let a_val = g.load(a_idx, DType::F32);
    let b_idx = g.index(b_ptr, idx);
    let b_val = g.load(b_idx, DType::F32);
    let sum = g.add_op(a_val, b_val);
    let result = g.mul(sum, two);
    let out_idx = g.index(out_ptr, idx);
    let store = g.store(out_idx, result);
    let end = g.end(idx);
    let sink = g.sink(vec![store, end]);

    // Render to C
    let code = ClangRenderer.render(&g, "fused_muladd", sink);
    println!("  Generated C:\n");
    for line in code.lines() {
        println!("    {line}");
    }

    // Compile and run
    let dev = CpuDevice;
    let program = dev
        .compile(&code, "fused_muladd", 3)
        .expect("compile failed");
    let mut a = Buffer::from_f32(&[1.0, 2.0, 3.0]);
    let mut b = Buffer::from_f32(&[10.0, 20.0, 30.0]);
    let mut out = dev.allocate(DType::F32, 3);

    dev.execute(&program, &mut [&mut out, &mut a, &mut b])
        .expect("execution failed");

    let output = out.to_f32();
    println!("\n  a      = [1.0, 2.0, 3.0]");
    println!("  b      = [10.0, 20.0, 30.0]");
    println!("  result = {output:?}");
    assert_eq!(output, vec![22.0, 44.0, 66.0]);
    println!("\n  ✓ IR → C → compile → run works!\n");
}
