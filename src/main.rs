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

pub mod device;
pub mod dtype;
pub mod uop;

use device::{Buffer, CpuDevice, Device};
use dtype::DType;

fn main() {
    println!("=== Milestone 2: Device Abstraction ===\n");

    let dev = CpuDevice;

    let source = r"
        void add(float* a, float* b, float* out, int n) {
            for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
        }
    ";
    let program = dev.compile(source, "add", 3).expect("compile failed");

    let mut a = Buffer::from_f32(&[1.0, 2.0, 3.0]);
    let mut b = Buffer::from_f32(&[4.0, 5.0, 6.0]);
    let mut out = dev.allocate(DType::F32, 3);

    dev.execute(&program, &mut [&mut a, &mut b, &mut out])
        .expect("execution failed");

    let result = out.to_f32();
    println!("  a      = [1.0, 2.0, 3.0]");
    println!("  b      = [4.0, 5.0, 6.0]");
    println!("  result = {result:?}");
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
    println!("\n  ✓ Device-abstracted add works!\n");
}
