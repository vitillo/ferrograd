//! Demonstrates the full pipeline: lazy tensor ops → fused kernel → execution.
//!
//! Run with `DEBUG=4` to see generated C source:
//! ```sh
//! DEBUG=4 cargo run --example demo
//! ```

#![allow(clippy::many_single_char_names)]

use rustgrad::tensor::{Tensor, cpu};

fn main() {
    let dev = cpu();

    println!("=== Tensor API: lazy evaluation + kernel fusion ===\n");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &dev);
    let b = Tensor::from_slice(&[10.0, 20.0, 30.0], &dev);
    let two = Tensor::from_slice(&[2.0, 2.0, 2.0], &dev);

    // Nothing has executed yet — just building a lazy graph.
    let c = a.add(&b).mul(&two);
    println!("  c is lazy: realized = {}\n", c.is_realized());

    // realize() lowers the graph to a single fused kernel, compiles, and runs.
    let result = c.to_vec();
    println!("  a       = [1.0, 2.0, 3.0]");
    println!("  b       = [10.0, 20.0, 30.0]");
    println!("  (a+b)*2 = {result:?}");
    assert_eq!(result, vec![22.0, 44.0, 66.0]);

    // Relu
    let x = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0], &dev);
    let r = x.relu().to_vec();
    println!("\n  x       = [1.0, -2.0, 3.0, -4.0]");
    println!("  relu(x) = {r:?}");
    assert_eq!(r, vec![1.0, 0.0, 3.0, 0.0]);

    println!("\n  ✓ Lazy eval + kernel fusion works!\n");
}
