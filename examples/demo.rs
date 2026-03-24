//! Demonstrates the full pipeline: lazy tensor ops → fused kernel → execution,
//! including autograd for computing gradients.
//!
//! Run with `DEBUG=4` to see generated C source:
//! ```sh
//! DEBUG=4 cargo run --example demo
//! ```

#![allow(clippy::many_single_char_names)]

use ferrograd::tensor::Tensor;

fn main() {
    println!("\n  --- Autograd: loss = sum(x @ w + b) ---");
    let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let w = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], &[2, 2]);
    let b = Tensor::from_slice(&[0.5, 0.6], &[1, 2]);

    let loss = x.matmul(&w).add(&b).sum(&[0, 1]);
    println!("  loss    = {:?}", loss.to_vec());

    let grads = loss.gradient(&[&w, &b]);
    println!("  dL/dw   = {:?}", grads[0].to_vec());
    println!("  dL/db   = {:?}", grads[1].to_vec());

    println!("\n  --- SGD step: w -= 0.1 * dL/dw ---");
    let lr = Tensor::from_slice(&[0.1], &[1, 1]);
    let w_new = w.sub(&grads[0].mul(&lr));
    println!("  w_new   = {:?}", w_new.to_vec());
}
