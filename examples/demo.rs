//! Demonstrates the full pipeline: lazy tensor ops → fused kernel → execution,
//! including autograd for computing gradients.
//!
//! Run with `DEBUG=4` to see generated C source:
//! ```sh
//! DEBUG=4 cargo run --example demo
//! ```

#![allow(clippy::many_single_char_names)]

use ferrograd::optim::Sgd;
use ferrograd::tensor::{cpu, Tensor};

fn main() {
    println!("\n  --- Autograd: loss = sum(x @ w + b) ---");
    let dev = cpu();
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], dev);
    let w = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[2, 2], dev).with_requires_grad(true);
    let b = Tensor::new(&[0.5, 0.6], &[1, 2], dev).with_requires_grad(true);

    let loss = x.matmul(&w).add(&b).sum(&[0, 1]);
    println!("  loss    = {:?}", loss.to_vec());

    loss.backward();
    println!("  dL/dw   = {:?}", w.grad().expect("weight grad").to_vec());
    println!("  dL/db   = {:?}", b.grad().expect("bias grad").to_vec());

    println!("\n  --- SGD step: w -= 0.1 * dL/dw ---");
    let optim = Sgd::new(vec![w.clone()], 0.1);
    optim.step();
    println!("  w_new   = {:?}", w.to_vec());
}
