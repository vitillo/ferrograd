//! Demonstrates the full pipeline: lazy tensor ops → fused kernel → execution.
//!
//! Run with `DEBUG=4` to see generated C source:
//! ```sh
//! DEBUG=4 cargo run --example demo
//! ```

#![allow(clippy::many_single_char_names)]

use ferrograd::tensor::{cpu, Tensor};

fn main() {
    let dev = cpu();

    /*let m = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev);
    let n = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &dev);
    let p = m.matmul(&n).to_vec();
    println!("\n  m       = [[1,2,3],[4,5,6]]");
    println!("  n       = [[1,2],[3,4],[5,6]]");
    println!("  m @ n   = {p:?}");
    assert_eq!(p, vec![22.0, 28.0, 49.0, 64.0]);*/

    // Sum reduction
    let s = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev);
    let row_sums = s.sum(&[1]).to_vec();
    println!("\n  s           = [[1,2,3],[4,5,6]]");
    println!("  s.sum(axis=1) = {row_sums:?}");
    assert_eq!(row_sums, vec![6.0, 15.0]);
}
