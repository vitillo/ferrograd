# ferrograd

A from-scratch tensor compiler in Rust, inspired by [tinygrad](https://github.com/tinygrad/tinygrad).

## Architecture

Follows the same pipeline as tinygrad, simplified:

```
Tensor API → Lazy UOp Graph → Scheduling → Codegen → Compilation → Execution
```

Each layer is a standalone module you can study independently:

| Module | What it does | Compiler concept |
|--------|-------------|-----------------|
| `tensor` | Lazy tensor API with kernel fusion | Lazy evaluation, scheduling |
| `uop` | DAG-based IR with hash-consing | Intermediate representations |
| `schedule` | Lazy graph → proto-kernels + rangeify | Scheduling, lowering |
| `optimize` | Symbolic simplification, upcast, unroll | Compiler optimizations |
| `codegen` | UOp graph → C source | Code emission |
| `gradient` | Reverse-mode autograd as graph transforms | Automatic differentiation |
| `rewrite` | Fixed-point graph simplification | Term rewriting |
| `nn` | Linear layers and loss functions | Neural network primitives |
| `dtype` | Data types bridging Rust, IR, and C | Type systems |
| `device` | Buffer, Device trait, CPU backend | Hardware abstraction |
| `shape` | Shape metadata and transformations | Tensor algebra |
| `dataset` | MNIST with auto-download and caching | Data loading |

## Example

```rust
use ferrograd::tensor::{Tensor, cpu};

let dev = cpu();

let a = Tensor::new(&[1.0, 2.0, 3.0], &[3], dev);
let b = Tensor::new(&[10.0, 20.0, 30.0], &[3], dev);

// Nothing executes yet — just builds a lazy graph.
let c = a.add(&b).mul(&Tensor::new(&[2.0, 2.0, 2.0], &[3], dev));

// realize() lowers to a single fused kernel, compiles, and runs.
let result = c.to_vec(); // [22.0, 44.0, 66.0]
```

Set `DEBUG=4` to see the generated C source:

```sh
DEBUG=4 cargo run --example demo
```

## Status

The compiler pipeline is functional end-to-end: lazy tensor graphs, multi-kernel
scheduling, reverse-mode autograd, and CPU code generation via clang. The MNIST
example trains a small MLP from scratch.

## Building

```sh
cargo clippy   # build + lint (clippy pedantic is on)
cargo test     # run all tests
```

## License

MIT
