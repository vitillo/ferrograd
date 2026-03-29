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

Train a two-layer MLP on MNIST from scratch:

```rust
use ferrograd::dataset::MNISTDataset;
use ferrograd::nn::{Linear, Parameters};
use ferrograd::optim::Sgd;
use ferrograd::tensor::{cpu, Tensor};

// Define a model
struct Mlp { l1: Linear, l2: Linear }

impl Mlp {
    fn forward(&self, x: &Tensor) -> Tensor {
        let h = self.l1.forward(x).relu();
        self.l2.forward(&h)
    }
}

impl Parameters for Mlp {
    fn parameters(&self) -> Vec<Tensor> {
        [self.l1.parameters(), self.l2.parameters()].concat()
    }
}

// Train
let dataset = MNISTDataset::load().unwrap();
let model = Mlp { l1: Linear::new(784, 128), l2: Linear::new(128, 10) };
let optim = Sgd::new(model.parameters(), 0.01);

let batch_x = dataset.train_images.narrow(0, 0, 256);
let batch_t = one_hot(&dataset.train_labels[..256], 10);

let loss = model.forward(&batch_x).cross_entropy(&batch_t);
loss.backward();  // reverse-mode autograd
optim.step();     // SGD update
```

Everything is lazy — `forward`, `cross_entropy`, and `backward` just build a
graph. `optim.step()` fuses it into kernels, compiles C via clang, and executes.

```sh
cargo run --example mnist --release   # full training loop
DEBUG=4 cargo run --example demo      # see generated C source
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
