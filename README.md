# ferrograd

A from-scratch tensor compiler in Rust, inspired by [tinygrad](https://github.com/tinygrad/tinygrad).
Built for learning how tensor compilers work, one milestone at a time.

## Architecture

Follows the same pipeline as tinygrad, simplified:

```
Tensor API → Lazy UOp Graph → Scheduling → Codegen → Compilation → Execution
```

Each layer is a standalone module you can study independently:

| Module | What it does | Compiler concept |
|--------|-------------|-----------------|
| `dtype` | Data types bridging Rust, IR, and C | Type systems |
| `device` | Buffer, Device trait, CPU backend | Hardware abstraction |
| `uop` | DAG-based IR with hash-consing | Intermediate representations |
| `codegen` | UOp graph → C source | Code emission / lowering |
| `rewrite` | Pattern matching + algebraic simplification | Term rewriting |
| `tensor` | Lazy tensor API with kernel fusion | Lazy evaluation, scheduling |
| `lower` | Tensor ops → kernel-level UOps | Lowering |

## Example

```rust
use ferrograd::tensor::{Tensor, cpu};

let dev = cpu();

let a = Tensor::from_slice(&[1.0, 2.0, 3.0], &dev);
let b = Tensor::from_slice(&[10.0, 20.0, 30.0], &dev);

// Nothing executes yet — just builds a lazy graph.
let c = a.add(&b).mul(&Tensor::from_slice(&[2.0, 2.0, 2.0], &dev));

// realize() lowers to a single fused kernel, compiles, and runs.
let result = c.to_vec(); // [22.0, 44.0, 66.0]
```

Set `DEBUG=4` to see the generated C source:

```sh
DEBUG=4 cargo run --example demo
```

## Roadmap

The project follows a 10-milestone plan (see [`dev/PLAN.md`](dev/PLAN.md)):

1. **JIT Hello World** — emit C → compile → dlopen → run
2. **DType + Buffer + Device** — type system and hardware abstraction
3. **UOp IR** — DAG-based intermediate representation with interning
4. **Code Generation** — IR to C
5. **Graph Rewriting** — pattern matching and algebraic simplification
6. **Tensor + Lazy Eval** — lazy evaluation and kernel fusion
7. **Reductions + Matmul** — loop nests and index arithmetic
8. **Autograd** — reverse-mode AD as graph transformation
9. **CUDA Backend** — GPU codegen (thread grids replace loops)
10. **MNIST** — train an MLP end-to-end

Milestones 1–6 are implemented.

## Building

```sh
cargo clippy   # build + lint (clippy pedantic is on)
cargo test     # run all tests
```

## License

MIT
