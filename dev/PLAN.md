# Rebuilding Tinygrad in Rust -- Educational Plan

## Context

The goal is to learn how a tensor compiler works by building one from scratch in Rust. The target is CPU + CUDA backends, enough to train an MLP on MNIST. Each milestone introduces a compiler concept, is independently testable, and builds on the previous one.

This plan is based on a deep exploration of the tinygrad codebase. Tinygrad's pipeline is: **Tensor API → Lazy UOp Graph → Scheduling → Codegen → Compilation → Execution**. We follow the same architecture but simplified heavily.

## Rust Crates

| Crate | Purpose |
|-------|---------|
| `libloading` | dlopen/dlsym for compiled .so files |
| `cudarc` | CUDA driver API (cuModuleLoad, cuLaunchKernel, cuMemAlloc) |
| `tempfile` | Temp files for C source / compiled objects |
| `bytemuck` | Safe f32 ↔ bytes transmutes for data loading |

---

## Milestone 1: JIT Compilation Hello World

**Compiler concept:** The emit → compile → load → run loop that every JIT uses.

**Build:**
- Write a hardcoded C string to a temp file (element-wise add of two float arrays)
- Shell out to `clang -shared -O2 -o kernel.so kernel.c`
- Use `libloading` to dlopen and call the function pointer
- Verify `[1,2,3] + [4,5,6] = [5,7,9]`

**Tinygrad reference:** `tinygrad/runtime/support/compiler_cpu.py` (ClangJITCompiler), `tinygrad/runtime/ops_cpu.py` (CPUProgram)

**Test:** Assert output matches expected values.

---

## Milestone 2: DType + Buffer + Device Abstraction

**Compiler concept:** Type systems and hardware abstraction layers.

**Build:**
- `DType` enum: `F32`, `I32`, `Bool` -- each knows its byte size and C type name
- `Buffer` struct: wraps `Vec<u8>`, has `alloc(dtype, numel)`, `copyin(&[u8])`, `copyout() -> Vec<u8>`
- `Device` trait: `alloc`, `free`, `copyin`, `copyout`, `compile(src) -> Program`, `exec(program, bufs)`
- `CpuDevice` implementing the trait using M1's approach

**Tinygrad reference:** `tinygrad/dtype.py`, `tinygrad/device.py` (Buffer, Allocator, Compiled)

**Test:** Roundtrip f32 data through Buffer. Run the M1 add kernel through the device abstraction.

---

## Milestone 3: UOp IR -- The Core Graph

**Compiler concept:** Intermediate representations -- DAG-based IR and hash-consing (interning).

**Build:**
- `UOp` node: `{ op: Op, dtype: DType, srcs: Vec<UOpId>, arg: Arg }` with arena-based storage (`Vec<UOp>` + index newtype)
- `Op` enum (small subset):
  - Compute: `Add`, `Mul`, `Neg`, `Exp2`, `Log2`, `Sqrt`, `Reciprocal`, `Max`, `CmpLt`, `Where`
  - Memory: `Load`, `Store`, `Index`, `Const`
  - Structure: `Param`, `Range`, `End`, `Sink`
- Interning: `HashMap<(Op, DType, Vec<UOpId>, Arg), UOpId>` ensures identical subgraphs share nodes
- Builder API: `let idx = graph.range(n); let a = graph.load(buf_a, idx); ...`

**Tinygrad reference:** `tinygrad/uop/__init__.py` (Ops enum), `tinygrad/uop/ops.py` (UOp class, UOpMetaClass interning)

**Test:** Build an add graph by hand, verify topology and deduplication.

---

## Milestone 4: Code Generation -- IR to C

**Compiler concept:** Code emission / lowering -- turning abstract IR into concrete source code.

**Build:**
- `CRenderer` that walks a toposorted UOp list and emits C:
  - `Param` → `float* data0` (function arg)
  - `Range` → `for (int idx0 = 0; idx0 < N; idx0++) {`
  - `End` → `}`
  - `Index` → pointer arithmetic `(data0 + idx0)`
  - `Load` → `float val0 = *(data0 + idx0);`
  - `Add` → `float alu0 = (val0 + val1);`
  - `Store` → `*(data2 + idx0) = alu0;`
  - `Const` → literal like `3.14f`
- Wrap in `void kernel_name(float* data0, float* data1, ...) { ... }`

**Tinygrad reference:** `tinygrad/renderer/cstyle.py` (CStyleLanguage._render, ClangRenderer)

**Test:** Render an add graph to C, compile via M2 device, run, verify output. Also test mul and fused multiply-add.

---

## Milestone 5: Graph Rewriting

**Compiler concept:** Term rewriting -- pattern-match-and-replace as the universal compiler optimization strategy.

**Build:**
- `UPat` struct: a pattern that matches UOp nodes by op, dtype, and source patterns. Named captures let the replacement function access matched subtrees.
- `PatternMatcher`: holds a list of `(UPat, replacement_fn)` rules. Given a UOp, tries each rule and returns the first match. Dispatches by op for O(1) lookup.
- `graph_rewrite(root, pm)`: walks the graph bottom-up (children first, then parent), applying the PatternMatcher at each node until no more rules fire (fixed-point iteration).
- Starter rules to validate the machinery:
  - Constant folding: `add(const(a), const(b))` → `const(a + b)` (and mul, neg, etc.)
  - Algebraic identities: `x + 0` → `x`, `x * 1` → `x`, `x * 0` → `0`

**Key insight:** Tinygrad does nearly all optimization through graph rewrites -- constant folding, algebraic simplification, scheduling, movement op lowering, GPU-specific transforms. It's all `PatternMatcher` rules applied by `graph_rewrite`. Building this machinery once means M6-M10 just add rules instead of writing ad-hoc passes.

**Tinygrad reference:** `tinygrad/uop/ops.py` (PatternMatcher, UPat, graph_rewrite), `tinygrad/uop/symbolic.py` (algebraic rules)

**Test:** Build a graph with `x + 0`, rewrite it, verify the add is eliminated. Build `const(2) + const(3)`, verify it folds to `const(5)`. Verify fixed-point: `(x + 0) * 1` should simplify to just `x` in one pass.

---

## Milestone 6: Tensor API + Lazy Evaluation

**Compiler concept:** Lazy evaluation and kernel fusion -- why ML frameworks don't execute eagerly.

**Build:**
- `Tensor` struct: holds `UOpId` (lazy graph root), shape, dtype, device
- Tensor ops that build graph instead of executing: `add`, `mul`, `neg`, `relu` (`max(x, 0)`), `reshape`, `permute`, `expand`
- Movement ops change index computation, not data layout
- `realize()`: linearize graph → render to C → compile → execute
- Simple scheduler: one kernel per realize (all fused stores in one loop nest)

**Key insight:** `(a + b) * 2` lazily builds one graph. On realize, it becomes a *single* fused kernel instead of two separate ones with a memory roundtrip.

**Tinygrad reference:** `tinygrad/tensor.py` (Tensor class), `tinygrad/engine/schedule.py`

**Test:** `Tensor::from_vec(vec![1., 2., 3.]).add(Tensor::from_vec(vec![4., 5., 6.])).realize()` returns `[5., 7., 9.]`. Verify `(a + b) * 2` produces a single kernel.

---

## Milestone 7: Reductions + Matmul

**Compiler concept:** Loop nest generation for reductions, multi-dimensional index arithmetic.

**Build:**
- `ReduceAxis` op with `ADD` and `MAX` reduction types. Generates an inner loop + accumulator:
  ```c
  float acc = 0.0f;
  for (int ridx0 = 0; ridx0 < K; ridx0++) {
    acc += *(data0 + idx0*K + ridx0);
  }
  ```
- Multi-dimensional index computation from shape + strides
- `Tensor::matmul` via reshape + multiply + reduce: `(a[M,1,K] * b[1,N,K].permute).sum(axis=2)` -- no special matmul op needed
- `Tensor::sum(axis)` and `Tensor::max(axis)`

**Tinygrad reference:** `tinygrad/schedule/rangeify.py` (loop generation), tinygrad implements matmul the same way

**Test:** `ones([3,4]).matmul(ones([4,5]))` → `[3,5]` tensor of `4.0`. Test sum/max along various axes.

---

## Milestone 8: Autograd

**Compiler concept:** Automatic differentiation as graph transformation (the chain rule on a DAG).

**Build:**
- `backward()` on Tensor: walk UOp graph in reverse topological order, apply gradient rules per op, accumulate gradients
- Gradient rules (from tinygrad's `tinygrad/gradient.py` lines 48-79):
  - `Add`: grad passes through to both inputs
  - `Mul`: product rule -- `(y * grad, x * grad)`
  - `Reciprocal`: `-grad * (1/x)^2`
  - `Exp2`: `2^x * ln(2) * grad`
  - `Log2`: `grad / (x * ln(2))`
  - `Sqrt`: `grad / (2 * sqrt(x))`
  - `Max(x, 0)` (relu): `grad * (x > 0)`
  - `ReduceAxis(ADD)`: broadcast (expand) grad back to input shape
  - `Reshape/Permute/Expand`: apply inverse movement op
- `SGD` optimizer: `param -= lr * param.grad`

**Key insight:** Gradient rules produce *new UOps*. The backward pass builds more graph, which gets compiled and fused like any forward computation. No separate "backward kernel" concept.

**Tinygrad reference:** `tinygrad/gradient.py` (pm_gradient, compute_gradient)

**Test:** Compute `loss = (x * w + b).sum()`, backward, verify `w.grad` and `b.grad` numerically against finite differences `(f(x+eps) - f(x-eps)) / 2eps`.

---

## Milestone 9: CUDA Backend

**Compiler concept:** GPU programming model -- thread grids replace loops.

**Build:**
- `CudaDevice` implementing the `Device` trait: `cuMemAlloc`/`cuMemFree`/`cuMemcpyHtoD`/`cuMemcpyDtoH` via `cudarc`
- `CudaRenderer` that emits CUDA C instead of plain C. Differences from `CRenderer`:
  - `extern "C" __global__ void kernel(...)`
  - `int idx = blockIdx.x * blockDim.x + threadIdx.x;` replaces the outer for-loop
  - Grid-stride loop for large tensors
  - Reductions use atomics (simplest approach) or shared memory
- Compile via `nvcc --cubin` or `cudarc`'s built-in NVRTC
- Launch via `cuLaunchKernel` with grid/block dims from tensor shape

**Key insight:** What was a `for` loop on CPU becomes a thread grid on GPU. Same IR, different renderer. This is exactly how tinygrad works -- `ClangRenderer` and `CUDARenderer` share `CStyleLanguage`.

**Tinygrad reference:** `tinygrad/renderer/cstyle.py` (CUDARenderer), `tinygrad/runtime/ops_cuda.py`

**Test:** Run the same operations from M6/M7 on CUDA, verify results match CPU.

---

## Milestone 10: Train an MLP on MNIST

**Compiler concept:** End-to-end integration -- every layer of the stack exercised.

**Build:**
- Data loading: MNIST IDX format (16-byte header + raw u8 pixels), normalize to f32
- Model: `Linear(784, 128) → ReLU → Linear(128, 10)` where `Linear(x) = x.matmul(w) + b`
- Loss: cross-entropy = `(-log_softmax(logits) * one_hot(labels)).sum() / batch_size`
  - Softmax = `exp(x - x.max()) / exp(x - x.max()).sum()`
  - Requires: `exp2`, `log2`, `max`, `sum`, `sub` (all available)
- Training loop: forward → loss → backward → SGD step → repeat
- Batching: iterate dataset in batches of 64-128

**Test:** Loss decreases over epochs. ~95% accuracy after 5 epochs is the baseline for this architecture.

---

## Milestone Dependency Graph

```
M1 (JIT) → M2 (Device) → M3 (IR) → M4 (Codegen) → M5 (Rewrite) → M6 (Tensor) → M7 (Reduce/Matmul) → M8 (Autograd) → M10 (MNIST)
                                                                                                          ↘ M9 (CUDA) ↗
```

## Compiler Concepts Summary

| # | Milestone | Concept |
|---|-----------|---------|
| 1 | JIT Hello World | JIT compilation pipeline |
| 2 | DType + Buffer + Device | Type systems, hardware abstraction |
| 3 | UOp IR | DAG-based intermediate representation, interning |
| 4 | IR to C | Code emission / lowering |
| 5 | Graph Rewriting | Term rewriting, pattern matching, algebraic simplification |
| 6 | Tensor + Lazy Eval | Lazy evaluation, kernel fusion, scheduling |
| 7 | Reduce + Matmul | Loop nest generation, index arithmetic |
| 8 | Autograd | Reverse-mode AD as graph transformation |
| 9 | CUDA | GPU codegen (thread grids replace loops) |
| 10 | MNIST | Full integration |
