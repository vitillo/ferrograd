# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar25`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `CLAUDE.md` — repository context and design principles.
   - `src/tensor.rs` — tensor API, realize, gradient. Most changes flow through here.
   - `src/schedule.rs` and `src/schedule/` — scheduling and rangeify (lazy graph → kernel IR).
   - `src/codegen/clang.rs` — C code generation from kernel IR.
   - `src/device/cpu.rs` — the CPU backend (clang compile, dlopen, libffi execute).
   - `src/runtime.rs` — device state, buffer store, kernel cache.
   - `src/rewrite.rs` — symbolic simplification rules.
   - `src/gradient.rs` — reverse-mode autograd.
   - `examples/mnist.rs` — the MNIST training example (the benchmark).
4. **Read the reference implementation**: Read `~/projects/tinygrad` for how tinygrad implements its CPU backend and compiler optimizations. Key files:
   - `tinygrad/codegen/uopgraph.py` — UOp graph rewriting and optimization passes.
   - `tinygrad/codegen/lowerer.py` — lowering from schedule to kernel IR.
   - `tinygrad/renderer/cstyle.py` — C-style code generation.
   - `tinygrad/runtime/ops_cpu.py` — CPU backend (ClangJIT).
   - `tinygrad/engine/schedule.py` — kernel scheduling and fusion.
   These are read-only references. Do not modify anything in the tinygrad repo.
5. **Establish the baseline**: Build and run the benchmark as-is.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The goal

**Make ferrograd's MNIST training batch faster**, closing the 140x gap with optimized frameworks. This is an educational project — optimizations should be elegant, simple, and high-ROI. Don't add complexity that doesn't pay for itself handsomely. Always look at how tinygrad (`~/projects/tinygrad`) solves the same problem before implementing. The benchmark is:

```bash
MAX_BATCHES=5 cargo run --example mnist --release 2>&1
```

This runs 5 batches of MNIST training. Batch 0 is a warmup — it pays the cost of kernel compilation (clang invocations, dlopen). Batches 1–4 are steady-state: all kernels are cached, so they measure pure execution time. The evaluation step at epoch end compiles new kernels too, so ignore it.

**The metric is the steady-state batch time (batch 1 or later, NOT batch 0).** Lower is better. Track the ratio vs baseline to measure progress.

## What you CAN do

- Modify any `src/` files — the compiler pipeline is fair game: scheduling, rangeify, codegen, rewrite rules, runtime, device backend.
- Modify `examples/mnist.rs` only for benchmark instrumentation (timing), not to change the model architecture or training logic.
- Add new source files in `src/` if needed for new compiler passes or optimizations.

## What you CANNOT do

- Change the model architecture, loss function, or optimizer in `examples/mnist.rs`. The training logic is the ground truth.
- Add new crate dependencies (you can only use what's already in `Cargo.toml`).
- Break existing tests. Run `cargo test` to verify after each change.

## Known bottlenecks and opportunities

Based on analysis of the ferrograd compiler pipeline, the single-batch MNIST training is ~140x slower than optimized frameworks. Here are the main areas to investigate (roughly ordered by expected impact):

### 1. Naive matmul via reduce (HIGH impact)

Matmul is implemented as reshape → expand → element-wise multiply → reduce-sum. This generates a naive triple-nested loop with no tiling, no vectorization, and terrible cache locality. For [256,784] × [784,128], the inner loop scans 784 elements with stride-1 on one input and large stride on the other.

**What tinygrad does**: Pattern-matches matmul in the scheduler and emits optimized loop nests with tiling, shared memory (on GPU), and SIMD grouping.

**What to try**: Pattern-match the matmul reduction in codegen and emit a tiled loop nest. Even basic loop reordering (i,k,j instead of i,j,k) improves cache behavior dramatically. Adding tiling (e.g. 32×32 blocks) on top of that is another big win. The generated C code with `-O2` will auto-vectorize well-structured inner loops.

### 2. Kernel compilation overhead (HIGH impact on first batch)

Each unique kernel requires: write temp .c file → `clang -shared -O2` → dlopen → dlsym. The first batch compiles many kernels (forward + backward + update). Clang invocation is ~10-50ms per kernel.

**What tinygrad does**: Caches compiled kernels across runs (disk cache). Also batches multiple kernels into a single compilation unit.

**What to try**:
- Batch all kernels into a single .c file with multiple functions, compile once.
- Cache compiled .dylib files on disk (keyed by source hash) to skip clang on subsequent runs.
- Try `-O1` or `-O0` to see how much time is compilation vs execution (diagnostic only).

### 3. Too many separate kernels (MEDIUM-HIGH impact)

The scheduler materializes intermediate results at every multi-consumer node and nested reduction. This means a simple forward pass generates many small kernels instead of a few fused ones. Each kernel has overhead: schedule, codegen, compile (if uncached), allocate output buffer, dispatch via libffi.

**What tinygrad does**: Aggressive kernel fusion — multiple ops get merged into a single kernel when they form a fusible chain (element-wise ops, broadcasts, reshapes are free to fuse).

**What to try**:
- Count how many kernels are generated for one training batch (add logging).
- Identify which materializations are unnecessary and relax the scheduler's materialization rules.
- Fuse chains of element-wise ops (add, mul, neg, relu) into single kernels.

### 4. libffi dispatch overhead (LOW-MEDIUM impact)

Every kernel call goes through libffi's dynamic calling convention. This is flexible but adds overhead per call: building the `Cif`, marshaling arguments, indirect call. For many small kernels, this adds up.

**What to try**:
- Generate a fixed-signature wrapper that takes a `void**` args array, avoiding libffi entirely.
- Or generate the dispatch code as part of the compiled kernel (a main-like entry point).

### 5. Buffer allocation (LOW impact)

Every kernel output gets a fresh `Vec<u8>` allocation. No buffer reuse across operations.

**What to try**: A simple buffer pool that recycles freed buffers by size. For the MNIST batch, the same buffer sizes are allocated repeatedly.

### 6. Symbolic simplification (LOW impact, but check)

The rewrite pass runs fixed-point simplification on every kernel's index arithmetic. If the rules don't converge quickly, this could be slow for large kernels.

**What to try**: Profile the rewrite pass. If it's significant, add early-exit conditions or limit iteration count.

## Experimentation

Each experiment modifies code, rebuilds, and runs the benchmark. The benchmark itself should take under a few seconds.

**What you CAN do:**
- Modify any `src/` files.
- Add debug/timing instrumentation.

**What you CANNOT do:**
- Change the model or training logic.
- Add dependencies.
- Break tests.

**The first run**: Your very first run should always be to establish the baseline, so run the benchmark as-is.

## Output format

The benchmark prints per-batch timing like:

```
  epoch 0 batch    0/234  loss=2.3012  850.0ms   ← warmup (includes compilation)
  epoch 0 batch    1/234  loss=2.2991  142.5ms   ← steady-state
  epoch 0 batch    2/234  loss=2.2810  141.8ms   ← steady-state
  epoch 0 batch    3/234  loss=2.2654  142.1ms   ← steady-state
  epoch 0 batch    4/234  loss=2.2501  141.9ms   ← steady-state
```

Ignore batch 0 (compilation overhead). Use the average of batches 1–4 as the metric.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 4 columns:

```
commit	batch_ms	status	description
```

1. git commit hash (short, 7 chars)
2. single-batch time in milliseconds
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	batch_ms	status	description
a1b2c3d	142.5	keep	baseline
b2c3d4e	95.3	keep	batch kernel compilation
c3d4e5f	0	crash	tiled matmul (shader compile error)
d4e5f6g	42.1	keep	loop reorder i,k,j for matmul
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar25`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. **Consult tinygrad** (`~/projects/tinygrad`): before picking your next optimization, read the relevant tinygrad source to understand how they solve the same problem. This is not optional — every experiment should be informed by how tinygrad handles it. Use `python3.14` at `/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/bin/python3.14` with `DEBUG=4` to inspect tinygrad's behavior on the same workload when helpful.
3. Pick an optimization from the opportunities list (or your own ideas based on profiling/reading tinygrad).
4. Implement it by modifying the code.
4. Run `cargo test` to make sure nothing is broken. If tests fail, fix before proceeding.
5. git commit.
6. Run the benchmark: `MAX_BATCHES=5 cargo run --example mnist --release 2>&1 | tee bench.log`
7. Read the results: `grep "batch " bench.log`
8. If the grep output is empty or shows an error, the run crashed. Run `tail -n 30 bench.log` to read the error and attempt a fix.
9. Record the results in the tsv (NOTE: do not commit results.tsv, leave it untracked).
10. If the batch time improved (lower), you "advance" the branch, keeping the commit.
11. If it's equal or worse, git reset back to where you started.

**Timeout**: Each benchmark run should take under 30 seconds. If it hangs, kill it and treat as failure.

**Crashes**: If a run crashes (compiler error, wrong buffer size, etc.), use your judgment: easy fix → fix and re-run. Fundamentally broken → log as crash and move on.

**Tests**: Always run `cargo test` before benchmarking. A "faster" compiler that produces wrong results is worthless.

**Simplicity is paramount**: This is an educational project. Every optimization must earn its complexity. Prefer changes that are high-ROI: huge performance improvement with minimal, elegant code. A 10-line change that gives 5x speedup is far better than a 200-line change that gives 6x. If an optimization requires adding significant complexity, skip it and find a simpler approach. The code should remain readable and instructive after your changes.

**Order of attack**: Start with the highest-impact, lowest-risk changes first. Suggested order:
1. Diagnostic: count kernels and measure compile vs execute time
2. Batch kernel compilation (single clang invocation)
3. Loop reordering for matmul kernels
4. Kernel fusion for element-wise chains
5. Tiled matmul
6. Disk-based kernel cache

But use your judgment — if you see a quick win, take it. Always check tinygrad first to see how they approach the same problem.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, re-read tinygrad's compiler for new patterns, try combining previous changes, or try more aggressive optimizations. The loop runs until the human interrupts you, period.
