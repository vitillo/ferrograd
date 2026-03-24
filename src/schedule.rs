//! # Scheduling — turning lazy tensor graphs into executable kernels
//!
//! This module converts the lazy `UOp` graph built by [`crate::tensor`] into
//! kernel-level IR that the codegen can render to C. The pipeline mirrors
//! tinygrad's `schedule/` package.
//!
//! ## Why scheduling exists
//!
//! The tensor API builds a lazy graph of high-level operations (Add, Reshape,
//! `ReduceAxis`, etc.). These ops describe *what* to compute, but not *how* to
//! iterate over memory. Scheduling bridges that gap:
//!
//! 1. **`schedule()`** (this file) — replaces device-level `Buffer` nodes with
//!    abstract `Param` slots and wraps everything in `Store`/`Sink`. This
//!    separates "what data lives where" from "what computation to do", which
//!    is essential because the same buffer might be shared across expressions.
//!
//! 2. **`rangeify`** — creates loop nests (`Range`/`End`) and pushes `Index`
//!    nodes down through the graph. Movement ops (Reshape, Permute, Expand)
//!    transform the index expressions as they pass through, then disappear.
//!    When Index reaches a Param leaf, it becomes a `Load` with a flat offset.
//!    Reductions get their own inner loops with accumulator patterns.
//!
//! 3. **`symbolic_simple`** (in [`crate::rewrite`]) — cleans up redundant
//!    arithmetic (`x+0`, `x*1`, constant folding) left over from index
//!    generation.
//!
//! After these steps, the graph contains only kernel-level ops that map
//! directly to C code: Range/End (loops), Load/Store (memory), and ALU ops.
//!
//! ## Pipeline
//!
//! ```text
//! Tensor ops (lazy UOp graph)
//!   │  Buffers, Reshape, Permute, Expand, ReduceAxis, ALU ops
//!   ▼
//! schedule()         Buffer → Param, wrap in Store/Sink
//!   │  Params, Reshape, Permute, Expand, ReduceAxis, ALU ops
//!   ▼
//! rangeify()         Add Range loops, push Index down, expand Reduce
//!   │  Range, End, Load, Store, DefineAcc, Assign, After, ALU ops
//!   ▼
//! symbolic_simple()  x+0→x, x*1→x, const fold
//!   │  (same ops, simplified index arithmetic)
//!   ▼
//! codegen            Render to C source
//! ```
//!
//! ## Example: `a[2,3] + b[2,3]`
//!
//! **After `schedule()`** — Buffers replaced with Params, wrapped in Store/Sink:
//! ```text
//! Sink(Store(Param(0), Add(Reshape([2,3], Param(1)), Reshape([2,3], Param(2)))))
//! ```
//!
//! **After `rangeify()`** — loops created, Index pushed down to Loads:
//! ```text
//! for i in 0..2:
//!   for j in 0..3:
//!     val0 = Load(Param(1), i*3+j)
//!     val1 = Load(Param(2), i*3+j)
//!     Store(Param(0), i*3+j, val0 + val1)
//! ```
//!
//! ## What's not here yet
//!
//! - **Fusion decisions**: currently every `realize()` produces exactly one
//!   kernel. For training loops with multiple outputs, we'll need a realize
//!   map that decides fusion boundaries.
//! - **Multi-consumer range merging**: when one op feeds two consumers that
//!   need different ranges.
//! - **Buffer cost analysis**: deciding which intermediates to materialize
//!   vs. recompute.
//!
//! ## Submodules
//!
//! - [`indexing`] — core index transformation rules (see module docs for details)
//! - [`rangeify`] — orchestrates all rewrite rules into a single pass

pub mod indexing;
pub mod rangeify;

use std::collections::HashMap;
use std::rc::Rc;

use crate::device::Buffer;
use crate::rewrite::{graph_rewrite, Captures, PatternMatcher, RewriteFn, UPat};
use crate::uop::{Arg, Op, UOp};

/// Schedule a tensor expression for execution as a single kernel.
///
/// This is the first step of lowering: it separates the computation graph
/// from the concrete device buffers. Every `Buffer` node (which holds an
/// `Rc<Buffer>` pointing to actual device memory) is replaced with a `Param`
/// node (which just has a slot number). This way, rangeify and codegen work
/// with abstract parameters, and the runtime binds actual buffers at launch.
///
/// The same `Buffer` identity (by `Rc` pointer) always maps to the same
/// `Param` slot, so `a + a` correctly shares a single input parameter.
///
/// Returns `(scheduled_sink, input_buffers)` where slot 0 is the output
/// and slots 1+ correspond to the collected input buffers.
///
/// # Panics
///
/// Panics if buffer extraction fails (internal error).
#[must_use]
pub fn schedule(expr: &UOp) -> (UOp, Vec<Buffer>) {
    use std::cell::RefCell;

    let input_bufs: Rc<RefCell<Vec<Buffer>>> = Rc::new(RefCell::new(Vec::new()));
    let buf_params: Rc<RefCell<HashMap<usize, UOp>>> = Rc::new(RefCell::new(HashMap::new()));

    let bufs = input_bufs.clone();
    let params = buf_params.clone();

    let pm = PatternMatcher::new(vec![(
        UPat::named(Op::Buffer, "buf"),
        Box::new(move |caps: &Captures| {
            let buf_uop = caps.get("buf");
            let Arg::Buffer(ref rc) = buf_uop.arg() else { return None };
            let ptr = Rc::as_ptr(rc) as usize;
            let dtype = buf_uop.dtype();
            let numel = rc.numel();
            let mut params_map = params.borrow_mut();
            Some(
                params_map
                    .entry(ptr)
                    .or_insert_with(|| {
                        let mut bufs_vec = bufs.borrow_mut();
                        let slot = bufs_vec.len() + 1;
                        bufs_vec.push(Buffer::clone(rc));
                        UOp::param(slot, dtype, numel)
                    })
                    .clone(),
            )
        }) as RewriteFn,
    )]);

    let parameterized = {
        let result = graph_rewrite(expr, &pm, "schedule");
        drop(pm);
        result
    };

    let out_numel = parameterized.shape().map_or(1, |s| s.iter().product());
    let out_param = UOp::param(0, expr.dtype(), out_numel);
    let store = UOp::store(out_param, parameterized);
    let sink = UOp::sink(vec![store]);

    let bufs = Rc::try_unwrap(input_bufs).unwrap().into_inner();
    (sink, bufs)
}
