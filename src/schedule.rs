//! # Scheduling — turning lazy tensor graphs into executable kernels
//!
//! This module converts the lazy `UOp` graph built by [`crate::tensor`] into
//! kernel-level IR that the codegen can render to C. The pipeline mirrors
//! tinygrad's `schedule/` package.
//!
//! ## Pipeline
//!
//! ```text
//! Tensor ops (lazy UOp graph)
//!   │
//!   ▼
//! schedule           Create Sink(Store(Param, expr)) — the proto-kernel
//!   │
//!   ▼
//! rangeify           Add Range loops, push Index through the graph,
//!   │                convert ReduceAxis → Reduce → DefineAcc/Assign/End
//!   ▼
//! symbolic_simple    Simplify index arithmetic (x+0→x, x*1→x, const fold)
//!   │
//!   ▼
//! codegen            Render kernel-level UOps to C source
//! ```
//!
//! ## How rangeify works
//!
//! 1. Matches `Store(Param, expr)` and creates `Range` loops for each
//!    dimension of the expression's shape.
//! 2. Wraps the expression in `Index(expr, ranges...)` and pushes `Index`
//!    down through the graph until it reaches the leaf `Buffer` nodes.
//!    As it passes through each op, the index expressions are adjusted:
//!    movement ops (Reshape, Permute, Expand) transform the indices to
//!    account for how they rearrange data, while math ops (Add, Mul, etc.)
//!    are left unchanged. When Index reaches a Buffer, it becomes a `Load`
//!    with a flat memory offset computed from the index expressions.
//! 3. `Reduce` nodes are expanded into accumulator loops (`DefineAcc` +
//!    `Assign` + `End`), with `After` nodes to ensure correct ordering.
//!
//! ## Example: `a[2,3] + b[2,3]`
//!
//! **Input** (scheduled proto-kernel):
//! ```text
//! Sink(Store(Param(0), Add(Reshape([2,3], Buffer(6)), Reshape([2,3], Buffer(6)))))
//! ```
//!
//! **After rangeify** — Store matched, ranges created, Index pushed to buffers:
//! ```text
//! for i in 0..2:
//!   for j in 0..3:
//!     val0 = Load(a, i*3+j)
//!     val1 = Load(b, i*3+j)
//!     Store(out, i*3+j, val0 + val1)
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
//! - **Buffer → Param in scheduler**: currently rangeify converts Buffers to
//!   Params during Index pushing. Ideally the scheduler would do this
//!   (matching tinygrad), but it requires solving how shape info flows
//!   when flat Buffers wrapped in Reshape become Params.
//!
//! ## Submodules
//!
//! - [`indexing`] — core index transformation rules and helpers
//! - [`rangeify`] — rewrite rules that add ranges and lower to kernel IR

pub mod indexing;
pub mod rangeify;

use std::collections::HashMap;
use std::rc::Rc;

use crate::device::Buffer;
use crate::rewrite::{graph_rewrite, Captures, PatternMatcher, RewriteFn, UPat};
use crate::uop::{Arg, Op, UOp};

/// Schedule a tensor expression for execution as a single kernel.
///
/// Converts all `Buffer` nodes to `Param` nodes (slot 0 = output, 1+ = inputs),
/// collects the input buffers, and wraps in `Store`/`Sink`.
/// This matches tinygrad where the scheduled AST has PARAM nodes for all buffers.
///
/// Returns `(scheduled_sink, input_buffers)`.
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

    // Replace Buffer → Param via graph_rewrite.
    let pm = PatternMatcher::new(vec![(
        UPat {
            op: Some(vec![Op::Buffer]),
            name: Some("buf".into()),
            arg: None,
            src: None,
            commutative: false,
        },
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
