//! # Scheduling — turning lazy tensor graphs into executable kernels
//!
//! This module converts the lazy `UOp` graph built by [`crate::tensor`] into
//! a list of kernel-level IR items ready for codegen + execution.
//!
//! ## Why scheduling exists
//!
//! The tensor API builds a lazy graph of high-level operations (Add, Reshape,
//! `ReduceAxis`, etc.). These ops describe *what* to compute, but not *how* to
//! iterate over memory. Scheduling bridges that gap:
//!
//! 1. **Splitting**: identifies where the graph must be cut into separate
//!    kernels (when a `ReduceAxis` feeds another `ReduceAxis`, the inner
//!    one must be materialized to a buffer).
//! 2. **Parameterization**: replaces device-level `Buffer` nodes with abstract
//!    `Param` slots, separating data placement from computation.
//! 3. **Wrapping**: wraps each subgraph in `Store`/`Sink` — the proto-kernel.
//!
//! The result is a `Vec<ScheduleItem>` in dependency order: each item is a
//! single kernel that reads from Buffers and writes one output. The executor
//! compiles and runs them sequentially.
//!
//! ## Pipeline
//!
//! ```text
//! Tensor ops (lazy UOp graph)
//!   │  Buffers, Reshape, Permute, Expand, ReduceAxis, ALU ops
//!   ▼
//! schedule()         Split at reduction boundaries, Buffer → Param, Store/Sink
//!   │  Returns Vec<ScheduleItem> in execution order
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
//! ## Submodules
//!
//! - [`indexing`] — core index transformation rules (see module docs for details)
//! - [`rangeify`] — orchestrates all rewrite rules into a single pass

pub mod indexing;
pub mod rangeify;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::device::{Buffer, Device};
use crate::dtype::DType;
use crate::rewrite::graph_rewrite;
use crate::uop::{Arg, Op, UOp};

/// A single kernel to compile and execute.
pub struct ScheduleItem {
    /// Kernel-ready `UOp` graph (`Sink(Store(Param, expr))`).
    pub sink: UOp,
    /// Input buffers (slot 1+), shared via `Rc` so intermediate outputs
    /// from earlier kernels are visible to later ones.
    pub input_bufs: Vec<Rc<Buffer>>,
    /// Pre-allocated output buffer. For intermediate kernels, downstream
    /// kernels already hold `Rc` clones of this buffer in their `input_bufs`.
    /// The executor writes into this buffer so data flows automatically.
    pub out_buf: Rc<Buffer>,
    /// Output shape.
    pub out_shape: Vec<usize>,
    /// Output element type.
    pub out_dtype: DType,
}

/// Analyze a lazy `UOp` graph and produce kernels in dependency order.
///
/// Automatically splits at `ReduceAxis` boundaries when a reduction feeds
/// into another reduction (directly or through element-wise ops). Each split
/// produces an intermediate `ScheduleItem` whose output buffer becomes an
/// input to subsequent kernels.
///
/// This is a pure graph transformation — no Device, no compilation.
///
/// # Panics
///
/// Panics if buffer extraction or shape inference fails.
#[must_use]
pub fn schedule(expr: &UOp, device: &dyn Device) -> Vec<ScheduleItem> {
    let mut items = Vec::new();
    let mut uop = expr.clone();

    // Materialize inner reductions that feed outer reductions.
    loop {
        let Some(cut) = find_cut_point(&uop) else {
            break;
        };

        let item = parameterize(&cut, device);

        // Replace the cut subtree with a Buffer pointing to item.out_buf.
        // Later kernels' parameterize will pick up this Rc<Buffer> as an input.
        let buf_uop = UOp::new(Op::Buffer, item.out_dtype, vec![], Arg::Buffer(item.out_buf.clone()));
        let replacement = UOp::new(Op::Reshape, item.out_dtype, vec![buf_uop], Arg::Dims(item.out_shape.clone()));
        uop = substitute_uop(&uop, &cut, &replacement);

        items.push(item);
    }

    // Final kernel.
    items.push(parameterize(&uop, device));

    items
}

/// Convert a subgraph to a parameterized kernel: Buffer→Param, Store/Sink.
fn parameterize(expr: &UOp, device: &dyn Device) -> ScheduleItem {
    use std::cell::RefCell;

    let input_bufs: Rc<RefCell<Vec<Rc<Buffer>>>> = Rc::new(RefCell::new(Vec::new()));
    let buf_params: Rc<RefCell<HashMap<usize, UOp>>> = Rc::new(RefCell::new(HashMap::new()));

    let bufs = input_bufs.clone();
    let params = buf_params.clone();

    let rewrite_buf = move |node: &UOp| -> Option<UOp> {
        if node.op() != Op::Buffer {
            return None;
        }
        let Arg::Buffer(ref rc) = node.arg() else {
            return None;
        };
        let ptr = Rc::as_ptr(rc) as usize;
        let dtype = node.dtype();
        let numel = rc.numel();
        let mut params_map = params.borrow_mut();
        Some(
            params_map
                .entry(ptr)
                .or_insert_with(|| {
                    let mut bufs_vec = bufs.borrow_mut();
                    let slot = bufs_vec.len() + 1;
                    bufs_vec.push(rc.clone());
                    UOp::param(slot, dtype, numel)
                })
                .clone(),
        )
    };

    let parameterized = graph_rewrite(expr, &rewrite_buf, "schedule");
    drop(rewrite_buf);

    let out_shape = parameterized.shape().unwrap_or_else(|| vec![1]);
    let out_numel: usize = out_shape.iter().product();
    let out_dtype = expr.dtype();
    let out_param = UOp::param(0, out_dtype, out_numel);
    let store = UOp::store(out_param, parameterized);
    let sink = UOp::sink(vec![store]);

    let bufs = Rc::try_unwrap(input_bufs).unwrap().into_inner();
    let out_buf = Rc::new(device.allocate(out_dtype, out_numel));
    ScheduleItem {
        sink,
        input_bufs: bufs,
        out_buf,
        out_shape,
        out_dtype,
    }
}

// ── Multi-kernel splitting ──────────────────────────────────────────────

/// Find the innermost `ReduceAxis` that has a `ReduceAxis` ancestor.
///
/// This identifies where to split: the inner reduction must be materialized
/// to a buffer so the outer reduction can randomly access its results.
/// Returns `None` if no nested reductions exist (single kernel suffices).
fn find_cut_point(root: &UOp) -> Option<UOp> {
    let order = root.toposort();

    // Mark nodes whose output eventually feeds into a ReduceAxis.
    let mut feeds_reduce: HashSet<UOp> = HashSet::new();
    for node in order.iter().rev() {
        if node.op() == Op::ReduceAxis || feeds_reduce.contains(node) {
            for src in node.srcs() {
                feeds_reduce.insert(src.clone());
            }
        }
    }

    // Walk bottom-up: the first ReduceAxis whose output feeds another
    // ReduceAxis is the cut point.
    for node in &order {
        if node.op() == Op::ReduceAxis && feeds_reduce.contains(node) {
            return Some(node.clone());
        }
    }
    None
}

/// Replace all occurrences of `old` with `new` in the graph rooted at `root`.
fn substitute_uop(root: &UOp, old: &UOp, new: &UOp) -> UOp {
    let order = root.toposort();
    let mut replace: HashMap<UOp, UOp> = HashMap::new();
    replace.insert(old.clone(), new.clone());

    for node in &order {
        if replace.contains_key(node) {
            continue;
        }
        let new_srcs: Vec<UOp> = node
            .srcs()
            .iter()
            .map(|s| replace.get(s).cloned().unwrap_or_else(|| s.clone()))
            .collect();
        let changed = node.srcs().iter().zip(&new_srcs).any(|(o, n)| o != n);
        if changed {
            replace.insert(
                node.clone(),
                UOp::new(node.op(), node.dtype(), new_srcs, node.arg().clone()),
            );
        }
    }

    replace.get(root).cloned().unwrap_or_else(|| root.clone())
}
