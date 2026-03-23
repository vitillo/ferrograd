//! # Scheduling — lower lazy tensor graphs to kernel-level `UOp`s
//!
//! Copies the tensor subgraph, then uses `graph_rewrite` to transform
//! `Buffer` → `Param + Index + Load`. ALU ops pass through with sources
//! auto-remapped by the rewrite engine.

use std::cell::RefCell;
use std::rc::Rc;

use crate::device::Buffer;
use crate::rewrite::{Captures, PatternMatcher, RewriteFn, UPat, graph_rewrite};
use crate::uop::{Arg, Op, UOp};

/// Lower a tensor-level `UOp` into a kernel-level `UOp` (Sink).
///
/// Returns `(sink, input_buffers)`.
///
/// # Panics
///
/// Panics if a `Buffer` node doesn't have `Arg::Buffer` data.
#[must_use]
pub fn lower_to_kernel(root: &UOp, numel: usize) -> (UOp, Vec<Buffer>) {
    // Kernel scaffolding.
    let root_dtype = root.dtype();
    let out_param = UOp::param(0, root_dtype);
    #[allow(clippy::cast_possible_wrap)]
    let n = UOp::const_int(numel as i64, crate::dtype::DType::I32);
    let idx = UOp::range(0, n);

    // Rewrite Buffer → Param + Index + Load.
    let input_bufs: Rc<RefCell<Vec<Buffer>>> = Rc::new(RefCell::new(Vec::new()));
    let lowered_expr = {
        let pm = build_lowering_rules(&idx, &input_bufs);
        graph_rewrite(root, &pm)
    };

    // Wrap in Store + End + Sink.
    let out_idx = UOp::index(out_param, idx.clone());
    let store = UOp::store(out_idx, lowered_expr);
    let end = UOp::end(idx);
    let sink = UOp::sink(vec![store, end]);

    let bufs = Rc::try_unwrap(input_bufs)
        .expect("pm dropped, no other refs")
        .into_inner();
    (sink, bufs)
}

fn build_lowering_rules(idx: &UOp, input_bufs: &Rc<RefCell<Vec<Buffer>>>) -> PatternMatcher {
    let bufs = input_bufs.clone();
    let idx = idx.clone();
    PatternMatcher::new(vec![(
        UPat {
            op: Some(vec![Op::Buffer]),
            name: Some("buf".into()),
            arg: None,
            src: None,
            commutative: false,
        },
        Box::new(move |caps: &Captures| {
            let buf_uop = caps.get("buf");
            let dtype = buf_uop.dtype();
            let buf = match buf_uop.arg() {
                Arg::Buffer(rc) => Buffer::clone(rc),
                _ => panic!("Buffer UOp must have Arg::Buffer"),
            };
            let mut bufs = bufs.borrow_mut();
            let slot = bufs.len() + 1;
            bufs.push(buf);
            let param = UOp::param(slot, dtype);
            let index = UOp::index(param, idx.clone());
            Some(UOp::load(index, dtype))
        }) as RewriteFn,
    )])
}
