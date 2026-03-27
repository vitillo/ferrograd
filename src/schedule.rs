//! # Scheduling — turning lazy tensor graphs into executable kernels
//!
//! This module converts the lazy tensor-level graph into a list of proto-kernel
//! graphs, each with explicit input parameters and a reserved output buffer.
//!
//! ## Why scheduling exists
//!
//! Lazy tensor operations build up an unbounded DAG of math. We can't compile
//! the whole thing as one kernel — shared subexpressions and chained reductions
//! need separate kernels with intermediate buffers. Scheduling decides *where*
//! to cut the graph by choosing which nodes to **materialize** (write to a
//! device buffer), then extracts each resulting subgraph into a self-contained
//! kernel with numbered parameter slots.
//!
//! This mirrors tinygrad's `schedule.py`: walk the lazy graph in topological
//! order, materialize nodes that are multi-consumer or chained reductions, and
//! parameterize each kernel subgraph so it's ready for lowering.

pub mod indexing;
pub mod rangeify;

use std::collections::{HashMap, HashSet};

use crate::device::{self, Buffer};
use crate::dtype::DType;
use crate::rewrite::graph_rewrite;
use crate::shape::Shape;
use crate::uop::{self, Arg, Op, UOp};

#[derive(Debug, Clone)]
/// A runtime input to a compiled kernel.
pub enum KernelInput {
    /// A realized tensor buffer referenced by a shared slot handle.
    Buffer(Buffer),
    /// A scalar `i32` input.
    I32(i32),
    /// A scalar `f32` input.
    F32(f32),
    /// A scalar boolean input.
    Bool(bool),
}

/// A single kernel to compile and execute.
#[derive(Debug)]
pub struct ScheduleItem {
    /// Kernel-ready `Sink(Store(Param(0), expr))`.
    pub sink: UOp,
    /// Runtime inputs in parameter-slot order, excluding output slot 0.
    pub inputs: Vec<KernelInput>,
    /// Output buffer slot reserved for slot 0, or `None` for in-place stores.
    pub output_buffer: Option<Buffer>,
    /// Output shape for allocated outputs.
    pub out_shape: Option<Shape>,
}

/// A full execution plan for realizing one or more lazy roots.
#[derive(Debug)]
pub struct SchedulePlan {
    /// Kernels to execute in dependency order.
    pub items: Vec<ScheduleItem>,
    /// Replacements to apply to live tensor graphs after execution.
    pub replacements: HashMap<UOp, UOp>,
}

/// Analyze a lazy `UOp` graph and produce kernels in dependency order.
#[must_use]
pub fn schedule(expr: &UOp) -> Vec<ScheduleItem> {
    schedule_many(std::slice::from_ref(expr)).items
}

/// Analyze several lazy roots together and produce one shared execution plan.
///
/// # Panics
///
/// Panics if `exprs` is empty.
#[must_use]
pub fn schedule_many(exprs: &[UOp]) -> SchedulePlan {
    assert!(
        !exprs.is_empty(),
        "schedule_many requires at least one root"
    );

    let union = UOp::sink(exprs.to_vec());
    let order = union.toposort();
    let consumer_map = uop::build_consumer_map(&order);
    let nested_reduce_inputs = nested_reduce_inputs(&order);
    let roots: HashSet<UOp> = exprs.iter().cloned().collect();

    let mut replacements: HashMap<UOp, UOp> = HashMap::new();
    let mut items = Vec::new();

    // Phase 1: materialize intermediates.
    //
    // Walk the graph in dependency order and create kernels for internal nodes
    // that need their own buffer (multi-consumer, chained reductions, etc.).
    // Roots are excluded here — they're handled in phase 2.
    //
    // Each materialized node gets a replacement entry mapping the original UOp
    // to a Buffer UOp, so downstream nodes (and the roots) will see realized
    // buffers instead of the original lazy subgraph.
    for node in &order {
        if !should_materialize(node, &roots, &consumer_map, &nested_reduce_inputs) {
            continue;
        }

        let kernel_expr = substitute_materialized(node, &replacements);
        let item = parameterize(&kernel_expr);
        let replacement = buffer_replacement(&item);
        replacements.insert(node.clone(), replacement);
        items.push(item);
    }

    // Phase 2: compile the requested roots.
    //
    // This runs *after* all intermediates are materialized, so
    // `substitute_materialized` can replace them with buffer references —
    // the root's kernel only contains its own math, not the intermediates'.
    //
    // Roots need separate handling because:
    // - They always produce a kernel (intermediates use heuristics).
    // - They may be assignments (`After(value, Store(dest, expr))`), which
    //   write into an existing buffer instead of allocating a new one.
    for expr in exprs {
        let final_expr = substitute_materialized(expr, &replacements);
        if let Some((store, replacement)) = assignment_effect(&final_expr) {
            items.push(parameterize_store(&store));
            replacements.insert(expr.clone(), replacement);
        } else {
            let item = parameterize(&final_expr);
            let replacement = buffer_replacement(&item);
            replacements.insert(expr.clone(), replacement);
            items.push(item);
        }
    }
    SchedulePlan {
        items,
        replacements,
    }
}

/// Turn a lazy expression into a kernel that allocates a fresh output buffer.
/// Wraps the expression in `Sink(Store(Param(0), expr))` and assigns numbered
/// parameter slots to every buffer and scalar input.
fn parameterize(expr: &UOp) -> ScheduleItem {
    let backend = device::get(expr.device());
    let (parameterized, inputs) = parameterize_inputs(expr, 1);
    let out_shape = parameterized.shape().unwrap_or_else(|| Shape::flat(1));
    let output_buffer = backend.reserve_buffer(expr.dtype(), out_shape.numel());
    let out_param = UOp::param_buffer(0, expr.dtype(), out_shape.numel(), expr.device());
    let store = UOp::new(
        Op::Store,
        DType::Void,
        vec![out_param, parameterized],
        Arg::None,
    );
    let sink = UOp::sink(vec![store]);

    ScheduleItem {
        sink,
        inputs,
        output_buffer: Some(output_buffer),
        out_shape: Some(out_shape),
    }
}

/// Like [`parameterize`], but for in-place stores (e.g. assignment into an
/// existing buffer). No output buffer is allocated — the destination is already
/// present in the graph as a regular buffer input at slot 0.
fn parameterize_store(store: &UOp) -> ScheduleItem {
    assert_eq!(
        store.op(),
        Op::Store,
        "parameterize_store requires a Store root"
    );
    let (parameterized_store, inputs) = parameterize_inputs(store, 0);
    let sink = UOp::sink(vec![parameterized_store]);
    ScheduleItem {
        sink,
        inputs,
        output_buffer: None,
        out_shape: None,
    }
}

/// Walk `root` and replace every `Buffer`, `Bind`, and `Shrink` offset with
/// numbered `Param` nodes, collecting the corresponding runtime inputs.
/// `slot_offset` reserves slot 0 for the output in allocating kernels (1) or
/// starts at 0 for in-place stores where the destination is a regular input.
fn parameterize_inputs(root: &UOp, slot_offset: usize) -> (UOp, Vec<KernelInput>) {
    let device = root.device();
    let mut inputs: Vec<KernelInput> = Vec::new();
    let mut params: HashMap<Buffer, UOp> = HashMap::new();
    let mut scalar_params: HashMap<UOp, UOp> = HashMap::new();

    // Allocating kernels reserve slot 0 for their output buffer; in-place store
    // kernels start their inputs at slot 0 because the destination is already in
    // the graph as a regular buffer input.
    let mut rewrite_inputs = |node: &UOp| -> Option<UOp> {
        match node.op() {
            Op::Buffer => {
                let Arg::Buffer(handle) = node.arg() else {
                    return None;
                };
                let dtype = node.dtype();
                Some(
                    params
                        .entry(handle.clone())
                        .or_insert_with(|| {
                            let slot = inputs.len() + slot_offset;
                            inputs.push(KernelInput::Buffer(handle.clone()));
                            UOp::param_buffer(slot, dtype, handle.numel(), device)
                        })
                        .clone(),
                )
            }
            Op::Bind => {
                let variable = node.srcs()[0].clone();
                let literal = scalar_input(node.srcs()[1].arg());
                Some(
                    scalar_params
                        .entry(variable)
                        .or_insert_with(|| {
                            let slot = inputs.len() + slot_offset;
                            inputs.push(literal.clone());
                            UOp::param_scalar(slot, node.dtype(), device)
                        })
                        .clone(),
                )
            }
            Op::Shrink => {
                let Arg::Bounds(lengths) = node.arg() else {
                    return None;
                };
                let src_shape = node.srcs()[0].shape()?;

                let mut changed = false;
                let mut new_srcs = vec![node.srcs()[0].clone()];
                for ((axis, start), &len) in node.srcs()[1..].iter().enumerate().zip(lengths.iter())
                {
                    if len != src_shape[axis] && start.op() == Op::Const {
                        let slot = inputs.len() + slot_offset;
                        inputs.push(scalar_input(start.arg()));
                        let param = UOp::param_scalar(slot, start.dtype(), device);
                        new_srcs.push(param);
                        changed = true;
                        continue;
                    }
                    new_srcs.push(start.clone());
                }

                changed.then(|| UOp::new(Op::Shrink, node.dtype(), new_srcs, node.arg().clone()))
            }
            _ => None,
        }
    };

    (graph_rewrite(root, &mut rewrite_inputs), inputs)
}
/// Convert a `Const` literal arg into the corresponding [`KernelInput`] scalar variant.
fn scalar_input(literal: &Arg) -> KernelInput {
    match literal {
        #[allow(clippy::cast_possible_truncation)]
        Arg::Float(value) => KernelInput::F32(*value as f32),
        Arg::Int(value) => {
            KernelInput::I32(i32::try_from(*value).expect("kernel scalar int must fit in i32"))
        }
        Arg::Bool(value) => KernelInput::Bool(*value),
        _ => panic!("kernel scalar input must be a literal"),
    }
}

/// Identify nodes that feed into a reduction which itself feeds another
/// reduction (e.g. `sum(sum(x))`). Chained reductions must be materialized
/// between stages because a single kernel can't nest two independent
/// accumulator loops — the inner reduce's output must be written to a buffer
/// before the outer reduce reads it.
fn nested_reduce_inputs(order: &[UOp]) -> HashSet<UOp> {
    let mut feeds_reduce = HashSet::new();
    for node in order.iter().rev() {
        if node.op() == Op::ReduceAxis || feeds_reduce.contains(node) {
            for src in node.srcs() {
                feeds_reduce.insert(src.clone());
            }
        }
    }
    feeds_reduce
}

/// Decide whether `node` needs its own kernel with a materialized output buffer.
///
/// A node is materialized when fusing it into its consumers' kernels would be
/// incorrect or wasteful:
/// - **Multi-consumer**: recomputing would duplicate work across kernels.
/// - **Reduce consumed by ALU**: the reduce result must be stored before the
///   elementwise op can read it (tinygrad splits these the same way).
/// - **Chained reductions**: a reduce whose output feeds another reduce needs
///   an intermediate buffer (see [`nested_reduce_inputs`]).
///
/// Nodes that are already leaf-like (Buffer, Const, Bind) or structural
/// (Sink, Store) are never materialized — they're either inputs or handled
/// by the kernel extraction logic directly. Movement ops are also never
/// materialized: they are views, so forcing them into their own kernel just
/// copies data instead of letting consumers inline the indexing math.
fn should_materialize(
    node: &UOp,
    roots: &HashSet<UOp>,
    consumer_map: &HashMap<UOp, Vec<UOp>>,
    nested_reduce_inputs: &HashSet<UOp>,
) -> bool {
    if roots.contains(node) || node.shape().is_none() {
        return false;
    }
    if matches!(
        node.op(),
        Op::Sink
            | Op::Store
            | Op::After
            | Op::Assign
            | Op::DefineAcc
            | Op::Buffer
            | Op::Const
            | Op::DefineVar
            | Op::Bind
    ) {
        return false;
    }
    if node.op().is_movement() {
        return false;
    }
    if node.has_buffer_identity() {
        return false;
    }
    if consumer_map
        .get(node)
        .is_some_and(|consumers| consumers.len() > 1)
    {
        return true;
    }
    if node.op() == Op::ReduceAxis
        && consumer_map
            .get(node)
            .is_some_and(|consumers| consumers.iter().any(|consumer| consumer.op().is_alu()))
    {
        return true;
    }
    node.op() == Op::ReduceAxis && nested_reduce_inputs.contains(node)
}

/// Create the `Buffer` node that replaces a materialized expression in the
/// lazy graph. Downstream consumers will see this buffer instead of the
/// original computation, creating the kernel boundary.
fn buffer_replacement(item: &ScheduleItem) -> UOp {
    let output_buffer = item
        .output_buffer
        .clone()
        .expect("buffer_replacement requires an allocated output");
    let out_shape = item
        .out_shape
        .clone()
        .expect("buffer_replacement requires an allocated output shape");
    let buffer = UOp::buffer(
        output_buffer.clone(),
        output_buffer.dtype(),
        item.sink.device(),
    );
    UOp::reshape(buffer, out_shape)
}

/// Check if `root` is an `After(value, Store)` — an in-place assignment that
/// writes into an existing buffer as a side effect. Returns `(store, value)`
/// so the caller can schedule the store separately and use `value` as the
/// replacement in the lazy graph.
fn assignment_effect(root: &UOp) -> Option<(UOp, UOp)> {
    if root.op() != Op::After || root.srcs().len() != 2 {
        return None;
    }
    let store = root.srcs()[1].clone();
    (store.op() == Op::Store).then(|| (store, root.srcs()[0].clone()))
}

/// Rewrite `root` by swapping every materialized subexpression with its
/// replacement buffer. This isolates the subgraph for the current kernel,
/// cutting it off at the buffer boundaries established by earlier scheduling
/// decisions.
fn substitute_materialized(root: &UOp, replacements: &HashMap<UOp, UOp>) -> UOp {
    let order = root.toposort();
    let mut substituted: HashMap<UOp, UOp> = HashMap::new();

    for node in &order {
        if node != root {
            if let Some(replacement) = replacements.get(node) {
                substituted.insert(node.clone(), replacement.clone());
                continue;
            }
        }

        let new_srcs: Vec<UOp> = node
            .srcs()
            .iter()
            .map(|src| substituted.get(src).cloned().unwrap_or_else(|| src.clone()))
            .collect();
        let changed = node
            .srcs()
            .iter()
            .zip(&new_srcs)
            .any(|(old, new)| old != new);
        let rewritten = if changed {
            UOp::new(node.op(), node.dtype(), new_srcs, node.arg().clone())
        } else {
            node.clone()
        };
        substituted.insert(node.clone(), rewritten);
    }

    substituted
        .get(root)
        .cloned()
        .unwrap_or_else(|| root.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{self, DeviceId};

    fn buffer_uop(data: &[f32], shape: &[usize]) -> UOp {
        let device = DeviceId::Cpu;
        let backend = device::get(device);
        let buffer = backend.allocate(DType::F32, data.len());
        backend.copy_from_host(&buffer, bytemuck::cast_slice(data));
        let buffer = UOp::buffer(buffer, DType::F32, device);
        UOp::reshape(buffer, Shape::from(shape))
    }

    #[test]
    fn test_nested_reductions_materialize_via_shared_buffer_handles() {
        device::clear_for_tests(DeviceId::Cpu);
        let input = buffer_uop(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let inner = UOp::reduce_axis(input, Op::Add, &[1]);
        let outer = UOp::reduce_axis(inner, Op::Add, &[0]);

        let items = schedule(&outer);
        assert_eq!(items.len(), 2);
        assert!(matches!(
            items[1].inputs[0],
            KernelInput::Buffer(ref handle) if Some(handle.clone()) == items[0].output_buffer
        ));
    }

    #[test]
    fn test_shared_subexpression_materializes_once() {
        device::clear_for_tests(DeviceId::Cpu);
        let left = buffer_uop(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let right = buffer_uop(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let shared = UOp::add(left, right);
        let sum = UOp::reduce_axis(shared.clone(), Op::Add, &[1]);
        let max = UOp::reduce_axis(shared.clone(), Op::Max, &[1]);
        let root = UOp::add(sum, max);

        let items = schedule(&root);
        assert_eq!(items.len(), 4);
        assert!(matches!(
            items[1].inputs[0],
            KernelInput::Buffer(ref handle) if Some(handle.clone()) == items[0].output_buffer
        ));
        assert!(matches!(
            items[2].inputs[0],
            KernelInput::Buffer(ref handle) if Some(handle.clone()) == items[0].output_buffer
        ));
        assert!(matches!(
            items[3].inputs[0],
            KernelInput::Buffer(ref handle) if Some(handle.clone()) == items[1].output_buffer
        ));
        assert!(matches!(
            items[3].inputs[1],
            KernelInput::Buffer(ref handle) if Some(handle.clone()) == items[2].output_buffer
        ));
    }

    #[test]
    fn test_parameterize_gives_each_shrink_occurrence_its_own_scalar_input() {
        device::clear_for_tests(DeviceId::Cpu);
        let input = buffer_uop(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let zero = UOp::const_int(0, DType::I32, DeviceId::Cpu);

        let left = UOp::shrink(input.clone(), &[zero.clone(), zero.clone()], &[2, 2]);
        let reshaped = UOp::reshape(input, Shape::from([3, 2]));
        let right = UOp::shrink(reshaped, &[zero.clone(), zero], &[2, 2]);
        let expr = UOp::add(left, right);

        let items = schedule(&expr);
        let scalar_count = items
            .iter()
            .flat_map(|item| item.inputs.iter())
            .filter(|input| {
                matches!(
                    input,
                    KernelInput::I32(_) | KernelInput::F32(_) | KernelInput::Bool(_)
                )
            })
            .count();

        assert_eq!(scalar_count, 2);
    }

    #[test]
    fn test_parameterize_store_turns_shrink_start_into_scalar_input() {
        device::clear_for_tests(DeviceId::Cpu);
        let src = buffer_uop(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let dst = buffer_uop(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let one = UOp::const_int(1, DType::I32, DeviceId::Cpu);
        let zero = UOp::const_int(0, DType::I32, DeviceId::Cpu);
        let narrowed = UOp::shrink(src, &[one, zero], &[2, 2]);

        let item = parameterize_store(&UOp::store(dst, narrowed));

        assert!(matches!(item.inputs[0], KernelInput::Buffer(_)));
        assert!(matches!(item.inputs[1], KernelInput::Buffer(_)));
        assert!(matches!(item.inputs[2], KernelInput::I32(1)));
    }

    #[test]
    fn test_shared_movement_view_does_not_materialize() {
        // Arrange
        device::clear_for_tests(DeviceId::Cpu);
        let left = buffer_uop(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let right = buffer_uop(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let base = UOp::add(left, right);
        let moved = UOp::reshape(base, Shape::from([2, 2]));
        let sum = UOp::reduce_axis(moved.clone(), Op::Add, &[1]);
        let max = UOp::reduce_axis(moved, Op::Max, &[1]);
        let root = UOp::add(sum, max);

        // Act
        let items = schedule(&root);

        // Assert
        assert_eq!(items.len(), 3);
        assert!(matches!(items[0].inputs[0], KernelInput::Buffer(_)));
        assert!(matches!(items[0].inputs[1], KernelInput::Buffer(_)));
        assert!(matches!(items[1].inputs[0], KernelInput::Buffer(_)));
        assert!(matches!(items[1].inputs[1], KernelInput::Buffer(_)));
        assert!(matches!(
            items[2].inputs[0],
            KernelInput::Buffer(ref handle) if Some(handle.clone()) == items[0].output_buffer
        ));
        assert!(matches!(
            items[2].inputs[1],
            KernelInput::Buffer(ref handle) if Some(handle.clone()) == items[1].output_buffer
        ));
    }

    #[test]
    fn test_shared_movement_view_stays_inlined_for_assignments() {
        // Arrange
        device::clear_for_tests(DeviceId::Cpu);
        let left = buffer_uop(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let right = buffer_uop(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let dst_a = buffer_uop(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let dst_b = buffer_uop(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let base = UOp::add(left, right);
        let moved = UOp::permute(base, &[1, 0]);
        let assign_a = UOp::after(dst_a.clone(), UOp::store(dst_a, moved.clone()));
        let assign_b = UOp::after(dst_b.clone(), UOp::store(dst_b, moved));

        // Act
        let plan = schedule_many(&[assign_a, assign_b]);

        // Assert
        assert_eq!(plan.items.len(), 2);
        assert!(plan.items.iter().all(|item| item.output_buffer.is_none()));
        assert!(plan
            .items
            .iter()
            .all(|item| matches!(item.inputs[0], KernelInput::Buffer(_))));
        assert!(plan
            .items
            .iter()
            .all(|item| matches!(item.inputs[1], KernelInput::Buffer(_))));
        assert!(plan
            .items
            .iter()
            .all(|item| matches!(item.inputs[2], KernelInput::Buffer(_))));
    }
}
