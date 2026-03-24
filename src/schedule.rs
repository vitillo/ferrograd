//! # Scheduling — turning lazy tensor graphs into executable kernels
//!
//! This module converts the lazy tensor-level graph into a list of proto-kernel
//! graphs, each with explicit input parameters and a reserved output buffer.

pub mod indexing;
pub mod rangeify;

use std::collections::{HashMap, HashSet};

use crate::dtype::DType;
use crate::device::Buffer;
use crate::runtime::{self, BufferId};
use crate::rewrite::graph_rewrite;
use crate::shape::Shape;
use crate::uop::{Arg, Op, UOp};

/// A runtime input to a compiled kernel.
pub enum KernelInput {
    /// A realized tensor buffer identified in device state.
    Buffer(BufferId),
    /// A one-element scalar buffer materialized for this execution.
    Scalar(Buffer),
}

/// A single kernel to compile and execute.
pub struct ScheduleItem {
    /// Kernel-ready `Sink(Store(Param(0), expr))`.
    pub sink: UOp,
    /// Runtime inputs in parameter-slot order, excluding output slot 0.
    pub inputs: Vec<KernelInput>,
    /// Output buffer id reserved for slot 0.
    pub output_id: BufferId,
    /// Output shape.
    pub out_shape: Shape,
    /// Output dtype.
    pub out_dtype: DType,
}

/// Analyze a lazy `UOp` graph and produce kernels in dependency order.
#[must_use]
pub fn schedule(expr: &UOp) -> Vec<ScheduleItem> {
    let order = expr.toposort();
    let consumer_map = build_consumer_map(&order);
    let nested_reduce_inputs = nested_reduce_inputs(&order);

    let mut replacements: HashMap<UOp, UOp> = HashMap::new();
    let mut items = Vec::new();

    for node in &order {
        if !should_materialize(node, expr, &consumer_map, &nested_reduce_inputs) {
            continue;
        }

        let kernel_expr = substitute_materialized(node, &replacements);
        let item = parameterize(&kernel_expr);
        let replacement = buffer_replacement(&item);
        replacements.insert(node.clone(), replacement);
        items.push(item);
    }

    let final_expr = substitute_materialized(expr, &replacements);
    items.push(parameterize(&final_expr));
    items
}

fn parameterize(expr: &UOp) -> ScheduleItem {
    use std::cell::RefCell;

    let state = runtime::state(expr.device());
    let inputs: RefCell<Vec<KernelInput>> = RefCell::new(Vec::new());
    let params: RefCell<HashMap<BufferId, UOp>> = RefCell::new(HashMap::new());
    let scalar_params: RefCell<HashMap<UOp, UOp>> = RefCell::new(HashMap::new());

    let rewrite_inputs = |node: &UOp| -> Option<UOp> {
        match node.op() {
            Op::Buffer => {
                let Arg::Buffer(id, numel) = node.arg() else {
                    return None;
                };
                let dtype = node.dtype();
                let mut params = params.borrow_mut();
                Some(
                    params
                        .entry(*id)
                        .or_insert_with(|| {
                            let mut inputs = inputs.borrow_mut();
                            let slot = inputs.len() + 1;
                            inputs.push(KernelInput::Buffer(*id));
                            UOp::param(slot, dtype, *numel, expr.device())
                        })
                        .clone(),
                )
            }
            Op::Bind => {
                let variable = node.srcs()[0].clone();
                let literal = node.srcs()[1].arg().clone();
                let mut scalar_params = scalar_params.borrow_mut();
                let param = scalar_params
                    .entry(variable)
                    .or_insert_with(|| {
                        let mut inputs = inputs.borrow_mut();
                        let slot = inputs.len() + 1;
                        inputs.push(KernelInput::Scalar(literal_buffer(node.dtype(), &literal)));
                        UOp::param(slot, node.dtype(), 1, expr.device())
                    })
                    .clone();
                let zero = UOp::const_int(0, DType::I32, expr.device());
                let index = UOp::new(Op::Index, node.dtype(), vec![param, zero], Arg::Index(0));
                Some(UOp::new(Op::Load, node.dtype(), vec![index], Arg::None))
            }
            _ => None,
        }
    };

    let parameterized = graph_rewrite(expr, &rewrite_inputs, "schedule");
    let out_shape = parameterized.shape().unwrap_or_else(|| Shape::flat(1));
    let output_id = state.reserve_buffer(expr.dtype(), out_shape.numel());
    let out_param = UOp::param(0, expr.dtype(), out_shape.numel(), expr.device());
    let store = UOp::new(
        Op::Store,
        DType::Void,
        vec![out_param, parameterized],
        Arg::None,
    );
    let sink = UOp::sink(vec![store]);

    ScheduleItem {
        sink,
        inputs: inputs.into_inner(),
        output_id,
        out_shape,
        out_dtype: expr.dtype(),
    }
}

fn literal_buffer(dtype: DType, literal: &Arg) -> Buffer {
    match literal {
        #[allow(clippy::cast_possible_truncation)]
        Arg::Float(value) => Buffer::from_f32(&[*value as f32]),
        Arg::Int(value) => {
            let value = i32::try_from(*value).expect("kernel scalar int must fit in i32");
            let bytes = value.to_ne_bytes().to_vec();
            Buffer::new(dtype, 1, crate::device::Storage::Cpu(bytes))
        }
        Arg::Bool(value) => {
            Buffer::new(dtype, 1, crate::device::Storage::Cpu(vec![u8::from(*value)]))
        }
        _ => panic!("kernel scalar input must be a literal"),
    }
}

fn build_consumer_map(order: &[UOp]) -> HashMap<UOp, Vec<UOp>> {
    let mut consumers: HashMap<UOp, Vec<UOp>> = HashMap::new();
    for node in order {
        for src in node.srcs() {
            consumers.entry(src.clone()).or_default().push(node.clone());
        }
    }
    consumers
}

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

fn should_materialize(
    node: &UOp,
    root: &UOp,
    consumer_map: &HashMap<UOp, Vec<UOp>>,
    nested_reduce_inputs: &HashSet<UOp>,
) -> bool {
    if node == root || node.shape().is_none() {
        return false;
    }
    if matches!(node.op(), Op::Buffer | Op::Const | Op::DefineVar | Op::Bind) {
        return false;
    }
    if consumer_map.get(node).is_some_and(|consumers| consumers.len() > 1) {
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

fn buffer_replacement(item: &ScheduleItem) -> UOp {
    let buffer = UOp::buffer(
        item.output_id,
        item.out_dtype,
        item.out_shape.numel(),
        item.sink.device(),
    );
    UOp::reshape(buffer, item.out_shape.clone())
}

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
        let changed = node.srcs().iter().zip(&new_srcs).any(|(old, new)| old != new);
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
    use crate::device::DeviceId;
    use crate::device::Buffer;
    use crate::runtime;

    fn buffer_uop(data: &[f32], shape: &[usize]) -> UOp {
        let device = DeviceId::Cpu;
        let id = runtime::state(device).store_buffer(Buffer::from_f32(data));
        let buffer = UOp::buffer(id, DType::F32, data.len(), device);
        UOp::reshape(buffer, Shape::from(shape))
    }

    #[test]
    fn test_nested_reductions_materialize_via_buffer_ids() {
        runtime::clear_for_tests(DeviceId::Cpu);
        let input = buffer_uop(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let inner = UOp::reduce_axis(input, Op::Add, &[1]);
        let outer = UOp::reduce_axis(inner, Op::Add, &[0]);

        let items = schedule(&outer);
        assert_eq!(items.len(), 2);
        assert!(matches!(items[1].inputs[0], KernelInput::Buffer(id) if id == items[0].output_id));
    }

    #[test]
    fn test_shared_subexpression_materializes_once() {
        runtime::clear_for_tests(DeviceId::Cpu);
        let left = buffer_uop(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let right = buffer_uop(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let shared = UOp::add(left, right);
        let sum = UOp::reduce_axis(shared.clone(), Op::Add, &[1]);
        let max = UOp::reduce_axis(shared.clone(), Op::Max, &[1]);
        let root = UOp::add(sum, max);

        let items = schedule(&root);
        assert_eq!(items.len(), 4);
        assert!(matches!(items[1].inputs[0], KernelInput::Buffer(id) if id == items[0].output_id));
        assert!(matches!(items[2].inputs[0], KernelInput::Buffer(id) if id == items[0].output_id));
        assert!(matches!(items[3].inputs[0], KernelInput::Buffer(id) if id == items[1].output_id));
        assert!(matches!(items[3].inputs[1], KernelInput::Buffer(id) if id == items[2].output_id));
    }
}
