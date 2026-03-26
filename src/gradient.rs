//! # Autograd — reverse-mode automatic differentiation
//!
//! Implements the standard reverse-mode AD algorithm: walk the forward graph
//! in reverse topological order, applying the chain rule at each op to
//! propagate gradients from outputs back to inputs.
//!
//! The key insight is that gradient rules produce new lazy `UOp` nodes —
//! they don't compute numbers, they build more graph. This means the
//! backward pass reuses the exact same scheduling → rangeify → codegen
//! pipeline as the forward pass, with no special runtime support needed.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/autograd/gradient.py` — same reverse-toposort + chain-rule
//! structure, same trick of emitting lazy ops for gradients.

use std::collections::{HashMap, HashSet};

use crate::shape::Shape;
use crate::uop::{self, Arg, Op, UOp};

/// Compute gradients of `root` with respect to `targets`.
#[must_use]
pub fn compute_gradient(root: &UOp, root_grad: &UOp, targets: &[UOp]) -> HashMap<UOp, UOp> {
    let order = root.toposort();
    let consumer_map = uop::build_consumer_map(&order);
    let needed = needed_nodes(root, targets, &consumer_map);
    let mut grads: HashMap<UOp, UOp> = HashMap::new();
    grads.insert(root.clone(), root_grad.clone());

    for node in order.iter().rev() {
        if !needed.contains(node) {
            continue;
        }
        let Some(grad) = grads.get(node).cloned() else {
            continue;
        };
        let local_grads = gradient_for_op(node, &grad);
        for (src, src_grad) in node.srcs().iter().zip(local_grads) {
            let Some(src_grad) = src_grad else {
                continue;
            };
            if !needed.contains(src) {
                continue;
            }
            grads
                .entry(src.clone())
                .and_modify(|existing| *existing = UOp::add(existing.clone(), src_grad.clone()))
                .or_insert(src_grad);
        }
    }

    let mut result = HashMap::new();
    for target in targets {
        let grad = grads
            .get(target)
            .cloned()
            .unwrap_or_else(|| zero_like(target));
        result.insert(target.clone(), grad);
    }
    result
}

/// Find all nodes on any path between `targets` and `root` via consumers.
///
/// Only nodes reachable upward from the targets need gradients computed —
/// this avoids wasting work on branches that don't influence any target.
fn needed_nodes(
    root: &UOp,
    targets: &[UOp],
    consumer_map: &HashMap<UOp, Vec<UOp>>,
) -> HashSet<UOp> {
    let mut needed = HashSet::new();
    let mut stack = targets.to_vec();
    while let Some(node) = stack.pop() {
        if !needed.insert(node.clone()) {
            continue;
        }
        if let Some(consumers) = consumer_map.get(&node) {
            stack.extend(consumers.iter().cloned());
        }
    }
    needed.insert(root.clone());
    needed
}

fn const_float_like(node: &UOp, value: f64) -> UOp {
    UOp::const_float(value, node.dtype(), node.device())
}

fn zero_like(node: &UOp) -> UOp {
    let shape = node.shape().unwrap_or_else(|| Shape::from([1]));
    UOp::full(&shape, node.dtype(), node.device(), 0.0)
}

/// Return per-source gradients for a single op, given the upstream gradient.
///
/// Each entry corresponds to one source of `node`. `None` means the op is
/// not differentiable w.r.t. that source (e.g., the condition in `Where`).
#[allow(clippy::too_many_lines)]
fn gradient_for_op(node: &UOp, grad: &UOp) -> Vec<Option<UOp>> {
    match node.op() {
        Op::Add => vec![Some(grad.clone()), Some(grad.clone())],
        Op::Mul => {
            let (x, y) = (&node.srcs()[0], &node.srcs()[1]);
            vec![
                Some(UOp::mul(y.clone(), grad.clone())),
                Some(UOp::mul(x.clone(), grad.clone())),
            ]
        }
        Op::Neg => vec![Some(UOp::neg(grad.clone()))],
        Op::Reciprocal => {
            let neg_grad = UOp::neg(grad.clone());
            let ret_sq = UOp::mul(node.clone(), node.clone());
            vec![Some(UOp::mul(neg_grad, ret_sq))]
        }
        Op::Exp2 => {
            let ln2 = const_float_like(node, std::f64::consts::LN_2);
            vec![Some(UOp::mul(UOp::mul(node.clone(), ln2), grad.clone()))]
        }
        Op::Log2 => {
            let x = &node.srcs()[0];
            let ln2 = const_float_like(node, std::f64::consts::LN_2);
            vec![Some(UOp::mul(
                grad.clone(),
                UOp::reciprocal(UOp::mul(x.clone(), ln2)),
            ))]
        }
        Op::Sqrt => {
            let two = const_float_like(node, 2.0);
            vec![Some(UOp::mul(
                grad.clone(),
                UOp::reciprocal(UOp::mul(two, node.clone())),
            ))]
        }
        Op::Max => {
            let (x, y) = (&node.srcs()[0], &node.srcs()[1]);
            let zero = const_float_like(node, 0.0);
            let half_grad = UOp::mul(grad.clone(), const_float_like(node, 0.5));
            let x_wins = UOp::cmplt(y.clone(), x.clone());
            let y_wins = UOp::cmplt(x.clone(), y.clone());
            let x_tie = UOp::where_(y_wins.clone(), zero.clone(), half_grad.clone());
            let x_grad = UOp::where_(x_wins.clone(), grad.clone(), x_tie);
            let y_tie = UOp::where_(x_wins, zero, half_grad);
            let y_grad = UOp::where_(y_wins, grad.clone(), y_tie);
            vec![Some(x_grad), Some(y_grad)]
        }
        Op::CmpLt => vec![None, None],
        Op::Where => {
            let cond = &node.srcs()[0];
            let zero = const_float_like(node, 0.0);
            vec![
                None,
                Some(UOp::where_(cond.clone(), grad.clone(), zero.clone())),
                Some(UOp::where_(cond.clone(), zero, grad.clone())),
            ]
        }
        Op::Reshape => {
            let src_shape = node.srcs()[0].shape().expect("reshape src must have shape");
            vec![Some(UOp::reshape(grad.clone(), src_shape))]
        }
        Op::Permute => {
            let Arg::Axes(order) = node.arg() else {
                panic!("Permute must have Axes arg");
            };
            let inverse = Shape::invert_permutation(order);
            vec![Some(UOp::permute(grad.clone(), &inverse))]
        }
        Op::Expand => {
            let src_shape = node.srcs()[0].shape().expect("expand src must have shape");
            let Arg::Shape(expanded) = node.arg() else {
                panic!("Expand must have Shape arg");
            };
            let reduce_axes = src_shape.expand_gradient_axes(expanded);
            if reduce_axes.is_empty() {
                vec![Some(grad.clone())]
            } else {
                vec![Some(UOp::reduce_axis(grad.clone(), Op::Add, &reduce_axes))]
            }
        }
        Op::ReduceAxis => {
            let Arg::Reduce(reduce_op, axes) = node.arg() else {
                panic!("ReduceAxis must have Reduce arg");
            };
            let src = &node.srcs()[0];
            let src_shape = src.shape().expect("reduce src must have shape");
            match *reduce_op {
                Op::Add => vec![Some(UOp::expand(grad.clone(), src_shape))],
                Op::Max => {
                    let dt = grad.dtype();
                    let max_expanded = UOp::expand(node.clone(), src_shape.clone());
                    let grad_expanded = UOp::expand(grad.clone(), src_shape.clone());
                    let is_not_max = UOp::cmplt(src.clone(), max_expanded);
                    let zero = UOp::const_float(0.0, dt, node.device());
                    let one = UOp::const_float(1.0, dt, node.device());
                    let mask = UOp::where_(is_not_max, zero, one);
                    let count = UOp::reduce_axis(mask.clone(), Op::Add, axes);
                    let count_expanded = UOp::expand(count, src_shape);
                    let weighted = UOp::mul(mask, UOp::reciprocal(count_expanded));
                    vec![Some(UOp::mul(weighted, grad_expanded))]
                }
                _ => panic!("unsupported reduce op gradient: {reduce_op:?}"),
            }
        }
        Op::Buffer | Op::Const => vec![],
        _ => panic!("gradient not implemented for {:?}", node.op()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device;
    use crate::device::DeviceId;
    use crate::dtype::DType;

    #[allow(clippy::cast_possible_truncation)]
    fn scalar_uop(val: f64) -> UOp {
        let device_id = DeviceId::Cpu;
        let backend = device::get(device_id);
        let buffer = backend.allocate(DType::F32, 1);
        backend.copy_from_host(&buffer, bytemuck::cast_slice(&[val as f32]));
        let buf = UOp::buffer(buffer, DType::F32, device_id);
        UOp::reshape(buf, Shape::from([1]))
    }

    #[test]
    fn test_grad_add() {
        let x = scalar_uop(3.0);
        let y = scalar_uop(5.0);
        let sum = UOp::add(x.clone(), y.clone());
        let grads = compute_gradient(&sum, &scalar_uop(1.0), &[x.clone(), y.clone()]);
        assert!(grads.contains_key(&x));
        assert!(grads.contains_key(&y));
    }
}
