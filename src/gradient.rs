//! # Autograd — reverse-mode automatic differentiation
//!
//! Computes gradients by walking the `UOp` graph in reverse topological order
//! and applying the chain rule at each node. Each op has a gradient rule that
//! takes the upstream gradient and produces gradients for its inputs.
//!
//! The key insight: gradient rules produce *new `UOp`s*. The backward pass builds
//! more lazy graph, which gets compiled and fused just like the forward pass.
//! There's no separate "backward kernel" — it's all the same compiler pipeline.
//!
//! ## How it works
//!
//! 1. Toposort the root expression to get dependency order.
//! 2. Walk nodes in reverse (outputs before inputs).
//! 3. For each node, look up its gradient rule and produce local gradients.
//! 4. Accumulate gradients: if a node feeds multiple consumers, their
//!    gradient contributions are summed (the multivariate chain rule).
//! 5. Return the accumulated gradient for each requested target.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/gradient.py` — `pm_gradient`, `compute_gradient`

use std::collections::HashMap;

use crate::uop::{Arg, Op, UOp};

/// Compute gradients of `root` with respect to `targets`.
///
/// `root_grad` is the gradient of the loss with respect to `root` — typically
/// a scalar 1.0 for the loss tensor.
///
/// Returns a map from each target `UOp` to its gradient `UOp`. Targets not
/// reachable from root get a zero gradient.
#[must_use]
pub fn compute_gradient(root: &UOp, root_grad: &UOp, targets: &[UOp]) -> HashMap<UOp, UOp> {
    let order = root.toposort();

    let mut grads: HashMap<UOp, UOp> = HashMap::new();
    grads.insert(root.clone(), root_grad.clone());

    for node in order.iter().rev() {
        let Some(grad) = grads.get(node).cloned() else {
            continue;
        };
        let local_grads = gradient_for_op(node, &grad);

        for (src, src_grad) in node.srcs().iter().zip(local_grads) {
            let Some(g) = src_grad else { continue };
            grads
                .entry(src.clone())
                .and_modify(|existing| {
                    *existing = UOp::add(existing.clone(), g.clone());
                })
                .or_insert(g);
        }
    }

    let mut result = HashMap::new();
    for target in targets {
        let grad = grads
            .get(target)
            .cloned()
            .unwrap_or_else(|| UOp::const_float(0.0, target.dtype()));
        result.insert(target.clone(), grad);
    }
    result
}

/// Gradient rules for each op.
///
/// Returns one `Option<UOp>` per source of `node`. `None` means "no gradient
/// flows to this source" (e.g. the condition input of Where).
///
/// Each rule implements the chain rule: given the upstream gradient `grad`
/// (∂loss/∂output), produce ∂loss/∂input for each input.
#[allow(clippy::too_many_lines)]
fn gradient_for_op(node: &UOp, grad: &UOp) -> Vec<Option<UOp>> {
    match node.op() {
        // ── Elementwise math ────────────────────────────────────────────

        // d/d{x,y} (x + y) = {1, 1}
        Op::Add => vec![Some(grad.clone()), Some(grad.clone())],

        // d/d{x,y} (x * y) = {y, x}
        Op::Mul => {
            let (x, y) = (&node.srcs()[0], &node.srcs()[1]);
            vec![
                Some(UOp::mul(y.clone(), grad.clone())),
                Some(UOp::mul(x.clone(), grad.clone())),
            ]
        }

        // d/dx (-x) = -1
        Op::Neg => vec![Some(UOp::neg(grad.clone()))],

        // d/dx (1/x) = -ret²
        Op::Reciprocal => {
            let neg_grad = UOp::neg(grad.clone());
            let ret_sq = UOp::mul(node.clone(), node.clone());
            vec![Some(UOp::mul(neg_grad, ret_sq))]
        }

        // d/dx (2^x) = ret · ln(2)
        Op::Exp2 => {
            let ln2 = UOp::const_float(std::f64::consts::LN_2, node.dtype());
            vec![Some(UOp::mul(UOp::mul(node.clone(), ln2), grad.clone()))]
        }

        // d/dx (log₂(x)) = 1/(x · ln(2))
        Op::Log2 => {
            let x = &node.srcs()[0];
            let ln2 = UOp::const_float(std::f64::consts::LN_2, node.dtype());
            vec![Some(UOp::mul(grad.clone(), UOp::reciprocal(UOp::mul(x.clone(), ln2))))]
        }

        // d/dx (√x) = 1/(2·ret)
        Op::Sqrt => {
            let two = UOp::const_float(2.0, node.dtype());
            vec![Some(UOp::mul(grad.clone(), UOp::reciprocal(UOp::mul(two, node.clone()))))]
        }

        // d/d{x,y} max(x,y): winner gets grad, ties split evenly at 0.5.
        Op::Max => {
            let (x, y) = (&node.srcs()[0], &node.srcs()[1]);
            let zero = UOp::const_float(0.0, node.dtype());
            let half_grad = UOp::mul(grad.clone(), UOp::const_float(0.5, node.dtype()));
            let x_wins = UOp::cmplt(y.clone(), x.clone());
            let y_wins = UOp::cmplt(x.clone(), y.clone());
            // x_grad = (y < x) ? grad : ((x < y) ? 0 : grad*0.5)
            let x_tie = UOp::where_(y_wins.clone(), zero.clone(), half_grad.clone());
            let x_grad = UOp::where_(x_wins.clone(), grad.clone(), x_tie);
            let y_tie = UOp::where_(x_wins, zero, half_grad);
            let y_grad = UOp::where_(y_wins, grad.clone(), y_tie);
            vec![Some(x_grad), Some(y_grad)]
        }

        // Comparisons produce booleans — no meaningful gradient.
        Op::CmpLt => vec![None, None],

        // cond: no gradient (discrete), t/f: route grad through Where.
        Op::Where => {
            let cond = &node.srcs()[0];
            let zero = UOp::const_float(0.0, node.dtype());
            vec![
                None,
                Some(UOp::where_(cond.clone(), grad.clone(), zero.clone())),
                Some(UOp::where_(cond.clone(), zero, grad.clone())),
            ]
        }

        // ── Movement ops — apply inverse transformation ─────────────────

        Op::Reshape => {
            let src_shape = node.srcs()[0].shape().expect("reshape src must have shape");
            vec![Some(UOp::new(Op::Reshape, grad.dtype(), vec![grad.clone()], Arg::Dims(src_shape)))]
        }

        // Inverse permutation.
        Op::Permute => {
            let Arg::Dims(ref perm) = node.arg() else {
                panic!("Permute must have Dims arg");
            };
            let mut inv = vec![0; perm.len()];
            for (i, &p) in perm.iter().enumerate() {
                inv[p] = i;
            }
            vec![Some(UOp::new(Op::Permute, grad.dtype(), vec![grad.clone()], Arg::Dims(inv)))]
        }

        // Expand broadcasts size-1 dims → gradient sums over those dims.
        Op::Expand => {
            let src_shape = node.srcs()[0].shape().expect("expand src must have shape");
            let Arg::Dims(ref expanded) = node.arg() else {
                panic!("Expand must have Dims arg");
            };
            let reduce_axes: Vec<usize> = src_shape
                .iter()
                .zip(expanded)
                .enumerate()
                .filter(|(_, (&s, &e))| s == 1 && e > 1)
                .map(|(i, _)| i)
                .collect();
            if reduce_axes.is_empty() {
                vec![Some(grad.clone())]
            } else {
                vec![Some(UOp::new(
                    Op::ReduceAxis, grad.dtype(),
                    vec![grad.clone()], Arg::Reduce(Op::Add, reduce_axes),
                ))]
            }
        }

        // ── Reduction ───────────────────────────────────────────────────

        // ReduceAxis(Add): expand gradient back to source shape.
        // ReduceAxis(Max): gradient to max-achieving elements, split on ties.
        Op::ReduceAxis => {
            let Arg::Reduce(ref reduce_op, ref axes) = node.arg() else {
                panic!("ReduceAxis must have Reduce arg");
            };
            let src = &node.srcs()[0];
            let src_shape = src.shape().expect("reduce src must have shape");
            match *reduce_op {
                Op::Add => {
                    vec![Some(UOp::new(
                        Op::Expand, grad.dtype(),
                        vec![grad.clone()], Arg::Dims(src_shape),
                    ))]
                }
                Op::Max => {
                    let dt = grad.dtype();
                    let max_expanded = UOp::new(
                        Op::Expand, node.dtype(),
                        vec![node.clone()], Arg::Dims(src_shape.clone()),
                    );
                    let grad_expanded = UOp::new(
                        Op::Expand, dt,
                        vec![grad.clone()], Arg::Dims(src_shape.clone()),
                    );
                    let is_not_max = UOp::cmplt(src.clone(), max_expanded);
                    let zero = UOp::const_float(0.0, dt);
                    let one = UOp::const_float(1.0, dt);
                    let mask = UOp::where_(is_not_max, zero, one);
                    // count = number of elements achieving max
                    let count = UOp::new(
                        Op::ReduceAxis, dt,
                        vec![mask.clone()], Arg::Reduce(Op::Add, axes.clone()),
                    );
                    let count_expanded = UOp::new(
                        Op::Expand, dt, vec![count], Arg::Dims(src_shape),
                    );
                    // mask / count * grad — split evenly among ties
                    let weighted = UOp::mul(mask, UOp::reciprocal(count_expanded));
                    vec![Some(UOp::mul(weighted, grad_expanded))]
                }
                _ => panic!("unsupported reduce op gradient: {reduce_op:?}"),
            }
        }

        // ── Leaves ──────────────────────────────────────────────────────
        Op::Buffer | Op::Const => vec![],

        _ => panic!("gradient not implemented for {:?}", node.op()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    fn scalar_uop(val: f64) -> UOp {
        let buf = UOp::new(
            Op::Buffer, DType::F32, vec![],
            Arg::Buffer(std::rc::Rc::new(crate::device::Buffer::from_f32(&[val as f32]))),
        );
        UOp::new(Op::Reshape, DType::F32, vec![buf], Arg::Dims(vec![1]))
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

    #[test]
    fn test_grad_mul() {
        let x = scalar_uop(3.0);
        let y = scalar_uop(5.0);
        let prod = UOp::mul(x.clone(), y.clone());
        let grads = compute_gradient(&prod, &scalar_uop(1.0), &[x.clone(), y.clone()]);
        assert!(grads.contains_key(&x));
        assert!(grads.contains_key(&y));
    }

    #[test]
    fn test_grad_shared_input() {
        // d/dx (x * x) = 2x — gradient accumulates from both sources
        let x = scalar_uop(3.0);
        let sq = UOp::mul(x.clone(), x.clone());
        let grads = compute_gradient(&sq, &scalar_uop(1.0), &[x.clone()]);
        assert_eq!(grads.get(&x).unwrap().op(), Op::Add);
    }
}
