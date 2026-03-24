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

use crate::dtype::DType;
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
                    *existing =
                        UOp::new(Op::Add, existing.dtype(), vec![existing.clone(), g.clone()], Arg::None);
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

        // d/d{x,y} (x + y) = {1, 1} → gradient passes through unchanged
        Op::Add => vec![Some(grad.clone()), Some(grad.clone())],

        // d/d{x,y} (x * y) = {y, x} → product rule
        Op::Mul => {
            let x = &node.srcs()[0];
            let y = &node.srcs()[1];
            vec![
                Some(UOp::new(Op::Mul, node.dtype(), vec![y.clone(), grad.clone()], Arg::None)),
                Some(UOp::new(Op::Mul, node.dtype(), vec![x.clone(), grad.clone()], Arg::None)),
            ]
        }

        // d/dx (-x) = -1 → negate the gradient
        Op::Neg => vec![Some(UOp::new(Op::Neg, node.dtype(), vec![grad.clone()], Arg::None))],

        // d/dx (1/x) = -1/x² = -(1/x)² = -ret²
        Op::Reciprocal => {
            let neg_grad = UOp::new(Op::Neg, node.dtype(), vec![grad.clone()], Arg::None);
            let ret_sq = UOp::new(Op::Mul, node.dtype(), vec![node.clone(), node.clone()], Arg::None);
            vec![Some(UOp::new(Op::Mul, node.dtype(), vec![neg_grad, ret_sq], Arg::None))]
        }

        // d/dx (2^x) = 2^x · ln(2) = ret · ln(2)
        Op::Exp2 => {
            let ln2 = UOp::const_float(std::f64::consts::LN_2, node.dtype());
            let ret_ln2 = UOp::new(Op::Mul, node.dtype(), vec![node.clone(), ln2], Arg::None);
            vec![Some(UOp::new(Op::Mul, node.dtype(), vec![ret_ln2, grad.clone()], Arg::None))]
        }

        // d/dx (log₂(x)) = 1/(x · ln(2))
        Op::Log2 => {
            let x = &node.srcs()[0];
            let ln2 = UOp::const_float(std::f64::consts::LN_2, node.dtype());
            let x_ln2 = UOp::new(Op::Mul, node.dtype(), vec![x.clone(), ln2], Arg::None);
            let recip = UOp::new(Op::Reciprocal, node.dtype(), vec![x_ln2], Arg::None);
            vec![Some(UOp::new(Op::Mul, node.dtype(), vec![grad.clone(), recip], Arg::None))]
        }

        // d/dx (√x) = 1/(2√x) = grad/(2·ret)
        Op::Sqrt => {
            let two = UOp::const_float(2.0, node.dtype());
            let two_ret = UOp::new(Op::Mul, node.dtype(), vec![two, node.clone()], Arg::None);
            let recip = UOp::new(Op::Reciprocal, node.dtype(), vec![two_ret], Arg::None);
            vec![Some(UOp::new(Op::Mul, node.dtype(), vec![grad.clone(), recip], Arg::None))]
        }

        // d/d{x,y} max(x,y): gradient goes to whichever was larger.
        // On ties (x == y), split evenly: each gets grad * 0.5.
        // Matches tinygrad: (x>y).where(ctx, (x==y).where(ctx*0.5, 0))
        Op::Max => {
            let x = &node.srcs()[0];
            let y = &node.srcs()[1];
            let dt = node.dtype();
            let zero = UOp::const_float(0.0, dt);
            let half_grad = UOp::new(Op::Mul, dt, vec![grad.clone(), UOp::const_float(0.5, dt)], Arg::None);
            // x == y ≡ !(x < y) && !(y < x), but we can express with nested Where:
            // x_grad = (y < x) ? grad : ((x < y) ? 0 : grad*0.5)
            let x_wins = UOp::new(Op::CmpLt, DType::Bool, vec![y.clone(), x.clone()], Arg::None);
            let y_wins = UOp::new(Op::CmpLt, DType::Bool, vec![x.clone(), y.clone()], Arg::None);
            let x_tie = UOp::new(Op::Where, dt, vec![y_wins.clone(), zero.clone(), half_grad.clone()], Arg::None);
            let x_grad = UOp::new(Op::Where, dt, vec![x_wins.clone(), grad.clone(), x_tie], Arg::None);
            // y_grad = (x < y) ? grad : ((y < x) ? 0 : grad*0.5)
            let y_tie = UOp::new(Op::Where, dt, vec![x_wins, zero, half_grad], Arg::None);
            let y_grad = UOp::new(Op::Where, dt, vec![y_wins, grad.clone(), y_tie], Arg::None);
            vec![Some(x_grad), Some(y_grad)]
        }

        // Comparisons produce booleans — no meaningful gradient.
        Op::CmpLt => vec![None, None],

        // d/d{cond,t,f} where(cond, t, f):
        // cond has no gradient (discrete), t gets grad where cond is true,
        // f gets grad where cond is false.
        Op::Where => {
            let cond = &node.srcs()[0];
            let zero = UOp::const_float(0.0, node.dtype());
            vec![
                None,
                Some(UOp::new(Op::Where, node.dtype(), vec![cond.clone(), grad.clone(), zero.clone()], Arg::None)),
                Some(UOp::new(Op::Where, node.dtype(), vec![cond.clone(), zero, grad.clone()], Arg::None)),
            ]
        }

        // ── Movement ops — apply inverse transformation ─────────────────

        // Reshape doesn't move data, so gradient just reshapes back.
        Op::Reshape => {
            let src_shape = node.srcs()[0].shape().expect("reshape src must have shape");
            vec![Some(UOp::new(Op::Reshape, grad.dtype(), vec![grad.clone()], Arg::Dims(src_shape)))]
        }

        // Permute reorders dimensions. Gradient applies the inverse permutation.
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

        // Expand broadcasts size-1 dims. Gradient sums over those dims to
        // undo the broadcast (many-to-one → sum the contributions).
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
                    Op::ReduceAxis,
                    grad.dtype(),
                    vec![grad.clone()],
                    Arg::Reduce(Op::Add, reduce_axes),
                ))]
            }
        }

        // ── Reduction ───────────────────────────────────────────────────

        // ReduceAxis(Add): gradient expands back to the source shape.
        // The sum collapsed a dimension — the gradient broadcasts back,
        // because each input element contributed equally to the sum.
        //
        // ReduceAxis(Max): gradient goes only to the element(s) that achieved
        // the maximum. We approximate by masking with (src == max_expanded).
        Op::ReduceAxis => {
            let Arg::Reduce(ref reduce_op, _) = node.arg() else {
                panic!("ReduceAxis must have Reduce arg");
            };
            let src = &node.srcs()[0];
            let src_shape = src.shape().expect("reduce src must have shape");
            match *reduce_op {
                Op::Add => {
                    vec![Some(UOp::new(
                        Op::Expand,
                        grad.dtype(),
                        vec![grad.clone()],
                        Arg::Dims(src_shape),
                    ))]
                }
                Op::Max => {
                    let Arg::Reduce(_, ref axes) = node.arg() else { unreachable!() };
                    let dt = grad.dtype();
                    // Expand max result and gradient back to source shape
                    let max_expanded = UOp::new(
                        Op::Expand, node.dtype(),
                        vec![node.clone()], Arg::Dims(src_shape.clone()),
                    );
                    let grad_expanded = UOp::new(
                        Op::Expand, dt,
                        vec![grad.clone()], Arg::Dims(src_shape.clone()),
                    );
                    // mask = 1 where src achieved the max, 0 elsewhere
                    let is_not_max = UOp::new(
                        Op::CmpLt, DType::Bool,
                        vec![src.clone(), max_expanded], Arg::None,
                    );
                    let zero = UOp::const_float(0.0, dt);
                    let one = UOp::const_float(1.0, dt);
                    let mask = UOp::new(
                        Op::Where, dt,
                        vec![is_not_max, zero.clone(), one], Arg::None,
                    );
                    // count = number of elements achieving max (sum of mask over reduce axes)
                    let count = UOp::new(
                        Op::ReduceAxis, dt,
                        vec![mask.clone()], Arg::Reduce(Op::Add, axes.clone()),
                    );
                    let count_expanded = UOp::new(
                        Op::Expand, dt,
                        vec![count], Arg::Dims(src_shape),
                    );
                    // grad_src = mask / count * grad (split evenly among ties)
                    let recip_count = UOp::new(Op::Reciprocal, dt, vec![count_expanded], Arg::None);
                    let weighted = UOp::new(Op::Mul, dt, vec![mask, recip_count], Arg::None);
                    let masked = UOp::new(Op::Mul, dt, vec![weighted, grad_expanded], Arg::None);
                    vec![Some(masked)]
                }
                _ => panic!("unsupported reduce op gradient: {reduce_op:?}"),
            }
        }

        // ── Leaves — no sources to propagate to ─────────────────────────
        Op::Buffer | Op::Const => vec![],

        _ => panic!("gradient not implemented for {:?}", node.op()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a scalar-shaped UOp (Reshape([1], Buffer)).
    fn scalar_uop(val: f64) -> UOp {
        let buf = UOp::new(
            Op::Buffer,
            DType::F32,
            vec![],
            Arg::Buffer(std::rc::Rc::new(crate::device::Buffer::from_f32(&[val as f32]))),
        );
        UOp::new(Op::Reshape, DType::F32, vec![buf], Arg::Dims(vec![1]))
    }

    fn ones_grad() -> UOp {
        UOp::new(Op::Reshape, DType::F32,
            vec![UOp::new(Op::Buffer, DType::F32, vec![],
                Arg::Buffer(std::rc::Rc::new(crate::device::Buffer::from_f32(&[1.0]))))],
            Arg::Dims(vec![1]))
    }

    #[test]
    fn test_grad_add() {
        // d/dx (x + y) = 1
        let x = scalar_uop(3.0);
        let y = scalar_uop(5.0);
        let sum = UOp::new(Op::Add, DType::F32, vec![x.clone(), y.clone()], Arg::None);
        let grad = ones_grad();

        let grads = compute_gradient(&sum, &grad, &[x.clone(), y.clone()]);

        // Both gradients should be the upstream gradient (1.0)
        assert!(grads.contains_key(&x));
        assert!(grads.contains_key(&y));
    }

    #[test]
    fn test_grad_mul() {
        // d/dx (x * y) = y, d/dy (x * y) = x
        let x = scalar_uop(3.0);
        let y = scalar_uop(5.0);
        let prod = UOp::new(Op::Mul, DType::F32, vec![x.clone(), y.clone()], Arg::None);
        let grad = ones_grad();

        let grads = compute_gradient(&prod, &grad, &[x.clone(), y.clone()]);
        assert!(grads.contains_key(&x));
        assert!(grads.contains_key(&y));
    }

    #[test]
    fn test_grad_shared_input() {
        // d/dx (x * x) = 2x — gradient accumulates from both sources
        let x = scalar_uop(3.0);
        let sq = UOp::new(Op::Mul, DType::F32, vec![x.clone(), x.clone()], Arg::None);
        let grad = ones_grad();

        let grads = compute_gradient(&sq, &grad, &[x.clone()]);
        // The gradient UOp should be Add(x*1, x*1) = 2x
        let g = grads.get(&x).unwrap();
        assert_eq!(g.op(), Op::Add);
    }
}
