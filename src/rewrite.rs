//! # Graph Rewriting — fixed-point rewrite engine for `UOp` graphs
//!
//! This is the compiler's workhorse: a general-purpose engine for
//! transforming IR graphs. You define rewrite rules as plain functions
//! (`&UOp → Option<UOp>`), and `graph_rewrite` applies them bottom-up
//! in a fixed-point loop until no more rules fire.
//!
//! Almost every transformation in the compiler is expressed as rewrite rules:
//! - **Scheduling**: `Buffer → Param` (in [`crate::schedule`])
//! - **Rangeify**: Store → loops, Index pushing, Reduce expansion
//!   (in [`crate::schedule::rangeify`])
//! - **Simplification**: `x+0 → x`, constant folding (in [`symbolic_simple`])
//!
//! Tinygrad uses `PatternMatcher` + `UPat` patterns for the same purpose.
//! We use Rust's native `match` instead — same fixed-point loop, same
//! declarative rules, but with compile-time type checking and no framework
//! to learn.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/uop/ops.py` — `graph_rewrite` (the fixed-point loop concept).

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::uop::{Arg, Op, UOp};

static DEBUG: LazyLock<u8> = LazyLock::new(|| {
    std::env::var("DEBUG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
});

// ── graph_rewrite ───────────────────────────────────────────────────────────

/// Rewrite a graph bottom-up until no more rules fire (fixed-point).
///
/// `rewrite` is called on each node after its children have been updated.
/// If it returns `Some(replacement)`, the node is replaced and another
/// iteration begins. The loop terminates when a full pass produces no
/// changes.
///
/// The `name` parameter identifies the pass in debug output. When
/// `DEBUG >= 3`, prints the graph before and after the rewrite.
#[must_use]
pub fn graph_rewrite(root: &UOp, rewrite: &dyn Fn(&UOp) -> Option<UOp>, name: &str) -> UOp {
    let debug = *DEBUG;
    if debug >= 3 {
        eprintln!("━━━ {name} [before] ━━━\n{}", root.dump());
    }

    let mut current = root.clone();

    loop {
        let order = current.toposort();
        let mut replace: HashMap<UOp, UOp> = HashMap::new();
        let mut changed = false;

        for node in &order {
            let new_srcs: Vec<UOp> = node
                .srcs()
                .iter()
                .map(|s| replace.get(s).cloned().unwrap_or_else(|| s.clone()))
                .collect();

            let srcs_changed = node.srcs().iter().zip(&new_srcs).any(|(old, new)| old != new);
            let rebuilt = if srcs_changed {
                UOp::new(node.op(), node.dtype(), new_srcs, node.arg().clone())
            } else {
                node.clone()
            };

            if let Some(replacement) = rewrite(&rebuilt) {
                if replacement != rebuilt {
                    replace.insert(node.clone(), replacement);
                    changed = true;
                    continue;
                }
            }
            replace.insert(node.clone(), rebuilt);
        }

        current = replace.get(&current).cloned().unwrap_or(current);

        if !changed {
            if debug >= 3 {
                eprintln!("━━━ {name} [after] ━━━\n{}", current.dump());
            }
            return current;
        }
    }
}

// ── Symbolic simplification ─────────────────────────────────────────────────
//
// These rules clean up the index arithmetic that rangeify generates.
// For example, when a Reshape inserts a size-1 dim, flat_index produces
// `idx * 1 + 0` — these rules simplify that to just `idx`.
//
// Tinygrad has a much larger set of symbolic rules (see `tinygrad/uop/symbolic.py`).
// We start with the essentials: constant folding and identity elimination.

/// Evaluate a binary op on two constant `Arg` values at compile time.
/// Returns `None` if the types or op are unsupported for folding.
fn fold_binary(op: Op, a: &Arg, b: &Arg) -> Option<Arg> {
    match (op, a, b) {
        (Op::Add, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x + y)),
        (Op::Add, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x + y)),
        (Op::Mul, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x * y)),
        (Op::Mul, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x * y)),
        _ => None,
    }
}

/// Reconstruct a `Const` `UOp` from a folded `Arg`, inheriting dtype/device from `node`.
fn const_from_arg(node: &UOp, arg: &Arg) -> UOp {
    match arg {
        Arg::Float(value) => UOp::const_float(*value, node.dtype(), node.device()),
        Arg::Int(value) => UOp::const_int(*value, node.dtype(), node.device()),
        Arg::Bool(value) => UOp::const_bool(*value, node.dtype(), node.device()),
        _ => panic!("constant folding produced non-literal arg"),
    }
}

/// Algebraic simplification rules for index arithmetic.
///
/// These run as a separate pass after rangeify to clean up redundant
/// operations in the generated index expressions. The rules are:
///
/// - **Constant folding**: `3 + 4` → `7`, `2 * 3` → `6`
/// - **Additive identity**: `x + 0` → `x` (common from broadcast dims with stride 0)
/// - **Multiplicative identity**: `x * 1` → `x` (common from innermost-dim stride)
/// - **Multiplicative zero**: `x * 0` → `0` (dead index arithmetic)
///
/// Each commutative rule has two arms to handle both operand orderings.
#[must_use]
pub fn symbolic_simple(node: &UOp) -> Option<UOp> {
    match (node.op(), node.srcs()) {
        // const + const → const
        (Op::Add, [a, b]) if a.is_const() && b.is_const() => {
            fold_binary(Op::Add, a.arg(), b.arg()).as_ref().map(|arg| const_from_arg(a, arg))
        }
        // const * const → const
        (Op::Mul, [a, b]) if a.is_const() && b.is_const() => {
            fold_binary(Op::Mul, a.arg(), b.arg()).as_ref().map(|arg| const_from_arg(a, arg))
        }
        // x + 0 → x
        (Op::Add, [x, y]) if y.is_zero() => Some(x.clone()),
        (Op::Add, [x, y]) if x.is_zero() => Some(y.clone()),
        // x * 1 → x
        (Op::Mul, [x, y]) if y.is_one() => Some(x.clone()),
        (Op::Mul, [x, y]) if x.is_one() => Some(y.clone()),
        // x * 0 → 0
        (Op::Mul, [_, y]) if y.is_zero() => Some(y.clone()),
        (Op::Mul, [x, _]) if x.is_zero() => Some(x.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceId;
    use crate::dtype::DType;

    #[test]
    fn test_rewrite_add_zero_eliminated() {
        let x = UOp::const_float(5.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(x, zero);
        let result = graph_rewrite(&sum, &symbolic_simple, "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_zero_plus_x_eliminated() {
        let x = UOp::const_float(5.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(zero, x);
        let result = graph_rewrite(&sum, &symbolic_simple, "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_constant_folding_add() {
        let two = UOp::const_float(2.0, DType::F32, DeviceId::Cpu);
        let three = UOp::const_float(3.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(two, three);
        let result = graph_rewrite(&sum, &symbolic_simple, "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_fixed_point() {
        let x = UOp::const_float(7.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let one = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(x, zero);
        let prod = UOp::mul(sum, one);
        let result = graph_rewrite(&prod, &symbolic_simple, "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(7.0));
    }

    #[test]
    fn test_rewrite_no_match_unchanged() {
        let x = UOp::const_float(3.0, DType::F32, DeviceId::Cpu);
        let y = UOp::const_float(4.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(x, y);
        let result = graph_rewrite(&sum, &|_| None, "test");
        assert_eq!(result, sum);
    }

    #[test]
    fn test_rewrite_mul_zero() {
        let x = UOp::const_float(42.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let prod = UOp::mul(x, zero);
        let result = graph_rewrite(&prod, &symbolic_simple, "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(0.0));
    }
}
