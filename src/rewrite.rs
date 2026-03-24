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

fn fold_binary(op: Op, a: &Arg, b: &Arg) -> Option<Arg> {
    match (op, a, b) {
        (Op::Add, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x + y)),
        (Op::Add, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x + y)),
        (Op::Mul, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x * y)),
        (Op::Mul, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x * y)),
        _ => None,
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
            fold_binary(Op::Add, a.arg(), b.arg())
                .map(|r| UOp::new(Op::Const, a.dtype(), vec![], r))
        }
        // const * const → const
        (Op::Mul, [a, b]) if a.is_const() && b.is_const() => {
            fold_binary(Op::Mul, a.arg(), b.arg())
                .map(|r| UOp::new(Op::Const, a.dtype(), vec![], r))
        }
        // x + 0 → x
        (Op::Add, [x, y]) if y.is_const_float(0.0) || y.is_const_int(0) => Some(x.clone()),
        (Op::Add, [x, y]) if x.is_const_float(0.0) || x.is_const_int(0) => Some(y.clone()),
        // x * 1 → x
        (Op::Mul, [x, y]) if y.is_const_float(1.0) || y.is_const_int(1) => Some(x.clone()),
        (Op::Mul, [x, y]) if x.is_const_float(1.0) || x.is_const_int(1) => Some(y.clone()),
        // x * 0 → 0
        (Op::Mul, [_, y]) if y.is_const_float(0.0) => Some(y.clone()),
        (Op::Mul, [x, _]) if x.is_const_float(0.0) => Some(x.clone()),
        (Op::Mul, [_, y]) if y.is_const_int(0) => Some(y.clone()),
        (Op::Mul, [x, _]) if x.is_const_int(0) => Some(x.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn test_rewrite_add_zero_eliminated() {
        // Arrange
        let x = UOp::const_float(5.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![x, zero], Arg::None);

        // Act
        let result = graph_rewrite(&sum, &symbolic_simple, "test");

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_zero_plus_x_eliminated() {
        // Arrange
        let x = UOp::const_float(5.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![zero, x], Arg::None);

        // Act
        let result = graph_rewrite(&sum, &symbolic_simple, "test");

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_constant_folding_add() {
        // Arrange
        let two = UOp::const_float(2.0, DType::F32);
        let three = UOp::const_float(3.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![two, three], Arg::None);

        // Act
        let result = graph_rewrite(&sum, &symbolic_simple, "test");

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_fixed_point() {
        // Arrange — (x + 0) * 1 should simplify in two steps
        let x = UOp::const_float(7.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let one = UOp::const_float(1.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![x, zero], Arg::None);
        let prod = UOp::new(Op::Mul, DType::F32, vec![sum, one], Arg::None);

        // Act
        let result = graph_rewrite(&prod, &symbolic_simple, "test");

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(7.0));
    }

    #[test]
    fn test_rewrite_no_match_unchanged() {
        // Arrange
        let x = UOp::const_float(3.0, DType::F32);
        let y = UOp::const_float(4.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![x, y], Arg::None);

        // Act — no-op rewrite function
        let result = graph_rewrite(&sum, &|_| None, "test");

        // Assert
        assert_eq!(result, sum);
    }

    #[test]
    fn test_rewrite_mul_zero() {
        // Arrange
        let x = UOp::const_float(42.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let prod = UOp::new(Op::Mul, DType::F32, vec![x, zero], Arg::None);

        // Act
        let result = graph_rewrite(&prod, &symbolic_simple, "test");

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(0.0));
    }
}
