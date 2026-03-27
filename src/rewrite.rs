//! # Graph Rewriting — fixed-point rewrite engine for `UOp` graphs
//!
//! This is the compiler's workhorse: a general-purpose engine for
//! transforming IR graphs. You define rewrite rules as plain functions
//! (`&UOp → Option<UOp>`), and `graph_rewrite` applies them bottom-up
//! in a fixed-point loop until no more rules fire.
//!
//! Almost every transformation in the compiler is expressed as rewrite rules:
//! - **Scheduling**: `Buffer → Param` (in [`crate::schedule`])
//! - **Rangeify**: Store → loops, Index pushing, kernel `Reduce` creation
//!   (in [`crate::schedule::rangeify`])
//! - **Optimization**: symbolic simplification and loop unrolling
//!   (in [`crate::optimize`])
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

use crate::uop::UOp;

// ── graph_rewrite ───────────────────────────────────────────────────────────

/// Rewrite a graph bottom-up until no more rules fire (fixed-point).
///
/// `rewrite` is called on each node after its children have been updated.
/// If it returns `Some(replacement)`, the node is replaced and another
/// iteration begins. The loop terminates when a full pass produces no
/// changes.
///
#[must_use]
pub fn graph_rewrite(root: &UOp, rewrite: &mut dyn FnMut(&UOp) -> Option<UOp>) -> UOp {
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

            let srcs_changed = node
                .srcs()
                .iter()
                .zip(&new_srcs)
                .any(|(old, new)| old != new);
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
            return current;
        }
    }
}

/// Rebuild `root` while substituting any node found in `replacements`.
///
/// This is a one-pass bottom-up rebuild, not a fixed-point rewrite: every node
/// is visited once in dependency order and either replaced directly or rebuilt
/// from already-substituted sources.
#[must_use]
pub(crate) fn substitute_with_map(root: &UOp, replacements: &HashMap<UOp, UOp>) -> UOp {
    let order = root.toposort();
    let mut substituted: HashMap<UOp, UOp> = HashMap::new();

    for node in &order {
        if let Some(replacement) = replacements.get(node) {
            substituted.insert(node.clone(), replacement.clone());
            continue;
        }

        let new_srcs: Vec<UOp> = node
            .srcs()
            .iter()
            .map(|src| substituted.get(src).cloned().unwrap_or_else(|| src.clone()))
            .collect();
        let srcs_changed = node
            .srcs()
            .iter()
            .zip(&new_srcs)
            .any(|(old, new)| old != new);
        let rebuilt = if srcs_changed {
            UOp::new(node.op(), node.dtype(), new_srcs, node.arg().clone())
        } else {
            node.clone()
        };
        substituted.insert(node.clone(), rebuilt);
    }

    substituted
        .get(root)
        .cloned()
        .unwrap_or_else(|| root.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_no_match_unchanged() {
        let sum = UOp::sink(vec![]);
        let result = graph_rewrite(&sum, &mut |_| None);
        assert_eq!(result, sum);
    }
}
