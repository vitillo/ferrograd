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
//!   (in the `optimize` module)
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

use crate::dtype::DType;
use crate::uop::{Arg, Op, UOp};

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

// ── shared structural flattening rules ──────────────────────────────────────

/// Flattens `End(range, Sink(body0, body1))` into `End(range, body0, body1)`.
///
/// Both the expander and devectorizer can produce `End` nodes whose body slot
/// wraps multiple effects in a temporary `Sink`. This rule inlines the `Sink`
/// children directly into the `End`, keeping its source list uniform for
/// downstream passes (linearizer, renderer).
pub(crate) fn flatten_end_bodies(node: &UOp) -> Option<UOp> {
    if node.op() != Op::End || node.srcs().len() < 2 {
        return None;
    }

    let mut changed = false;
    let mut srcs = vec![node.srcs()[0].clone()];
    for body in &node.srcs()[1..] {
        if body.op() == Op::Sink {
            changed = true;
            srcs.extend(body.srcs().iter().cloned());
            continue;
        }
        srcs.push(body.clone());
    }
    changed.then(|| UOp::new(Op::End, DType::Void, srcs, Arg::None))
}

/// Flattens `Sink(Sink(..), ..)` into a single `Sink(..)`.
///
/// Inlining effects during expansion or store scalarization can produce
/// nested sinks. This rule collapses them into one flat effect list.
pub(crate) fn flatten_nested_sinks(node: &UOp) -> Option<UOp> {
    if node.op() != Op::Sink {
        return None;
    }

    let mut changed = false;
    let mut srcs = Vec::new();
    for src in node.srcs() {
        if src.op() == Op::Sink {
            changed = true;
            srcs.extend(src.srcs().iter().cloned());
            continue;
        }
        srcs.push(src.clone());
    }
    changed.then(|| UOp::sink(srcs))
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
