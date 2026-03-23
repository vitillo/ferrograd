//! # Graph Rewriting — pattern-match-and-replace on `UOp` graphs
//!
//! The universal optimization strategy: define patterns that match subgraphs,
//! and replacement functions that produce simplified equivalents. A fixed-point
//! loop applies all rules bottom-up until no more fire.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/uop/ops.py` — `UPat`, `PatternMatcher`, `graph_rewrite`.
//! `tinygrad/uop/symbolic.py` — algebraic simplification rules.

use std::collections::HashMap;

use crate::uop::{Arg, Op, UOpGraph, UOpId};

// ── Captures ────────────────────────────────────────────────────────────────

/// Named bindings from a successful pattern match.
///
/// When a `UPat` with a name matches a node, that node's `UOpId` is stored
/// here. The replacement function reads these to build the rewritten subgraph.
#[derive(Clone)]
pub struct Captures(HashMap<String, UOpId>);

impl Captures {
    #[must_use]
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    fn insert(&mut self, name: String, id: UOpId) {
        self.0.insert(name, id);
    }

    fn get_existing(&self, name: &str) -> Option<UOpId> {
        self.0.get(name).copied()
    }

    /// Look up a capture by name.
    ///
    /// # Panics
    ///
    /// Panics if `name` was not captured during pattern matching.
    #[must_use]
    pub fn get(&self, name: &str) -> UOpId {
        self.0
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("capture '{name}' not found"))
    }
}

// ── Permutations ────────────────────────────────────────────────────────────

/// Generates all permutations of `[0, 1, ..., n-1]` using Heap's algorithm.
struct Permutations {
    n: usize,
    state: Vec<usize>,
    indices: Vec<usize>,
    i: usize,
    started: bool,
}

impl Permutations {
    fn new(n: usize) -> Self {
        Self {
            n,
            state: vec![0; n],
            indices: (0..n).collect(),
            i: 0,
            started: false,
        }
    }

    fn next(&mut self) -> Option<&[usize]> {
        if !self.started {
            self.started = true;
            return Some(&self.indices);
        }
        while self.i < self.n {
            if self.state[self.i] < self.i {
                if self.i.is_multiple_of(2) {
                    self.indices.swap(0, self.i);
                } else {
                    self.indices.swap(self.state[self.i], self.i);
                }
                self.state[self.i] += 1;
                self.i = 0;
                return Some(&self.indices);
            }
            self.state[self.i] = 0;
            self.i += 1;
        }
        None
    }
}

// ── UPat ────────────────────────────────────────────────────────────────────

/// A pattern that matches `UOp` nodes in the graph.
///
/// Maps to tinygrad's `UPat`. Each field constrains one aspect of the node;
/// `None` means "match anything." Named captures let the replacement
/// function refer to matched subtrees.
///
/// For binary ops, set `commutative = true` to automatically try both
/// source orderings (like tinygrad's list-based permutation matching).
pub struct UPat {
    /// Operations to match. `None` = any op.
    pub op: Option<Vec<Op>>,
    /// Capture name. Same name appearing twice in a pattern tree means
    /// both positions must match the same `UOpId`.
    pub name: Option<String>,
    /// Argument to match. `None` = any argument.
    pub arg: Option<Arg>,
    /// Source (children) patterns. `None` = any children.
    pub src: Option<Vec<UPat>>,
    /// Try both source orderings for 2-source patterns.
    pub commutative: bool,
}

impl UPat {
    /// Match any node, capture it under `name`.
    #[must_use]
    pub fn var(name: &str) -> Self {
        Self {
            op: None,
            name: Some(name.to_string()),
            arg: None,
            src: None,
            commutative: false,
        }
    }

    /// Match a `Const` node with a specific argument value.
    #[must_use]
    pub fn cst(arg: Arg) -> Self {
        Self {
            op: Some(vec![Op::Const]),
            name: None,
            arg: Some(arg),
            src: None,
            commutative: false,
        }
    }

    /// Match a `Const` node with any value, capture it under `name`.
    #[must_use]
    pub fn any_const(name: &str) -> Self {
        Self {
            op: Some(vec![Op::Const]),
            name: Some(name.to_string()),
            arg: None,
            src: None,
            commutative: false,
        }
    }

    /// Match a specific op with the given source patterns.
    #[must_use]
    pub fn op(op: Op, src: Vec<Self>) -> Self {
        Self {
            op: Some(vec![op]),
            name: None,
            arg: None,
            src: Some(src),
            commutative: false,
        }
    }

    /// Match a commutative binary op — tries both source orderings.
    #[must_use]
    pub fn comm(op: Op, src: Vec<Self>) -> Self {
        Self {
            op: Some(vec![op]),
            name: None,
            arg: None,
            src: Some(src),
            commutative: true,
        }
    }

    /// Try to match this pattern against a node in the graph.
    pub fn matches(&self, graph: &UOpGraph, id: UOpId, captures: &mut Captures) -> bool {
        let node = graph.get(id);

        if let Some(ops) = &self.op {
            if !ops.contains(&node.op) {
                return false;
            }
        }

        if let Some(expected) = &self.arg {
            if *expected != node.arg {
                return false;
            }
        }

        // If this name was already captured, it must be the same node.
        if let Some(name) = &self.name {
            if let Some(existing) = captures.get_existing(name) {
                if existing != id {
                    return false;
                }
            }
        }

        let matched = match &self.src {
            None => true,
            Some(pats) => {
                let srcs = node.srcs.clone();
                if srcs.len() != pats.len() {
                    return false;
                }

                if self.commutative {
                    let snapshot = captures.clone();
                    let mut perm = Permutations::new(srcs.len());
                    let mut found = false;
                    while let Some(order) = perm.next() {
                        *captures = snapshot.clone();
                        let permuted: Vec<UOpId> = order.iter().map(|&i| srcs[i]).collect();
                        if Self::match_srcs(graph, &permuted, pats, captures) {
                            found = true;
                            break;
                        }
                    }
                    found
                } else {
                    Self::match_srcs(graph, &srcs, pats, captures)
                }
            }
        };

        // Only record the capture after children match successfully.
        if matched {
            if let Some(name) = &self.name {
                captures.insert(name.clone(), id);
            }
        }

        matched
    }

    fn match_srcs(
        graph: &UOpGraph,
        srcs: &[UOpId],
        pats: &[UPat],
        captures: &mut Captures,
    ) -> bool {
        for (src_id, pat) in srcs.iter().zip(pats.iter()) {
            if !pat.matches(graph, *src_id, captures) {
                return false;
            }
        }
        true
    }
}

// ── PatternMatcher ──────────────────────────────────────────────────────────

/// Rewrite function: receives captures and mutable graph, returns replacement.
pub type RewriteFn = Box<dyn Fn(&Captures, &mut UOpGraph) -> Option<UOpId>>;

/// A collection of rewrite rules, dispatched by `Op`.
pub struct PatternMatcher {
    rules: HashMap<Op, Vec<Rule>>,
}

struct Rule {
    pattern: UPat,
    rewrite: RewriteFn,
}

impl PatternMatcher {
    /// Build from a list of `(pattern, rewrite_fn)` pairs.
    ///
    /// # Panics
    ///
    /// Panics if any pattern has `op = None` or matches multiple ops.
    #[must_use]
    pub fn new(pairs: Vec<(UPat, RewriteFn)>) -> Self {
        let mut rules: HashMap<Op, Vec<Rule>> = HashMap::new();
        for (pat, rewrite) in pairs {
            let ops = pat
                .op
                .as_ref()
                .expect("PatternMatcher rules must have a concrete op");
            assert_eq!(
                ops.len(),
                1,
                "multi-op patterns not yet supported in PatternMatcher"
            );
            let op = ops[0];
            rules.entry(op).or_default().push(Rule {
                pattern: pat,
                rewrite,
            });
        }
        Self { rules }
    }

    /// Try to rewrite a node by applying the first matching rule.
    pub fn rewrite(&self, graph: &mut UOpGraph, id: UOpId) -> Option<UOpId> {
        let op = graph.get(id).op;

        let rules = self.rules.get(&op)?;
        let mut captures = Captures::new();
        for rule in rules {
            captures.clear();
            if rule.pattern.matches(graph, id, &mut captures) {
                if let Some(result) = (rule.rewrite)(&captures, graph) {
                    if result != id {
                        return Some(result);
                    }
                }
            }
        }
        None
    }
}

// ── graph_rewrite ───────────────────────────────────────────────────────────

/// Rewrite a graph bottom-up until no more rules fire (fixed-point).
///
/// Walks in topological order, rebuilds each node with rewritten sources,
/// then tries all matching rules. Repeats until a full pass changes nothing.
pub fn graph_rewrite(graph: &mut UOpGraph, root: UOpId, pm: &PatternMatcher) -> UOpId {
    let mut current_root = root;

    loop {
        let order = graph.toposort(current_root);
        let mut replace: HashMap<UOpId, UOpId> = HashMap::new();
        let mut changed = false;

        for &id in &order {
            let (op, dtype, srcs, arg) = {
                let node = graph.get(id);
                (node.op, node.dtype, node.srcs.clone(), node.arg.clone())
            };

            let new_srcs: Vec<UOpId> = srcs
                .iter()
                .map(|s| replace.get(s).copied().unwrap_or(*s))
                .collect();

            let rebuilt = if new_srcs == srcs {
                id
            } else {
                changed = true;
                graph.add(op, dtype, new_srcs, arg)
            };

            if let Some(replacement) = pm.rewrite(graph, rebuilt) {
                replace.insert(id, replacement);
                changed = true;
            } else {
                replace.insert(id, rebuilt);
            }
        }

        current_root = replace.get(&current_root).copied().unwrap_or(current_root);

        if !changed {
            return current_root;
        }
    }
}

// ── Starter rules ───────────────────────────────────────────────────────────

/// Evaluate a binary ALU op on two constant arguments.
fn fold_binary(op: Op, a: &Arg, b: &Arg) -> Option<Arg> {
    match (op, a, b) {
        (Op::Add, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x + y)),
        (Op::Add, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x + y)),
        (Op::Mul, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x * y)),
        (Op::Mul, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x * y)),
        _ => None,
    }
}

/// Starter rules: constant folding and algebraic identities.
///
/// - `const + const` → folded constant
/// - `const * const` → folded constant
/// - `x + 0` → `x`
/// - `x * 1` → `x`
/// - `x * 0` → `0`
#[must_use]
pub fn symbolic_simple() -> PatternMatcher {
    PatternMatcher::new(vec![
        // const + const → const
        (
            UPat::comm(Op::Add, vec![UPat::any_const("a"), UPat::any_const("b")]),
            Box::new(|caps, g| {
                let (dtype, a_arg, b_arg) = {
                    let a = g.get(caps.get("a"));
                    let b = g.get(caps.get("b"));
                    (a.dtype, a.arg.clone(), b.arg.clone())
                };
                let result = fold_binary(Op::Add, &a_arg, &b_arg)?;
                Some(g.add(Op::Const, dtype, vec![], result))
            }),
        ),
        // const * const → const
        (
            UPat::comm(Op::Mul, vec![UPat::any_const("a"), UPat::any_const("b")]),
            Box::new(|caps, g| {
                let (dtype, a_arg, b_arg) = {
                    let a = g.get(caps.get("a"));
                    let b = g.get(caps.get("b"));
                    (a.dtype, a.arg.clone(), b.arg.clone())
                };
                let result = fold_binary(Op::Mul, &a_arg, &b_arg)?;
                Some(g.add(Op::Const, dtype, vec![], result))
            }),
        ),
        // x + 0 → x
        (
            UPat::comm(Op::Add, vec![UPat::var("x"), UPat::cst(Arg::Float(0.0))]),
            Box::new(|caps, _g| Some(caps.get("x"))),
        ),
        (
            UPat::comm(Op::Add, vec![UPat::var("x"), UPat::cst(Arg::Int(0))]),
            Box::new(|caps, _g| Some(caps.get("x"))),
        ),
        // x * 1 → x
        (
            UPat::comm(Op::Mul, vec![UPat::var("x"), UPat::cst(Arg::Float(1.0))]),
            Box::new(|caps, _g| Some(caps.get("x"))),
        ),
        (
            UPat::comm(Op::Mul, vec![UPat::var("x"), UPat::cst(Arg::Int(1))]),
            Box::new(|caps, _g| Some(caps.get("x"))),
        ),
        // x * 0 → 0
        (
            UPat::comm(Op::Mul, vec![UPat::var("_x"), UPat::cst(Arg::Float(0.0))]),
            Box::new(|caps, g| {
                let dtype = g.get(caps.get("_x")).dtype;
                Some(g.add(Op::Const, dtype, vec![], Arg::Float(0.0)))
            }),
        ),
        (
            UPat::comm(Op::Mul, vec![UPat::var("_x"), UPat::cst(Arg::Int(0))]),
            Box::new(|caps, g| {
                let dtype = g.get(caps.get("_x")).dtype;
                Some(g.add(Op::Const, dtype, vec![], Arg::Int(0)))
            }),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::uop::UOpGraph;

    #[test]
    fn test_pattern_match_var_captures_any_node() {
        // Arrange
        let mut g = UOpGraph::new();
        let a = g.const_float(42.0, DType::F32);

        // Act
        let pat = UPat::var("x");
        let mut caps = Captures::new();
        let matched = pat.matches(&g, a, &mut caps);

        // Assert
        assert!(matched);
        assert_eq!(caps.get("x"), a);
    }

    #[test]
    fn test_pattern_match_const_exact() {
        // Arrange
        let mut g = UOpGraph::new();
        let zero = g.const_float(0.0, DType::F32);
        let one = g.const_float(1.0, DType::F32);

        // Act/Assert
        let pat = UPat::cst(Arg::Float(0.0));
        assert!(pat.matches(&g, zero, &mut Captures::new()));
        assert!(!pat.matches(&g, one, &mut Captures::new()));
    }

    #[test]
    fn test_pattern_match_same_name_must_bind_same_node() {
        // Arrange
        let mut g = UOpGraph::new();
        let a = g.const_float(1.0, DType::F32);
        let b = g.const_float(2.0, DType::F32);
        let sum_same = g.add_op(a, a);
        let sum_diff = g.add_op(a, b);

        let pat = UPat::op(Op::Add, vec![UPat::var("x"), UPat::var("x")]);

        // Act/Assert
        assert!(pat.matches(&g, sum_same, &mut Captures::new()));
        assert!(!pat.matches(&g, sum_diff, &mut Captures::new()));
    }

    #[test]
    fn test_commutative_pattern_matches_both_orderings() {
        // Arrange — pattern is comm(Add, [var("x"), cst(0)]) which should match 0+x too
        let mut g = UOpGraph::new();
        let x = g.const_float(5.0, DType::F32);
        let zero = g.const_float(0.0, DType::F32);
        let x_plus_zero = g.add_op(x, zero);
        let zero_plus_x = g.add_op(zero, x);

        let pat = UPat::comm(Op::Add, vec![UPat::var("x"), UPat::cst(Arg::Float(0.0))]);

        // Act/Assert — both orderings match
        let mut caps = Captures::new();
        assert!(pat.matches(&g, x_plus_zero, &mut caps));
        assert_eq!(caps.get("x"), x);

        let mut caps = Captures::new();
        assert!(pat.matches(&g, zero_plus_x, &mut caps));
        assert_eq!(caps.get("x"), x);
    }

    #[test]
    fn test_rewrite_add_zero_eliminated() {
        // Arrange
        let mut g = UOpGraph::new();
        let x = g.const_float(5.0, DType::F32);
        let zero = g.const_float(0.0, DType::F32);
        let sum = g.add_op(x, zero);

        let pm = symbolic_simple();

        // Act
        let result = graph_rewrite(&mut g, sum, &pm);

        // Assert
        assert_eq!(result, x);
    }

    #[test]
    fn test_rewrite_zero_plus_x_eliminated() {
        // Arrange — commutative: 0 + x should also simplify
        let mut g = UOpGraph::new();
        let x = g.const_float(5.0, DType::F32);
        let zero = g.const_float(0.0, DType::F32);
        let sum = g.add_op(zero, x);

        let pm = symbolic_simple();

        // Act
        let result = graph_rewrite(&mut g, sum, &pm);

        // Assert
        assert_eq!(result, x);
    }

    #[test]
    fn test_rewrite_mul_one_eliminated() {
        // Arrange
        let mut g = UOpGraph::new();
        let x = g.const_float(5.0, DType::F32);
        let one = g.const_float(1.0, DType::F32);
        let prod = g.mul(x, one);

        let pm = symbolic_simple();

        // Act
        let result = graph_rewrite(&mut g, prod, &pm);

        // Assert
        assert_eq!(result, x);
    }

    #[test]
    fn test_rewrite_constant_folding_add() {
        // Arrange
        let mut g = UOpGraph::new();
        let two = g.const_float(2.0, DType::F32);
        let three = g.const_float(3.0, DType::F32);
        let sum = g.add_op(two, three);

        let pm = symbolic_simple();

        // Act
        let result = graph_rewrite(&mut g, sum, &pm);

        // Assert
        let node = g.get(result);
        assert_eq!(node.op, Op::Const);
        assert_eq!(node.arg, Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_constant_folding_mul() {
        // Arrange
        let mut g = UOpGraph::new();
        let three = g.const_float(3.0, DType::F32);
        let four = g.const_float(4.0, DType::F32);
        let prod = g.mul(three, four);

        let pm = symbolic_simple();

        // Act
        let result = graph_rewrite(&mut g, prod, &pm);

        // Assert
        let node = g.get(result);
        assert_eq!(node.op, Op::Const);
        assert_eq!(node.arg, Arg::Float(12.0));
    }

    #[test]
    fn test_rewrite_fixed_point_add_zero_then_mul_one() {
        // Arrange — (x + 0) * 1 should simplify to x
        let mut g = UOpGraph::new();
        let x = g.const_float(7.0, DType::F32);
        let zero = g.const_float(0.0, DType::F32);
        let one = g.const_float(1.0, DType::F32);
        let sum = g.add_op(x, zero);
        let prod = g.mul(sum, one);

        let pm = symbolic_simple();

        // Act
        let result = graph_rewrite(&mut g, prod, &pm);

        // Assert
        assert_eq!(result, x);
    }

    #[test]
    fn test_rewrite_no_match_leaves_graph_unchanged() {
        // Arrange
        let mut g = UOpGraph::new();
        let x = g.const_float(3.0, DType::F32);
        let y = g.const_float(4.0, DType::F32);
        let sum = g.add_op(x, y);

        let pm = PatternMatcher::new(vec![]);

        // Act
        let result = graph_rewrite(&mut g, sum, &pm);

        // Assert
        assert_eq!(result, sum);
    }

    #[test]
    fn test_rewrite_mul_zero_produces_zero() {
        // Arrange
        let mut g = UOpGraph::new();
        let x = g.const_float(42.0, DType::F32);
        let zero = g.const_float(0.0, DType::F32);
        let prod = g.mul(x, zero);

        let pm = symbolic_simple();

        // Act
        let result = graph_rewrite(&mut g, prod, &pm);

        // Assert
        let node = g.get(result);
        assert_eq!(node.op, Op::Const);
        assert_eq!(node.arg, Arg::Float(0.0));
    }
}
