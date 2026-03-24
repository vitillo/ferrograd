//! # Graph Rewriting — pattern-match-and-replace on `UOp` graphs
//!
//! Define patterns that match subgraphs and replacement functions that
//! produce simplified equivalents. A fixed-point loop applies all rules
//! bottom-up until no more fire.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/uop/ops.py` — `UPat`, `PatternMatcher`, `graph_rewrite`.

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::uop::{Arg, Op, UOp};

static DEBUG: LazyLock<u8> = LazyLock::new(|| {
    std::env::var("DEBUG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
});

// ── Captures ────────────────────────────────────────────────────────────────

/// Named bindings from a successful pattern match.
#[derive(Clone)]
pub struct Captures(HashMap<String, UOp>);

impl Captures {
    #[must_use]
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    fn insert(&mut self, name: String, uop: UOp) {
        self.0.insert(name, uop);
    }

    fn get_existing(&self, name: &str) -> Option<&UOp> {
        self.0.get(name)
    }

    /// Look up a capture by name.
    ///
    /// # Panics
    ///
    /// Panics if `name` was not captured during pattern matching.
    #[must_use]
    pub fn get(&self, name: &str) -> UOp {
        self.0
            .get(name)
            .cloned()
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

/// A pattern that matches `UOp` nodes.
pub struct UPat {
    /// Operations to match. `None` = any op.
    pub op: Option<Vec<Op>>,
    /// Capture name.
    pub name: Option<String>,
    /// Argument to match. `None` = any argument.
    pub arg: Option<Arg>,
    /// Source patterns. `None` = any children.
    pub src: Option<Vec<UPat>>,
    /// Try all source permutations for commutative ops.
    pub commutative: bool,
}

impl UPat {
    /// Match any node, capture it under `name`.
    #[must_use]
    pub fn var(name: &str) -> Self {
        Self { op: None, name: Some(name.to_string()), arg: None, src: None, commutative: false }
    }

    /// Match a `Const` node with a specific argument value.
    #[must_use]
    pub fn cst(arg: Arg) -> Self {
        Self { op: Some(vec![Op::Const]), name: None, arg: Some(arg), src: None, commutative: false }
    }

    /// Match a `Const` node with any value, capture it.
    #[must_use]
    pub fn any_const(name: &str) -> Self {
        Self { op: Some(vec![Op::Const]), name: Some(name.to_string()), arg: None, src: None, commutative: false }
    }

    /// Match a specific op with source patterns.
    #[must_use]
    pub fn op(op: Op, src: Vec<Self>) -> Self {
        Self { op: Some(vec![op]), name: None, arg: None, src: Some(src), commutative: false }
    }

    /// Match a commutative binary op — tries all source permutations.
    #[must_use]
    pub fn comm(op: Op, src: Vec<Self>) -> Self {
        Self { op: Some(vec![op]), name: None, arg: None, src: Some(src), commutative: true }
    }

    /// Try to match this pattern against a `UOp`.
    pub fn matches(&self, uop: &UOp, captures: &mut Captures) -> bool {
        if let Some(ops) = &self.op {
            if !ops.contains(&uop.op()) {
                return false;
            }
        }

        if let Some(expected) = &self.arg {
            if expected != uop.arg() {
                return false;
            }
        }

        if let Some(name) = &self.name {
            if let Some(existing) = captures.get_existing(name) {
                if existing != uop {
                    return false;
                }
            }
        }

        let matched = match &self.src {
            None => true,
            Some(pats) => {
                if uop.srcs().len() != pats.len() {
                    return false;
                }
                if self.commutative {
                    let snapshot = captures.clone();
                    let mut perm = Permutations::new(uop.srcs().len());
                    let mut found = false;
                    while let Some(order) = perm.next() {
                        *captures = snapshot.clone();
                        let permuted: Vec<&UOp> = order.iter().map(|&i| &uop.srcs()[i]).collect();
                        if Self::match_srcs(&permuted, pats, captures) {
                            found = true;
                            break;
                        }
                    }
                    found
                } else {
                    let srcs: Vec<&UOp> = uop.srcs().iter().collect();
                    Self::match_srcs(&srcs, pats, captures)
                }
            }
        };

        if matched {
            if let Some(name) = &self.name {
                captures.insert(name.clone(), uop.clone());
            }
        }

        matched
    }

    fn match_srcs(srcs: &[&UOp], pats: &[UPat], captures: &mut Captures) -> bool {
        for (src, pat) in srcs.iter().zip(pats.iter()) {
            if !pat.matches(src, captures) {
                return false;
            }
        }
        true
    }
}

// ── PatternMatcher ──────────────────────────────────────────────────────────

/// Rewrite function: receives captures, returns replacement.
pub type RewriteFn = Box<dyn Fn(&Captures) -> Option<UOp>>;

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
            let ops = pat.op.as_ref().expect("rules must have a concrete op");
            assert_eq!(ops.len(), 1, "multi-op patterns not yet supported");
            let op = ops[0];
            rules.entry(op).or_default().push(Rule { pattern: pat, rewrite });
        }
        Self { rules }
    }

    /// Try to rewrite a node by applying the first matching rule.
    #[must_use]
    pub fn rewrite(&self, uop: &UOp) -> Option<UOp> {
        let rules = self.rules.get(&uop.op())?;
        let mut captures = Captures::new();
        for rule in rules {
            captures.clear();
            if rule.pattern.matches(uop, &mut captures) {
                if let Some(result) = (rule.rewrite)(&captures) {
                    if result != *uop {
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
/// The `name` parameter identifies the pass in debug output (like tinygrad's
/// `name=` on `graph_rewrite`). When `DEBUG >= 3`, prints the graph before
/// and after the rewrite.
#[must_use]
pub fn graph_rewrite(root: &UOp, pm: &PatternMatcher, name: &str) -> UOp {
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
            // Rebuild with replaced children.
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

            if let Some(replacement) = pm.rewrite(&rebuilt) {
                replace.insert(node.clone(), replacement);
                changed = true;
            } else {
                replace.insert(node.clone(), rebuilt);
            }
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

// ── Starter rules ───────────────────────────────────────────────────────────

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
#[must_use]
pub fn symbolic_simple() -> PatternMatcher {
    PatternMatcher::new(vec![
        // const + const → const
        (
            UPat::comm(Op::Add, vec![UPat::any_const("a"), UPat::any_const("b")]),
            Box::new(|caps| {
                let a = caps.get("a");
                let b = caps.get("b");
                let result = fold_binary(Op::Add, a.arg(), b.arg())?;
                Some(UOp::new(Op::Const, a.dtype(), vec![], result))
            }),
        ),
        // const * const → const
        (
            UPat::comm(Op::Mul, vec![UPat::any_const("a"), UPat::any_const("b")]),
            Box::new(|caps| {
                let a = caps.get("a");
                let b = caps.get("b");
                let result = fold_binary(Op::Mul, a.arg(), b.arg())?;
                Some(UOp::new(Op::Const, a.dtype(), vec![], result))
            }),
        ),
        // x + 0 → x
        (
            UPat::comm(Op::Add, vec![UPat::var("x"), UPat::cst(Arg::Float(0.0))]),
            Box::new(|caps| Some(caps.get("x"))),
        ),
        (
            UPat::comm(Op::Add, vec![UPat::var("x"), UPat::cst(Arg::Int(0))]),
            Box::new(|caps| Some(caps.get("x"))),
        ),
        // x * 1 → x
        (
            UPat::comm(Op::Mul, vec![UPat::var("x"), UPat::cst(Arg::Float(1.0))]),
            Box::new(|caps| Some(caps.get("x"))),
        ),
        (
            UPat::comm(Op::Mul, vec![UPat::var("x"), UPat::cst(Arg::Int(1))]),
            Box::new(|caps| Some(caps.get("x"))),
        ),
        // x * 0 → 0
        (
            UPat::comm(Op::Mul, vec![UPat::var("x"), UPat::cst(Arg::Float(0.0))]),
            Box::new(|caps| {
                let dtype = caps.get("x").dtype();
                Some(UOp::new(Op::Const, dtype, vec![], Arg::Float(0.0)))
            }),
        ),
        (
            UPat::comm(Op::Mul, vec![UPat::var("x"), UPat::cst(Arg::Int(0))]),
            Box::new(|caps| {
                let dtype = caps.get("x").dtype();
                Some(UOp::new(Op::Const, dtype, vec![], Arg::Int(0)))
            }),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    #[test]
    fn test_pattern_match_var_captures_any_node() {
        let a = UOp::const_float(42.0, DType::F32);
        let pat = UPat::var("x");
        let mut caps = Captures::new();
        assert!(pat.matches(&a, &mut caps));
        assert_eq!(caps.get("x"), a);
    }

    #[test]
    fn test_pattern_match_const_exact() {
        let zero = UOp::const_float(0.0, DType::F32);
        let one = UOp::const_float(1.0, DType::F32);
        let pat = UPat::cst(Arg::Float(0.0));
        assert!(pat.matches(&zero, &mut Captures::new()));
        assert!(!pat.matches(&one, &mut Captures::new()));
    }

    #[test]
    fn test_pattern_match_same_name_must_bind_same_node() {
        let a = UOp::const_float(1.0, DType::F32);
        let b = UOp::const_float(2.0, DType::F32);
        let sum_same = UOp::new(Op::Add, DType::F32, vec![a.clone(), a.clone()], Arg::None);
        let sum_diff = UOp::new(Op::Add, DType::F32, vec![a, b], Arg::None);

        let pat = UPat::op(Op::Add, vec![UPat::var("x"), UPat::var("x")]);
        assert!(pat.matches(&sum_same, &mut Captures::new()));
        assert!(!pat.matches(&sum_diff, &mut Captures::new()));
    }

    #[test]
    fn test_commutative_pattern_matches_both_orderings() {
        let x = UOp::const_float(5.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let x_plus_zero = UOp::new(Op::Add, DType::F32, vec![x.clone(), zero.clone()], Arg::None);
        let zero_plus_x = UOp::new(Op::Add, DType::F32, vec![zero, x.clone()], Arg::None);

        let pat = UPat::comm(Op::Add, vec![UPat::var("x"), UPat::cst(Arg::Float(0.0))]);

        let mut caps = Captures::new();
        assert!(pat.matches(&x_plus_zero, &mut caps));
        assert_eq!(caps.get("x"), x);

        let mut caps = Captures::new();
        assert!(pat.matches(&zero_plus_x, &mut caps));
        assert_eq!(caps.get("x"), x);
    }

    #[test]
    fn test_rewrite_add_zero_eliminated() {
        let x = UOp::const_float(5.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![x, zero], Arg::None);

        let result = graph_rewrite(&sum, &symbolic_simple(), "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_constant_folding_add() {
        let two = UOp::const_float(2.0, DType::F32);
        let three = UOp::const_float(3.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![two, three], Arg::None);

        let result = graph_rewrite(&sum, &symbolic_simple(), "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn test_rewrite_fixed_point() {
        let x = UOp::const_float(7.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let one = UOp::const_float(1.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![x, zero], Arg::None);
        let prod = UOp::new(Op::Mul, DType::F32, vec![sum, one], Arg::None);

        let result = graph_rewrite(&prod, &symbolic_simple(), "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(7.0));
    }

    #[test]
    fn test_rewrite_no_match_unchanged() {
        let x = UOp::const_float(3.0, DType::F32);
        let y = UOp::const_float(4.0, DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![x, y], Arg::None);

        let pm = PatternMatcher::new(vec![]);
        let result = graph_rewrite(&sum, &pm, "test");
        assert_eq!(result, sum);
    }

    #[test]
    fn test_rewrite_mul_zero() {
        let x = UOp::const_float(42.0, DType::F32);
        let zero = UOp::const_float(0.0, DType::F32);
        let prod = UOp::new(Op::Mul, DType::F32, vec![x, zero], Arg::None);

        let result = graph_rewrite(&prod, &symbolic_simple(), "test");
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(0.0));
    }
}
