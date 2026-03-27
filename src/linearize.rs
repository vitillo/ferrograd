//! # Late kernel linearization
//!
//! Turns a lowered kernel DAG into a flat, ordered list of `UOp`s that the
//! renderer can print as straight-line code. This is the last transformation
//! before codegen.
//!
//! ## Why a separate linearization pass?
//!
//! After devectorization the kernel is still a DAG — nodes reference their
//! inputs, but nothing specifies the *order* they should appear in the
//! generated C. A naïve topological sort would work for purely data-flow
//! graphs, but kernels have *control flow*: `Range`/`End` pairs must nest
//! correctly, sibling loops must not interleave, and accumulators must be
//! declared before the loop that updates them.
//!
//! Tinygrad solves this the same way: a late `linearize` pass builds
//! explicit control-flow edges between related nodes and then runs a
//! priority-aware topological sort that respects those edges.
//!
//! ## What the pass does
//!
//! 1. **Discover nesting**: walk the DAG to figure out which `End` nodes
//!    are nested inside which parent `End` (or the root `Sink`).
//! 2. **Inject sibling edges**: if two loops share a parent, add the first
//!    loop's `End` as an extra source on the second loop's `Range`. This
//!    bakes the ordering into the graph itself (matching tinygrad's
//!    `pm_add_control_flow`), so the linearizer needs no side tables.
//! 3. **Priority sort**: assign tinygrad-style priorities (params first,
//!    then loads, then ALU, then stores, then loop open/close) and run a
//!    max-heap topological traversal that pops the highest-priority ready
//!    node at each step.
//!
//! ```text
//!         Sink                        ParamBuffer data0
//!        ╱    ╲                       ParamBuffer data1
//!     End₀    End₁        ──►         for (idx0 …) { … }   // End₀ body
//!      │        │                     for (idx1 …) { … }   // End₁ body
//!    Range₀  Range₁
//! ```
//!
//! ## Tinygrad reference
//!
//! `tinygrad/codegen/linearize.py` — `linearize_uop` builds the same
//! CFG edges and runs the same priority-aware reverse topological sort.

use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::uop::{Arg, Op, UOp};

/// Extract the iteration count from a Range node's bound argument.
/// Used to compute how many times a node executes, which drives scheduling
/// priority -- tinygrad schedules cheaper (fewer iterations) nodes first.
fn range_extent(range: &UOp) -> usize {
    let Some(bound) = range.srcs().first() else {
        return 1;
    };
    match bound.arg() {
        Arg::Int(v) if *v > 0 => usize::try_from(*v).unwrap_or(1),
        _ => 1,
    }
}

/// Compute which Range loops are "active" (open but not yet closed) at a given
/// node. End nodes remove their Range from the active set. A node is inside a
/// loop if that loop's Range is in its active set.
fn active_ranges(node: &UOp, cache: &mut HashMap<UOp, HashSet<UOp>>) -> HashSet<UOp> {
    if let Some(ranges) = cache.get(node) {
        return ranges.clone();
    }

    let mut ranges = HashSet::new();
    for src in node.srcs() {
        ranges.extend(active_ranges(src, cache));
    }
    for ended in node.ended_ranges() {
        if ended.op() == Op::Range {
            ranges.remove(&ended);
            continue;
        }
        for range in active_ranges(&ended, cache) {
            ranges.remove(&range);
        }
    }
    if node.op() == Op::Range {
        ranges.insert(node.clone());
    }

    cache.insert(node.clone(), ranges.clone());
    ranges
}

/// Add a CFG ordering edge: `node` must be scheduled after `predecessor`.
/// Deduplicates to avoid redundant edges in the dependency graph.
fn push_edge(edges: &mut HashMap<UOp, Vec<UOp>>, node: UOp, predecessor: UOp) {
    let node_edges = edges.entry(node).or_default();
    if !node_edges.contains(&predecessor) {
        node_edges.push(predecessor);
    }
}

/// How many times a node executes at runtime -- the product of all enclosing
/// loop extents. Tinygrad uses this as the primary scheduling tiebreaker:
/// nodes that run fewer times (cheaper) are scheduled first, keeping them
/// outside inner loops where possible.
fn run_count(node: &UOp) -> usize {
    active_ranges(node, &mut HashMap::new())
        .iter()
        .fold(1_usize, |acc, range| {
            acc.saturating_mul(range_extent(range))
        })
}

/// Compute tinygrad's multi-level scheduling priority for a node.
/// Returns `(run_count, op_priority, extra, topo_order)` -- sorted ascending,
/// so lower values are scheduled first. Op-level priorities ensure params come
/// before computation, Ends close loops early, Stores happen late, and Ranges
/// open last. The `extra` field orders params by buffer slot for determinism.
fn priority_key(
    node: &UOp,
    order_idx: usize,
) -> (usize, i32, i64, usize) {
    let (priority, extra) = match node.op() {
        Op::ParamBuffer => match node.arg() {
            Arg::ParamBuffer(slot, _) => (-20, i64::try_from(*slot).unwrap_or(i64::MAX)),
            _ => unreachable!("ParamBuffer must carry Arg::ParamBuffer"),
        },
        Op::ParamScalar => match node.arg() {
            Arg::ParamScalar(slot) => (-20, i64::try_from(*slot).unwrap_or(i64::MAX)),
            _ => unreachable!("ParamScalar must carry Arg::ParamScalar"),
        },
        Op::DefineVar => (-19, 0),
        Op::DefineAcc => (-17, 0),
        Op::Load => (-1, 0),
        Op::Store => (1, 0),
        Op::Range => (5, 0),
        Op::End => (-5, 0),
        _ => (0, 0),
    };
    (run_count(node), priority, extra, order_idx)
}

/// Walk the toposorted DAG to discover two things: transitive dependencies for
/// each node, and the loop nesting structure. An End node is "nested" inside
/// another End (or Sink) if its dependencies include the parent's Range -- this
/// means one loop is fully contained within another. The nesting map drives
/// sibling detection: End nodes that share the same parent are siblings.
fn build_deps_and_nesting(order: &[UOp]) -> (HashMap<UOp, HashSet<UOp>>, HashMap<UOp, UOp>) {
    let mut deps: HashMap<UOp, HashSet<UOp>> = HashMap::new();
    let mut nesting: HashMap<UOp, UOp> = HashMap::new();

    for node in order {
        let mut node_deps = HashSet::new();
        for src in node.srcs() {
            if let Some(src_deps) = deps.get(src) {
                node_deps.extend(src_deps.iter().cloned());
            }
        }

        if matches!(node.op(), Op::End | Op::Sink) {
            for dep in node_deps.iter().filter(|dep| dep.op() == Op::End) {
                if nesting.contains_key(dep) {
                    continue;
                }
                let is_nested = if node.op() == Op::Sink {
                    true
                } else {
                    let parent_range = &node.srcs()[0];
                    deps.get(dep)
                        .is_some_and(|dep_deps| dep_deps.contains(parent_range))
                };
                if is_nested {
                    nesting.insert(dep.clone(), node.clone());
                }
            }
        }

        if matches!(node.op(), Op::Range | Op::End) {
            node_deps.insert(node.clone());
        }
        deps.insert(node.clone(), node_deps);
    }

    (deps, nesting)
}

/// Group End nodes by their nesting parent. "Siblings" are End nodes that close
/// loops at the same nesting depth -- e.g., two independent loops inside the
/// same outer loop, or two top-level loops under the Sink. These siblings need
/// sequencing edges so the linearizer doesn't interleave them.
fn build_siblings(order: &[UOp], nesting: &HashMap<UOp, UOp>) -> HashMap<UOp, Vec<UOp>> {
    let mut siblings: HashMap<UOp, Vec<UOp>> = HashMap::new();
    for node in order {
        if let Some(parent) = nesting.get(node) {
            siblings
                .entry(parent.clone())
                .or_default()
                .push(node.clone());
        }
    }
    siblings
}

/// Chain sibling loops so they execute sequentially: each loop's Range depends
/// on the previous sibling's End (or the parent Range for the first child).
/// Without these edges, independent loops could interleave in the linear output,
/// producing invalid code like `for A { for B { } end A; } end B;`. Siblings
/// are sorted by dependency count to respect any existing data-flow ordering.
fn add_sibling_edges(
    edges: &mut HashMap<UOp, Vec<UOp>>,
    deps: &HashMap<UOp, HashSet<UOp>>,
    siblings: HashMap<UOp, Vec<UOp>>,
) {
    for (parent, children) in siblings {
        let mut ordered = children;
        let sibling_counts: HashMap<UOp, usize> = ordered
            .iter()
            .cloned()
            .map(|end| {
                let count = ordered
                    .iter()
                    .filter(|candidate| {
                        deps.get(&end)
                            .is_some_and(|end_deps| end_deps.contains(*candidate))
                    })
                    .count();
                (end, count)
            })
            .collect();
        ordered.sort_by_key(|end| {
            sibling_counts
                .get(end)
                .copied()
                .expect("linearize: missing sibling dependency count")
        });

        if parent.op() == Op::Sink {
            for pair in ordered.windows(2) {
                let prev = &pair[0];
                let next = &pair[1];
                let next_range = next.srcs()[0].clone();
                assert!(
                    !deps
                        .get(prev)
                        .is_some_and(|prev_deps| prev_deps.contains(&next_range)),
                    "linearize: control-flow cycle between sibling loops"
                );
                push_edge(edges, next_range, prev.clone());
            }
            continue;
        }

        let mut predecessor = parent.srcs()[0].clone();
        for end in &ordered {
            let next_range = end.srcs()[0].clone();
            assert!(
                !deps
                    .get(&predecessor)
                    .is_some_and(|pred_deps| pred_deps.contains(&next_range)),
                "linearize: nested control-flow cycle"
            );
            push_edge(edges, next_range, predecessor.clone());
            predecessor = end.clone();
        }
    }
}

/// Inject sibling ordering edges directly into the graph by adding extra
/// sources to `Range` nodes. After this, the graph's natural `srcs()` encode
/// all ordering constraints and the linearizer needs no side tables.
///
/// This mirrors tinygrad's `pm_add_control_flow` pattern matcher, which
/// patches Range nodes with CFG predecessors before linearization.
fn inject_control_flow(sink: &UOp) -> UOp {
    let order = sink.toposort();
    let (deps, nesting) = build_deps_and_nesting(&order);
    let siblings = build_siblings(&order, &nesting);

    let mut edges: HashMap<UOp, Vec<UOp>> = HashMap::new();
    add_sibling_edges(&mut edges, &deps, siblings);

    if edges.is_empty() {
        return sink.clone();
    }

    // Patch Range nodes: add CFG predecessors as extra sources.
    crate::rewrite::graph_rewrite(sink, &mut |node| {
        let extra = edges.get(node)?;
        let mut srcs = node.srcs().to_vec();
        srcs.extend(extra.iter().cloned());
        Some(UOp::new(node.op(), node.dtype(), srcs, node.arg().clone()))
    })
}

/// Convert a lowered kernel DAG into an ordered program.
///
/// This is the ferrograd equivalent of tinygrad's late linearizer: it turns
/// a scoped DAG into a linear list that the renderer can print directly.
#[must_use]
pub(crate) fn linearize(root: &UOp) -> Vec<UOp> {
    assert_eq!(root.op(), Op::Sink, "linearize: root must be a Sink node");

    let root = inject_control_flow(root);
    let order = root.toposort();
    let idx_of: HashMap<UOp, usize> = order
        .iter()
        .cloned()
        .enumerate()
        .map(|(idx, node)| (node, idx))
        .collect();

    let mut out_degree: HashMap<UOp, usize> = HashMap::new();
    for node in order.iter().rev() {
        for src in node.srcs() {
            *out_degree.entry(src.clone()).or_default() += 1;
        }
    }

    let mut ranked: Vec<usize> = (0..order.len()).collect();
    ranked.sort_by_key(|idx| priority_key(&order[*idx], *idx));

    let mut nkey = vec![0_usize; order.len()];
    for (rank, idx) in ranked.into_iter().enumerate() {
        nkey[idx] = rank;
    }

    let root_idx = *idx_of
        .get(&root)
        .expect("linearize: sink should appear in toposort");
    let mut heap = BinaryHeap::from([(nkey[root_idx], root_idx)]);
    let mut visited = HashSet::new();
    let mut linear = Vec::with_capacity(order.len());

    while let Some((_, idx)) = heap.pop() {
        if !visited.insert(idx) {
            continue;
        }
        linear.push(order[idx].clone());
        for src in order[idx].srcs() {
            let remaining = out_degree
                .get_mut(src)
                .expect("linearize: out-degree missing for source");
            *remaining -= 1;
            if *remaining == 0 {
                let src_idx = *idx_of
                    .get(src)
                    .expect("linearize: source should appear in toposort");
                heap.push((nkey[src_idx], src_idx));
            }
        }
    }

    assert_eq!(
        linear.len(),
        order.len(),
        "linearize: failed to order the full kernel graph"
    );
    linear.reverse();
    linear
        .into_iter()
        .filter(|node| node.op() != Op::Sink)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::linearize;
    use crate::device::DeviceId;
    use crate::dtype::DType;
    use crate::uop::{Arg, AxisKind, Op, UOp};

    fn build_two_loop_graph() -> UOp {
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 2, device);
        let n = UOp::const_int(1, DType::I32, device);

        let idx0 = UOp::new(
            Op::Range,
            DType::I32,
            vec![n.clone()],
            Arg::Range(0, AxisKind::Loop),
        );
        let out0 = UOp::new(
            Op::Index,
            out.dtype(),
            vec![out.clone(), idx0.clone()],
            Arg::None,
        );
        let store0 = UOp::new(
            Op::Store,
            DType::Void,
            vec![out0, UOp::const_float(1.0, DType::F32, device)],
            Arg::None,
        );
        let end0 = UOp::new(Op::End, DType::Void, vec![idx0, store0], Arg::None);

        let idx1 = UOp::new(
            Op::Range,
            DType::I32,
            vec![n],
            Arg::Range(1, AxisKind::Loop),
        );
        let one = UOp::const_int(1, DType::I32, device);
        let out1 = UOp::new(Op::Index, out.dtype(), vec![out, one], Arg::None);
        let store1 = UOp::new(
            Op::Store,
            DType::Void,
            vec![out1, UOp::const_float(2.0, DType::F32, device)],
            Arg::None,
        );
        let end1 = UOp::new(Op::End, DType::Void, vec![idx1, store1], Arg::None);

        UOp::sink(vec![end0, end1])
    }

    fn build_reduce_graph() -> UOp {
        let device = DeviceId::Cpu;
        let out = UOp::param_buffer(0, DType::F32, 1, device);
        let input = UOp::param_buffer(1, DType::F32, 4, device);
        let outer_bound = UOp::const_int(1, DType::I32, device);
        let reduce_bound = UOp::const_int(4, DType::I32, device);
        let outer = UOp::new(
            Op::Range,
            DType::I32,
            vec![outer_bound],
            Arg::Range(0, AxisKind::Loop),
        );
        let reduce = UOp::new(
            Op::Range,
            DType::I32,
            vec![reduce_bound],
            Arg::Range(1, AxisKind::Reduce),
        );
        let idx = UOp::new(
            Op::Index,
            input.dtype(),
            vec![input, reduce.clone()],
            Arg::None,
        );
        let val = UOp::new(Op::Load, DType::F32, vec![idx], Arg::None);
        let zero = UOp::const_float(0.0, DType::F32, device);
        let acc = UOp::new(Op::DefineAcc, DType::F32, vec![zero], Arg::None);
        let update = UOp::new(
            Op::Assign,
            DType::F32,
            vec![
                acc.clone(),
                UOp::new(Op::Add, DType::F32, vec![acc.clone(), val], Arg::None),
            ],
            Arg::None,
        );
        let reduce_end = UOp::new(Op::End, DType::Void, vec![reduce, update], Arg::None);
        let out_idx = UOp::new(Op::Index, out.dtype(), vec![out, outer.clone()], Arg::None);
        let store = UOp::new(
            Op::Store,
            DType::Void,
            vec![out_idx, UOp::after(acc.clone(), reduce_end)],
            Arg::None,
        );
        let outer_end = UOp::new(Op::End, DType::Void, vec![outer.clone(), store], Arg::None);
        UOp::sink(vec![outer_end])
    }

    #[test]
    fn test_linearize_closes_first_loop_before_second_opens() {
        let sink = build_two_loop_graph();
        let linear = linearize(&sink);

        // After inject_control_flow, Range nodes may be rebuilt with extra
        // CFG sources, so find them by arg rather than identity.
        let find = |op: Op, arg: &Arg| {
            linear
                .iter()
                .position(|node| node.op() == op && node.arg() == arg)
                .expect("node should exist in linearized output")
        };
        let range0 = find(Op::Range, &Arg::Range(0, AxisKind::Loop));
        let range1 = find(Op::Range, &Arg::Range(1, AxisKind::Loop));
        let ends: Vec<usize> = linear
            .iter()
            .enumerate()
            .filter(|(_, n)| n.op() == Op::End)
            .map(|(i, _)| i)
            .collect();

        assert!(range0 < ends[0]);
        assert!(ends[0] < range1);
    }

    #[test]
    fn test_linearize_places_accumulator_before_reduce_loop() {
        let sink = build_reduce_graph();
        let linear = linearize(&sink);

        let reduce = linear
            .iter()
            .find(|node| matches!(node.arg(), Arg::Range(_, AxisKind::Reduce)))
            .cloned()
            .expect("reduce range should exist");
        let acc = linear
            .iter()
            .find(|node| node.op() == Op::DefineAcc)
            .cloned()
            .expect("accumulator should exist");

        let pos = |u: &UOp| linear.iter().position(|node| node == u).unwrap();
        assert!(pos(&acc) < pos(&reduce));
    }
}
