//! Shared lane-pack primitives for late lowering.
//!
//! Both [`crate::expand`] and [`crate::devectorize`] manipulate the
//! `Unroll(Vectorize(..))` lane-pack representation. This module only holds
//! the pure lane math they genuinely share: recognizing packs, rebuilding
//! them, and broadcasting/scalarizing over lane coordinates.

use std::collections::HashMap;

use crate::dtype::DType;
use crate::uop::{Arg, Op, UOp};

/// Axis index and lane count pairs describing the shape of a lane pack.
pub(crate) type LaneMeta = Box<[(usize, usize)]>;
/// The individual scalar `UOp`s inside a `Vectorize` node, one per lane.
pub(crate) type LaneValues = Vec<UOp>;

/// Recognizes an `Unroll(Vectorize(..))` structure -- the canonical lane pack
/// form produced by expansion. Returns the individual lane values and their
/// axis metadata so callers can manipulate lanes without pattern-matching
/// the two-node wrapper each time.
pub(crate) fn lane_value(node: &UOp) -> Option<(LaneValues, LaneMeta)> {
    if node.op() != Op::Unroll || node.srcs().len() != 1 {
        return None;
    }
    let Arg::Lanes(lanes) = node.arg() else {
        return None;
    };
    let vector = &node.srcs()[0];
    if vector.op() != Op::Vectorize {
        return None;
    }
    Some((vector.srcs().to_vec(), lanes.clone()))
}

/// Constructs the canonical `Op(Vectorize(lane0, lane1, ..))` wrapper that
/// represents a lane pack. All expansion helpers emit this form so that
/// later passes have a single pattern to match.
pub(crate) fn make_lane_pack(op: Op, dtype: DType, lanes: LaneValues, meta: LaneMeta) -> UOp {
    let vector = UOp::new(Op::Vectorize, dtype, lanes, Arg::None);
    UOp::new(op, dtype, vec![vector], Arg::Lanes(meta))
}

/// Builds the cartesian product of all lane coordinates from the metadata.
/// A lane pack over axes `(0,2), (1,3)` produces 6 coordinate vectors --
/// one for each position in the 2x3 lane grid. Used to iterate over every
/// scalar position when expanding or broadcasting lane packs.
pub(crate) fn coord_product(meta: &[(usize, usize)]) -> Vec<Vec<usize>> {
    if meta.is_empty() {
        return vec![Vec::new()];
    }

    let mut out = vec![Vec::new()];
    for (_, size) in meta {
        let mut next = Vec::new();
        for prefix in &out {
            for idx in 0..*size {
                let mut coords = prefix.clone();
                coords.push(idx);
                next.push(coords);
            }
        }
        out = next;
    }
    out
}

/// Converts multi-dimensional lane coordinates to a flat index into the
/// `LaneValues` vector, using row-major (C) order over the metadata shape.
pub(crate) fn coord_to_index(meta: &[(usize, usize)], coords: &[usize]) -> usize {
    meta.iter()
        .zip(coords)
        .fold(0_usize, |acc, ((_, size), coord)| {
            acc.saturating_mul(*size).saturating_add(*coord)
        })
}

/// Merge multiple lane metadata shapes into one superset shape.
///
/// This is the lane-level equivalent of broadcast shape inference: if one
/// operand carries axis 0 lanes and another carries axis 1 lanes, the result
/// must carry both.
pub(crate) fn merged_lane_meta(metas: &[LaneMeta]) -> LaneMeta {
    let mut merged = Vec::new();
    for meta in metas {
        for &(axis, size) in meta {
            if merged
                .iter()
                .any(|(existing_axis, _)| *existing_axis == axis)
            {
                continue;
            }
            merged.push((axis, size));
        }
    }
    merged.sort_by_key(|(axis, _)| *axis);
    merged.into_boxed_slice()
}

/// Broadcasts a lane pack from a narrower metadata shape into a wider one by
/// repeating values along missing axes.
pub(crate) fn expand_lanes_to_meta(
    lanes: &[UOp],
    src_meta: &[(usize, usize)],
    dst_meta: &[(usize, usize)],
) -> Vec<UOp> {
    if src_meta == dst_meta {
        return lanes.to_vec();
    }

    let mut dst_positions = HashMap::new();
    for (idx, (axis, _)) in dst_meta.iter().enumerate() {
        dst_positions.insert(*axis, idx);
    }

    coord_product(dst_meta)
        .into_iter()
        .map(|dst_coords| {
            let src_coords: Vec<usize> = src_meta
                .iter()
                .map(|(axis, _)| {
                    dst_coords[*dst_positions
                        .get(axis)
                        .expect("source lane axis should exist in merged metadata")]
                })
                .collect();
            lanes[coord_to_index(src_meta, &src_coords)].clone()
        })
        .collect()
}

/// Unifies the lane shape across all operands of a node.
///
/// Lane-valued inputs are broadcast to the merged metadata shape, while scalar
/// inputs remain singletons that callers can replicate lane-by-lane.
pub(crate) fn combine_lane_sources(node: &UOp) -> Option<(Vec<LaneValues>, LaneMeta)> {
    let mut lane_metas = Vec::new();
    let mut lanes: Vec<LaneValues> = Vec::new();

    for src in node.srcs() {
        if let Some((src_lanes, src_meta)) = lane_value(src) {
            lane_metas.push(src_meta.clone());
            lanes.push(src_lanes);
        } else {
            lanes.push(vec![src.clone()]);
        }
    }

    if lane_metas.is_empty() {
        return None;
    }

    let meta = merged_lane_meta(&lane_metas);
    let lane_count = meta.iter().map(|(_, size)| *size).product::<usize>();
    let lanes = node
        .srcs()
        .iter()
        .zip(lanes)
        .map(|(src, src_lanes)| {
            if let Some((_, src_meta)) = lane_value(src) {
                expand_lanes_to_meta(&src_lanes, &src_meta, &meta)
            } else {
                vec![src_lanes[0].clone()]
            }
        })
        .collect::<Vec<_>>();
    if lanes
        .iter()
        .any(|src_lanes| src_lanes.len() != 1 && src_lanes.len() != lane_count)
    {
        return None;
    }
    Some((lanes, meta))
}
