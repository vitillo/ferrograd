//! # Kernel optimization — loop splitting and symbolic simplification
//!
//! After rangeify produces loop-and-load kernel IR, this module reshapes loop
//! structure and simplifies index arithmetic before codegen. Three passes run
//! in a fixed order:
//!
//! 1. **[`upcast`]** — split an output loop into outer `GLOBAL` + inner
//!    `UPCAST` so the C backend can auto-vectorize consecutive stores.
//! 2. **[`unroll`]** — split a reduction loop into outer `REDUCE` + inner
//!    `UNROLL` to expose instruction-level parallelism.
//! 3. **[`symbolic`]** — fold constants (`2+3 → 5`) and eliminate identities
//!    (`x+0`, `x*1`, `x*0`) in the resulting index arithmetic.
//!
//! Upcast and unroll are mechanically identical — split a range into an outer
//! loop plus a small inner loop (width 4), then let [`crate::expand`] fully
//! unroll the inner portion. The difference is which axis kind they target,
//! which changes what happens downstream. See each submodule for details.
//!
//! ## Configuration
//!
//! The `OPT` environment variable controls which passes run:
//! - unset / `all` / `1`: all passes (default)
//! - `none` / `off` / `0`: no passes (useful for debugging raw IR)
//! - comma-separated names: `symbolic,upcast,unroll`


mod symbolic;
mod upcast;
mod unroll;

#[allow(unused_imports)]
pub(crate) use symbolic::symbolic_simple;
#[allow(unused_imports)]
pub(crate) use upcast::upcast_loops;
#[allow(unused_imports)]
pub(crate) use unroll::unroll_reduce_loops;

use std::sync::LazyLock;

use crate::dtype::DType;
use crate::rewrite::graph_rewrite;
use crate::uop::{Arg, UOp};

/// Extract a constant `I32` bound from a `Range` node, returning [`None`] for
/// non-constant or non-`I32` bounds. Used by both loop-splitting passes to
/// check divisibility before splitting.
fn const_i32(node: &UOp) -> Option<i64> {
    let Arg::Int(value) = node.arg() else {
        return None;
    };
    (node.dtype() == DType::I32).then_some(*value)
}

/// Controls which optimization passes run on the kernel IR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OptimizeConfig {
    symbolic: bool,
    upcast: bool,
    unroll: bool,
}

impl OptimizeConfig {
    pub(crate) const fn all() -> Self {
        Self {
            symbolic: true,
            upcast: true,
            unroll: true,
        }
    }

    pub(crate) const fn none() -> Self {
        Self {
            symbolic: false,
            upcast: false,
            unroll: false,
        }
    }
}

/// Parse the `OPT` environment variable into an [`OptimizeConfig`].
///
/// Accepts `"0"` / `"off"` / `"none"` to disable everything, `"1"` / `"on"` /
/// `"all"` (or unset) to enable everything, or a comma-separated list of pass
/// names like `"symbolic,unroll"` for selective control.
pub(crate) fn parse_optimize_config(raw: Option<&str>) -> OptimizeConfig {
    let Some(raw) = raw.map(str::trim) else {
        return OptimizeConfig::all();
    };
    if raw.is_empty() {
        return OptimizeConfig::all();
    }

    match raw {
        "0" | "off" | "none" => return OptimizeConfig::none(),
        "1" | "on" | "all" => return OptimizeConfig::all(),
        _ => {}
    }

    let mut config = OptimizeConfig::none();
    for pass in raw.split(',').map(str::trim) {
        match pass {
            "symbolic" => config.symbolic = true,
            "upcast" => config.upcast = true,
            "unroll" => config.unroll = true,
            _ => {}
        }
    }
    config
}

static OPTIMIZE: LazyLock<OptimizeConfig> =
    LazyLock::new(|| parse_optimize_config(std::env::var("OPT").ok().as_deref()));

/// Run enabled kernel IR optimization passes, configured by `OPT` env var.
#[must_use]
pub(crate) fn optimize(kernel: &UOp) -> UOp {
    optimize_with_config(kernel, *OPTIMIZE)
}

/// Run kernel IR optimization passes using an explicit config.
///
/// Order is **upcast → unroll → symbolic**, matching the intended interaction:
/// loop structure is finalized first, then index cleanup runs on the result.
#[must_use]
pub(crate) fn optimize_with_config(kernel: &UOp, config: OptimizeConfig) -> UOp {
    let mut current = kernel.clone();

    if config.upcast {
        current = upcast::upcast_loops(&current);
    }

    if config.unroll {
        current = unroll::unroll_reduce_loops(&current);
    }

    if config.symbolic {
        current = graph_rewrite(&current, &mut symbolic::symbolic_simple);
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_config_defaults_to_all_passes() {
        assert_eq!(parse_optimize_config(None), OptimizeConfig::all());
        assert_eq!(parse_optimize_config(Some("")), OptimizeConfig::all());
        assert_eq!(parse_optimize_config(Some("all")), OptimizeConfig::all());
    }

    #[test]
    fn parse_config_can_disable_all_passes() {
        assert_eq!(parse_optimize_config(Some("0")), OptimizeConfig::none());
        assert_eq!(parse_optimize_config(Some("off")), OptimizeConfig::none());
        assert_eq!(parse_optimize_config(Some("none")), OptimizeConfig::none());
    }

    #[test]
    fn parse_config_enables_named_passes() {
        assert_eq!(
            parse_optimize_config(Some("symbolic")),
            OptimizeConfig {
                symbolic: true,
                upcast: false,
                unroll: false,
            }
        );
        assert_eq!(
            parse_optimize_config(Some("symbolic,upcast,unroll,unknown")),
            OptimizeConfig {
                symbolic: true,
                upcast: true,
                unroll: true,
            }
        );
    }
}
