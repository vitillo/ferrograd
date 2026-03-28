//! # Symbolic simplification on kernel index IR
//!
//! Rangeify generates stride arithmetic with many literal zeros and ones
//! (`idx*1 + 0`, `stride*0`, etc.). This pass cleans those up so codegen
//! sees smaller, simpler expressions — the same role tinygrad’s symbolic
//! layer plays on kernel math.
//!
//! Two kinds of rewrites:
//! - **Constant folding** — evaluate `Const op Const` at compile time
//!   (`2+3 → 5`, `4*8 → 32`).
//! - **Identity elimination** — remove no-op operands
//!   (`x+0 → x`, `x*1 → x`, `x*0 → 0`).
//!
//! Runs as the **last** optimization pass so upcast and unroll have already
//! finalized the loop structure before index cleanup simplifies the result.

use crate::uop::{Arg, Op, UOp};

/// Fold `Const op Const` into a single [`Arg`], or [`None`] if unsupported.
fn fold_binary(op: Op, a: &Arg, b: &Arg) -> Option<Arg> {
    match (op, a, b) {
        (Op::Add, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x + y)),
        (Op::Add, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x + y)),
        (Op::Mul, Arg::Float(x), Arg::Float(y)) => Some(Arg::Float(x * y)),
        (Op::Mul, Arg::Int(x), Arg::Int(y)) => Some(Arg::Int(x * y)),
        _ => None,
    }
}

/// Build a `Const` node from a folded [`Arg`], inheriting dtype and device.
fn const_from_arg(node: &UOp, arg: &Arg) -> UOp {
    match arg {
        Arg::Float(value) => UOp::const_float(*value, node.dtype(), node.device()),
        Arg::Int(value) => UOp::const_int(*value, node.dtype(), node.device()),
        Arg::Bool(value) => UOp::const_bool(*value, node.dtype(), node.device()),
        _ => panic!("constant folding produced non-literal arg"),
    }
}

/// [`graph_rewrite`](crate::rewrite::graph_rewrite) callback that applies
/// constant folding and identity elimination to a single node.
#[must_use]
pub(crate) fn symbolic_simple(node: &UOp) -> Option<UOp> {
    match (node.op(), node.srcs()) {
        (op @ (Op::Add | Op::Mul), [a, b]) if a.is_const() && b.is_const() => {
            fold_binary(op, a.arg(), b.arg())
                .as_ref()
                .map(|arg| const_from_arg(a, arg))
        }
        (Op::Add, [x, y]) if y.is_zero() => Some(x.clone()),
        (Op::Add, [x, y]) if x.is_zero() => Some(y.clone()),
        (Op::Mul, [x, y]) if y.is_one() => Some(x.clone()),
        (Op::Mul, [x, y]) if x.is_one() => Some(y.clone()),
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
    use crate::rewrite::graph_rewrite;
    use crate::uop::{Arg, Op, UOp};

    #[test]
    fn add_zero_eliminated() {
        // Arrange
        let x = UOp::const_float(5.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(x, zero);

        // Act
        let result = graph_rewrite(&sum, &mut symbolic_simple);

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn zero_plus_x_eliminated() {
        // Arrange
        let x = UOp::const_float(5.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(zero, x);

        // Act
        let result = graph_rewrite(&sum, &mut symbolic_simple);

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn constant_folding_add() {
        // Arrange
        let two = UOp::const_float(2.0, DType::F32, DeviceId::Cpu);
        let three = UOp::const_float(3.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(two, three);

        // Act
        let result = graph_rewrite(&sum, &mut symbolic_simple);

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(5.0));
    }

    #[test]
    fn fixed_point_chains_identities() {
        // Arrange
        let x = UOp::const_float(7.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let one = UOp::const_float(1.0, DType::F32, DeviceId::Cpu);
        let sum = UOp::add(x, zero);
        let prod = UOp::mul(sum, one);

        // Act
        let result = graph_rewrite(&prod, &mut symbolic_simple);

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(7.0));
    }

    #[test]
    fn mul_zero() {
        // Arrange
        let x = UOp::const_float(42.0, DType::F32, DeviceId::Cpu);
        let zero = UOp::const_float(0.0, DType::F32, DeviceId::Cpu);
        let prod = UOp::mul(x, zero);

        // Act
        let result = graph_rewrite(&prod, &mut symbolic_simple);

        // Assert
        assert_eq!(result.op(), Op::Const);
        assert_eq!(*result.arg(), Arg::Float(0.0));
    }
}
