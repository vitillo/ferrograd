//! # Clang renderer — `UOp` IR to C
//!
//! Walks a toposorted `UOp` graph and emits scalar C with for-loops.

use std::collections::HashMap;
use std::fmt::Write;

use crate::codegen::Renderer;
use crate::dtype::DType;
use crate::uop::{Arg, Op, UOp};

/// Renders `UOp` graphs to C, compiled by clang.
pub struct ClangRenderer;

impl Renderer for ClangRenderer {
    #[allow(clippy::too_many_lines)]
    fn render(&self, root: &UOp, name: &str) -> String {
        assert_eq!(root.op(), Op::Sink, "render: root must be a Sink node");

        let order = root.toposort();

        // Collect Param nodes for function signature.
        let mut params: Vec<(usize, DType)> = Vec::new();
        for node in &order {
            if node.op() == Op::Param {
                if let Arg::Param(slot, _) = node.arg() {
                    params.push((*slot, node.dtype()));
                }
            }
        }
        params.sort_by_key(|(slot, _)| *slot);

        let args: Vec<String> = params
            .iter()
            .map(|(slot, dtype)| format!("{}* restrict data{slot}", dtype.c_type()))
            .collect();
        let mut out = format!("#include <math.h>\nvoid {name}({}) {{\n", args.join(", "));

        // Variable name map: UOp → C expression string.
        let mut names: HashMap<&UOp, String> = HashMap::new();
        let mut depth: usize = 1;
        let mut val_count: usize = 0;
        let mut alu_count: usize = 0;

        let indent = |d: usize| "  ".repeat(d);

        let name_of = |u: &UOp, names: &HashMap<&UOp, String>| -> String {
            names
                .get(u)
                .cloned()
                .unwrap_or_else(|| panic!("node referenced before being rendered"))
        };

        for node in &order {
            // Side-effect-only nodes — no C expression to name.
            match node.op() {
                Op::Sink => continue,
                Op::End => {
                    depth -= 1;
                    let _ = writeln!(out, "{ind}}}", ind = indent(depth));
                    continue;
                }
                Op::Store => {
                    let idx_expr = name_of(&node.srcs()[0], &names);
                    let value = name_of(&node.srcs()[1], &names);
                    let _ = writeln!(out, "{ind}*{idx_expr} = {value};", ind = indent(depth));
                    continue;
                }
                Op::After => {
                    // Passthrough: use src[0]'s value, src[1] is ordering only.
                    let val = name_of(&node.srcs()[0], &names);
                    names.insert(node, val);
                    continue;
                }
                Op::Buffer => unreachable!("Buffer is a tensor-level op"),
                _ => {}
            }

            let srcs: Vec<String> = node
                .srcs()
                .iter()
                .map(|s| name_of(s, &names))
                .collect();

            match node.op() {
                Op::Param => {
                    if let Arg::Param(slot, _) = node.arg() {
                        names.insert(node, format!("data{slot}"));
                    }
                }
                Op::Const => {
                    let expr = match node.arg() {
                        Arg::Float(v) => {
                            if v.is_infinite() {
                                if v.is_sign_positive() { "INFINITY".to_string() }
                                else { "(-INFINITY)".to_string() }
                            } else if v.is_nan() {
                                "NAN".to_string()
                            } else {
                                let s = if v.fract() == 0.0 {
                                    format!("{v:.1}")
                                } else {
                                    format!("{v}")
                                };
                                if node.dtype() == DType::F32 { format!("{s}f") } else { s }
                            }
                        }
                        Arg::Int(v) => format!("{v}"),
                        Arg::Bool(v) => {
                            if *v { "1".to_string() } else { "0".to_string() }
                        }
                        _ => panic!("Const with unexpected arg: {:?}", node.arg()),
                    };
                    names.insert(node, expr);
                }
                Op::Range => {
                    let Arg::Index(axis) = node.arg() else {
                        panic!("Range without axis arg");
                    };
                    let bound = srcs[0].clone();
                    let var = format!("idx{axis}");
                    let _ = writeln!(
                        out,
                        "{ind}for (int {var} = 0; {var} < {bound}; {var}++) {{",
                        ind = indent(depth),
                    );
                    names.insert(node, var);
                    depth += 1;
                }
                Op::Index => {
                    let ptr = srcs[0].clone();
                    let offset = srcs[1].clone();
                    names.insert(node, format!("({ptr}+{offset})"));
                }
                Op::Load => {
                    let index_expr = srcs[0].clone();
                    let var = format!("val{val_count}");
                    val_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = *{index_expr};",
                        ind = indent(depth),
                        ctype = node.dtype().c_type(),
                    );
                    names.insert(node, var);
                }
                // Unary
                Op::Neg => {
                    let x = srcs[0].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = (-{x});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::Exp2 => {
                    let x = srcs[0].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = exp2({x});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::Log2 => {
                    let x = srcs[0].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = log2({x});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::Sqrt => {
                    let x = srcs[0].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = __builtin_sqrtf({x});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::Reciprocal => {
                    let x = srcs[0].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = (1/{x});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }

                // Binary
                Op::Add => {
                    let a = srcs[0].clone();
                    let b = srcs[1].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = ({a}+{b});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::Mul => {
                    let a = srcs[0].clone();
                    let b = srcs[1].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = ({a}*{b});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::Max => {
                    let a = srcs[0].clone();
                    let b = srcs[1].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = (({a}>{b})?{a}:{b});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::CmpLt => {
                    let a = srcs[0].clone();
                    let b = srcs[1].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = ({a}<{b});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }

                // Ternary
                Op::Where => {
                    let cond = srcs[0].clone();
                    let true_val = srcs[1].clone();
                    let false_val = srcs[2].clone();
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = ({cond}?{true_val}:{false_val});", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::DefineAcc => {
                    let init = srcs[0].clone();
                    let var = format!("acc{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(out, "{ind}{ctype} {var} = {init};", ind = indent(depth), ctype = node.dtype().c_type());
                    names.insert(node, var);
                }
                Op::Assign => {
                    let acc_var = name_of(&node.srcs()[0], &names);
                    let new_val = name_of(&node.srcs()[1], &names);
                    let _ = writeln!(out, "{ind}{acc_var} = {new_val};", ind = indent(depth));
                    names.insert(node, acc_var);
                }
                // Sink, End, Store, After, Buffer handled above.
                Op::Sink | Op::End | Op::Store | Op::After | Op::Buffer => unreachable!(),
                // Tensor-level and unexpanded ops should be lowered before codegen.
                Op::Reshape | Op::Permute | Op::Expand | Op::ReduceAxis | Op::Reduce => {
                    unreachable!("{op:?} should be lowered before codegen", op = node.op())
                }
            }
        }

        out.push_str("}\n");
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{CpuDevice, Device};

    fn build_add_graph(n: i64) -> UOp {
        let out_ptr = UOp::param(0, DType::F32, 3);
        let a_ptr = UOp::param(1, DType::F32, 3);
        let b_ptr = UOp::param(2, DType::F32, 3);
        let bound = UOp::const_int(n, DType::I32);
        let idx = UOp::range(0, bound);
        let a_val = UOp::load(UOp::index(a_ptr, idx.clone()), DType::F32);
        let b_val = UOp::load(UOp::index(b_ptr, idx.clone()), DType::F32);
        let sum = UOp::new(Op::Add, DType::F32, vec![a_val, b_val], Arg::None);
        let store = UOp::store(UOp::index(out_ptr, idx.clone()), sum);
        let end = UOp::end(idx);
        UOp::sink(vec![store, end])
    }

    #[test]
    fn test_render_add_kernel_structure() {
        // Arrange
        let sink = build_add_graph(3);

        // Act
        let code = ClangRenderer.render(&sink, "add");

        // Assert
        assert!(code.contains("void add("));
        assert!(code.contains("float* restrict data0"));
        assert!(code.contains("float* restrict data1"));
        assert!(code.contains("float* restrict data2"));
        assert!(code.contains("for (int idx0 = 0; idx0 < 3; idx0++)"));
        assert!(code.contains("float val"));
        assert!(code.contains("float alu"));
    }

    #[test]
    fn test_render_add_kernel_compiles_and_runs() {
        // Arrange
        let sink = build_add_graph(3);
        let code = ClangRenderer.render(&sink, "add");

        let dev = CpuDevice;
        let program = dev.compile(&code, "add", 3).expect("compile failed");
        let mut a = crate::device::Buffer::from_f32(&[1.0, 2.0, 3.0]);
        let mut b = crate::device::Buffer::from_f32(&[4.0, 5.0, 6.0]);
        let mut out = dev.allocate(DType::F32, 3);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a, &mut b]).unwrap();

        // Assert
        assert_eq!(out.to_f32(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_render_negate_kernel() {
        // Arrange
        let out_ptr = UOp::param(0, DType::F32, 3);
        let a_ptr = UOp::param(1, DType::F32, 3);
        let n = UOp::const_int(3, DType::I32);
        let idx = UOp::range(0, n);
        let a_val = UOp::load(UOp::index(a_ptr, idx.clone()), DType::F32);
        let neg = UOp::new(Op::Neg, DType::F32, vec![a_val], Arg::None);
        let store = UOp::store(UOp::index(out_ptr, idx.clone()), neg);
        let end = UOp::end(idx);
        let sink = UOp::sink(vec![store, end]);

        let code = ClangRenderer.render(&sink, "negate");
        let dev = CpuDevice;
        let program = dev.compile(&code, "negate", 2).expect("compile failed");
        let mut a = crate::device::Buffer::from_f32(&[1.0, -2.0, 3.0]);
        let mut out = dev.allocate(DType::F32, 3);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a]).unwrap();

        // Assert
        assert_eq!(out.to_f32(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_render_relu_kernel() {
        // Arrange
        let out_ptr = UOp::param(0, DType::F32, 4);
        let a_ptr = UOp::param(1, DType::F32, 4);
        let n = UOp::const_int(4, DType::I32);
        let zero = UOp::const_float(0.0, DType::F32);
        let idx = UOp::range(0, n);
        let a_val = UOp::load(UOp::index(a_ptr, idx.clone()), DType::F32);
        let cond = UOp::new(Op::CmpLt, DType::Bool, vec![zero.clone(), a_val.clone()], Arg::None);
        let relu = UOp::new(Op::Where, DType::F32, vec![cond, a_val, zero], Arg::None);
        let store = UOp::store(UOp::index(out_ptr, idx.clone()), relu);
        let end = UOp::end(idx);
        let sink = UOp::sink(vec![store, end]);

        let code = ClangRenderer.render(&sink, "relu");
        let dev = CpuDevice;
        let program = dev.compile(&code, "relu", 2).expect("compile failed");
        let mut a = crate::device::Buffer::from_f32(&[1.0, -2.0, 3.0, -4.0]);
        let mut out = dev.allocate(DType::F32, 4);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a]).unwrap();

        // Assert
        assert_eq!(out.to_f32(), vec![1.0, 0.0, 3.0, 0.0]);
    }
}
