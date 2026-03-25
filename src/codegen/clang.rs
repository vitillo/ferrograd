//! # Clang renderer — `UOp` IR to C
//!
//! Walks a toposorted `UOp` graph and emits scalar C with for-loops.
//! No vectorization or tiling — every element is computed one at a time
//! inside nested `for` loops, which makes the output easy to read and debug.
//!
//! The mapping from IR to C is mostly 1:1:
//! - `Range`/`End` → `for` loop open/close
//! - `DefineAcc`/`Assign` → accumulator variable + update statement
//! - `Load`/`Store` → pointer dereference through `Index` expressions
//! - ALU ops (`Add`, `Mul`, …) → C operators or builtins (`exp2`, `log2`)
//! - `ParamBuffer`/`ParamScalar` → function parameters (`float* restrict data0`)
//!
//! ## Tinygrad reference
//!
//! `tinygrad/renderer/cstyle.py` — `ClangRenderer`. Tinygrad's version
//! shares this same structure but also handles vectorized types and
//! local/group memory, which we don't need yet.

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

        // Collect kernel parameter nodes for function signature.
        let mut params: Vec<(usize, DType, bool)> = Vec::new();
        for node in &order {
            match node.op() {
                Op::ParamBuffer => {
                    if let Arg::ParamBuffer(slot, _) = node.arg() {
                        params.push((*slot, node.dtype(), true));
                    }
                }
                Op::ParamScalar => {
                    if let Arg::ParamScalar(slot) = node.arg() {
                        params.push((*slot, node.dtype(), false));
                    }
                }
                _ => {}
            }
        }
        params.sort_by_key(|(slot, _, _)| *slot);

        let args: Vec<String> = params
            .iter()
            .map(|(slot, dtype, is_buffer)| {
                if *is_buffer {
                    format!("{}* restrict data{slot}", dtype.c_type())
                } else {
                    format!("{} data{slot}", dtype.c_type())
                }
            })
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
                Op::Device | Op::DefineVar => {
                    names.insert(node, String::new());
                    continue;
                }
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
                Op::ParamBuffer => {
                    if let Arg::ParamBuffer(slot, _) = node.arg() {
                        names.insert(node, format!("data{slot}"));
                    }
                }
                Op::ParamScalar => {
                    if let Arg::ParamScalar(slot) = node.arg() {
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
                Op::Sink
                | Op::End
                | Op::Store
                | Op::After
                | Op::Buffer
                | Op::Device
                | Op::DefineVar => {
                    unreachable!()
                }
                // Tensor-level and unexpanded ops should be lowered before codegen.
                Op::Bind
                | Op::Shrink
                | Op::Reshape
                | Op::Permute
                | Op::Expand
                | Op::ReduceAxis
                | Op::Reduce => {
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
    use crate::device::{CpuDevice, Device, DeviceId, KernelArg};

    fn build_add_graph(n: i64) -> UOp {
        let device = DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 3, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, 3, device);
        let b_ptr = UOp::param_buffer(2, DType::F32, 3, device);
        let bound = UOp::const_int(n, DType::I32, device);
        let idx = UOp::new(Op::Range, DType::I32, vec![bound], Arg::Index(0));
        let a_idx = UOp::new(Op::Index, a_ptr.dtype(), vec![a_ptr, idx.clone()], Arg::None);
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let b_idx = UOp::new(Op::Index, b_ptr.dtype(), vec![b_ptr, idx.clone()], Arg::None);
        let b_val = UOp::new(Op::Load, DType::F32, vec![b_idx], Arg::None);
        let sum = UOp::add(a_val, b_val);
        let out_idx = UOp::new(Op::Index, out_ptr.dtype(), vec![out_ptr, idx.clone()], Arg::None);
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, sum], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store.clone()], Arg::None);
        UOp::sink(vec![store, end])
    }

    #[test]
    fn test_render_add_kernel_structure() {
        let sink = build_add_graph(3);
        let code = ClangRenderer.render(&sink, "add");
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
        let sink = build_add_graph(3);
        let code = ClangRenderer.render(&sink, "add");

        let dev = CpuDevice;
        let program = dev.compile(&code, "add", 3).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 3)),
            KernelArg::Buffer(crate::device::Buffer::from_f32(&[1.0, 2.0, 3.0])),
            KernelArg::Buffer(crate::device::Buffer::from_f32(&[4.0, 5.0, 6.0])),
        ];

        dev.execute(&program, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(out.to_f32(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_render_negate_kernel() {
        let device = DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 3, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, 3, device);
        let n = UOp::const_int(3, DType::I32, device);
        let idx = UOp::new(Op::Range, DType::I32, vec![n], Arg::Index(0));
        let a_idx = UOp::new(Op::Index, a_ptr.dtype(), vec![a_ptr, idx.clone()], Arg::None);
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let neg = UOp::neg(a_val);
        let out_idx = UOp::new(Op::Index, out_ptr.dtype(), vec![out_ptr, idx.clone()], Arg::None);
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, neg], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store.clone()], Arg::None);
        let sink = UOp::sink(vec![store, end]);

        let code = ClangRenderer.render(&sink, "negate");
        let dev = CpuDevice;
        let program = dev.compile(&code, "negate", 2).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 3)),
            KernelArg::Buffer(crate::device::Buffer::from_f32(&[1.0, -2.0, 3.0])),
        ];

        dev.execute(&program, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(out.to_f32(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_render_relu_kernel() {
        let device = DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 4, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, 4, device);
        let n = UOp::const_int(4, DType::I32, device);
        let zero = UOp::const_float(0.0, DType::F32, device);
        let idx = UOp::new(Op::Range, DType::I32, vec![n], Arg::Index(0));
        let a_idx = UOp::new(Op::Index, a_ptr.dtype(), vec![a_ptr, idx.clone()], Arg::None);
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let cond = UOp::cmplt(zero.clone(), a_val.clone());
        let relu = UOp::where_(cond, a_val, zero);
        let out_idx = UOp::new(Op::Index, out_ptr.dtype(), vec![out_ptr, idx.clone()], Arg::None);
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, relu], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store.clone()], Arg::None);
        let sink = UOp::sink(vec![store, end]);

        let code = ClangRenderer.render(&sink, "relu");
        let dev = CpuDevice;
        let program = dev.compile(&code, "relu", 2).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 4)),
            KernelArg::Buffer(crate::device::Buffer::from_f32(&[1.0, -2.0, 3.0, -4.0])),
        ];

        dev.execute(&program, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(out.to_f32(), vec![1.0, 0.0, 3.0, 0.0]);
    }
}
