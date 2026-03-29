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


use std::collections::HashMap;
use std::fmt::Write;

use crate::codegen::Renderer;
use crate::dtype::DType;
use crate::uop::{Arg, AxisKind, Op, UOp};

/// Renders `UOp` graphs to C, compiled by clang.
#[derive(Debug)]
pub struct ClangRenderer;

impl Renderer for ClangRenderer {
    #[allow(clippy::too_many_lines)]
    fn render(&self, uops: &[UOp], name: &str) -> String {
        // Collect kernel parameter nodes for function signature.
        let mut params: Vec<(usize, DType, bool)> = Vec::new();
        for node in uops {
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
            names.get(u).cloned().unwrap_or_else(|| {
                panic!(
                    "node referenced before being rendered: op={:?} arg={:?}",
                    u.op(),
                    u.arg()
                )
            })
        };
        for node in uops {
            match node.op() {
                Op::Device | Op::DefineVar => {
                    names.insert(node, String::new());
                }
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
                                if v.is_sign_positive() {
                                    "INFINITY".to_string()
                                } else {
                                    "(-INFINITY)".to_string()
                                }
                            } else if v.is_nan() {
                                "NAN".to_string()
                            } else {
                                let s = if v.fract() == 0.0 {
                                    format!("{v:.1}")
                                } else {
                                    format!("{v}")
                                };
                                if node.dtype() == DType::F32 {
                                    format!("{s}f")
                                } else {
                                    s
                                }
                            }
                        }
                        Arg::Int(v) => format!("{v}"),
                        Arg::Bool(v) => {
                            if *v {
                                "1".to_string()
                            } else {
                                "0".to_string()
                            }
                        }
                        _ => panic!("Const with unexpected arg: {:?}", node.arg()),
                    };
                    names.insert(node, expr);
                }
                _ => {}
            }
        }

        for node in uops {
            // Leaf nodes already have names from the first pass.
            match node.op() {
                Op::Sink
                | Op::Device
                | Op::DefineVar
                | Op::ParamBuffer
                | Op::ParamScalar
                | Op::Const => continue,
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
                // Range may carry extra CFG ordering sources beyond the bound;
                // only srcs[0] (the bound) matters for rendering.
                Op::Range => {
                    let Arg::Range(axis, kind) = node.arg() else {
                        panic!("Range without axis arg");
                    };
                    let loop_prefix = match kind {
                        AxisKind::Loop | AxisKind::Global => "idx",
                        AxisKind::Local => "lidx",
                        AxisKind::Reduce | AxisKind::GroupReduce => "ridx",
                        AxisKind::Upcast | AxisKind::Unroll => "uidx",
                        AxisKind::Thread => "tidx",
                    };
                    let bound = name_of(&node.srcs()[0], &names);
                    let var = format!("{loop_prefix}{axis}");
                    let _ = writeln!(
                        out,
                        "{ind}for (int {var} = 0; {var} < {bound}; {var}++) {{",
                        ind = indent(depth),
                    );
                    names.insert(node, var);
                    depth += 1;
                    continue;
                }
                Op::Buffer => unreachable!("Buffer is a tensor-level op"),
                _ => {}
            }
            let srcs: Vec<String> = node.srcs().iter().map(|s| name_of(s, &names)).collect();

            match node.op() {
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
                // ALU ops and DefineAcc all follow the same pattern: declare a
                // typed local variable and assign an expression to it.
                Op::Neg
                | Op::Exp2
                | Op::Log2
                | Op::Sqrt
                | Op::Reciprocal
                | Op::Add
                | Op::Mul
                | Op::Max
                | Op::CmpLt
                | Op::Where
                | Op::DefineAcc => {
                    let (prefix, expr) = match node.op() {
                        Op::Neg => ("alu", format!("(-{})", srcs[0])),
                        Op::Exp2 => ("alu", format!("exp2({})", srcs[0])),
                        Op::Log2 => ("alu", format!("log2({})", srcs[0])),
                        Op::Sqrt => ("alu", format!("__builtin_sqrtf({})", srcs[0])),
                        Op::Reciprocal => ("alu", format!("(1/{})", srcs[0])),
                        Op::Add => ("alu", format!("({}+{})", srcs[0], srcs[1])),
                        Op::Mul => ("alu", format!("({}*{})", srcs[0], srcs[1])),
                        Op::Max => ("alu", format!("(({}>{1})?{0}:{1})", srcs[0], srcs[1])),
                        Op::CmpLt => ("alu", format!("({}<{})", srcs[0], srcs[1])),
                        Op::Where => ("alu", format!("({}?{}:{})", srcs[0], srcs[1], srcs[2])),
                        Op::DefineAcc => ("acc", srcs[0].clone()),
                        _ => unreachable!(),
                    };
                    let var = format!("{prefix}{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = {expr};",
                        ind = indent(depth),
                        ctype = node.dtype().c_type(),
                    );
                    names.insert(node, var);
                }
                Op::Assign => {
                    let acc_var = name_of(&node.srcs()[0], &names);
                    let new_val = name_of(&node.srcs()[1], &names);
                    let _ = writeln!(out, "{ind}{acc_var} = {new_val};", ind = indent(depth));
                    names.insert(node, acc_var);
                }
                // Leaf, side-effect, and control-flow nodes handled above.
                Op::Sink
                | Op::End
                | Op::Store
                | Op::After
                | Op::Range
                | Op::Buffer
                | Op::Device
                | Op::DefineVar
                | Op::ParamBuffer
                | Op::ParamScalar
                | Op::Const => {
                    unreachable!()
                }
                // Tensor-level and unexpanded ops should be lowered before codegen.
                Op::Bind
                | Op::Shrink
                | Op::Reshape
                | Op::Permute
                | Op::Expand
                | Op::Contiguous
                | Op::Vectorize
                | Op::Unroll
                | Op::Contract
                | Op::Gep
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
    use crate::linearize::linearize;

    fn render_sink(sink: &UOp, name: &str) -> String {
        let linear = linearize(sink);
        ClangRenderer.render(&linear, name)
    }

    fn f32_buffer(dev: &CpuDevice, data: &[f32]) -> crate::device::Buffer {
        let buffer = dev.allocate(DType::F32, data.len());
        dev.copy_from_host(&buffer, bytemuck::cast_slice(data));
        buffer
    }

    fn read_f32(dev: &CpuDevice, buffer: &crate::device::Buffer) -> Vec<f32> {
        bytemuck::cast_slice::<u8, f32>(&dev.copy_to_host(buffer)).to_vec()
    }

    fn build_add_graph(n: i64) -> UOp {
        let device = DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 3, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, 3, device);
        let b_ptr = UOp::param_buffer(2, DType::F32, 3, device);
        let bound = UOp::const_int(n, DType::I32, device);
        let idx = UOp::new(
            Op::Range,
            DType::I32,
            vec![bound],
            Arg::Range(0, AxisKind::Loop),
        );
        let a_idx = UOp::new(
            Op::Index,
            a_ptr.dtype(),
            vec![a_ptr, idx.clone()],
            Arg::None,
        );
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let b_idx = UOp::new(
            Op::Index,
            b_ptr.dtype(),
            vec![b_ptr, idx.clone()],
            Arg::None,
        );
        let b_val = UOp::new(Op::Load, DType::F32, vec![b_idx], Arg::None);
        let sum = UOp::add(a_val, b_val);
        let out_idx = UOp::new(
            Op::Index,
            out_ptr.dtype(),
            vec![out_ptr, idx.clone()],
            Arg::None,
        );
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, sum], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store.clone()], Arg::None);
        UOp::sink(vec![store, end])
    }

    fn build_shared_reduce_multi_acc_graph() -> UOp {
        let device = DeviceId::Cpu;
        let out_buf = UOp::param_buffer(0, DType::F32, 2, device);
        let in_buf = UOp::param_buffer(1, DType::F32, 6, device);
        let outer_bound = UOp::const_int(1, DType::I32, device);
        let width = UOp::const_int(2, DType::I32, device);
        let reduce_bound = UOp::const_int(3, DType::I32, device);
        let zero = UOp::const_int(0, DType::I32, device);
        let one = UOp::const_int(1, DType::I32, device);
        let neg_inf = UOp::const_float(f64::NEG_INFINITY, DType::F32, device);

        let outer = UOp::new(
            Op::Range,
            DType::I32,
            vec![outer_bound],
            Arg::Range(0, AxisKind::Loop),
        );
        let reduce = UOp::new(
            Op::Range,
            DType::I32,
            vec![reduce_bound.clone()],
            Arg::Range(1, AxisKind::Reduce),
        );

        let base = UOp::mul(outer.clone(), width.clone());
        let lane0 = UOp::add(base.clone(), zero.clone());
        let lane1 = UOp::add(base.clone(), one.clone());

        let stride = reduce_bound.clone();
        let in0 = UOp::add(UOp::mul(lane0.clone(), stride.clone()), reduce.clone());
        let in1 = UOp::add(UOp::mul(lane1.clone(), stride), reduce.clone());
        let in0_idx = UOp::new(
            Op::Index,
            in_buf.dtype(),
            vec![in_buf.clone(), in0],
            Arg::None,
        );
        let in1_idx = UOp::new(
            Op::Index,
            in_buf.dtype(),
            vec![in_buf.clone(), in1],
            Arg::None,
        );
        let in0_val = UOp::new(Op::Load, DType::F32, vec![in0_idx], Arg::None);
        let in1_val = UOp::new(Op::Load, DType::F32, vec![in1_idx], Arg::None);

        let acc0 = UOp::new_tagged(
            Op::DefineAcc,
            DType::F32,
            vec![neg_inf.clone()],
            Arg::None,
            1,
        );
        let acc1 = UOp::new_tagged(Op::DefineAcc, DType::F32, vec![neg_inf], Arg::None, 2);
        let upd0 = UOp::new(
            Op::Assign,
            DType::F32,
            vec![
                acc0.clone(),
                UOp::new(Op::Max, DType::F32, vec![acc0.clone(), in0_val], Arg::None),
            ],
            Arg::None,
        );
        let upd1 = UOp::new(
            Op::Assign,
            DType::F32,
            vec![
                acc1.clone(),
                UOp::new(Op::Max, DType::F32, vec![acc1.clone(), in1_val], Arg::None),
            ],
            Arg::None,
        );
        let reduce_end = UOp::new(
            Op::End,
            DType::Void,
            vec![reduce, upd0.clone(), upd1.clone()],
            Arg::None,
        );

        let out0_idx = UOp::new(
            Op::Index,
            out_buf.dtype(),
            vec![out_buf.clone(), lane0],
            Arg::None,
        );
        let out1_idx = UOp::new(
            Op::Index,
            out_buf.dtype(),
            vec![out_buf.clone(), lane1],
            Arg::None,
        );
        let store0 = UOp::new(
            Op::Store,
            DType::Void,
            vec![out0_idx, UOp::after(acc0, reduce_end.clone())],
            Arg::None,
        );
        let store1 = UOp::new(
            Op::Store,
            DType::Void,
            vec![out1_idx, UOp::after(acc1, reduce_end.clone())],
            Arg::None,
        );
        let outer_end = UOp::new(Op::End, DType::Void, vec![outer, store0, store1], Arg::None);
        UOp::sink(vec![outer_end])
    }

    #[test]
    fn test_render_add_kernel_structure() {
        let sink = build_add_graph(3);
        let code = render_sink(&sink, "add");
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
        let code = render_sink(&sink, "add");

        let dev = CpuDevice::new();
        let program = dev.compile(&code, "add", 3).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 3)),
            KernelArg::Buffer(f32_buffer(&dev, &[1.0, 2.0, 3.0])),
            KernelArg::Buffer(f32_buffer(&dev, &[4.0, 5.0, 6.0])),
        ];

        dev.execute(&program, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, out), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_render_negate_kernel() {
        let device = DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 3, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, 3, device);
        let n = UOp::const_int(3, DType::I32, device);
        let idx = UOp::new(
            Op::Range,
            DType::I32,
            vec![n],
            Arg::Range(0, AxisKind::Loop),
        );
        let a_idx = UOp::new(
            Op::Index,
            a_ptr.dtype(),
            vec![a_ptr, idx.clone()],
            Arg::None,
        );
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let neg = UOp::neg(a_val);
        let out_idx = UOp::new(
            Op::Index,
            out_ptr.dtype(),
            vec![out_ptr, idx.clone()],
            Arg::None,
        );
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, neg], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store.clone()], Arg::None);
        let sink = UOp::sink(vec![store, end]);

        let code = render_sink(&sink, "negate");
        let dev = CpuDevice::new();
        let program = dev.compile(&code, "negate", 2).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 3)),
            KernelArg::Buffer(f32_buffer(&dev, &[1.0, -2.0, 3.0])),
        ];

        dev.execute(&program, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, out), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_render_relu_kernel() {
        let device = DeviceId::Cpu;
        let out_ptr = UOp::param_buffer(0, DType::F32, 4, device);
        let a_ptr = UOp::param_buffer(1, DType::F32, 4, device);
        let n = UOp::const_int(4, DType::I32, device);
        let zero = UOp::const_float(0.0, DType::F32, device);
        let idx = UOp::new(
            Op::Range,
            DType::I32,
            vec![n],
            Arg::Range(0, AxisKind::Loop),
        );
        let a_idx = UOp::new(
            Op::Index,
            a_ptr.dtype(),
            vec![a_ptr, idx.clone()],
            Arg::None,
        );
        let a_val = UOp::new(Op::Load, DType::F32, vec![a_idx], Arg::None);
        let cond = UOp::cmplt(zero.clone(), a_val.clone());
        let relu = UOp::where_(cond, a_val, zero);
        let out_idx = UOp::new(
            Op::Index,
            out_ptr.dtype(),
            vec![out_ptr, idx.clone()],
            Arg::None,
        );
        let store = UOp::new(Op::Store, DType::Void, vec![out_idx, relu], Arg::None);
        let end = UOp::new(Op::End, DType::Void, vec![idx, store.clone()], Arg::None);
        let sink = UOp::sink(vec![store, end]);

        let code = render_sink(&sink, "relu");
        let dev = CpuDevice::new();
        let program = dev.compile(&code, "relu", 2).expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 4)),
            KernelArg::Buffer(f32_buffer(&dev, &[1.0, -2.0, 3.0, -4.0])),
        ];

        dev.execute(&program, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, out), vec![1.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn test_render_reduce_range_uses_same_loop_codegen() {
        let device = DeviceId::Cpu;
        let bound = UOp::const_int(4, DType::I32, device);
        let reduce = UOp::new(
            Op::Range,
            DType::I32,
            vec![bound],
            Arg::Range(1, AxisKind::Reduce),
        );
        let sink = UOp::sink(vec![UOp::new(
            Op::End,
            DType::Void,
            vec![reduce],
            Arg::None,
        )]);

        let code = render_sink(&sink, "noop_reduce");
        assert!(code.contains("for (int ridx1 = 0; ridx1 < 4; ridx1++)"));
    }

    #[test]
    fn test_render_loop_and_reduce_axes_use_distinct_names() {
        let device = DeviceId::Cpu;
        let loop_bound = UOp::const_int(3, DType::I32, device);
        let reduce_bound = UOp::const_int(2, DType::I32, device);
        let loop_range = UOp::new(
            Op::Range,
            DType::I32,
            vec![loop_bound],
            Arg::Range(0, AxisKind::Loop),
        );
        let reduce_range = UOp::new(
            Op::Range,
            DType::I32,
            vec![reduce_bound],
            Arg::Range(0, AxisKind::Reduce),
        );
        let nested = UOp::new(
            Op::End,
            DType::Void,
            vec![
                loop_range,
                UOp::new(Op::End, DType::Void, vec![reduce_range], Arg::None),
            ],
            Arg::None,
        );
        let sink = UOp::sink(vec![nested]);

        let code = render_sink(&sink, "nested_ranges");
        assert!(code.contains("for (int idx0 = 0; idx0 < 3; idx0++)"));
        assert!(code.contains("for (int ridx0 = 0; ridx0 < 2; ridx0++)"));
    }

    #[test]
    fn test_render_shared_reduce_multi_acc_declares_accumulators_before_loop() {
        let sink = build_shared_reduce_multi_acc_graph();

        let code = render_sink(&sink, "shared_reduce");
        let loop_pos = code
            .find("for (int ridx1 = 0; ridx1 < 3; ridx1++)")
            .expect("missing reduce loop");
        let acc_positions = code
            .lines()
            .filter(|line| line.contains("float acc") && line.contains("(-INFINITY);"))
            .map(|line| code.find(line).expect("accumulator line should exist"))
            .collect::<Vec<_>>();

        assert!(acc_positions.len() >= 2);
        assert!(acc_positions.iter().all(|pos| *pos < loop_pos));
    }

    #[test]
    fn test_render_shared_reduce_multi_acc_compiles_and_runs() {
        let sink = build_shared_reduce_multi_acc_graph();
        let code = render_sink(&sink, "shared_reduce");

        let dev = CpuDevice::new();
        let program = dev
            .compile(&code, "shared_reduce", 2)
            .expect("compile failed");
        let mut args = [
            KernelArg::Buffer(dev.allocate(DType::F32, 2)),
            KernelArg::Buffer(f32_buffer(&dev, &[1.0, 5.0, 3.0, -1.0, 0.0, 7.0])),
        ];

        dev.execute(&program, &mut args).unwrap();
        let KernelArg::Buffer(out) = &args[0] else {
            panic!("output arg should stay a buffer");
        };
        assert_eq!(read_f32(&dev, out), vec![5.0, 7.0]);
    }
}
