//! # Clang renderer — `UOp` IR to C
//!
//! Walks a toposorted `UOp` graph and emits scalar C with for-loops.
//! The output goes straight to the CPU backend's clang pipeline.
//!
//! ## Tinygrad reference
//!
//! `tinygrad/renderer/cstyle.py` — `CStyleLanguage`, `ClangRenderer`.
//! Our version is simpler: no vectorization, no image types, no address spaces.

use std::fmt::Write;

use crate::codegen::Renderer;
use crate::dtype::DType;
use crate::uop::{Arg, Op, UOpGraph, UOpId};

/// Renders `UOp` graphs to C, compiled by clang.
///
/// Maps to tinygrad's `ClangRenderer(CStyleLanguage)`. Emits scalar C
/// with for-loops — no vectorization or GPU features.
pub struct ClangRenderer;

impl Renderer for ClangRenderer {
    #[allow(clippy::too_many_lines)]
    fn render(&self, graph: &UOpGraph, name: &str, root: UOpId) -> String {
        assert_eq!(
            graph.get(root).op,
            Op::Sink,
            "render: root must be a Sink node"
        );

        let order = graph.toposort(root);

        // First pass: collect Param nodes to build the function signature.
        let mut params: Vec<(usize, DType)> = Vec::new();
        for &id in &order {
            let node = graph.get(id);
            if node.op == Op::Param {
                if let Arg::Index(slot) = node.arg {
                    params.push((slot, node.dtype));
                }
            }
        }
        params.sort_by_key(|(slot, _)| *slot);

        // Function signature: void name(float* data0, float* data1, ...)
        let args: Vec<String> = params
            .iter()
            .map(|(slot, dtype)| format!("{}* data{slot}", dtype.c_type()))
            .collect();
        let mut out = format!("void {name}({}) {{\n", args.join(", "));

        // Variable name map: UOpId → C expression string.
        // Some nodes are rendered inline (constants, index arithmetic),
        // others get a named variable (loads, ALU ops).
        let mut names: Vec<Option<String>> = vec![None; graph.len()];
        let mut depth: usize = 1;
        let mut val_count: usize = 0;
        let mut alu_count: usize = 0;

        let indent = |d: usize| "  ".repeat(d);

        // Helper: get the C expression for a source node.
        let name_of = |id: UOpId, names: &[Option<String>]| -> String {
            names[id.idx()]
                .clone()
                .unwrap_or_else(|| panic!("node {id} referenced before being rendered"))
        };

        for &id in &order {
            let node = graph.get(id);

            match node.op {
                Op::Param => {
                    if let Arg::Index(slot) = node.arg {
                        names[id.idx()] = Some(format!("data{slot}"));
                    }
                }

                Op::Const => {
                    let expr = match &node.arg {
                        Arg::Float(v) => {
                            // Ensure the literal always has a decimal point (e.g. "2.0f" not "2f").
                            let s = if v.fract() == 0.0 {
                                format!("{v:.1}")
                            } else {
                                format!("{v}")
                            };
                            if node.dtype == DType::F32 {
                                format!("{s}f")
                            } else {
                                s
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
                        _ => panic!("Const node with unexpected arg: {:?}", node.arg),
                    };
                    // Constants are rendered inline (no separate statement).
                    names[id.idx()] = Some(expr);
                }

                Op::Range => {
                    let Arg::Index(axis) = node.arg else {
                        panic!("Range without axis arg");
                    };
                    let bound = name_of(node.srcs[0], &names);
                    let var = format!("idx{axis}");
                    let _ = writeln!(
                        out,
                        "{ind}for (int {var} = 0; {var} < {bound}; {var}++) {{",
                        ind = indent(depth),
                    );
                    names[id.idx()] = Some(var);
                    depth += 1;
                }

                Op::End => {
                    depth -= 1;
                    let _ = writeln!(out, "{ind}}}", ind = indent(depth));
                }

                Op::Index => {
                    let ptr = name_of(node.srcs[0], &names);
                    let offset = name_of(node.srcs[1], &names);
                    // Rendered inline — no separate statement.
                    names[id.idx()] = Some(format!("({ptr}+{offset})"));
                }

                Op::Load => {
                    let index_expr = name_of(node.srcs[0], &names);
                    let var = format!("val{val_count}");
                    val_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = *{index_expr};",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::Store => {
                    let index_expr = name_of(node.srcs[0], &names);
                    let value = name_of(node.srcs[1], &names);
                    let _ = writeln!(
                        out,
                        "{ind}*{index_expr} = {value};",
                        ind = indent(depth),
                    );
                }

                Op::Sink => {
                    // No code emitted for Sink — it's just the graph root.
                }

                // Unary math
                Op::Neg => {
                    let x = name_of(node.srcs[0], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = (-{x});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::Exp2 => {
                    let x = name_of(node.srcs[0], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = exp2({x});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::Log2 => {
                    let x = name_of(node.srcs[0], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = log2({x});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::Sqrt => {
                    let x = name_of(node.srcs[0], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = __builtin_sqrtf({x});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::Reciprocal => {
                    let x = name_of(node.srcs[0], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = (1/{x});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                // Binary math
                Op::Add => {
                    let a = name_of(node.srcs[0], &names);
                    let b = name_of(node.srcs[1], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = ({a}+{b});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::Mul => {
                    let a = name_of(node.srcs[0], &names);
                    let b = name_of(node.srcs[1], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = ({a}*{b});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::Max => {
                    let a = name_of(node.srcs[0], &names);
                    let b = name_of(node.srcs[1], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = (({a}>{b})?{a}:{b});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                Op::CmpLt => {
                    let a = name_of(node.srcs[0], &names);
                    let b = name_of(node.srcs[1], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = ({a}<{b});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
                }

                // Ternary
                Op::Where => {
                    let cond = name_of(node.srcs[0], &names);
                    let true_val = name_of(node.srcs[1], &names);
                    let false_val = name_of(node.srcs[2], &names);
                    let var = format!("alu{alu_count}");
                    alu_count += 1;
                    let _ = writeln!(
                        out,
                        "{ind}{ctype} {var} = ({cond}?{true_val}:{false_val});",
                        ind = indent(depth),
                        ctype = node.dtype.c_type(),
                    );
                    names[id.idx()] = Some(var);
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
    use crate::uop::UOpGraph;

    /// Helper: build add kernel graph (out[i] = a[i] + b[i])
    fn build_add_graph(n: i64) -> (UOpGraph, UOpId) {
        let mut g = UOpGraph::new();
        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let b_ptr = g.param(2, DType::F32);
        let bound = g.const_int(n, DType::I32);
        let idx = g.range(0, bound);
        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let b_idx = g.index(b_ptr, idx);
        let b_val = g.load(b_idx, DType::F32);
        let sum = g.add_op(a_val, b_val);
        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, sum);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);
        (g, sink)
    }

    #[test]
    fn test_render_add_kernel_structure() {
        // Arrange
        let (g, sink) = build_add_graph(3);

        // Act
        let code = ClangRenderer.render(&g, "add", sink);

        // Assert — check key structural elements
        assert!(code.contains("void add("));
        assert!(code.contains("float* data0"));
        assert!(code.contains("float* data1"));
        assert!(code.contains("float* data2"));
        assert!(code.contains("for (int idx0 = 0; idx0 < 3; idx0++)"));
        assert!(code.contains("float val"));
        assert!(code.contains("float alu"));
        println!("{code}");
    }

    #[test]
    fn test_render_add_kernel_compiles_and_runs() {
        // Arrange — build graph and render to C
        let (g, sink) = build_add_graph(3);
        let code = ClangRenderer.render(&g, "add", sink);
        println!("{code}");

        // Compile through the device layer
        let dev = CpuDevice;
        let program = dev.compile(&code, "add", 3).expect("compile failed");
        let mut a = crate::device::Buffer::from_f32(&[1.0, 2.0, 3.0]);
        let mut b = crate::device::Buffer::from_f32(&[4.0, 5.0, 6.0]);
        let mut out = dev.allocate(DType::F32, 3);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a, &mut b])
            .unwrap();

        // Assert
        assert_eq!(out.to_f32(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_render_mul_kernel_compiles_and_runs() {
        // Arrange
        let mut g = UOpGraph::new();
        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let b_ptr = g.param(2, DType::F32);
        let n = g.const_int(3, DType::I32);
        let idx = g.range(0, n);
        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let b_idx = g.index(b_ptr, idx);
        let b_val = g.load(b_idx, DType::F32);
        let prod = g.mul(a_val, b_val);
        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, prod);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);

        let code = ClangRenderer.render(&g, "mul", sink);
        println!("{code}");

        let dev = CpuDevice;
        let program = dev.compile(&code, "mul", 3).expect("compile failed");
        let mut a = crate::device::Buffer::from_f32(&[2.0, 3.0, 4.0]);
        let mut b = crate::device::Buffer::from_f32(&[5.0, 6.0, 7.0]);
        let mut out = dev.allocate(DType::F32, 3);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a, &mut b])
            .unwrap();

        // Assert
        assert_eq!(out.to_f32(), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_render_fused_muladd_single_kernel() {
        // Arrange — out[i] = (a[i] + b[i]) * 2.0
        let mut g = UOpGraph::new();
        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let b_ptr = g.param(2, DType::F32);
        let n = g.const_int(4, DType::I32);
        let two = g.const_float(2.0, DType::F32);
        let idx = g.range(0, n);
        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let b_idx = g.index(b_ptr, idx);
        let b_val = g.load(b_idx, DType::F32);
        let sum = g.add_op(a_val, b_val);
        let result = g.mul(sum, two);
        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, result);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);

        let code = ClangRenderer.render(&g, "fused_muladd", sink);
        println!("{code}");

        let dev = CpuDevice;
        let program = dev
            .compile(&code, "fused_muladd", 3)
            .expect("compile failed");
        let mut a = crate::device::Buffer::from_f32(&[1.0, 2.0, 3.0, 4.0]);
        let mut b = crate::device::Buffer::from_f32(&[10.0, 20.0, 30.0, 40.0]);
        let mut out = dev.allocate(DType::F32, 4);

        // Act
        dev.execute(&program, &mut [&mut out, &mut a, &mut b])
            .unwrap();

        // Assert — (1+10)*2=22, (2+20)*2=44, (3+30)*2=66, (4+40)*2=88
        assert_eq!(out.to_f32(), vec![22.0, 44.0, 66.0, 88.0]);
    }

    #[test]
    fn test_render_negate_kernel() {
        // Arrange — out[i] = -a[i]
        let mut g = UOpGraph::new();
        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let n = g.const_int(3, DType::I32);
        let idx = g.range(0, n);
        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let neg = g.neg(a_val);
        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, neg);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);

        let code = ClangRenderer.render(&g, "negate", sink);
        println!("{code}");

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
        // Arrange — out[i] = max(a[i], 0) via where(0 < a[i], a[i], 0)
        let mut g = UOpGraph::new();
        let out_ptr = g.param(0, DType::F32);
        let a_ptr = g.param(1, DType::F32);
        let n = g.const_int(4, DType::I32);
        let zero = g.const_float(0.0, DType::F32);
        let idx = g.range(0, n);
        let a_idx = g.index(a_ptr, idx);
        let a_val = g.load(a_idx, DType::F32);
        let cond = g.cmplt(zero, a_val);
        let relu = g.where_op(cond, a_val, zero);
        let out_idx = g.index(out_ptr, idx);
        let store = g.store(out_idx, relu);
        let end = g.end(idx);
        let sink = g.sink(vec![store, end]);

        let code = ClangRenderer.render(&g, "relu", sink);
        println!("{code}");

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
