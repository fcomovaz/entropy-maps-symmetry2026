"""Deterministic C99 code generation for CNN graphs, registry header, golden CSVs, and reports."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .c_writer import format_c_array, format_c_float, write_c_source, write_header
from .cnn_graph import CnnIrGraph, CnnIrOp, run_cnn_graph_f32_batch
from .cnn_quantize import QuantizedCnnModel, QuantizedCnnOp, quantize_cnn_graph, run_cnn_graph_q15_batch
from .config import CImpConfig
from .memory_plan import MemoryPlan, build_memory_plan


@dataclass(frozen=True, slots=True)
class ExportedCnnModel:
    """Artifacts and parity summary produced for one exported CNN model."""

    model_name: str
    header_path: Path
    source_path: Path
    header_sha256: str
    source_sha256: str
    golden_csv_path: Path
    checkpoint_path: Path | None
    checkpoint_sha256: str
    weight_format: str
    memory_plan: MemoryPlan
    q15_label_mismatches: int


def _emit_conv_params_c(
    prefix: str,
    idx: int,
    op: CnnIrOp,
    qop: QuantizedCnnOp,
) -> list[str]:
    """Generate `static const` weight and bias C arrays for one Conv2d/DWConv2d/Linear op."""
    assert op.weights is not None and op.bias is not None
    assert qop.bias_acc is not None
    blocks = [
        format_c_array(f"{prefix}_OP{idx}_W_F32", op.weights, "float", per_line=8),
        format_c_array(f"{prefix}_OP{idx}_B_F32", op.bias, "float", per_line=8),
    ]
    if qop.weight_format == "int8_per_channel":
        assert qop.w_q8 is not None and qop.mul_per_ch is not None and qop.shift_per_ch is not None
        blocks.extend(
            [
                format_c_array(f"{prefix}_OP{idx}_W_Q8", qop.w_q8, "int8_t", per_line=16),
                format_c_array(f"{prefix}_OP{idx}_B_ACC", qop.bias_acc, "int64_t", per_line=6),
                format_c_array(f"{prefix}_OP{idx}_MUL_CH", qop.mul_per_ch, "int32_t", per_line=8),
                format_c_array(f"{prefix}_OP{idx}_SHIFT_CH", qop.shift_per_ch, "int8_t", per_line=16),
            ]
        )
    else:
        assert qop.w_q15 is not None
        blocks.extend(
            [
                format_c_array(f"{prefix}_OP{idx}_W_Q15", qop.w_q15, "q15_t", per_line=12),
                format_c_array(f"{prefix}_OP{idx}_B_ACC", qop.bias_acc, "int64_t", per_line=6),
            ]
        )
    return blocks


def _emit_f32_op_call(
    prefix: str,
    idx: int,
    op: CnnIrOp,
    plan: MemoryPlan,
) -> str:
    """Return C99 call statement executing `op` in float32 against static arena offsets."""
    in0 = f"(const float *)(const void *)(arena + {plan.allocations[op.inputs[0]].offset_f32})"
    out_ptr = f"(float *)(void *)(arena + {plan.allocations[op.output].offset_f32})"
    cin, hin, win = op.in_shapes[0]
    cout, hout, wout = op.out_shape

    if op.op_type == "conv2d":
        kh, kw = op.kernel_size
        return (
            f"    cimp_conv2d_f32({in0}, {cin}, {hin}, {win}, "
            f"{prefix}_OP{idx}_W_F32, {prefix}_OP{idx}_B_F32, "
            f"{cout}, {kh}, {kw}, {op.stride}, {op.padding}, {1 if op.relu_fused else 0}, "
            f"{out_ptr}, {hout}, {wout});"
        )
    if op.op_type == "dwconv2d":
        kh, kw = op.kernel_size
        return (
            f"    cimp_dwconv2d_f32({in0}, {cin}, {hin}, {win}, "
            f"{prefix}_OP{idx}_W_F32, {prefix}_OP{idx}_B_F32, "
            f"{kh}, {kw}, {op.stride}, {op.padding}, {1 if op.relu_fused else 0}, "
            f"{out_ptr}, {hout}, {wout});"
        )
    if op.op_type == "relu":
        numel = plan.allocations[op.output].numel
        return f"    cimp_relu_f32({in0}, {out_ptr}, {numel});"
    if op.op_type == "hardswish":
        numel = plan.allocations[op.output].numel
        return f"    cimp_hardswish_f32({in0}, {out_ptr}, {numel});"
    if op.op_type == "hardsigmoid":
        numel = plan.allocations[op.output].numel
        return f"    cimp_hardsigmoid_f32({in0}, {out_ptr}, {numel});"
    if op.op_type == "maxpool2d":
        return f"    cimp_maxpool2d_2x2_f32({in0}, {cin}, {hin}, {win}, {out_ptr}, {hout}, {wout});"
    if op.op_type == "global_avgpool2d":
        return f"    cimp_global_avgpool2d_f32({in0}, {cin}, {hin}, {win}, {out_ptr});"
    if op.op_type == "linear":
        return (
            f"    cimp_linear_f32({in0}, {prefix}_OP{idx}_W_F32, {prefix}_OP{idx}_B_F32, "
            f"{cin}, {cout}, {1 if op.relu_fused else 0}, {out_ptr});"
        )
    if op.op_type == "add":
        in1 = f"(const float *)(const void *)(arena + {plan.allocations[op.inputs[1]].offset_f32})"
        numel = plan.allocations[op.output].numel
        return f"    cimp_add_f32({in0}, {in1}, {out_ptr}, {numel}, {1 if op.relu_fused else 0});"
    if op.op_type == "scale_channels":
        scale_ptr = f"(const float *)(const void *)(arena + {plan.allocations[op.inputs[1]].offset_f32})"
        return f"    cimp_scale_channels_f32({in0}, {scale_ptr}, {cin}, {hin * win}, {out_ptr});"
    raise ValueError(f"Unsupported op_type for f32 C codegen: {op.op_type}")


def _emit_q15_op_call(
    prefix: str,
    idx: int,
    op: CnnIrOp,
    qop: QuantizedCnnOp,
    plan: MemoryPlan,
) -> str:
    """Return C99 call statement executing `op` in Q15 fixed point against static arena offsets."""
    in0 = f"(const q15_t *)(const void *)(arena + {plan.allocations[op.inputs[0]].offset_q15})"
    out_q15 = f"(q15_t *)(void *)(arena + {plan.allocations[op.output].offset_q15})"
    cin, hin, win = op.in_shapes[0]
    cout, hout, wout = op.out_shape

    if op.op_type == "conv2d":
        kh, kw = op.kernel_size
        if qop.weight_format == "int8_per_channel":
            return (
                f"    cimp_conv2d_q15_w8({in0}, {cin}, {hin}, {win}, "
                f"{prefix}_OP{idx}_W_Q8, {prefix}_OP{idx}_B_ACC, "
                f"{prefix}_OP{idx}_MUL_CH, {prefix}_OP{idx}_SHIFT_CH, "
                f"{cout}, {kh}, {kw}, {op.stride}, {op.padding}, {1 if op.relu_fused else 0}, "
                f"{out_q15}, {hout}, {wout});"
            )
        return (
            f"    cimp_conv2d_q15({in0}, {cin}, {hin}, {win}, "
            f"{prefix}_OP{idx}_W_Q15, {prefix}_OP{idx}_B_ACC, "
            f"{cout}, {kh}, {kw}, {op.stride}, {op.padding}, {qop.out_shift}, {1 if op.relu_fused else 0}, "
            f"{out_q15}, {hout}, {wout});"
        )
    if op.op_type == "dwconv2d":
        kh, kw = op.kernel_size
        if qop.weight_format == "int8_per_channel":
            return (
                f"    cimp_dwconv2d_q15_w8({in0}, {cin}, {hin}, {win}, "
                f"{prefix}_OP{idx}_W_Q8, {prefix}_OP{idx}_B_ACC, "
                f"{prefix}_OP{idx}_MUL_CH, {prefix}_OP{idx}_SHIFT_CH, "
                f"{kh}, {kw}, {op.stride}, {op.padding}, {1 if op.relu_fused else 0}, "
                f"{out_q15}, {hout}, {wout});"
            )
        return (
            f"    cimp_dwconv2d_q15({in0}, {cin}, {hin}, {win}, "
            f"{prefix}_OP{idx}_W_Q15, {prefix}_OP{idx}_B_ACC, "
            f"{kh}, {kw}, {op.stride}, {op.padding}, {qop.out_shift}, {1 if op.relu_fused else 0}, "
            f"{out_q15}, {hout}, {wout});"
        )
    if op.op_type == "relu":
        numel = plan.allocations[op.output].numel
        return f"    cimp_relu_q15({in0}, {out_q15}, {numel}, {qop.out_shift});"
    if op.op_type == "hardswish":
        numel = plan.allocations[op.output].numel
        return f"    cimp_hardswish_q15({in0}, {out_q15}, {numel}, {qop.in_fracs[0]}, {qop.out_frac});"
    if op.op_type == "hardsigmoid":
        numel = plan.allocations[op.output].numel
        return f"    cimp_hardsigmoid_q15({in0}, {out_q15}, {numel}, {qop.in_fracs[0]}, {qop.out_frac});"
    if op.op_type == "maxpool2d":
        return f"    cimp_maxpool2d_2x2_q15({in0}, {cin}, {hin}, {win}, {out_q15}, {hout}, {wout});"
    if op.op_type == "global_avgpool2d":
        return f"    cimp_global_avgpool2d_q15({in0}, {cin}, {hin}, {win}, {qop.out_shift}, {out_q15});"
    if op.op_type == "linear":
        if op.is_final_logits:
            out_i32 = f"(int32_t *)(void *)(arena + {plan.allocations[op.output].offset_q15})"
            if qop.weight_format == "int8_per_channel":
                return (
                    f"    cimp_linear_logits_q15_w8({in0}, {prefix}_OP{idx}_W_Q8, {prefix}_OP{idx}_B_ACC, "
                    f"{prefix}_OP{idx}_MUL_CH, {prefix}_OP{idx}_SHIFT_CH, {cin}, {cout}, {out_i32});"
                )
            return (
                f"    cimp_linear_logits_q15({in0}, {prefix}_OP{idx}_W_Q15, {prefix}_OP{idx}_B_ACC, "
                f"{cin}, {cout}, {qop.out_shift}, {out_i32});"
            )
        if qop.weight_format == "int8_per_channel":
            return (
                f"    cimp_linear_q15_w8({in0}, {prefix}_OP{idx}_W_Q8, {prefix}_OP{idx}_B_ACC, "
                f"{prefix}_OP{idx}_MUL_CH, {prefix}_OP{idx}_SHIFT_CH, {cin}, {cout}, "
                f"{1 if op.relu_fused else 0}, {out_q15});"
            )
        return (
            f"    cimp_linear_q15({in0}, {prefix}_OP{idx}_W_Q15, {prefix}_OP{idx}_B_ACC, "
            f"{cin}, {cout}, {qop.out_shift}, {1 if op.relu_fused else 0}, {out_q15});"
        )
    if op.op_type == "add":
        in1 = f"(const q15_t *)(const void *)(arena + {plan.allocations[op.inputs[1]].offset_q15})"
        numel = plan.allocations[op.output].numel
        return (
            f"    cimp_add_q15({in0}, {in1}, {out_q15}, {numel}, "
            f"{qop.add_shift_a}, {qop.add_shift_b}, {1 if op.relu_fused else 0});"
        )
    if op.op_type == "scale_channels":
        scale_ptr = f"(const q15_t *)(const void *)(arena + {plan.allocations[op.inputs[1]].offset_q15})"
        return (
            f"    cimp_scale_channels_q15({in0}, {scale_ptr}, {cin}, {hin * win}, "
            f"{qop.out_shift}, {out_q15});"
        )
    raise ValueError(f"Unsupported op_type for q15 C codegen: {op.op_type}")


def emit_cnn_model(
    cfg: CImpConfig,
    graph: CnnIrGraph,
    qmodel: QuantizedCnnModel,
    plan: MemoryPlan,
    eval_file_names: tuple[str, ...],
    eval_true_labels: np.ndarray,
    eval_images_u8: np.ndarray,
    eval_images_f32: np.ndarray,
) -> ExportedCnnModel:
    """Emit `model_<cnn>.h`, `model_<cnn>.c`, and `golden_cnn_<cnn>.csv` for one CNN."""
    model_name = graph.model_name.lower()
    upper = model_name.upper()
    prefix = f"CIMP_CNN_{upper}"
    c_in, h_in, w_in = graph.input_shape
    n_pix = c_in * h_in * w_in
    num_classes = graph.ops[-1].out_shape[0]

    header_body = (
        f"#define {prefix}_INPUT_C {c_in}\n"
        f"#define {prefix}_INPUT_H {h_in}\n"
        f"#define {prefix}_INPUT_W {w_in}\n"
        f"#define {prefix}_NUM_CLASSES {num_classes}\n"
        f"#define {prefix}_ARENA_F32_BYTES {plan.peak_arena_f32_bytes}\n"
        f"#define {prefix}_ARENA_Q15_BYTES {plan.peak_arena_q15_bytes}\n"
        f"#define {prefix}_WEIGHT_FLASH_BYTES {plan.weight_flash_bytes}\n"
        f"#define {prefix}_THRESHOLD_PROB ({format_c_float(graph.selected_threshold)})\n"
        f"#define {prefix}_THRESHOLD_LOGIT_F32 ({format_c_float(graph.threshold_logit)})\n"
        f"#define {prefix}_OUT_FRAC {qmodel.output_frac}\n"
        f"#define {prefix}_THRESHOLD_LOGIT_Q ((int32_t){qmodel.threshold_logit_q})\n\n"
        f"int cimp_{model_name}_forward_f32(\n"
        f"    const uint8_t *img_u8,\n"
        f"    uint8_t *arena,\n"
        f"    size_t arena_bytes,\n"
        f"    float *out_logits\n"
        f");\n\n"
        f"int cimp_{model_name}_forward_q15(\n"
        f"    const uint8_t *img_u8,\n"
        f"    uint8_t *arena,\n"
        f"    size_t arena_bytes,\n"
        f"    int32_t *out_logits_q\n"
        f");\n\n"
        f"int cimp_{model_name}_classify_f32(\n"
        f"    const uint8_t *img_u8,\n"
        f"    uint8_t *arena,\n"
        f"    size_t arena_bytes,\n"
        f"    float *out_margin\n"
        f");\n\n"
        f"int cimp_{model_name}_classify_q15(\n"
        f"    const uint8_t *img_u8,\n"
        f"    uint8_t *arena,\n"
        f"    size_t arena_bytes,\n"
        f"    int32_t *out_margin_q\n"
        f");"
    )
    header_path = cfg.generated_dir / f"model_{model_name}.h"
    header_sha = write_header(
        path=header_path,
        guard=f"CIMP_MODEL_{upper}_H",
        includes=("<stddef.h>", "<stdint.h>"),
        body=header_body,
        provenance_lines=(
            f"CNN Model: {model_name} | target={cfg.effective_target_tag}",
            f"Weight format: {qmodel.weight_format}, flash={plan.weight_flash_bytes} B, "
            f"arena_f32={plan.peak_arena_f32_bytes} B, arena_q15={plan.peak_arena_q15_bytes} B",
            f"Checkpoint SHA-256: {graph.checkpoint_sha256 or 'n/a'}",
        ),
    )

    param_sections = [
        format_c_array(f"{prefix}_NORM_LUT_Q15", qmodel.norm_lut_q15, "q15_t", per_line=12)
    ]
    for idx, (op, qop) in enumerate(zip(graph.ops, qmodel.ops)):
        if op.op_type in ("conv2d", "dwconv2d", "linear"):
            param_sections.extend(_emit_conv_params_c(prefix, idx, op, qop))

    in_f32_off = plan.allocations[graph.input_name].offset_f32
    out_f32_off = plan.allocations[graph.output_name].offset_f32
    in_q15_off = plan.allocations[graph.input_name].offset_q15
    out_q15_off = plan.allocations[graph.output_name].offset_q15

    f32_calls = [
        f"    cimp_input_norm_u8_f32(img_u8, {n_pix}, {format_c_float(graph.norm_mean)}, {format_c_float(graph.norm_std)}, "
        f"(float *)(void *)(arena + {in_f32_off}));"
    ] + [_emit_f32_op_call(prefix, idx, op, plan) for idx, op in enumerate(graph.ops)]

    q15_calls = [
        f"    cimp_input_norm_u8_q15(img_u8, {n_pix}, {prefix}_NORM_LUT_Q15, "
        f"(q15_t *)(void *)(arena + {in_q15_off}));"
    ] + [
        _emit_q15_op_call(prefix, idx, op, qop, plan)
        for idx, (op, qop) in enumerate(zip(graph.ops, qmodel.ops))
    ]

    fns_body = (
        f"int cimp_{model_name}_forward_f32(\n"
        f"    const uint8_t *img_u8,\n"
        f"    uint8_t *arena,\n"
        f"    size_t arena_bytes,\n"
        f"    float *out_logits\n"
        f") {{\n"
        f"    if (img_u8 == NULL || arena == NULL || out_logits == NULL) {{\n"
        f"        return CIMP_ERR_INVALID_ARG;\n"
        f"    }}\n"
        f"    if (arena_bytes < (size_t){prefix}_ARENA_F32_BYTES) {{\n"
        f"        return CIMP_ERR_ARENA_TOO_SMALL;\n"
        f"    }}\n"
        + "\n".join(f32_calls)
        + "\n"
        + f"    const float *logits_src = (const float *)(const void *)(arena + {out_f32_off});\n"
        + f"    for (int32_t k = 0; k < {num_classes}; ++k) {{\n"
        + f"        out_logits[k] = logits_src[k];\n"
        + f"    }}\n"
        + "    return CIMP_OK;\n}\n\n"
        + f"int cimp_{model_name}_forward_q15(\n"
        + f"    const uint8_t *img_u8,\n"
        + f"    uint8_t *arena,\n"
        + f"    size_t arena_bytes,\n"
        + f"    int32_t *out_logits_q\n"
        + f") {{\n"
        + f"    if (img_u8 == NULL || arena == NULL || out_logits_q == NULL) {{\n"
        + f"        return CIMP_ERR_INVALID_ARG;\n"
        + f"    }}\n"
        + f"    if (arena_bytes < (size_t){prefix}_ARENA_Q15_BYTES) {{\n"
        + f"        return CIMP_ERR_ARENA_TOO_SMALL;\n"
        + f"    }}\n"
        + "\n".join(q15_calls)
        + "\n"
        + f"    const int32_t *logits_src = (const int32_t *)(const void *)(arena + {out_q15_off});\n"
        + f"    for (int32_t k = 0; k < {num_classes}; ++k) {{\n"
        + f"        out_logits_q[k] = logits_src[k];\n"
        + f"    }}\n"
        + "    return CIMP_OK;\n}\n\n"
        + f"int cimp_{model_name}_classify_f32(\n"
        + f"    const uint8_t *img_u8,\n"
        + f"    uint8_t *arena,\n"
        + f"    size_t arena_bytes,\n"
        + f"    float *out_margin\n"
        + f") {{\n"
        + f"    float logits[{num_classes}];\n"
        + f"    int rc = cimp_{model_name}_forward_f32(img_u8, arena, arena_bytes, logits);\n"
        + f"    if (rc != CIMP_OK) {{\n"
        + f"        return rc;\n"
        + f"    }}\n"
        + f"    float margin = logits[1] - logits[0];\n"
        + f"    if (out_margin != NULL) {{\n"
        + f"        *out_margin = margin;\n"
        + f"    }}\n"
        + f"    return (margin > {prefix}_THRESHOLD_LOGIT_F32) ? 1 : 0;\n"
        + "}\n\n"
        + f"int cimp_{model_name}_classify_q15(\n"
        + f"    const uint8_t *img_u8,\n"
        + f"    uint8_t *arena,\n"
        + f"    size_t arena_bytes,\n"
        + f"    int32_t *out_margin_q\n"
        + f") {{\n"
        + f"    int32_t logits[{num_classes}];\n"
        + f"    int rc = cimp_{model_name}_forward_q15(img_u8, arena, arena_bytes, logits);\n"
        + f"    if (rc != CIMP_OK) {{\n"
        + f"        return rc;\n"
        + f"    }}\n"
        + f"    int64_t margin64 = (int64_t)logits[1] - (int64_t)logits[0];\n"
        + f"    if (out_margin_q != NULL) {{\n"
        + f"        *out_margin_q = cimp_sat32(margin64);\n"
        + f"    }}\n"
        + f"    return (margin64 > (int64_t){prefix}_THRESHOLD_LOGIT_Q) ? 1 : 0;\n"
        + "}"
    )

    source_path = cfg.generated_dir / f"model_{model_name}.c"
    source_sha = write_c_source(
        path=source_path,
        includes=(f'"model_{model_name}.h"', '"cimp_tensor.h"', '"infra_pytorch.h"'),
        body="\n\n".join(param_sections + [fns_body]),
        provenance_lines=(
            f"Auto-generated CNN forward pass for {model_name}",
            f"Weight format: {qmodel.weight_format} | out_frac={qmodel.output_frac}",
        ),
    )

    # Golden per-sample CSV using batched float32 IR and Python q15 IR
    golden_csv = cfg.generated_dir / f"golden_cnn_{model_name}.csv"
    golden_csv.parent.mkdir(parents=True, exist_ok=True)
    mismatches = 0
    scale_inv = math.ldexp(1.0, -qmodel.output_frac)
    f32_logits_all, _ = run_cnn_graph_f32_batch(graph, eval_images_f32, batch_size=64)
    q_raw_all, _ = run_cnn_graph_q15_batch(qmodel, eval_images_u8, batch_size=64)

    with golden_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "file_name",
                "true_label",
                "py_f32_logit0",
                "py_f32_logit1",
                "py_f32_margin",
                "py_f32_label",
                "py_q15_logit0_raw",
                "py_q15_logit1_raw",
                "py_q15_margin_raw",
                "py_q15_margin_f64",
                "py_q15_label",
            ]
        )
        for i, fname in enumerate(eval_file_names):
            f32_logits = f32_logits_all[i]
            f32_margin = float(f32_logits[1] - f32_logits[0])
            f32_label = 1 if f32_margin > graph.threshold_logit else 0

            q_raw = q_raw_all[i]
            q_margin_raw = int(q_raw[1]) - int(q_raw[0])
            q_label = 1 if q_margin_raw > qmodel.threshold_logit_q else 0
            if q_label != f32_label:
                mismatches += 1
            writer.writerow(
                [
                    i,
                    fname,
                    int(eval_true_labels[i]),
                    f"{float(f32_logits[0]):.9g}",
                    f"{float(f32_logits[1]):.9g}",
                    f"{f32_margin:.9g}",
                    f32_label,
                    int(q_raw[0]),
                    int(q_raw[1]),
                    q_margin_raw,
                    f"{float(q_margin_raw * scale_inv):.9g}",
                    q_label,
                ]
            )

    return ExportedCnnModel(
        model_name=model_name,
        header_path=header_path,
        source_path=source_path,
        header_sha256=header_sha,
        source_sha256=source_sha,
        golden_csv_path=golden_csv,
        checkpoint_path=graph.checkpoint_path,
        checkpoint_sha256=graph.checkpoint_sha256,
        weight_format=qmodel.weight_format,
        memory_plan=plan,
        q15_label_mismatches=mismatches,
    )


def write_registry_and_reports(
    cfg: CImpConfig,
    exported_tabular: dict[str, object],
    exported_cnns: list[ExportedCnnModel],
    extra_manifest: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    """Emit `cimp_models_registry.h`, `memory_report.json`, and `export_manifest.json`."""
    all_cnn_names = (
        "ultra_tiny",
        "tiny_cnn",
        "ds_cnn",
        "small_cnn",
        "mobilenet_like",
        "tiny_resnet",
        "torchvision_mobilenet",
    )
    exported_cnn_map = {m.model_name: m for m in exported_cnns}

    has_lr = 1 if "lr_dot" in exported_tabular else 0
    has_mlp = 1 if "mlp_16_dot" in exported_tabular else 0

    lines: list[str] = [
        f"#define CIMP_HAS_MODEL_LR_DOT {has_lr}",
        f"#define CIMP_HAS_MODEL_MLP_16_DOT {has_mlp}",
    ]
    if has_lr:
        lines.append('#include "model_lr_dot.h"')
    if has_mlp:
        lines.append('#include "model_mlp_16_dot.h"')

    max_arena_f32 = 64
    max_arena_q15 = 64
    for cname in all_cnn_names:
        upper = cname.upper()
        if cname in exported_cnn_map:
            em = exported_cnn_map[cname]
            max_arena_f32 = max(max_arena_f32, em.memory_plan.peak_arena_f32_bytes)
            max_arena_q15 = max(max_arena_q15, em.memory_plan.peak_arena_q15_bytes)
            lines.append(f"#define CIMP_HAS_CNN_{upper} 1")
            lines.append(f'#include "model_{cname}.h"')
        else:
            lines.append(f"#define CIMP_HAS_CNN_{upper} 0")

    lines.append(f"#define CIMP_MAX_CNN_ARENA_F32_BYTES {max_arena_f32}")
    lines.append(f"#define CIMP_MAX_CNN_ARENA_Q15_BYTES {max_arena_q15}")
    lines.append(f"#define CIMP_MAX_CNN_ARENA_BYTES {max(max_arena_f32, max_arena_q15)}")

    registry_path = cfg.generated_dir / "cimp_models_registry.h"
    registry_sha = write_header(
        path=registry_path,
        guard="CIMP_MODELS_REGISTRY_H",
        includes=(),
        body="\n".join(lines),
        provenance_lines=(
            f"Generated model registry for target={cfg.effective_target_tag}",
        ),
    )

    mem_report = {
        "target_tag": cfg.effective_target_tag,
        "budgets": {
            "flash_budget_bytes": cfg.cnn.flash_budget_bytes,
            "sram_budget_bytes": cfg.cnn.sram_budget_bytes,
            "psram_budget_bytes": cfg.cnn.psram_budget_bytes,
        },
        "models": [
            {
                "model_name": em.model_name,
                "weight_format": em.weight_format,
                "weight_flash_bytes": em.memory_plan.weight_flash_bytes,
                "peak_arena_f32_bytes": em.memory_plan.peak_arena_f32_bytes,
                "peak_arena_q15_bytes": em.memory_plan.peak_arena_q15_bytes,
                "fits_internal_sram_q15": em.memory_plan.fits_internal_sram_q15,
                "fits_psram_q15": em.memory_plan.fits_psram_q15,
                "fits_flash": em.memory_plan.fits_flash,
            }
            for em in exported_cnns
        ],
    }
    mem_report_path = cfg.results_dir / "memory_report.json"
    mem_report_path.parent.mkdir(parents=True, exist_ok=True)
    mem_report_path.write_text(json.dumps(mem_report, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "target_tag": cfg.effective_target_tag,
        "config_path": str(cfg.config_path),
        "seed": cfg.seed,
        "registry_header": str(registry_path),
        "registry_sha256": registry_sha,
        "tabular_models": {
            name: {
                "header": str(getattr(obj, "header_path")),
                "header_sha256": getattr(obj, "header_sha256"),
                "weight_bits": getattr(obj, "weight_bits"),
                "weight_bytes": getattr(obj, "weight_bytes"),
                "q15_label_mismatches": getattr(obj, "q15_label_mismatches"),
            }
            for name, obj in exported_tabular.items()
        },
        "cnn_models": {
            em.model_name: {
                "header": str(em.header_path),
                "source": str(em.source_path),
                "header_sha256": em.header_sha256,
                "source_sha256": em.source_sha256,
                "checkpoint_sha256": em.checkpoint_sha256,
                "weight_format": em.weight_format,
                "weight_flash_bytes": em.memory_plan.weight_flash_bytes,
                "peak_arena_q15_bytes": em.memory_plan.peak_arena_q15_bytes,
                "q15_label_mismatches": em.q15_label_mismatches,
            }
            for em in exported_cnns
        },
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path = cfg.generated_dir / "export_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return registry_path, mem_report_path, manifest_path


def export_cnn_models(
    cfg: CImpConfig,
    calib_images_f32: np.ndarray,
    eval_file_names: tuple[str, ...],
    eval_true_labels: np.ndarray,
    eval_images_u8: np.ndarray,
    eval_images_f32: np.ndarray,
    load_cnn_ir_fn=None,
) -> list[ExportedCnnModel]:
    """Load, trace, fold, calibrate, quantize, plan memory, and emit C code for configured CNNs."""
    from .cnn_graph import load_cnn_ir

    loader = load_cnn_ir_fn if load_cnn_ir_fn is not None else load_cnn_ir
    exported: list[ExportedCnnModel] = []
    for model_name in cfg.cnn.models:
        graph = loader(cfg, model_name)
        qmodel = quantize_cnn_graph(graph, calib_images_f32, cfg)
        plan = build_memory_plan(graph, qmodel.weight_format, qmodel.weight_bytes, cfg)
        exported.append(
            emit_cnn_model(
                cfg=cfg,
                graph=graph,
                qmodel=qmodel,
                plan=plan,
                eval_file_names=eval_file_names,
                eval_true_labels=eval_true_labels,
                eval_images_u8=eval_images_u8,
                eval_images_f32=eval_images_f32,
            )
        )
    return exported
