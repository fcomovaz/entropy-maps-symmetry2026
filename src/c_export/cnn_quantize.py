"""Post-training fixed-point CNN quantizer (q15 / int8_per_channel) and bit-exact Python mirror."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .cnn_graph import CnnIrGraph, CnnIrOp, run_cnn_graph_f32_batch
from .config import CImpConfig
from .fixed_point import (
    choose_frac_bits,
    dequantize,
    quantize,
    round_shift_right_array,
    sat16,
    sat_array,
)


@dataclass(frozen=True, slots=True)
class QuantizedCnnOp:
    """Fixed-point quantization parameters for a single `CnnIrOp`."""

    op_id: str
    op_type: str
    weight_format: str            # "none", "q15", or "int8_per_channel"
    in_fracs: tuple[int, ...]
    out_frac: int
    w_frac: int = 0               # for uniform q15 weights
    out_shift: int = 0            # shift applied to raw int64 accumulator -> out_frac (for q15 / avgpool / scale)
    add_shift_a: int = 0
    add_shift_b: int = 0
    w_q15: np.ndarray | None = None
    w_q8: np.ndarray | None = None
    bias_acc: np.ndarray | None = None     # int64
    mul_per_ch: np.ndarray | None = None   # int32 (for w8)
    shift_per_ch: np.ndarray | None = None # int8 (for w8)


@dataclass(frozen=True, slots=True)
class QuantizedCnnModel:
    """Complete quantized CNN representation and calibration provenance."""

    graph: CnnIrGraph
    weight_format: str             # "q15" or "int8_per_channel"
    weight_bytes: int
    input_frac: int
    output_frac: int
    norm_lut_q15: np.ndarray       # [256] int16
    threshold_logit_q: int         # int32 threshold on (logit[1] - logit[0]) at output_frac
    tensor_fracs: dict[str, int]
    ops: tuple[QuantizedCnnOp, ...]
    q15_saturation_count: int


def _select_multiplier_and_shift(alpha: float) -> tuple[int, int]:
    """Express positive real `alpha` cleanly as `mul * 2^(-shift)` with `mul` in [2^19, 2^20)."""
    if alpha <= 0.0:
        return 0, 0
    # Choose shift so floor(alpha * 2^shift + 0.5) ~= 2^19 (keeps dot * mul well below 2^55)
    exp = math.floor(math.log2(alpha))
    shift = 19 - exp
    scale = math.ldexp(1.0, shift)
    mul = int(math.floor(alpha * scale + 0.5))
    return mul, shift


def _select_weight_format(graph: CnnIrGraph, cfg: CImpConfig) -> str:
    """Choose `q15` or `int8_per_channel` respecting `flash_budget_bytes`."""
    total_weights = 0
    total_biases = 0
    for op in graph.ops:
        if op.weights is not None:
            total_weights += int(op.weights.size)
        if op.bias is not None:
            total_biases += int(op.bias.size)

    q15_weight_bytes = total_weights * 2 + total_biases * 8
    if cfg.cnn.weight_granularity == "int8_per_channel":
        return "int8_per_channel"
    if (
        q15_weight_bytes > cfg.cnn.flash_budget_bytes
        and "int8_per_channel" in cfg.cnn.allowed_weight_formats
    ):
        return "int8_per_channel"
    return "q15"


def quantize_cnn_graph(
    graph: CnnIrGraph,
    calib_images_f32: np.ndarray,
    cfg: CImpConfig,
) -> QuantizedCnnModel:
    """Calibrate activations on `calib_images_f32` [N, C, H, W] and quantize `CnnIrGraph`."""
    n_calib = int(calib_images_f32.shape[0])
    if n_calib == 0:
        raise ValueError("Calibration image set cannot be empty")

    # 1. Calibrate tensor max-abs values across sample activations in batches
    _, intermediates = run_cnn_graph_f32_batch(
        graph, calib_images_f32, batch_size=64, collect_intermediates=True
    )

    pct = cfg.cnn.activation_range_percentile
    tensor_max_abs: dict[str, float] = {}
    for name, arr in intermediates.items():
        all_abs = np.abs(arr).ravel()
        if pct >= 99.999:
            val = float(np.max(all_abs))
        else:
            val = float(np.percentile(all_abs, pct))
        tensor_max_abs[name] = max(val, 1e-3)

    # 2. Assign Qm.n frac_bits to every tensor
    in_frac = cfg.cnn.input_frac_bits
    tensor_fracs: dict[str, int] = {graph.input_name: in_frac}
    op_map = {op.output: op for op in graph.ops}

    for op in graph.ops:
        if op.op_type in ("maxpool2d", "global_avgpool2d"):
            # Preserve incoming Q-format cleanly
            tensor_fracs[op.output] = tensor_fracs[op.inputs[0]]
        elif op.op_type == "hardsigmoid":
            # Strictly bounded in [0, 1]; Q2.14 represents 1.0 = 16384 cleanly without 1-LSB clamp
            tensor_fracs[op.output] = 14
        elif op.is_final_logits:
            # Final logits live in int32; keep high fractional precision (e.g., Q16 or Q12)
            tensor_fracs[op.output] = choose_frac_bits(
                tensor_max_abs[op.output], storage_bits=32, headroom_bits=2, min_frac=8
            )
            tensor_fracs[op.output] = min(tensor_fracs[op.output], 16)
        else:
            tensor_fracs[op.output] = choose_frac_bits(
                tensor_max_abs[op.output], storage_bits=16, headroom_bits=1, min_frac=2
            )

    _ = op_map
    weight_format = _select_weight_format(graph, cfg)

    # Build 256-entry normalized input LUT at Q(in_frac)
    lut_float = (np.arange(256, dtype=np.float64) / 255.0 - graph.norm_mean) / graph.norm_std
    norm_lut_q, _ = quantize(lut_float, in_frac, bits=16)

    q_ops: list[QuantizedCnnOp] = []
    total_weight_bytes = int(norm_lut_q.size * 2)
    total_sat = 0

    for op in graph.ops:
        in_fracs = tuple(tensor_fracs[inp] for inp in op.inputs)
        out_frac = tensor_fracs[op.output]

        if op.op_type in ("conv2d", "dwconv2d", "linear"):
            assert op.weights is not None and op.bias is not None
            w_f64 = op.weights.astype(np.float64)
            b_f64 = op.bias.astype(np.float64)

            if weight_format == "q15":
                max_w = float(np.max(np.abs(w_f64))) if w_f64.size else 1.0
                w_frac = choose_frac_bits(max_w, storage_bits=16, min_frac=2)
                w_q15, n_sat = quantize(w_f64, w_frac, bits=16)
                total_sat += n_sat
                acc_frac = in_fracs[0] + w_frac
                out_shift = acc_frac - out_frac
                b_acc, _ = quantize(b_f64, acc_frac, bits=64)
                total_weight_bytes += int(w_q15.size * 2 + b_acc.size * 8)
                q_ops.append(
                    QuantizedCnnOp(
                        op_id=op.op_id,
                        op_type=op.op_type,
                        weight_format="q15",
                        in_fracs=in_fracs,
                        out_frac=out_frac,
                        w_frac=w_frac,
                        out_shift=out_shift,
                        w_q15=w_q15.astype(np.int16),
                        bias_acc=b_acc.astype(np.int64),
                    )
                )
            else:
                cout = int(w_f64.shape[0])
                w_flat = w_f64.reshape(cout, -1)
                w_q8 = np.zeros_like(w_flat, dtype=np.int8)
                mul_ch = np.zeros(cout, dtype=np.int32)
                shift_ch = np.zeros(cout, dtype=np.int8)
                bias_acc = np.zeros(cout, dtype=np.int64)

                for oc in range(cout):
                    max_oc = float(np.max(np.abs(w_flat[oc])))
                    scale_c = max(max_oc / 127.0, 1e-12)
                    q_row = np.clip(np.floor(w_flat[oc] / scale_c + 0.5), -127, 127).astype(
                        np.int8
                    )
                    w_q8[oc] = q_row
                    alpha_c = scale_c * math.ldexp(1.0, out_frac - in_fracs[0])
                    mul_c, sh_c = _select_multiplier_and_shift(alpha_c)
                    mul_ch[oc] = mul_c
                    shift_ch[oc] = sh_c
                    bias_acc[oc] = int(
                        math.floor(b_f64[oc] * math.ldexp(1.0, out_frac + sh_c) + 0.5)
                    )

                total_weight_bytes += int(
                    w_q8.size * 1 + bias_acc.size * 8 + mul_ch.size * 4 + shift_ch.size * 1
                )
                q_ops.append(
                    QuantizedCnnOp(
                        op_id=op.op_id,
                        op_type=op.op_type,
                        weight_format="int8_per_channel",
                        in_fracs=in_fracs,
                        out_frac=out_frac,
                        w_q8=w_q8.reshape(op.weights.shape),
                        bias_acc=bias_acc,
                        mul_per_ch=mul_ch,
                        shift_per_ch=shift_ch,
                    )
                )

        elif op.op_type in ("relu", "global_avgpool2d"):
            q_ops.append(
                QuantizedCnnOp(
                    op_id=op.op_id,
                    op_type=op.op_type,
                    weight_format="none",
                    in_fracs=in_fracs,
                    out_frac=out_frac,
                    out_shift=in_fracs[0] - out_frac,
                )
            )
        elif op.op_type in ("hardswish", "hardsigmoid", "maxpool2d"):
            q_ops.append(
                QuantizedCnnOp(
                    op_id=op.op_id,
                    op_type=op.op_type,
                    weight_format="none",
                    in_fracs=in_fracs,
                    out_frac=out_frac,
                )
            )
        elif op.op_type == "add":
            q_ops.append(
                QuantizedCnnOp(
                    op_id=op.op_id,
                    op_type=op.op_type,
                    weight_format="none",
                    in_fracs=in_fracs,
                    out_frac=out_frac,
                    add_shift_a=in_fracs[0] - out_frac,
                    add_shift_b=in_fracs[1] - out_frac,
                )
            )
        elif op.op_type == "scale_channels":
            q_ops.append(
                QuantizedCnnOp(
                    op_id=op.op_id,
                    op_type=op.op_type,
                    weight_format="none",
                    in_fracs=in_fracs,
                    out_frac=out_frac,
                    out_shift=(in_fracs[0] + in_fracs[1]) - out_frac,
                )
            )
        else:
            raise ValueError(f"Unsupported IR op for quantization: {op.op_type}")

    out_frac = tensor_fracs[graph.output_name]
    thresh_q = int(math.floor(graph.threshold_logit * math.ldexp(1.0, out_frac) + 0.5))
    return QuantizedCnnModel(
        graph=graph,
        weight_format=weight_format,
        weight_bytes=total_weight_bytes,
        input_frac=in_frac,
        output_frac=out_frac,
        norm_lut_q15=norm_lut_q.astype(np.int16),
        threshold_logit_q=thresh_q,
        tensor_fracs=tensor_fracs,
        ops=tuple(q_ops),
        q15_saturation_count=total_sat,
    )


def _conv2d_q15_batch(
    x_q: np.ndarray,
    op: CnnIrOp,
    qop: QuantizedCnnOp,
) -> np.ndarray:
    """Bit-exact batched Python mirror of `cimp_conv2d_q15`, `_q15_w8`, `cimp_dwconv2d_q15`, `_q15_w8`.

    Computes the exact integer dot-product accumulation via `torch.nn.functional.conv2d`
    in `float64` (exact up to 2^53; maximum intermediate accumulation is < 2^40) and then applies
    exact `int64` bias addition, half-up rounding right-shift, optional fused ReLU, and int16 saturation.
    """
    cout = int(op.out_shape[0])
    is_w8 = qop.weight_format == "int8_per_channel"
    w_arr = qop.w_q8 if is_w8 else qop.w_q15  # type: ignore[union-attr]
    assert w_arr is not None and qop.bias_acc is not None

    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(x_q, dtype=np.float64))
        wt = torch.from_numpy(np.ascontiguousarray(w_arr, dtype=np.float64))
        acc_f64 = F.conv2d(
            xt,
            wt,
            bias=None,
            stride=op.stride,
            padding=op.padding,
            groups=op.groups,
        )
    acc = np.ascontiguousarray(acc_f64.numpy(), dtype=np.int64)

    if is_w8:
        assert qop.mul_per_ch is not None and qop.shift_per_ch is not None
        out = np.empty_like(acc, dtype=np.int16)
        for oc in range(cout):
            scaled = acc[:, oc] * int(qop.mul_per_ch[oc]) + int(qop.bias_acc[oc])
            shifted = round_shift_right_array(scaled, int(qop.shift_per_ch[oc]))
            if op.relu_fused:
                shifted = np.maximum(0, shifted)
            out[:, oc] = sat_array(shifted, 16).astype(np.int16)
        return out

    acc += qop.bias_acc.reshape(1, cout, 1, 1)
    shifted = round_shift_right_array(acc, qop.out_shift)
    if op.relu_fused:
        shifted = np.maximum(0, shifted)
    return sat_array(shifted, 16).astype(np.int16)


def _conv2d_q15_ref(
    x_q: np.ndarray,
    op: CnnIrOp,
    qop: QuantizedCnnOp,
) -> np.ndarray:
    """Bit-exact Python mirror of single-sample `cimp_conv2d_q15` and `cimp_dwconv2d_q15`."""
    return _conv2d_q15_batch(x_q[None, ...], op, qop)[0]


def _execute_q15_chunk(
    qmodel: QuantizedCnnModel,
    imgs_u8_chunk: np.ndarray,
) -> np.ndarray:
    """Execute quantized CNN graph over `[B, H, W]` uint8 batch and return raw `[B, C_out]` int32."""
    b_size = int(imgs_u8_chunk.shape[0])
    tensors: dict[str, np.ndarray] = {
        qmodel.graph.input_name: qmodel.norm_lut_q15[
            np.ascontiguousarray(imgs_u8_chunk, dtype=np.uint8)
        ][:, None, :, :]
    }

    for op, qop in zip(qmodel.graph.ops, qmodel.ops):
        if op.op_type in ("conv2d", "dwconv2d"):
            res = _conv2d_q15_batch(tensors[op.inputs[0]], op, qop)
        elif op.op_type == "relu":
            v = tensors[op.inputs[0]].astype(np.int64)
            pos = np.where(v > 0, round_shift_right_array(v, qop.out_shift), 0)
            res = sat_array(pos, 16).astype(np.int16)
        elif op.op_type == "hardswish":
            inv6_q24 = np.int64(2796203)
            three_q = np.int64(3 << qop.in_fracs[0])
            id_shift = qop.in_fracs[0] - qop.out_frac
            prod_shift = 2 * qop.in_fracs[0] + 24 - qop.out_frac
            x = tensors[op.inputs[0]].astype(np.int64)
            mid = round_shift_right_array(x * (x + three_q) * inv6_q24, prod_shift)
            high = round_shift_right_array(x, id_shift)
            out64 = np.where(x <= -three_q, 0, np.where(x >= three_q, high, mid))
            res = sat_array(out64, 16).astype(np.int16)
        elif op.op_type == "hardsigmoid":
            inv6_q24 = np.int64(2796203)
            three_q = np.int64(3 << qop.in_fracs[0])
            one_q = np.int64(32767 if qop.out_frac >= 15 else (1 << qop.out_frac))
            prod_shift = qop.in_fracs[0] + 24 - qop.out_frac
            x = tensors[op.inputs[0]].astype(np.int64)
            mid = np.clip(round_shift_right_array((x + three_q) * inv6_q24, prod_shift), 0, one_q)
            out64 = np.where(x <= -three_q, 0, np.where(x >= three_q, one_q, mid))
            res = sat_array(out64, 16).astype(np.int16)
        elif op.op_type == "maxpool2d":
            v = tensors[op.inputs[0]]
            b, c, h, w = v.shape
            res = v.reshape(b, c, h // 2, 2, w // 2, 2).max(axis=(3, 5)).astype(np.int16)
        elif op.op_type == "global_avgpool2d":
            v = tensors[op.inputs[0]].astype(np.int64)
            b, c, h, w = v.shape
            hw = h * w
            half_hw = hw // 2
            sums = np.sum(v.reshape(b, c, hw), axis=2, dtype=np.int64)
            means = np.where(sums >= 0, (sums + half_hw) // hw, -((-sums + half_hw) // hw))
            res = sat_array(round_shift_right_array(means, qop.out_shift), 16).astype(
                np.int16
            ).reshape(b, c, 1, 1)
        elif op.op_type == "linear":
            x_mat = tensors[op.inputs[0]].reshape(b_size, -1).astype(np.int64)
            assert qop.bias_acc is not None
            if qop.weight_format == "int8_per_channel":
                assert qop.w_q8 is not None and qop.mul_per_ch is not None and qop.shift_per_ch is not None
                dots = x_mat @ qop.w_q8.astype(np.int64).T
                shifted_arr = np.empty_like(dots, dtype=np.int64)
                for oc in range(dots.shape[1]):
                    scaled = dots[:, oc] * int(qop.mul_per_ch[oc]) + int(qop.bias_acc[oc])
                    sh = round_shift_right_array(scaled, int(qop.shift_per_ch[oc]))
                    if op.relu_fused:
                        sh = np.maximum(0, sh)
                    shifted_arr[:, oc] = sh
            else:
                assert qop.w_q15 is not None
                acc = (x_mat @ qop.w_q15.astype(np.int64).T) + qop.bias_acc[None, :]
                shifted_arr = round_shift_right_array(acc, qop.out_shift)
                if op.relu_fused:
                    shifted_arr = np.maximum(0, shifted_arr)
            if op.is_final_logits:
                res = sat_array(shifted_arr, 32).astype(np.int32).reshape(b_size, *op.out_shape)
            else:
                res = sat_array(shifted_arr, 16).astype(np.int16).reshape(b_size, *op.out_shape)
        elif op.op_type == "add":
            va = round_shift_right_array(tensors[op.inputs[0]].astype(np.int64), qop.add_shift_a)
            vb = round_shift_right_array(tensors[op.inputs[1]].astype(np.int64), qop.add_shift_b)
            s = va + vb
            if op.relu_fused:
                s = np.maximum(0, s)
            res = sat_array(s, 16).astype(np.int16)
        elif op.op_type == "scale_channels":
            feat = tensors[op.inputs[0]].astype(np.int64)
            scale = tensors[op.inputs[1]].astype(np.int64).reshape(b_size, -1, 1, 1)
            res = sat_array(round_shift_right_array(feat * scale, qop.out_shift), 16).astype(
                np.int16
            )
        else:
            raise ValueError(f"Unsupported IR op in q15 mirror: {op.op_type}")

        tensors[op.output] = res

    return np.ascontiguousarray(tensors[qmodel.graph.output_name].reshape(b_size, -1), dtype=np.int32)


def run_cnn_graph_q15(
    qmodel: QuantizedCnnModel,
    img_u8: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact Python fixed-point execution over `[H, W]` uint8 image.

    Returns:
        Tuple `(raw_int32_logits, float64_dequantized_logits)`.
    """
    raw_logits = _execute_q15_chunk(qmodel, img_u8[None, ...])[0]
    return raw_logits, dequantize(raw_logits, qmodel.output_frac)


def run_cnn_graph_q15_batch(
    qmodel: QuantizedCnnModel,
    imgs_u8: np.ndarray,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Bit-exact Python fixed-point batched execution over `[N, H, W]` uint8 images."""
    arr = np.ascontiguousarray(imgs_u8, dtype=np.uint8)
    n_total = int(arr.shape[0])
    if n_total == 0:
        num_classes = qmodel.graph.ops[-1].out_shape[0]
        return np.empty((0, num_classes), dtype=np.int32), np.empty((0, num_classes), dtype=np.float64)

    chunks: list[np.ndarray] = []
    step = max(1, int(batch_size))
    for start in range(0, n_total, step):
        end = min(n_total, start + step)
        chunks.append(_execute_q15_chunk(qmodel, arr[start:end]))
    raw_all = np.ascontiguousarray(np.vstack(chunks), dtype=np.int32)
    return raw_all, dequantize(raw_all, qmodel.output_frac)
