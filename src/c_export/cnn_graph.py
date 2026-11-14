"""FX-based PyTorch CNN graph extraction, analytical Conv+BN folding, and float32 IR interpreter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

from src.models.cnn.architectures.registry import create_model

from .c_writer import sha256_file
from .config import CImpConfig


@dataclass(frozen=True, slots=True)
class CnnIrOp:
    """Single fused/lowered CNN IR operation using NCHW [C, H, W] tensor identifiers."""

    op_id: str
    op_type: str  # conv2d, dwconv2d, batchnorm2d, relu, hardswish, hardsigmoid, maxpool2d, global_avgpool2d, linear, add, scale_channels
    inputs: tuple[str, ...]
    output: str
    in_shapes: tuple[tuple[int, int, int], ...]
    out_shape: tuple[int, int, int]
    weights: np.ndarray | None = None
    bias: np.ndarray | None = None
    bn_gamma: np.ndarray | None = None
    bn_beta: np.ndarray | None = None
    bn_mean: np.ndarray | None = None
    bn_var: np.ndarray | None = None
    bn_eps: float = 1e-5
    kernel_size: tuple[int, int] = (1, 1)
    stride: int = 1
    padding: int = 0
    groups: int = 1
    relu_fused: bool = False
    is_final_logits: bool = False


@dataclass(frozen=True, slots=True)
class CnnIrGraph:
    """Complete folded CNN IR graph ready for calibration, quantization, memory planning, and C codegen."""

    model_name: str
    checkpoint_path: Path | None
    checkpoint_sha256: str
    selected_threshold: float
    threshold_logit: float
    input_name: str
    input_shape: tuple[int, int, int]  # (C, H, W)
    output_name: str
    norm_mean: float
    norm_std: float
    ops: tuple[CnnIrOp, ...]


def _as_pair(val: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize an int or 2-tuple parameter to `(h, w)`."""
    if isinstance(val, tuple):
        return (int(val[0]), int(val[1]))
    return (int(val), int(val))


def _shape_to_chw(meta_shape: torch.Size | tuple[int, ...]) -> tuple[int, int, int]:
    """Convert `(B, C, H, W)` or `(B, C)` FX tensor shape into `(C, H, W)`."""
    if len(meta_shape) == 4:
        return (int(meta_shape[1]), int(meta_shape[2]), int(meta_shape[3]))
    if len(meta_shape) == 2:
        return (int(meta_shape[1]), 1, 1)
    raise ValueError(f"Unsupported CNN FX tensor rank {len(meta_shape)}: {tuple(meta_shape)}")


def _extract_raw_ops(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int],
) -> tuple[str, str, list[CnnIrOp], dict[str, str]]:
    """Trace `model` in eval mode with `torch.fx` and lower nodes into raw IR ops + tensor aliases."""
    model.eval()
    traced = fx.symbolic_trace(model)
    dummy = torch.zeros((1, *input_shape), dtype=torch.float32)
    ShapeProp(traced).propagate(dummy)

    modules: dict[str, nn.Module] = dict(traced.named_modules())
    placeholder_name: str | None = None
    output_tensor: str | None = None
    aliases: dict[str, str] = {}
    raw_ops: list[CnnIrOp] = []

    def resolve_alias(name: str) -> str:
        cur = name
        while cur in aliases:
            cur = aliases[cur]
        return cur

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            placeholder_name = str(node.name)
        elif node.op == "call_module":
            mod = modules[str(node.target)]
            arg_node = node.args[0]
            assert isinstance(arg_node, fx.Node)
            in_name = resolve_alias(str(arg_node.name))
            in_chw = _shape_to_chw(arg_node.meta["tensor_meta"].shape)
            out_chw = _shape_to_chw(node.meta["tensor_meta"].shape)

            if isinstance(mod, (nn.Dropout, nn.Identity)):
                aliases[str(node.name)] = in_name
            elif isinstance(mod, nn.Conv2d):
                kh, kw = _as_pair(mod.kernel_size)
                sh, sw = _as_pair(mod.stride)
                ph, pw = _as_pair(mod.padding)
                dh, dw = _as_pair(mod.dilation)
                if sh != sw or ph != pw or dh != 1 or dw != 1:
                    raise ValueError(
                        f"Unsupported Conv2d geometry in {model_name}:{node.name}: "
                        f"stride={(sh, sw)}, padding={(ph, pw)}, dilation={(dh, dw)}"
                    )
                groups = int(mod.groups)
                if groups != 1 and groups != int(mod.in_channels):
                    raise ValueError(
                        f"Only groups=1 or depthwise groups=in_channels supported, got groups={groups}"
                    )
                op_type = "dwconv2d" if (groups == int(mod.in_channels) and groups > 1) else "conv2d"
                w_np = np.ascontiguousarray(mod.weight.detach().cpu().numpy(), dtype=np.float32)
                b_np = (
                    np.ascontiguousarray(mod.bias.detach().cpu().numpy(), dtype=np.float32)
                    if mod.bias is not None
                    else np.zeros(int(mod.out_channels), dtype=np.float32)
                )
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type=op_type,
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                        weights=w_np,
                        bias=b_np,
                        kernel_size=(kh, kw),
                        stride=sh,
                        padding=ph,
                        groups=groups,
                    )
                )
            elif isinstance(mod, nn.BatchNorm2d):
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="batchnorm2d",
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                        bn_gamma=np.ascontiguousarray(mod.weight.detach().cpu().numpy(), dtype=np.float64),
                        bn_beta=np.ascontiguousarray(mod.bias.detach().cpu().numpy(), dtype=np.float64),
                        bn_mean=np.ascontiguousarray(mod.running_mean.detach().cpu().numpy(), dtype=np.float64),
                        bn_var=np.ascontiguousarray(mod.running_var.detach().cpu().numpy(), dtype=np.float64),
                        bn_eps=float(mod.eps),
                    )
                )
            elif isinstance(mod, nn.ReLU):
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="relu",
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                    )
                )
            elif isinstance(mod, nn.Hardswish):
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="hardswish",
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                    )
                )
            elif isinstance(mod, nn.Hardsigmoid):
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="hardsigmoid",
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                    )
                )
            elif isinstance(mod, nn.MaxPool2d):
                kh, kw = _as_pair(mod.kernel_size)
                sh, sw = _as_pair(mod.stride if mod.stride is not None else mod.kernel_size)
                ph, pw = _as_pair(mod.padding)
                if (kh, kw) != (2, 2) or (sh, sw) != (2, 2) or (ph, pw) != (0, 0):
                    raise ValueError(
                        f"Unsupported MaxPool2d in {model_name}:{node.name}: k={(kh, kw)} s={(sh, sw)} p={(ph, pw)}"
                    )
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="maxpool2d",
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                        kernel_size=(2, 2),
                        stride=2,
                        padding=0,
                    )
                )
            elif isinstance(mod, nn.AdaptiveAvgPool2d):
                out_sz = _as_pair(mod.output_size)
                if out_sz != (1, 1):
                    raise ValueError(f"Only AdaptiveAvgPool2d(1) supported, got {mod.output_size}")
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="global_avgpool2d",
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                    )
                )
            elif isinstance(mod, nn.Linear):
                w_np = np.ascontiguousarray(mod.weight.detach().cpu().numpy(), dtype=np.float32)
                b_np = (
                    np.ascontiguousarray(mod.bias.detach().cpu().numpy(), dtype=np.float32)
                    if mod.bias is not None
                    else np.zeros(int(mod.out_features), dtype=np.float32)
                )
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="linear",
                        inputs=(in_name,),
                        output=str(node.name),
                        in_shapes=(in_chw,),
                        out_shape=out_chw,
                        weights=w_np,
                        bias=b_np,
                    )
                )
            else:
                raise ValueError(f"Unsupported nn.Module {type(mod).__name__} in {model_name}:{node.name}")

        elif node.op in ("call_function", "call_method"):
            target_str = str(node.target)
            if (
                node.target is torch.flatten
                or (node.op == "call_method" and str(node.target) == "flatten")
            ):
                arg_node = node.args[0]
                assert isinstance(arg_node, fx.Node)
                aliases[str(node.name)] = resolve_alias(str(arg_node.name))
            elif "add" in target_str:
                a_node, b_node = node.args[0], node.args[1]
                assert isinstance(a_node, fx.Node) and isinstance(b_node, fx.Node)
                a_chw = _shape_to_chw(a_node.meta["tensor_meta"].shape)
                b_chw = _shape_to_chw(b_node.meta["tensor_meta"].shape)
                out_chw = _shape_to_chw(node.meta["tensor_meta"].shape)
                if a_chw != b_chw:
                    raise ValueError(f"Residual add shape mismatch: {a_chw} vs {b_chw}")
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="add",
                        inputs=(resolve_alias(str(a_node.name)), resolve_alias(str(b_node.name))),
                        output=str(node.name),
                        in_shapes=(a_chw, b_chw),
                        out_shape=out_chw,
                    )
                )
            elif "mul" in target_str:
                a_node, b_node = node.args[0], node.args[1]
                assert isinstance(a_node, fx.Node) and isinstance(b_node, fx.Node)
                a_chw = _shape_to_chw(a_node.meta["tensor_meta"].shape)
                b_chw = _shape_to_chw(b_node.meta["tensor_meta"].shape)
                out_chw = _shape_to_chw(node.meta["tensor_meta"].shape)
                a_name = resolve_alias(str(a_node.name))
                b_name = resolve_alias(str(b_node.name))
                if b_chw[1] == 1 and b_chw[2] == 1 and a_chw[0] == b_chw[0]:
                    feat_name, feat_chw, scale_name, scale_chw = a_name, a_chw, b_name, b_chw
                elif a_chw[1] == 1 and a_chw[2] == 1 and a_chw[0] == b_chw[0]:
                    feat_name, feat_chw, scale_name, scale_chw = b_name, b_chw, a_name, a_chw
                else:
                    raise ValueError(f"Unsupported mul shapes: {a_chw} * {b_chw}")
                raw_ops.append(
                    CnnIrOp(
                        op_id=str(node.name),
                        op_type="scale_channels",
                        inputs=(feat_name, scale_name),
                        output=str(node.name),
                        in_shapes=(feat_chw, scale_chw),
                        out_shape=out_chw,
                    )
                )
            else:
                raise ValueError(f"Unsupported FX op {node.op}:{node.target} in {model_name}:{node.name}")
        elif node.op == "output":
            ret_node = node.args[0]
            assert isinstance(ret_node, fx.Node)
            output_tensor = resolve_alias(str(ret_node.name))

    if placeholder_name is None or output_tensor is None:
        raise RuntimeError(f"Failed to locate input/output nodes in {model_name}")
    return placeholder_name, output_tensor, raw_ops, aliases


def _fold_batchnorms_and_relus(
    raw_ops: list[CnnIrOp],
    output_tensor: str,
    fuse_relu: bool,
) -> tuple[list[CnnIrOp], str]:
    """Fold every `BatchNorm2d` into its feeding `conv2d`/`dwconv2d` and optionally fuse `ReLU`."""
    producer: dict[str, CnnIrOp] = {op.output: op for op in raw_ops}
    remap: dict[str, str] = {}

    def resolve_t(t: str) -> str:
        cur = t
        while cur in remap:
            cur = remap[cur]
        return cur

    folded_ops: list[CnnIrOp] = []
    # 1. Fold BatchNorm2d into preceding Conv2d/DWConv2d
    op_by_out: dict[str, int] = {}
    for op in raw_ops:
        remapped_inputs = tuple(resolve_t(inp) for inp in op.inputs)
        if op.op_type == "batchnorm2d":
            parent_name = remapped_inputs[0]
            parent_idx = op_by_out.get(parent_name)
            if parent_idx is None:
                raise ValueError(f"BatchNorm {op.op_id} input '{parent_name}' has no producer")
            conv_op = folded_ops[parent_idx]
            if conv_op.op_type not in ("conv2d", "dwconv2d"):
                raise ValueError(
                    f"BatchNorm {op.op_id} follows '{conv_op.op_type}', expected conv2d/dwconv2d"
                )
            assert conv_op.weights is not None and conv_op.bias is not None
            assert op.bn_gamma is not None and op.bn_beta is not None
            assert op.bn_mean is not None and op.bn_var is not None

            scale = op.bn_gamma / np.sqrt(op.bn_var + op.bn_eps)
            w_f64 = conv_op.weights.astype(np.float64)
            w_folded = (w_f64 * scale[:, None, None, None]).astype(np.float32)
            b_folded = ((conv_op.bias.astype(np.float64) - op.bn_mean) * scale + op.bn_beta).astype(
                np.float32
            )
            folded_ops[parent_idx] = replace(conv_op, weights=w_folded, bias=b_folded)
            remap[op.output] = conv_op.output
        else:
            new_op = replace(op, inputs=remapped_inputs)
            op_by_out[new_op.output] = len(folded_ops)
            folded_ops.append(new_op)

    final_out = resolve_t(output_tensor)

    if fuse_relu:
        # Count tensor consumers in folded_ops
        consumer_count: dict[str, int] = {}
        for op in folded_ops:
            for inp in op.inputs:
                consumer_count[inp] = consumer_count.get(inp, 0) + 1
        consumer_count[final_out] = consumer_count.get(final_out, 0) + 1

        remap_relu: dict[str, str] = {}

        def res_r(t: str) -> str:
            cur = t
            while cur in remap_relu:
                cur = remap_relu[cur]
            return cur

        fused_ops: list[CnnIrOp] = []
        last_by_out: dict[str, int] = {}
        for op in folded_ops:
            ins = tuple(res_r(inp) for inp in op.inputs)
            if op.op_type == "relu":
                src = ins[0]
                src_idx = last_by_out.get(src)
                if (
                    src_idx is not None
                    and consumer_count.get(src, 0) == 1
                    and fused_ops[src_idx].op_type in ("conv2d", "dwconv2d", "linear", "add")
                ):
                    fused_ops[src_idx] = replace(fused_ops[src_idx], relu_fused=True)
                    remap_relu[op.output] = fused_ops[src_idx].output
                    continue
            new_op = replace(op, inputs=ins)
            last_by_out[new_op.output] = len(fused_ops)
            fused_ops.append(new_op)
        folded_ops = fused_ops
        final_out = res_r(final_out)

    # Mark final Linear logits layer
    for idx in range(len(folded_ops) - 1, -1, -1):
        if folded_ops[idx].output == final_out and folded_ops[idx].op_type == "linear":
            folded_ops[idx] = replace(folded_ops[idx], is_final_logits=True)
            break

    _ = producer
    return folded_ops, final_out


def extract_cnn_ir_from_module(
    model: nn.Module,
    model_name: str,
    input_shape: tuple[int, int, int],
    norm_mean: float,
    norm_std: float,
    selected_threshold: float = 0.5,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str = "",
    fuse_relu: bool = True,
) -> CnnIrGraph:
    """Trace an instantiated PyTorch CNN `nn.Module`, fold BatchNorms, and return `CnnIrGraph`."""
    in_name, raw_out, raw_ops, _ = _extract_raw_ops(model, model_name, input_shape)
    ops, final_out = _fold_batchnorms_and_relus(raw_ops, raw_out, fuse_relu=fuse_relu)
    p = min(max(float(selected_threshold), 1e-6), 1.0 - 1e-6)
    threshold_logit = float(math.log(p / (1.0 - p)))
    return CnnIrGraph(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        selected_threshold=float(selected_threshold),
        threshold_logit=threshold_logit,
        input_name=in_name,
        input_shape=input_shape,
        output_name=final_out,
        norm_mean=float(norm_mean),
        norm_std=float(norm_std),
        ops=tuple(ops),
    )


def load_cnn_ir(cfg: CImpConfig, model_name: str) -> CnnIrGraph:
    """Load trained checkpoint + calibrated threshold for `model_name` and return verified `CnnIrGraph`."""
    results_dir = cfg.cnn_cfg.effective_results_dir(cfg.object_name, model_name)
    ckpt_name = (
        "checkpoint.pt"
        if (results_dir / "checkpoint.pt").is_file()
        else f"{cfg.cnn.checkpoint_variant}_model.pt"
    )
    ckpt_path = results_dir / ckpt_name
    summary_path = results_dir / "metrics_summary.json"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"CNN checkpoint not found: {ckpt_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"CNN metrics_summary.json not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calib = summary.get("threshold_calibration", {})
    if "selected_threshold" not in calib:
        raise KeyError(f"Missing threshold_calibration.selected_threshold in {summary_path}")
    selected_threshold = float(calib["selected_threshold"])

    model = create_model(model_name, num_classes=2)
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
    elif isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
        state_dict = raw_ckpt["state_dict"]
    else:
        state_dict = raw_ckpt
    model.load_state_dict(state_dict)
    model.eval()

    c_in = 1
    img_sz = cfg.descriptor_spec.image_size
    graph = extract_cnn_ir_from_module(
        model=model,
        model_name=model_name,
        input_shape=(c_in, img_sz, img_sz),
        norm_mean=cfg.cnn_cfg.data.normalize_mean,
        norm_std=cfg.cnn_cfg.data.normalize_std,
        selected_threshold=selected_threshold,
        checkpoint_path=ckpt_path,
        checkpoint_sha256=sha256_file(ckpt_path),
        fuse_relu=cfg.cnn.fuse_conv_relu_maxpool,
    )

    # Verify extracted IR matches PyTorch eager forward pass within strict float32 tolerance
    rng = np.random.default_rng(cfg.seed)
    probe = rng.normal(loc=0.0, scale=0.5, size=(c_in, img_sz, img_sz)).astype(np.float32)
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(probe[None, ...])).detach().cpu().numpy()[0]
    ir_logits, _ = run_cnn_graph_f32(graph, probe)
    max_diff = float(np.max(np.abs(pt_logits - ir_logits)))
    if max_diff > cfg.verification.f32_logit_abs_tol:
        raise ValueError(
            f"Folded IR vs PyTorch mismatch for {model_name}: "
            f"max_abs_diff={max_diff:.3e} > tol={cfg.verification.f32_logit_abs_tol:.3e}"
        )
    return graph


def _conv2d_ref_f32_batch(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int,
    padding: int,
    groups: int,
    relu_fused: bool,
) -> np.ndarray:
    """Reference NCHW float32 batched convolution using float64 accumulation."""
    with torch.no_grad():
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
        wt = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float64))
        bt = torch.from_numpy(np.ascontiguousarray(b, dtype=np.float64))
        yt = F.conv2d(xt, wt, bias=bt, stride=stride, padding=padding, groups=groups)
        if relu_fused:
            yt = torch.relu(yt)
        return np.ascontiguousarray(yt.to(torch.float32).numpy(), dtype=np.float32)


def _conv2d_ref_f32(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    stride: int,
    padding: int,
    groups: int,
    relu_fused: bool,
) -> np.ndarray:
    """Reference NCHW float32 single-sample convolution matching `infra_pytorch.c` double accumulation."""
    return _conv2d_ref_f32_batch(x[None, ...], w, b, stride, padding, groups, relu_fused)[0]


def _execute_f32_chunk(
    graph: CnnIrGraph,
    x_chunk: np.ndarray,
    collect_intermediates: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Execute folded `CnnIrGraph` in float32 over one batch `[B, C, H, W]`."""
    b_size = int(x_chunk.shape[0])
    tensors: dict[str, np.ndarray] = {
        graph.input_name: np.ascontiguousarray(x_chunk, dtype=np.float32)
    }
    captured: dict[str, np.ndarray] = {}
    if collect_intermediates:
        captured[graph.input_name] = tensors[graph.input_name]

    for op in graph.ops:
        if op.op_type in ("conv2d", "dwconv2d"):
            assert op.weights is not None and op.bias is not None
            res = _conv2d_ref_f32_batch(
                tensors[op.inputs[0]],
                op.weights,
                op.bias,
                op.stride,
                op.padding,
                op.groups,
                op.relu_fused,
            )
        elif op.op_type == "relu":
            res = np.maximum(np.float32(0.0), tensors[op.inputs[0]])
        elif op.op_type == "hardswish":
            v = tensors[op.inputs[0]]
            res = np.where(v <= -3.0, 0.0, np.where(v >= 3.0, v, v * (v + 3.0) / 6.0)).astype(
                np.float32
            )
        elif op.op_type == "hardsigmoid":
            v = tensors[op.inputs[0]]
            res = np.where(v <= -3.0, 0.0, np.where(v >= 3.0, 1.0, v / 6.0 + 0.5)).astype(
                np.float32
            )
        elif op.op_type == "maxpool2d":
            v = tensors[op.inputs[0]]
            b, c, h, w = v.shape
            res = v.reshape(b, c, h // 2, 2, w // 2, 2).max(axis=(3, 5)).astype(np.float32)
        elif op.op_type == "global_avgpool2d":
            v = tensors[op.inputs[0]]
            res = v.astype(np.float64).mean(axis=(2, 3), keepdims=True).astype(np.float32)
        elif op.op_type == "linear":
            assert op.weights is not None and op.bias is not None
            v = tensors[op.inputs[0]].reshape(b_size, -1).astype(np.float64)
            out_mat = (v @ op.weights.astype(np.float64).T + op.bias.astype(np.float64)).astype(
                np.float32
            )
            if op.relu_fused:
                out_mat = np.maximum(np.float32(0.0), out_mat)
            res = out_mat.reshape(b_size, *op.out_shape)
        elif op.op_type == "add":
            res = (tensors[op.inputs[0]] + tensors[op.inputs[1]]).astype(np.float32)
            if op.relu_fused:
                res = np.maximum(np.float32(0.0), res)
        elif op.op_type == "scale_channels":
            feat = tensors[op.inputs[0]]
            scale = tensors[op.inputs[1]].reshape(b_size, -1, 1, 1)
            res = (feat * scale).astype(np.float32)
        else:
            raise ValueError(f"Unknown IR op_type: {op.op_type}")

        tensors[op.output] = np.ascontiguousarray(res, dtype=np.float32)
        if collect_intermediates:
            captured[op.output] = tensors[op.output]

    logits = np.ascontiguousarray(tensors[graph.output_name].reshape(b_size, -1), dtype=np.float32)
    return logits, captured


def run_cnn_graph_f32(
    graph: CnnIrGraph,
    x_norm: np.ndarray,
    collect_intermediates: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Execute folded `CnnIrGraph` in float32 over one input `[C, H, W]`."""
    logits_b, captured_b = _execute_f32_chunk(
        graph, x_norm[None, ...], collect_intermediates=collect_intermediates
    )
    if collect_intermediates:
        captured = {k: v[0] for k, v in captured_b.items()}
    else:
        captured = {}
    return logits_b[0], captured


def run_cnn_graph_f32_batch(
    graph: CnnIrGraph,
    x_norm_batch: np.ndarray,
    batch_size: int = 64,
    collect_intermediates: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Execute folded `CnnIrGraph` in float32 over `[N, C, H, W]` in chunks."""
    x_arr = np.ascontiguousarray(x_norm_batch, dtype=np.float32)
    n_total = int(x_arr.shape[0])
    if n_total == 0:
        num_classes = graph.ops[-1].out_shape[0]
        return np.empty((0, num_classes), dtype=np.float32), {}

    logit_chunks: list[np.ndarray] = []
    inter_chunks: dict[str, list[np.ndarray]] = {}
    step = max(1, int(batch_size))
    for start in range(0, n_total, step):
        end = min(n_total, start + step)
        chunk_logits, chunk_inter = _execute_f32_chunk(
            graph, x_arr[start:end], collect_intermediates=collect_intermediates
        )
        logit_chunks.append(chunk_logits)
        if collect_intermediates:
            for k, v in chunk_inter.items():
                inter_chunks.setdefault(k, []).append(v)

    logits_all = np.ascontiguousarray(np.vstack(logit_chunks), dtype=np.float32)
    intermediates = (
        {k: np.concatenate(vs, axis=0) for k, vs in inter_chunks.items()}
        if collect_intermediates
        else {}
    )
    return logits_all, intermediates
