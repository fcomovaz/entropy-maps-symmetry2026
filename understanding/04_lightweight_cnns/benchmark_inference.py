"""Latencia de forward local, batch 1 por defecto; excluye transferencias."""
import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from common import ROOT, input_size, model_stats, positive_int, seed_everything
from models import create_model, model_names


def measure(model: nn.Module, x: torch.Tensor, warmup: int, iterations: int) -> list[float]:
    """Mide cada forward con perf_counter, sincronizando CUDA a ambos lados."""
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        samples = []
        for _ in range(iterations):
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            start = time.perf_counter()
            model(x)
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            samples.append((time.perf_counter() - start) * 1000)
    return samples


def main() -> None:
    """Reporta media/mediana/mínimo y conserva muestras y entorno en JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-size", type=input_size, default=128)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--num-classes", type=positive_int, default=2)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print("GTX 1050 timing is NOT representative of ESP32/microcontroller performance.")
    print(f"float32; batch={args.batch_size}; size={args.input_size}; CPU threads={args.threads}; "
          f"warmup={args.warmup}; iterations={args.iterations}")
    print("Forward only: excludes image decoding, resize, transfers, softmax and region selection.")
    print("CUDA timings include Python dispatch and synchronization overhead; not pure kernel time.")
    if "cuda" not in devices:
        print("CUDA unavailable: GPU timings omitted.")
    print(f"{'Model':24} {'Device':7} {'Params':>10} {'MB':>8} {'Median ms':>11} {'Mean ms':>11} {'Min ms':>11}")
    rows = []
    for name in model_names():
        model = create_model(name, args.num_classes).eval()
        parameters, _, size_mb = model_stats(model)
        cpu_input = torch.randn(args.batch_size, 1, args.input_size, args.input_size)
        for device in devices:
            # Transferencia y asignación suceden ANTES del warmup y la medición.
            model = model.to(device)
            x = cpu_input.to(device)
            samples = measure(model, x, args.warmup, args.iterations)
            row = {"model": name, "device": device, "parameters": parameters,
                   "size_mb": size_mb, "median_ms": statistics.median(samples),
                   "mean_ms": statistics.mean(samples), "min_ms": min(samples),
                   "samples_ms": samples}
            rows.append(row)
            print(f"{name:24} {device:7} {parameters:10,d} {size_mb:8.4f} "
                  f"{row['median_ms']:11.4f} {row['mean_ms']:11.4f} {row['min_ms']:11.4f}")
            del x
        del model
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if "cuda" in devices else None,
        "settings": {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items()},
        "dtype": "float32", "cudnn_benchmark": False, "cudnn_deterministic": True,
        "scope": "model forward including host overhead; no preprocessing or transfers",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved measurements: {args.output}")


if __name__ == "__main__":
    main()
