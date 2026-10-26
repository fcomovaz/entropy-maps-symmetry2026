"""Verify project imports and real CUDA execution; never fall back to CPU."""

import sys
import traceback


def main() -> int:
    """Return a nonzero exit code if imports or required CUDA checks fail."""
    print("=== Environment verification ===")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    try:
        import numpy
        import scipy
        import pandas
        import matplotlib
        import sklearn
        import skimage
        import cv2
        import PIL
        import torch
        import torchvision
    except Exception as exc:
        print(f"IMPORT ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    packages = [
        numpy,
        scipy,
        pandas,
        matplotlib,
        sklearn,
        skimage,
        cv2,
        PIL,
        torch,
        torchvision,
    ]
    for package in packages:
        print(f"{package.__name__}: {getattr(package, '__version__', 'unknown')}")
    cuda = torch.cuda.is_available()
    print(f"CUDA available: {cuda}")
    print(f"PyTorch CUDA version: {torch.version.cuda or 'None (CPU build)'}")
    print(f"PyTorch path: {torch.__file__}")
    try:
        if torch.version.cuda is None:
            raise RuntimeError("A CUDA PyTorch build is required; CPU build detected.")
        # Explicit initialization preserves the actual driver/DLL error on failure.
        torch.cuda.init()
        if not cuda:
            raise RuntimeError(
                "CUDA is required but torch.cuda.is_available() is False."
            )
        print(f"Device count: {torch.cuda.device_count()}")
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        print(f"GPU: {props.name}")
        print(f"Compute capability: {torch.cuda.get_device_capability(index)}")
        print(f"VRAM total: {props.total_memory / (1024 ** 3):.2f} GiB")
        print(f"Compiled CUDA architectures: {torch.cuda.get_arch_list()}")

        # Small deterministic [32, 32] matrices: allocation, transfers and a kernel.
        device = torch.device("cuda", index)
        cpu = torch.arange(32 * 32, dtype=torch.float32).reshape(32, 32)
        gpu = cpu.to(device)
        if gpu.device.type != "cuda":
            raise RuntimeError(f"Expected CUDA allocation, got {gpu.device}.")
        print("CUDA ALLOCATION TEST: PASS")
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=0, atol=0)
        print("TRANSFER TEST: PASS")
        result = gpu @ torch.eye(32, device=device)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(result.cpu(), cpu, rtol=0, atol=0)
        print("CUDA EXECUTION TEST: PASS")

        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
        ).to(device)
        with torch.no_grad():
            output = model(torch.randn(64, 128, device=device))
        torch.cuda.synchronize(device)
        if output.shape != (64, 10) or output.device.type != "cuda":
            raise RuntimeError(
                f"Unexpected model output: {output.shape}, {output.device}"
            )
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Model output contains non-finite values.")
        print("MODEL CUDA TEST: PASS")
    except Exception as exc:
        print(f"CUDA ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("Environment verification: FAIL")
        return 1
    print("Environment verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
