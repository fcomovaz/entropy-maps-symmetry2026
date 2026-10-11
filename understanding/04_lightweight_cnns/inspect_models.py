"""Cuenta parámetros y ejecuta cada arquitectura sin dependencias de profiling."""
import argparse
import torch
from common import input_size, model_stats, positive_int, seed_everything, select_device
from models import create_model, model_names


def main() -> None:
    """Imprime una tabla y verifica shapes con un forward real."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-size", type=input_size, default=128)
    parser.add_argument("--num-classes", type=positive_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--threads", type=positive_int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(42)
    device = select_device(args.device)
    x = torch.randn(args.batch_size, 1, args.input_size, args.input_size, device=device)
    print(f"Device: {device}; float32; MB = 1,000,000 bytes (parameters only)")
    print(f"{'Model':24} {'Parameters':>11} {'Trainable':>11} {'Input shape':>20} {'Output shape':>14} {'Size MB':>10}")
    for name in model_names():
        model = create_model(name, args.num_classes).to(device).eval()
        with torch.inference_mode():
            y = model(x)
        assert y.shape == (args.batch_size, args.num_classes)
        assert torch.isfinite(y).all()
        count, trainable, size = model_stats(model)
        print(f"{name:24} {count:11,d} {trainable:11,d} {str(tuple(x.shape)):>20} {str(tuple(y.shape)):>14} {size:10.4f}")
        del model


if __name__ == "__main__":
    main()
