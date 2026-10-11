"""Arquitectura completa y flujo de shapes en bloques principales."""
import argparse
import torch
from common import input_size, positive_int, select_device
from models import create_model, model_names


def main() -> None:
    """Hooks observan bloques sin cambiar el cálculo de forward."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_names(), default="ds_cnn")
    parser.add_argument("--input-size", type=input_size, default=128)
    parser.add_argument("--num-classes", type=positive_int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--leaf", action="store_true", help="Observar cada operación hoja; incluye depthwise/pointwise.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = select_device(args.device)
    model = create_model(args.model, args.num_classes).to(device).eval()
    print(model)
    print("\nBlock | input -> output | parameters (including children)")
    handles = []

    def record(name: str):
        def hook(module, inputs, output):
            parameters = sum(p.numel() for p in module.parameters())
            print(f"{name}: {tuple(inputs[0].shape)} -> {tuple(output.shape)} | {parameters:,}")
        return hook

    if args.leaf:
        blocks = [(name, block) for name, block in model.named_modules()
                  if name and not list(block.children())]
    elif args.model == "torchvision_mobilenet":
        blocks = [(f"features.{name}", block)
                  for name, block in model.network.features.named_children()]
        blocks += [("avgpool", model.network.avgpool), ("classifier", model.network.classifier)]
    else:
        blocks = list(model.named_children())
    try:
        for name, block in blocks:
            handles.append(block.register_forward_hook(record(name)))
        with torch.inference_mode():
            output = model(torch.randn(1, 1, args.input_size, args.input_size, device=device))
        print(f"Final logits: {tuple(output.shape)}")
        print("flatten aparece entre pooling y Linear en forward; no es un módulo con hook.")
        print("En --leaf una ReLU reutilizada aparece por cada llamada; no sumar filas repetidas.")
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
