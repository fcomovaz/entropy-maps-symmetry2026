"""Smoke training: datos artificiales o carpetas train/val con clases."""
import argparse
from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import ROOT, image_to_tensor, input_size, positive_int, seed_everything, select_device
from models import create_model, model_names


def synthetic_dataset(samples: int, size: int, seed: int) -> TensorDataset:
    """Franjas suaves + ruido; clase 1 añade un cuadrado claro aleatorio."""
    generator = torch.Generator().manual_seed(seed)
    columns = torch.arange(size).float()
    stripes = 0.25 + 0.10 * torch.sin(columns * (2 * torch.pi / 8))
    images = stripes.reshape(1, 1, 1, size).expand(samples, 1, size, size).clone()
    images += 0.03 * torch.randn(images.shape, generator=generator)
    labels = torch.arange(samples) % 2
    side = max(4, size // 4)
    for index in range(samples):
        if labels[index] == 1:
            row, col = torch.randint(0, size - side + 1, (2,), generator=generator).tolist()
            images[index, 0, row:row + side, col:col + side] = 0.95
    return TensorDataset(images.clamp(0, 1), labels.long())


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    """Entrena si hay optimizer; en validación desactiva gradientes y BatchNorm updates."""
    model.train(optimizer is not None)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss no finita.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    if count == 0:
        raise ValueError("No hay batches; aumenta samples o reduce batch-size.")
    return total_loss / count, correct / count


def main() -> None:
    """Guarda state_dict y metadatos de preprocesado para predict_image.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_names(), default="tiny_cnn")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--data", type=Path)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--input-size", type=input_size, default=64)
    parser.add_argument("--samples", type=positive_int, default=64)
    parser.add_argument("--num-classes", type=positive_int, help="Inferido de los datos si se omite.")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--output", type=Path, help="Default: outputs/<model>_<mode>.pt; sobrescribe esa ruta.")
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("Usa batch-size >= 2: BatchNorm puede recibir mapas 1x1 a resolución 32.")
    if not 0 < args.lr < float("inf"):
        parser.error("lr debe ser positiva y finita.")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    if args.synthetic:
        train_data = synthetic_dataset(args.samples, args.input_size, args.seed)
        val_data = synthetic_dataset(max(16, args.samples // 4), args.input_size, args.seed + 1)
        class_to_idx = {"normal": 0, "artificial_anomaly": 1}
        print("SYNTHETIC SMOKE TEST ONLY: metrics are not scientific defect-detection results.")
    else:
        from torchvision.datasets import ImageFolder
        transform = partial(image_to_tensor, size=args.input_size)
        train_data = ImageFolder(args.data / "train", transform=transform)
        val_data = ImageFolder(args.data / "val", transform=transform)
        class_to_idx = train_data.class_to_idx
        if class_to_idx != val_data.class_to_idx:
            raise ValueError("train y val deben tener exactamente las mismas clases.")
    num_classes = len(class_to_idx)
    if num_classes < 2:
        raise ValueError("El demo requiere al menos dos clases.")
    if args.num_classes is not None and args.num_classes != num_classes:
        raise ValueError(f"Los datos tienen {num_classes} clases, no {args.num_classes}.")
    if len(train_data) < args.batch_size:
        raise ValueError("El conjunto train debe tener al menos batch-size imágenes.")
    # Solo descartamos un batch final de un elemento (BN en mapas 1x1).
    drop_last = len(train_data) % args.batch_size == 1
    if drop_last:
        print("Training: se omite el último batch de 1 imagen por BatchNorm.")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed),
                              num_workers=0, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=0)
    model = create_model(args.model, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Model: {args.model}; device: {device}; classes: {class_to_idx}; seed: {args.seed}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}: train loss={train_loss:.4f}, acc={train_accuracy:.3f}; "
              f"val loss={val_loss:.4f}, acc={val_accuracy:.3f}")
    data_mode = "synthetic" if args.synthetic else "folder"
    output = args.output or ROOT / "outputs" / f"{args.model}_{data_mode}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model": args.model, "num_classes": num_classes, "input_size": args.input_size,
        "class_to_idx": class_to_idx, "data_mode": data_mode,
        "preprocessing": "grayscale, bilinear square resize, float32 [0,1]",
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "torch_version": str(torch.__version__),
        "device": str(device),
    }, output)
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
