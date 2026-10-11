"""Inferencia sobre una imagen, con pesos aleatorios o checkpoint local."""
import argparse
from pathlib import Path
from PIL import Image
import torch
from common import image_to_tensor, input_size, positive_int, seed_everything, select_device
from models import create_model, model_names


def main() -> None:
    """Softmax se aplica aquí a los logits, nunca dentro del modelo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=model_names(), default="tiny_cnn")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--input-size", type=input_size, default=None, help="Default: checkpoint size, otherwise 128.")
    parser.add_argument("--num-classes", type=positive_int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(args.seed)
    device = select_device(args.device)
    model = create_model(args.model, args.num_classes)
    metadata = {}
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            metadata = checkpoint
            if metadata["model"] != args.model or metadata["num_classes"] != args.num_classes:
                raise ValueError("Checkpoint incompatible: usa el mismo --model y --num-classes.")
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint  # También admite un state_dict puro.
        model.load_state_dict(state, strict=True)
        print(f"Loaded checkpoint: {args.checkpoint}")
        if metadata.get("data_mode") == "synthetic":
            print("WARNING: checkpoint trained on synthetic toy data; not validated for real defects.")
        else:
            print("Checkpoint loaded; training history and predictive quality are not verified.")
    else:
        print("WARNING: model uses random/untrained weights. Predictions are not meaningful.")
    size = args.input_size or metadata.get("input_size", 128)
    with Image.open(args.image) as image:
        x = image_to_tensor(image, size).unsqueeze(0).to(device)
    model = model.to(device).eval()
    with torch.inference_mode():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
    predicted = int(probabilities.argmax(dim=1).item())
    print(f"Device: {device}; input: {tuple(x.shape)}; grayscale [0, 1]")
    print(f"Logits: {logits.cpu().tolist()[0]}")
    print(f"Probabilities: {probabilities.cpu().tolist()[0]}")
    print(f"Predicted class index: {predicted}")
    if metadata.get("class_to_idx"):
        names = {value: key for key, value in metadata["class_to_idx"].items()}
        print(f"Predicted class name: {names[predicted]}")


if __name__ == "__main__":
    main()
