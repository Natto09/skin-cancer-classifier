"""
Single-image inference from the command line. Modern replacement for
test_model.py (which hardcoded an old, no-longer-used 512-dim head shape --
see legacy/test_model_v1.py for the original).

Usage:
    python -m src.evaluate.predict_image \\
        --model-preset resnet_1m --checkpoint skin_cancer_best_resnet_1M.pth \\
        --image test_image.jpg
"""

import argparse
import os

from torchvision import transforms

from ..models.classifier import SkinLesionClassifier
from ..models.config import ModelConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser(description="Run one image through a trained checkpoint.")
    parser.add_argument("--model-preset", required=True, help="Name of a configs/model/*.json file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_cfg = ModelConfig.from_json(os.path.join(repo_root, "configs", "model", f"{args.model_preset}.json"))
    clf = SkinLesionClassifier.from_checkpoint(args.checkpoint, model_cfg)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    try:
        predicted, probabilities = clf.predict_image(args.image, transform)
        print("\nConfidence per class:")
        for name, pct in probabilities.items():
            print(f"- {name}: {pct:.2f}%")
        print(f"\nPredicted: {predicted}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
