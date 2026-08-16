"""
Per-class accuracy / confusion matrix evaluation for one trained checkpoint.

Reconstructs the exact same train/val/test split used during training (same
meta_csv, seed, val_fraction, test_fraction) and evaluates the saved best
model on the held-out test set.

Usage:
    python -m src.evaluate.confusion_matrix \\
        --model-preset resnet_1m --checkpoint skin_cancer_best_resnet_1M.pth \\
        --meta_csv all_augment_1M/lowmeta.csv
"""

import argparse
import os

from sklearn.metrics import confusion_matrix, classification_report

from ..data.dataset import AugmentedSkinDataset
from ..data.splits import load_meta_and_split
from ..models.classifier import SkinLesionClassifier
from ..models.config import ModelConfig

CANCER_RELATED_CLASSES = {"mel", "bcc", "akiec"}


def evaluate_checkpoint(clf: SkinLesionClassifier, meta_csv, val_fraction=0.15, test_fraction=0.10,
                         seed=42, metadata_csv=None, id_col="image_id", label_col="dx",
                         batch_size=128, workers=4):
    from torch.utils.data import DataLoader
    from torchvision import transforms

    _, _, test_rows, class_to_idx = load_meta_and_split(
        meta_csv, val_fraction, test_fraction, seed,
        metadata_csv=metadata_csv, id_col=id_col, label_col=label_col,
    )
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(AugmentedSkinDataset(test_rows, eval_tf), batch_size=batch_size,
                         shuffle=False, num_workers=workers)

    probs, labels = clf.predict_proba(loader)
    preds = probs.argmax(axis=1)

    print("\n" + "=" * 70)
    print("PER-CLASS REPORT (precision / recall / f1-score / support)")
    print("=" * 70)
    print(classification_report(labels, preds, target_names=classes, digits=3))

    print("=" * 70)
    print("CONFUSION MATRIX (rows = true label, columns = predicted label)")
    print("=" * 70)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    header = "        " + "".join(f"{c:>8}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:>8}" + "".join(f"{v:>8}" for v in row))

    print("\n" + "=" * 70)
    print("RECALL ON CANCER-RELATED CLASSES (mel / bcc / akiec)")
    print("=" * 70)
    for i, cls in enumerate(classes):
        if cls in CANCER_RELATED_CLASSES:
            true_count = sum(1 for l in labels if l == i)
            correct_count = sum(1 for l, p in zip(labels, preds) if l == i and p == i)
            recall = correct_count / true_count if true_count else float("nan")
            print(f"  {cls:>6}: recall = {recall:.3f}  ({correct_count}/{true_count} caught)")

    return cm, classes, labels, preds


def save_confusion_heatmap(cm, classes, out_path="confusion_matrix.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib not installed -- skipping the PNG heatmap.")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix -- Test Set")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved confusion matrix heatmap to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-class evaluation on the held-out test set.")
    parser.add_argument("--meta_csv", default="all_augment_1M/lowmeta.csv")
    parser.add_argument("--metadata_csv", default="data/ham10000/HAM10000_metadata.csv")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-preset", required=True, help="Name of a configs/model/*.json file")
    parser.add_argument("--checkpoint", required=True, help="Path to the .pth weights file")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--save_heatmap", default=None, help="Optional output path for a PNG heatmap")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_cfg = ModelConfig.from_json(os.path.join(repo_root, "configs", "model", f"{args.model_preset}.json"))
    clf = SkinLesionClassifier.from_checkpoint(args.checkpoint, model_cfg)

    cm, classes, labels, preds = evaluate_checkpoint(
        clf, args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
        args.metadata_csv, args.id_col, args.label_col, args.batch_size, args.workers,
    )
    if args.save_heatmap:
        save_confusion_heatmap(cm, classes, args.save_heatmap)


if __name__ == "__main__":
    main()
