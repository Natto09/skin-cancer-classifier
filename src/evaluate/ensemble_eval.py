"""
Fits the DAME-style meta-classifier ensemble (see src.models.ensemble) on
the VAL split and reports final numbers on the TEST split. Replaces
ensemble_meta_classifier.py.

METHODOLOGY: the meta-classifier is trained on VAL-split predictions, never
TRAIN-split -- backbones are close to memorized on their own training data,
so train-split softmax outputs are overconfident and would teach the
meta-classifier a distorted picture.

Usage:
    python -m src.evaluate.ensemble_eval \\
        --meta_csv all_augment_1M/lowmeta.csv \\
        --resnet_path skin_cancer_best_resnet_1M.pth \\
        --densenet_path skin_cancer_best_densenet_1M.pth \\
        --vit_path skin_cancer_best_vit_1M.pth
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, recall_score

from ..data.dataset import AugmentedSkinDataset
from ..data.splits import load_meta_and_split
from ..models.config import ModelConfig
from ..models.ensemble import EnsembleClassifier

CANCER_RELATED_CLASSES = {"mel", "bcc", "akiec"}


def save_confusion_heatmap(cm, classes, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("[INFO] matplotlib/seaborn not installed -- skipping the PNG heatmap. "
              "Install with: pip install matplotlib seaborn")
        return
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[artifacts] saved confusion matrix heatmap to {out_path}")


def save_misclassified_examples(test_rows, y_true, y_pred, class_to_idx, error_pairs, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    filenames = [r[0] for r in test_rows]
    for true_c, pred_c in error_pairs:
        true_idx, pred_idx = class_to_idx[true_c], class_to_idx[pred_c]
        mask = (y_true == true_idx) & (y_pred == pred_idx)
        rows = [filenames[i] for i in np.where(mask)[0]]
        out_path = os.path.join(out_dir, f"errors_{true_c}_as_{pred_c}.csv")
        pd.DataFrame({"filename": rows}).to_csv(out_path, index=False)
        print(f"[error-analysis] {true_c} -> {pred_c}: {len(rows)} cases saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit + evaluate the meta-classifier ensemble.")
    parser.add_argument("--meta_csv", default="all_augment_1M/lowmeta.csv")
    parser.add_argument("--metadata_csv", default="data/ham10000/HAM10000_metadata.csv")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resnet_path", default="skin_cancer_best_resnet_1M.pth")
    parser.add_argument("--densenet_path", default="skin_cancer_best_densenet_1M.pth")
    parser.add_argument("--vit_path", default="skin_cancer_best_vit_1M.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--meta_max_iter", type=int, default=1000)
    parser.add_argument("--artifacts_dir", default=".")
    parser.add_argument("--error_analysis", action="store_true")
    parser.add_argument("--error_analysis_dir", default="error_analysis")
    args = parser.parse_args()

    for path, label in [(args.resnet_path, "ResNet50"), (args.densenet_path, "DenseNet121"),
                         (args.vit_path, "ViT")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"[MISSING] {label} weights not found at '{path}'.")

    print(f"Loading {args.meta_csv} and rebuilding val/test splits (seed={args.seed}) ...")
    _, val_rows, test_rows, class_to_idx = load_meta_and_split(
        args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
        metadata_csv=args.metadata_csv, id_col=args.id_col, label_col=args.label_col,
    )
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(classes)
    print(f"Val set: {len(val_rows):,} rows | Test set: {len(test_rows):,} rows | Classes: {classes}")

    from torch.utils.data import DataLoader
    from torchvision import transforms
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_loader = DataLoader(AugmentedSkinDataset(val_rows, eval_tf), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(AugmentedSkinDataset(test_rows, eval_tf), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers)

    configs = {
        "resnet50": ModelConfig(backbone="resnet50", num_classes=num_classes, pretrained=False),
        "densenet121": ModelConfig(backbone="densenet121", num_classes=num_classes, pretrained=False),
        "vit_b16": ModelConfig(backbone="vit_b16", num_classes=num_classes, pretrained=False),
    }
    weight_paths = {"resnet50": args.resnet_path, "densenet121": args.densenet_path, "vit_b16": args.vit_path}
    ensemble = EnsembleClassifier.from_configs(configs, weight_paths, order=["resnet50", "densenet121", "vit_b16"])

    print(f"Training Logistic Regression meta-classifier (max_iter={args.meta_max_iter}) on VAL-set predictions ...")
    ensemble.fit_meta_classifier(val_loader, class_names=classes, max_iter=args.meta_max_iter)

    test_preds, test_labels, test_blocks = ensemble.predict(test_loader)

    print("\n" + "=" * 70)
    print("INDIVIDUAL BACKBONE ACCURACY ON TEST SET (for comparison)")
    print("=" * 70)
    for name, block in zip(ensemble.order, test_blocks):
        preds = block.argmax(axis=1)
        acc = (preds == test_labels).mean()
        macro_rec = recall_score(test_labels, preds, average="macro", zero_division=0)
        print(f"  {name:>12}: accuracy={acc:.4f}  macro_recall={macro_rec:.4f}")

    ensemble_acc = (test_preds == test_labels).mean()
    ensemble_macro_recall = recall_score(test_labels, test_preds, average="macro", zero_division=0)
    print(f"  {'ENSEMBLE':>12}: accuracy={ensemble_acc:.4f}  macro_recall={ensemble_macro_recall:.4f}")

    print("\n" + "=" * 70)
    print("PER-CLASS REPORT -- ENSEMBLE")
    print("=" * 70)
    print(classification_report(test_labels, test_preds, target_names=classes, digits=3))

    cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))
    print("=" * 70)
    print("CONFUSION MATRIX -- ENSEMBLE (rows = true label, columns = predicted label)")
    print("=" * 70)
    header = "        " + "".join(f"{c:>8}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:>8}" + "".join(f"{v:>8}" for v in row))

    save_confusion_heatmap(cm, classes, os.path.join(args.artifacts_dir, "ensemble_confusion_heatmap.png"))

    print("\n" + "=" * 70)
    print("RECALL ON CANCER-RELATED CLASSES (mel / bcc / akiec) -- ENSEMBLE")
    print("=" * 70)
    for i, cls in enumerate(classes):
        if cls in CANCER_RELATED_CLASSES:
            true_count = int((test_labels == i).sum())
            correct_count = int(((test_labels == i) & (test_preds == i)).sum())
            recall = correct_count / true_count if true_count > 0 else float("nan")
            print(f"  {cls:>6}: recall = {recall:.3f}  ({correct_count}/{true_count} caught)")

    ensemble.save(args.artifacts_dir, weight_paths, test_accuracy=ensemble_acc,
                  test_macro_recall=ensemble_macro_recall)

    if args.error_analysis:
        error_pairs = [("mel", "nv"), ("mel", "bkl"), ("akiec", "bkl")]
        save_misclassified_examples(test_rows, test_labels, test_preds, class_to_idx,
                                     error_pairs, args.error_analysis_dir)


if __name__ == "__main__":
    main()
