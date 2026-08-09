"""
Per-class accuracy / confusion matrix evaluation for the trained ResNet50
skin cancer classifier.

Reconstructs the EXACT same train/val/test split used during training
(same --meta_csv, --seed, --val_fraction, --test_fraction => same source
images end up in the test set) and evaluates the saved best model on it.

Why this matters for HAM10000 specifically: classes are heavily imbalanced
(nv is roughly two-thirds of all images), so a single "test accuracy: 85%"
number can hide a model that's great at nv and quietly bad at mel/bcc/akiec
-- the classes that actually matter for catching skin cancer. Per-class
recall tells you that; overall accuracy does not.

Usage (defaults match train_resnet_100K.py):
    python3 evaluate_confusion_matrix.py

Point it at a different dataset/checkpoint with:
    python3 evaluate_confusion_matrix.py \
        --meta_csv "all_augment_1M/lowmeta.csv" \
        --best_model_path "skin_cancer_best_resnet_1M.pth"
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Classes considered malignant/premalignant in HAM10000 -- the ones where a
# missed detection (false negative) is the costly kind of mistake, as
# opposed to nv/bkl/df/vasc where a false positive is more the annoyance.
CANCER_RELATED_CLASSES = {"mel", "bcc", "akiec"}


class AugmentedSkinDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label_idx = self.rows[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label_idx


def merge_labels_if_missing(df, metadata_csv, id_col, label_col):
    non_null = df["label"].dropna()
    has_labels = len(non_null) > 0 and (non_null.astype(str).str.strip() != "").any()
    if has_labels:
        return df
    if not metadata_csv:
        raise ValueError("meta.csv has no labels and no --metadata_csv was given.")
    labels_df = pd.read_csv(metadata_csv, usecols=[id_col, label_col], dtype=str)
    id_to_label = dict(zip(labels_df[id_col], labels_df[label_col]))
    original_id = df["original_image"].astype(str).str.rsplit(".", n=1).str[0]
    df = df.copy()
    df["label"] = original_id.map(id_to_label)
    return df


def three_way_split(per_image, val_fraction, test_fraction, seed):
    labels_by_image = per_image.set_index("original_image")["label"]
    train_val_images, test_images = train_test_split(
        per_image["original_image"], test_size=test_fraction,
        random_state=seed, shuffle=True, stratify=per_image["label"],
    )
    relative_val_fraction = val_fraction / (1 - test_fraction)
    train_val_labels = labels_by_image.loc[train_val_images]
    train_images, val_images = train_test_split(
        train_val_images, test_size=relative_val_fraction,
        random_state=seed, shuffle=True, stratify=train_val_labels,
    )
    return set(train_images), set(val_images), set(test_images)


def load_meta_and_split(meta_csv, val_fraction, test_fraction, seed,
                         metadata_csv=None, id_col="image_id", label_col="dx"):
    df = pd.read_csv(
        meta_csv, usecols=["filename", "original_image", "label"],
        dtype={"filename": "string", "original_image": "category", "label": "string"},
    )
    df = merge_labels_if_missing(df, metadata_csv, id_col, label_col)
    df = df[df["label"].notna() & (df["label"] != "")]

    classes = sorted(df["label"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    per_image = df.groupby("original_image", observed=True)["label"].first().reset_index()
    train_images, val_images, test_images = three_way_split(
        per_image, val_fraction, test_fraction, seed
    )

    test_df = df[df["original_image"].isin(test_images)]
    test_rows = list(zip(test_df["filename"].tolist(),
                          test_df["label"].map(class_to_idx).tolist()))
    return test_rows, class_to_idx


def build_resnet50(num_classes, device):
    model = models.resnet50(weights=None)  # weights loaded from checkpoint below
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(128, num_classes),
    )
    return model.to(device)


def build_densenet121(num_classes, device):
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(128, num_classes),
    )
    return model.to(device)


def build_vit(num_classes, device):
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(128, num_classes),
    )
    return model.to(device)


ARCH_BUILDERS = {
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "vit": build_vit,
}


def main():
    parser = argparse.ArgumentParser(description="Per-class evaluation on the held-out test set.")
    parser.add_argument("--meta_csv", default="all_augment_100K/lowestmeta.csv")
    parser.add_argument("--metadata_csv", default="data/ham10000/HAM10000_metadata.csv")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--best_model_path", default="skin_cancer_best_resnet_100K.pth")
    parser.add_argument("--arch", choices=list(ARCH_BUILDERS.keys()), default="resnet50",
                         help="Which architecture --best_model_path was trained with. Must "
                              "match exactly (resnet50/densenet121/vit) or loading the weights "
                              "will fail with a state_dict key mismatch.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.meta_csv} and rebuilding the test split "
          f"(seed={args.seed}, val_fraction={args.val_fraction}, "
          f"test_fraction={args.test_fraction}) ...")
    test_rows, class_to_idx = load_meta_and_split(
        args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
        metadata_csv=args.metadata_csv, id_col=args.id_col, label_col=args.label_col,
    )
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(f"Test set: {len(test_rows):,} rows across {len(classes)} classes: {classes}")

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_loader = DataLoader(
        AugmentedSkinDataset(test_rows, transform=val_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
    )

    print(f"Loading model weights from {args.best_model_path} (arch={args.arch}) ...")
    model = ARCH_BUILDERS[args.arch](len(classes), device)
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    print("\n" + "=" * 70)
    print("PER-CLASS REPORT (precision / recall / f1-score / support)")
    print("=" * 70)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report)

    print("=" * 70)
    print("CONFUSION MATRIX (rows = true label, columns = predicted label)")
    print("=" * 70)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    header = "        " + "".join(f"{c:>8}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:>8}" + "".join(f"{v:>8}" for v in row))

    print("\n" + "=" * 70)
    print("RECALL ON CANCER-RELATED CLASSES (mel / bcc / akiec)")
    print("-- this is the number that matters most for a screening tool: it's")
    print("   the fraction of ACTUAL cancer cases the model correctly flagged.")
    print("   A high overall accuracy with low recall here is a dangerous model.")
    print("=" * 70)
    for i, cls in enumerate(classes):
        if cls in CANCER_RELATED_CLASSES:
            true_count = sum(1 for l in all_labels if l == i)
            correct_count = sum(1 for l, p in zip(all_labels, all_preds) if l == i and p == i)
            recall = correct_count / true_count if true_count > 0 else float("nan")
            print(f"  {cls:>6}: recall = {recall:.3f}  ({correct_count}/{true_count} caught)")

    # Save a heatmap PNG for a quick visual read
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix -- Test Set")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out_path = "confusion_matrix.png"
        fig.savefig(out_path, dpi=150)
        print(f"\nSaved confusion matrix heatmap to {out_path}")
    except ImportError:
        print("\n[INFO] matplotlib not installed -- skipping the PNG heatmap "
              "(the printed confusion matrix above still has everything).")


if __name__ == "__main__":
    main()