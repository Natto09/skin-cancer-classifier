"""
train_specialist_mel_bkl.py -- a small specialist model trained ONLY on
mel (melanoma) vs bkl (benign keratosis) -- the single confusion pair
confirmed repeatedly in this project (confusion matrix, error analysis,
and the real Wikipedia test case in the live web app) to be the biggest
source of missed/false melanoma calls.

This model is meant to be called as a "second opinion" specifically when
the main ensemble's top-2 predictions are {mel, bkl} -- see
combined_pipeline_eval.py for how the three pieces (ensemble + gate +
this specialist) fit together.

*** --device FLAG (this version) ***
Added an explicit --device argument so this script can be pinned to a
specific GPU. This is what lets you run this script and
train_gate_model.py at the same time in two terminals, each on its own
GPU, instead of both defaulting to cuda:0 and fighting over the same
device.

*** --weight_decay / --label_smoothing (this version) ***
Added AdamW weight_decay (default 0.01) and CrossEntropyLoss label_smoothing
(default 0.05) to reduce overfitting, matching the same change made to
train_gate_model.py. Both apply uniformly across classes -- unlike lowering
mel_class_weight, they don't shift the decision boundary toward or away
from "mel", so they shouldn't cost recall the way a weight change would.
Set either to 0 via the CLI to disable.

Usage (2-GPU parallel run -- run this in terminal 2):
    python3 train_specialist_mel_bkl.py \\
      --meta_csv "all_augment_1M/lowmeta.csv" \\
      --arch densenet121 \\
      --device cuda:1

(terminal 1, at the same time):
    python3 train_gate_model.py \\
      --meta_csv "all_augment_1M/lowmeta.csv" \\
      --arch densenet121 \\
      --device cuda:0

Produces:
    skin_cancer_best_specialist_mel_bkl_1M.pth
    train_checkpoint_specialist_mel_bkl_1M.pth
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

TARGET_CLASSES = ["bkl", "mel"]  # index 0 = bkl, index 1 = mel


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

    # --- filter down to ONLY mel and bkl before splitting. Note: the split
    # is still done on the full per-image label distribution restricted to
    # these two classes, using the SAME seed=42 as every other split in this
    # project, so these rows are a subset of the same train/val/test
    # partitioning philosophy (no leakage introduced).
    df = df[df["label"].isin(TARGET_CLASSES)]

    class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}

    per_image = df.groupby("original_image", observed=True)["label"].first().reset_index()
    train_images, val_images, test_images = three_way_split(
        per_image, val_fraction, test_fraction, seed
    )

    train_df = df[df["original_image"].isin(train_images)]
    val_df = df[df["original_image"].isin(val_images)]
    test_df = df[df["original_image"].isin(test_images)]

    train_rows = list(zip(train_df["filename"].tolist(),
                           train_df["label"].map(class_to_idx).tolist()))
    val_rows = list(zip(val_df["filename"].tolist(),
                         val_df["label"].map(class_to_idx).tolist()))
    test_rows = list(zip(test_df["filename"].tolist(),
                         test_df["label"].map(class_to_idx).tolist()))
    return train_rows, val_rows, test_rows, class_to_idx


def build_model(arch, num_classes, device):
    if arch == "resnet50":
        model = models.resnet50(weights="DEFAULT")
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.8), nn.Linear(128, num_classes),
        )
    elif arch == "densenet121":
        model = models.densenet121(weights="DEFAULT")
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.8), nn.Linear(128, num_classes),
        )
    else:
        raise ValueError(f"Unsupported --arch '{arch}' (use resnet50 or densenet121)")
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for c in range(num_classes):
            mask = labels == c
            class_total[c] += mask.sum().item()
            class_correct[c] += ((preds == labels) & mask).sum().item()
    acc = correct / total if total else 0.0
    recalls = [class_correct[c] / class_total[c] if class_total[c] else 0.0 for c in range(num_classes)]
    return acc, recalls


def main():
    parser = argparse.ArgumentParser(description="Train the mel-vs-bkl specialist model.")
    parser.add_argument("--meta_csv", default="all_augment_1M/lowmeta.csv")
    parser.add_argument("--metadata_csv", default="data/ham10000/HAM10000_metadata.csv")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", default="densenet121", choices=["resnet50", "densenet121"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5,
                         help="Early stopping patience, measured on val mel-recall.")
    parser.add_argument("--mel_class_weight", type=float, default=2.0,
                         help="Extra loss weight on 'mel' beyond inverse-frequency, since "
                              "missing a real melanoma is the costliest error this specialist "
                              "is meant to catch.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                         help="AdamW weight decay (L2 regularization). Applies uniformly across "
                              "classes, so it reduces overfitting without shifting recall/precision "
                              "balance between classes the way changing mel_class_weight would.")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                         help="Label smoothing for CrossEntropyLoss. Small value (0.05) reduces "
                              "overconfidence/overfitting without meaningfully changing which class "
                              "the model favors -- unlike mel_class_weight, it doesn't push the "
                              "decision boundary toward or away from 'mel'. Set to 0 to disable.")
    parser.add_argument("--device", default=None,
                         help="Device to train on, e.g. 'cuda:0', 'cuda:1', or 'cpu'. "
                              "Defaults to cuda:0 if a GPU is available, else cpu. Set this "
                              "explicitly to run this script alongside train_gate_model.py "
                              "on a second GPU at the same time.")
    parser.add_argument("--best_model_path", default="skin_cancer_best_specialist_mel_bkl_1M.pth")
    parser.add_argument("--checkpoint_path", default="train_checkpoint_specialist_mel_bkl_1M.pth")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.meta_csv}, filtering to {TARGET_CLASSES}, "
          f"building splits (seed={args.seed}) ...")
    train_rows, val_rows, test_rows, class_to_idx = load_meta_and_split(
        args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
        metadata_csv=args.metadata_csv, id_col=args.id_col, label_col=args.label_col,
    )
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(class_to_idx)
    mel_idx = class_to_idx["mel"]
    print(f"Train: {len(train_rows):,} | Val: {len(val_rows):,} | Test: {len(test_rows):,} "
          f"| Classes: {class_to_idx}")

    train_labels = np.array([r[1] for r in train_rows])
    class_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"Train class counts: {dict(zip(idx_to_class.values(), class_counts))}")

    raw_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    loss_weights = raw_weights.copy()
    loss_weights[mel_idx] *= args.mel_class_weight
    class_weights_tensor = torch.tensor(loss_weights, dtype=torch.float32).to(device)
    print(f"Loss weights: {dict(zip(idx_to_class.values(), loss_weights.round(3)))}")

    sampler_weights = raw_weights.copy()
    sampler_weights[mel_idx] *= args.mel_class_weight
    per_row_weights = np.array([sampler_weights[label] for _, label in train_rows])
    sampler = WeightedRandomSampler(weights=per_row_weights, num_samples=len(train_rows))

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(AugmentedSkinDataset(train_rows, train_transforms),
                               batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    val_loader = DataLoader(AugmentedSkinDataset(val_rows, eval_transforms),
                             batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(AugmentedSkinDataset(test_rows, eval_transforms),
                              batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.arch, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Regularization: weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}")

    best_mel_recall = -1.0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        epoch_start = time.time()
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += inputs.size(0)
            if step % 50 == 0 or step == len(train_loader):
                pct = 100 * step / len(train_loader)
                print(f"[EPOCH {epoch}] step {step}/{len(train_loader)} ({pct:.1f}%) | "
                      f"loss: {running_loss/running_total:.4f} "
                      f"acc: {100*running_correct/running_total:.2f}%", flush=True)

        train_acc = running_correct / running_total
        val_acc, val_recalls = evaluate(model, val_loader, device, num_classes)
        val_mel_recall = val_recalls[mel_idx]
        elapsed = (time.time() - epoch_start) / 60
        print(f"[EPOCH {epoch}] done in {elapsed:.1f}m -- Train Acc: {100*train_acc:.2f}% | "
              f"Val Acc: {100*val_acc:.2f}% | Val mel Recall: {val_mel_recall:.4f}")

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_metric": best_mel_recall,
            "class_to_idx": class_to_idx,
        }, args.checkpoint_path)

        if val_mel_recall > best_mel_recall:
            best_mel_recall = val_mel_recall
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.best_model_path)
            print(f"  -> new best (val mel recall = {best_mel_recall:.4f}), saved to {args.best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(no val mel-recall improvement for {args.patience} epochs)")
                break

    total_minutes = (time.time() - start_time) / 60
    print(f"Training complete. Total time: {total_minutes:.2f} minutes")

    print(f"Loading best checkpoint ({args.best_model_path}) for final test evaluation ...")
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))
    test_acc, test_recalls = evaluate(model, test_loader, device, num_classes)
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS -- MEL-VS-BKL SPECIALIST")
    print("=" * 60)
    print(f"  Test Accuracy: {100*test_acc:.2f}%")
    for c, name in idx_to_class.items():
        print(f"  {name:>6} recall: {test_recalls[c]:.4f}")


if __name__ == "__main__":
    main()