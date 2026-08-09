"""
train_gate_model.py -- Stage 1 of the two-stage classifier: a binary
"cancer-related vs not" gate model.

Groups the original 7 classes into 2:
    cancer     = mel, bcc, akiec
    non_cancer = bkl, df, nv, vasc

Reuses the exact same dataset-loading / three-way-split logic as
ensemble_meta_classifier.py (same seed, same split-before-augmentation
approach) so this model's val/test splits line up with everything else
in this project -- no new data-leakage risk introduced.

WHY A SEPARATE GATE MODEL: the 7-class ensemble already "knows" which
classes are cancer-related, but it has to solve 7-way discrimination at
once. A binary gate only has to solve ONE decision (cancer vs not), which
is a much easier problem and tends to produce a much higher recall on the
positive (cancer) class -- exactly the number this project cares about
most for a screening tool.

*** --device FLAG (this version) ***
Added an explicit --device argument so this script can be pinned to a
specific GPU. This is what lets you run this script and
train_specialist_mel_bkl.py at the same time in two terminals, each on
its own GPU, instead of both defaulting to cuda:0 and fighting over the
same device.

*** --weight_decay / --label_smoothing (this version) ***
Added AdamW weight_decay (default 0.01) and CrossEntropyLoss label_smoothing
(default 0.05) to reduce overfitting (train/val gap was ~99.7%/90.7% in an
earlier run). Both apply uniformly across classes -- unlike lowering
cancer_class_weight, they don't shift the decision boundary toward or away
from "cancer", so they shouldn't cost recall the way a weight change would.
Set either to 0 via the CLI to disable.

CONFIRMED RESULT with defaults (weight_decay=0.01, label_smoothing=0.05):
test accuracy 88.30% (+1.91pp vs no-regularization run) BUT cancer recall
83.01% (-2.05pp, down from 85.06%) -- accuracy improved by trading away
some cancer recall, which violates this project's actual goal. Prime
suspect: label_smoothing dilutes the effect of cancer_class_weight (both
act on the same target distribution), even though in isolation label
smoothing "shouldn't" favor one class. Recommended follow-up: re-run with
--label_smoothing 0 (keep weight_decay 0.01) to isolate which regularizer
is responsible before deciding whether to keep, lower, or drop label
smoothing.

*** --multi_gpu FLAG (this version) ***
Added DataParallel support so a SINGLE training run can use all visible
GPUs at once (splits each batch across GPUs, roughly halving wall-clock
time on 2 GPUs) -- different from --device, which pins a whole run to ONE
GPU so you can run this script and train_specialist_mel_bkl.py side by
side on separate GPUs. --multi_gpu and --device are mutually exclusive.
Don't combine --multi_gpu with running the other script at the same time
unless you have more than 2 GPUs, since --multi_gpu claims all visible
ones. Checkpoints are always saved WITHOUT the DataParallel "module."
prefix, so they stay loadable by combined_pipeline_eval.py and by this
script itself when run single-GPU, regardless of how they were trained.
Consider also raising --batch_size (e.g. 128 instead of 64) when using
--multi_gpu with 2 GPUs, since DataParallel computes BatchNorm statistics
per-GPU shard -- at the default batch_size each GPU would only see 32
images per step instead of the 64 a single-GPU run saw, which can make
BatchNorm's running statistics noisier.

Usage (2-GPU parallel run -- run this in terminal 1):
    python3 train_gate_model.py \\
      --meta_csv "all_augment_1M/lowmeta.csv" \\
      --arch densenet121 \\
      --device cuda:0

(terminal 2, at the same time):
    python3 train_specialist_mel_bkl.py \\
      --meta_csv "all_augment_1M/lowmeta.csv" \\
      --arch densenet121 \\
      --device cuda:1

Usage (single run, both GPUs at once -- faster wall-clock, don't run
train_specialist_mel_bkl.py at the same time with only 2 GPUs):
    python3 train_gate_model.py \\
      --meta_csv "all_augment_1M/lowmeta.csv" \\
      --arch densenet121 \\
      --multi_gpu --batch_size 128 \\
      --weight_decay 0.01 --label_smoothing 0

Produces:
    skin_cancer_best_gate_1M.pth       (best checkpoint by val cancer-recall)
    train_checkpoint_gate_1M.pth       (full checkpoint incl. optimizer state)
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

CANCER_CLASSES = {"mel", "bcc", "akiec"}


# ---------------------------------------------------------------------------
# Same dataset / split logic as ensemble_meta_classifier.py (kept identical
# on purpose -- this is what prevents data leakage and keeps every model in
# this project comparable on the same splits).
# ---------------------------------------------------------------------------

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

    # --- the only real difference from ensemble_meta_classifier.py: remap
    # the original 7-class label to a binary gate label BEFORE splitting.
    df["gate_label"] = df["label"].apply(lambda c: "cancer" if c in CANCER_CLASSES else "non_cancer")

    per_image = df.groupby("original_image", observed=True)["label"].first().reset_index()
    train_images, val_images, test_images = three_way_split(
        per_image, val_fraction, test_fraction, seed
    )

    gate_classes = ["cancer", "non_cancer"]
    gate_to_idx = {c: i for i, c in enumerate(gate_classes)}

    train_df = df[df["original_image"].isin(train_images)]
    val_df = df[df["original_image"].isin(val_images)]
    test_df = df[df["original_image"].isin(test_images)]

    train_rows = list(zip(train_df["filename"].tolist(),
                           train_df["gate_label"].map(gate_to_idx).tolist()))
    val_rows = list(zip(val_df["filename"].tolist(),
                         val_df["gate_label"].map(gate_to_idx).tolist()))
    test_rows = list(zip(test_df["filename"].tolist(),
                          test_df["gate_label"].map(gate_to_idx).tolist()))
    return train_rows, val_rows, test_rows, gate_to_idx


# ---------------------------------------------------------------------------
# Model construction (same head pattern as the rest of the project:
# Linear -> BatchNorm1d(128) -> ReLU -> Dropout(0.8) -> Linear(128, n))
# ---------------------------------------------------------------------------

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


def unwrap_state_dict(model):
    """Always save the plain (non-DataParallel) state dict, whether or not this
    run used --multi_gpu, so checkpoints stay loadable by combined_pipeline_eval.py
    and by single-GPU runs of this same script without a 'module.' prefix mismatch."""
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def load_into_model(model, state_dict):
    """Counterpart to unwrap_state_dict -- loads a plain state dict into either a
    plain model or a DataParallel-wrapped one."""
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    # per-class recall (index 0 = "cancer", the class we care most about)
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
    parser = argparse.ArgumentParser(description="Train the binary cancer/non-cancer gate model.")
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
                         help="Early stopping patience, measured on val cancer-recall.")
    parser.add_argument("--cancer_class_weight", type=float, default=3.0,
                         help="Extra loss weight on the 'cancer' class beyond inverse-frequency "
                              "(the gate's whole job is to not miss cancer cases, so bias it hard "
                              "toward recall on that class).")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                         help="AdamW weight decay (L2 regularization). Applies uniformly across "
                              "classes, so it reduces overfitting without shifting recall/precision "
                              "balance between classes the way changing cancer_class_weight would.")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                         help="Label smoothing for CrossEntropyLoss. Small value (0.05) reduces "
                              "overconfidence/overfitting without meaningfully changing which class "
                              "the model favors -- unlike cancer_class_weight, it doesn't push the "
                              "decision boundary toward or away from 'cancer'. Set to 0 to disable.")
    parser.add_argument("--device", default=None,
                         help="Device to train on, e.g. 'cuda:0', 'cuda:1', or 'cpu'. "
                              "Defaults to cuda:0 if a GPU is available, else cpu. Set this "
                              "explicitly to run this script alongside "
                              "train_specialist_mel_bkl.py on a second GPU at the same time. "
                              "Mutually exclusive with --multi_gpu.")
    parser.add_argument("--multi_gpu", action="store_true",
                         help="Use DataParallel to split THIS run across all visible GPUs "
                              "(faster wall-clock on a single training run). Mutually exclusive "
                              "with --device -- don't set both. Don't combine with running "
                              "train_specialist_mel_bkl.py at the same time unless you have more "
                              "than 2 GPUs, since this claims all visible ones.")
    parser.add_argument("--best_model_path", default="skin_cancer_best_gate_1M.pth")
    parser.add_argument("--checkpoint_path", default="train_checkpoint_gate_1M.pth")
    args = parser.parse_args()

    if args.multi_gpu and args.device:
        parser.error("--multi_gpu and --device are mutually exclusive -- pick one.")

    if args.multi_gpu:
        if torch.cuda.device_count() < 2:
            print(f"WARNING: --multi_gpu requested but only {torch.cuda.device_count()} GPU(s) "
                  f"visible; continuing on a single device.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu_active = args.multi_gpu and torch.cuda.device_count() > 1
    print(f"Device: {device}" + (f" (DataParallel across {torch.cuda.device_count()} GPUs)"
                                   if multi_gpu_active else ""))

    print(f"Loading {args.meta_csv} and building gate (cancer/non-cancer) splits (seed={args.seed}) ...")
    train_rows, val_rows, test_rows, gate_to_idx = load_meta_and_split(
        args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
        metadata_csv=args.metadata_csv, id_col=args.id_col, label_col=args.label_col,
    )
    idx_to_gate = {i: c for c, i in gate_to_idx.items()}
    num_classes = len(gate_to_idx)
    cancer_idx = gate_to_idx["cancer"]
    print(f"Train: {len(train_rows):,} | Val: {len(val_rows):,} | Test: {len(test_rows):,} "
          f"| Classes: {gate_to_idx}")

    train_labels = np.array([r[1] for r in train_rows])
    class_counts = np.bincount(train_labels, minlength=num_classes)
    print(f"Train class counts: cancer={class_counts[cancer_idx]:,}, "
          f"non_cancer={class_counts[1 - cancer_idx]:,}")

    # Inverse-frequency loss weights, with an extra manual boost on "cancer"
    # since missing a cancer case is far costlier than a false alarm.
    raw_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    loss_weights = raw_weights.copy()
    loss_weights[cancer_idx] *= args.cancer_class_weight
    class_weights_tensor = torch.tensor(loss_weights, dtype=torch.float32).to(device)
    print(f"Loss weights: {dict(zip(idx_to_gate.values(), loss_weights.round(3)))}")

    # Weighted sampler so the model sees "cancer" examples proportionally
    # more often during training too (not just via the loss weight).
    sampler_weights = raw_weights.copy()
    sampler_weights[cancer_idx] *= args.cancer_class_weight
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
    if multi_gpu_active:
        model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Regularization: weight_decay={args.weight_decay}, label_smoothing={args.label_smoothing}")

    best_cancer_recall = -1.0
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
        val_cancer_recall = val_recalls[cancer_idx]
        elapsed = (time.time() - epoch_start) / 60
        print(f"[EPOCH {epoch}] done in {elapsed:.1f}m -- Train Acc: {100*train_acc:.2f}% | "
              f"Val Acc: {100*val_acc:.2f}% | Val Cancer Recall: {val_cancer_recall:.4f}")

        # Save the full checkpoint every epoch (for resuming if needed)
        torch.save({
            "model_state": unwrap_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_metric": best_cancer_recall,
            "class_to_idx": gate_to_idx,
        }, args.checkpoint_path)

        if val_cancer_recall > best_cancer_recall:
            best_cancer_recall = val_cancer_recall
            epochs_without_improvement = 0
            torch.save(unwrap_state_dict(model), args.best_model_path)
            print(f"  -> new best (val cancer recall = {best_cancer_recall:.4f}), saved to {args.best_model_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch} "
                      f"(no val cancer-recall improvement for {args.patience} epochs)")
                break

    total_minutes = (time.time() - start_time) / 60
    print(f"Training complete. Total time: {total_minutes:.2f} minutes")

    print(f"Loading best checkpoint ({args.best_model_path}) for final test evaluation ...")
    load_into_model(model, torch.load(args.best_model_path, map_location=device))
    test_acc, test_recalls = evaluate(model, test_loader, device, num_classes)
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS -- GATE MODEL")
    print("=" * 60)
    print(f"  Test Accuracy: {100*test_acc:.2f}%")
    for c, name in idx_to_gate.items():
        print(f"  {name:>12} recall: {test_recalls[c]:.4f}")


if __name__ == "__main__":
    main()