import argparse
import os
import ssl
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ssl._create_default_https_context = ssl._create_unverified_context

CHECKPOINT_EVERY_STEPS = 2000  # save a resumable checkpoint this often within an epoch


# ---------------------------------------------------------------------------
# Dataset: reads directly from meta.csv produced by pump_data_augment.py
# ---------------------------------------------------------------------------

class AugmentedSkinDataset(Dataset):
    def __init__(self, rows, transform=None):
        # rows: list of (filepath, label_idx)
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
    """
    meta.csv from pump_data_augment.py only has labels baked in if
    --metadata_csv was passed during augmentation. If the label column
    came out empty, merge labels in now from a separate id,label CSV
    (e.g. HAM10000_metadata.csv), matching on original_image with its
    file extension stripped off.
    """
    has_labels = df["label"].astype(str).str.strip().replace("nan", "").ne("").any()
    if has_labels:
        return df

    if not metadata_csv:
        raise ValueError(
            "meta.csv has no labels in it, and no --metadata_csv was given to fill "
            "them in. Pass --metadata_csv pointing at your HAM10000_metadata.csv "
            "(or equivalent) with --id_col/--label_col set correctly."
        )

    print(f"[LABELS] meta.csv has no labels -- merging in from {metadata_csv} ...")
    labels_df = pd.read_csv(metadata_csv, usecols=[id_col, label_col], dtype=str)
    id_to_label = dict(zip(labels_df[id_col], labels_df[label_col]))

    original_id = df["original_image"].astype(str).str.rsplit(".", n=1).str[0]
    df = df.copy()
    df["label"] = original_id.map(id_to_label)

    matched = df["label"].notna().sum()
    print(f"[LABELS] matched {matched:,} / {len(df):,} rows to a label")
    return df


def load_meta_and_split(meta_csv, val_fraction, seed, metadata_csv=None, id_col="image_id", label_col="dx"):
    """
    Reads meta.csv and splits at the ORIGINAL IMAGE level (not row level) so
    that all 600 augmented variants of a given source image end up entirely
    in train or entirely in val -- never split across both. Splitting at the
    row level would leak near-duplicate images between train/val and give you
    an artificially inflated validation accuracy.
    """
    print(f"Loading {meta_csv} ...")
    df = pd.read_csv(
        meta_csv,
        usecols=["filename", "original_image", "label"],
        dtype={"filename": "string", "original_image": "category", "label": "string"},
    )
    df = merge_labels_if_missing(df, metadata_csv, id_col, label_col)
    df = df[df["label"].notna() & (df["label"] != "")]

    classes = sorted(df["label"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes found ({len(classes)}): {classes}")

    # one label per original source image (should be identical across all its variants)
    per_image = df.groupby("original_image", observed=True)["label"].first().reset_index()

    train_images, val_images = train_test_split(
        per_image["original_image"],
        test_size=val_fraction,
        random_state=seed,
        stratify=per_image["label"],
    )
    train_images = set(train_images)
    val_images = set(val_images)

    train_df = df[df["original_image"].isin(train_images)]
    val_df = df[df["original_image"].isin(val_images)]

    train_rows = list(zip(train_df["filename"].tolist(),
                           train_df["label"].map(class_to_idx).tolist()))
    val_rows = list(zip(val_df["filename"].tolist(),
                         val_df["label"].map(class_to_idx).tolist()))

    print(f"Source images   -> train: {len(train_images):,}  val: {len(val_images):,}")
    print(f"Output rows     -> train: {len(train_rows):,}  val: {len(val_rows):,}")

    return train_rows, val_rows, class_to_idx


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_val_loss, counter, class_to_idx):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
        "counter": counter,
        "class_to_idx": class_to_idx,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["step"], ckpt["best_val_loss"], ckpt["counter"], ckpt["class_to_idx"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on the augmented skin cancer dataset.")
    parser.add_argument("--meta_csv", default="all augment/meta.csv",
                         help='Path to meta.csv produced by pump_data_augment.py')
    parser.add_argument("--metadata_csv", default=None,
                         help="Optional id,label CSV (e.g. HAM10000_metadata.csv) used to fill in "
                              "labels if meta.csv's label column came out empty")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="Fraction of SOURCE IMAGES (not rows) held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--best_model_path", default="skin_cancer_best_resnet.pth")
    parser.add_argument("--checkpoint_path", default="train_checkpoint.pth")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from --checkpoint_path if it exists")
    args = parser.parse_args()

    # --- 1. Transforms ---
    # No RandomHorizontalFlip / RandomVerticalFlip / ColorJitter here: those
    # augmentations are already baked into the dataset as physical files, so
    # applying them again live would be redundant.
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- 2. Load meta.csv and split at the source-image level ---
    train_rows, val_rows, class_to_idx = load_meta_and_split(
        args.meta_csv, args.val_fraction, args.seed,
        metadata_csv=args.metadata_csv, id_col=args.id_col, label_col=args.label_col,
    )
    num_classes = len(class_to_idx)

    dataset_train = AugmentedSkinDataset(train_rows, transform=train_transforms)
    dataset_val = AugmentedSkinDataset(val_rows, transform=val_transforms)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )

    # --- 3. Model (ResNet50 + hybrid middle layer, same as before) ---
    model = models.resnet50(weights="DEFAULT")

    for name, child in model.named_children():
        if name in ["layer4", "fc"]:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(128, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    start_epoch = 0
    resume_step = 0
    best_val_loss = float("inf")
    counter = 0

    if args.resume and os.path.exists(args.checkpoint_path):
        start_epoch, resume_step, best_val_loss, counter, saved_class_to_idx = load_checkpoint(
            args.checkpoint_path, model, optimizer, scheduler
        )
        if saved_class_to_idx != class_to_idx:
            print("[WARN] class_to_idx from checkpoint differs from current data. "
                  "Proceeding with current data's class_to_idx.")
        print(f"[RESUME] Resuming from epoch {start_epoch}, step {resume_step}, "
              f"best_val_loss={best_val_loss:.4f}, patience_counter={counter}")

    print("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        skip_until = resume_step if epoch == start_epoch else 0

        for step, (inputs, labels) in enumerate(train_loader):
            if step < skip_until:
                continue  # fast-skip batches already done before the crash

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (step + 1) % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(args.checkpoint_path, model, optimizer, scheduler,
                                 epoch, step + 1, best_val_loss, counter, class_to_idx)

        train_loss = running_loss / max(1, (len(train_loader) - skip_until))
        train_acc = 100 * correct / max(1, total)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        scheduler.step()

        # end-of-epoch checkpoint (step resets to 0 for the next epoch)
        save_checkpoint(args.checkpoint_path, model, optimizer, scheduler,
                         epoch + 1, 0, best_val_loss, counter, class_to_idx)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), args.best_model_path)
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Total time: {(time.time() - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()