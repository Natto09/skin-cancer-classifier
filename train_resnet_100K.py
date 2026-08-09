import argparse
import copy
import os
import random
import ssl
import time

import numpy as np

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from PIL import Image
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ssl._create_default_https_context = ssl._create_unverified_context

CHECKPOINT_EVERY_STEPS = 2000  # save a resumable checkpoint this often within an epoch

# =============================================================================
# >>> DATASET CONFIG -- EDIT THIS WHEN SWITCHING AUGMENTED DATASET SIZES <<<
#
# Both pump_data_augment.py (600 combos/image) and lowpump_data_augment.py
# (135 combos/image) write the SAME column layout, just different filenames:
#   filename, original_image, flip, rotation_deg, cool_level, warm_level, label
# So switching datasets is just pointing --meta_csv (or this default) at the
# right file -- nothing else in this script needs to change.
#
#   1M dataset (10 combos/image):    "all_augment_1M/lowmeta.csv"
#   100K dataset (10 combos/image):  "all_augment_100K/lowestmeta.csv"
#   6M dataset (600 combos/image):   "all_augment_6M/meta.csv"
#
# Currently set to the 100K dataset. To train on a different dataset instead,
# either change the line below, or pass --meta_csv on the command line
# (the CLI flag always overrides this default).
# =============================================================================
DEFAULT_META_CSV = "all_augment_100K/lowestmeta.csv"


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


class FocalLoss(nn.Module):
    """
    Multi-class focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Unlike plain class-weighted CrossEntropy (which scales every mistake on
    a given class by a fixed amount regardless of how confidently wrong the
    model was), focal loss additionally down-weights EASY examples (already
    classified with high confidence) and concentrates gradient on HARD ones
    -- e.g. an mel image that looks a lot like nv, which is exactly the kind
    of mistake that matters for a cancer screen. gamma controls how strongly
    easy examples get down-weighted (0 = reduces to weighted CrossEntropy).
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(smooth_targets * log_probs).sum(dim=-1)
            pt = torch.exp(-ce)
        else:
            ce = torch.nn.functional.nll_loss(log_probs, targets, reduction="none")
            pt = torch.exp(-ce)

        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()


def merge_labels_if_missing(df, metadata_csv, id_col, label_col):
    """
    meta.csv from pump_data_augment.py only has labels baked in if
    --metadata_csv was passed during augmentation. If the label column
    came out empty, merge labels in now from a separate id,label CSV
    (e.g. HAM10000_metadata.csv), matching on original_image with its
    file extension stripped off.
    """
    non_null = df["label"].dropna()
    has_labels = len(non_null) > 0 and (non_null.astype(str).str.strip() != "").any()
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


def three_way_split(per_image, val_fraction, test_fraction, seed):
    """
    Splits SOURCE IMAGES (not rows) into train/val/test, shuffled and
    stratified by class label at every step, using the given fractions
    (e.g. 0.75/0.15/0.10). Splitting at the row level instead would leak
    augmented variants of the same source image across sets.
    """
    labels_by_image = per_image.set_index("original_image")["label"]

    # Step 1: carve off the test set from everything
    train_val_images, test_images = train_test_split(
        per_image["original_image"],
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=per_image["label"],
    )

    # Step 2: split what's left into train/val, re-expressing val_fraction
    # relative to the remaining (1 - test_fraction) pool
    relative_val_fraction = val_fraction / (1 - test_fraction)
    train_val_labels = labels_by_image.loc[train_val_images]

    train_images, val_images = train_test_split(
        train_val_images,
        test_size=relative_val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )

    return set(train_images), set(val_images), set(test_images)


def load_meta_and_split(meta_csv, val_fraction, test_fraction, seed,
                         metadata_csv=None, id_col="image_id", label_col="dx"):
    """
    Reads meta.csv/lowmeta.csv and splits at the ORIGINAL IMAGE level (not row
    level) so that every augmented variant of a given source image (however
    many combos-per-image this particular dataset has -- 135, 600, or any
    other count) ends up entirely in one of train/val/test -- never split
    across more than one. Splitting at the row level would leak near-duplicate
    images between sets and give you artificially inflated validation/test
    accuracy.
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

    print(f"Source images   -> train: {len(train_images):,}  val: {len(val_images):,}  "
          f"test: {len(test_images):,}")
    print(f"Output rows     -> train: {len(train_rows):,}  val: {len(val_rows):,}  "
          f"test: {len(test_rows):,}")

    return train_rows, val_rows, test_rows, class_to_idx


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_val_metric, counter,
                     class_to_idx, scaler=None):
    torch.save({
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val_metric": best_val_metric,  # macro recall on val set (higher is better)
        "counter": counter,
        "class_to_idx": class_to_idx,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    unwrap_model(model).load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    # Older checkpoints tracked "best_val_loss" (lower-is-better) instead of
    # "best_val_metric" (macro recall, higher-is-better). If resuming from one
    # of those, there's no way to translate the old value onto the new scale,
    # so just restart the "best" tracking from scratch rather than silently
    # misinterpreting a loss value as a recall value.
    if "best_val_metric" in ckpt:
        best_val_metric = ckpt["best_val_metric"]
    else:
        print("[WARN] Checkpoint predates macro-recall-based model selection -- "
              "restarting best-model tracking from scratch (0.0).")
        best_val_metric = 0.0
    return ckpt["epoch"], ckpt["step"], best_val_metric, ckpt["counter"], ckpt["class_to_idx"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    """
    Fixes every source of randomness we control so reruns use the same
    train/val/test split, the same shuffle order each epoch, and the same
    initial conditions -- so you get the same set of images every run
    instead of a different random subset/order each time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Gives each DataLoader worker process its own deterministic seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def unwrap_model(model):
    """Returns the underlying model whether or not it's wrapped in DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


def print_gpu_info():
    if not torch.cuda.is_available():
        print("[GPU] No CUDA device available -- running on CPU.")
        return
    n = torch.cuda.device_count()
    print(f"[GPU] {n} CUDA device(s) visible:")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"       [{i}] {props.name} -- {total_gb:.1f} GB total memory")


def find_max_batch_size(model, optimizer, criterion, device, num_classes,
                         start_batch, max_batch, use_amp=True,
                         growth_factor=2, safety_margin=0.9, refine_steps=6):
    """
    Probes how large a batch size fits in GPU memory in two passes:

    1. COARSE: doubles the batch size (128, 256, 512, 1024, 2048, ...) until
       it hits an out-of-memory error. This is fast but leaves a big gap --
       e.g. if 1024 works and 2048 OOMs, the true max could be anywhere in
       between (it might be 1638, might be 1100), and doubling alone would
       never find that.
    2. REFINE: binary-searches within that gap for `refine_steps` rounds to
       narrow in on the actual boundary, so the final number reflects what
       the GPU can really do instead of being stuck at the last power of 2.

    Probing is done under the SAME autocast/AMP settings that real training
    will use, since mixed precision changes how much memory a given batch
    size actually needs.

    `safety_margin` then backs off a bit further from that boundary, since
    real training adds overhead this synthetic probe doesn't see: DataLoader
    worker processes, pinned-memory buffers, CUDA memory fragmentation over
    many steps, etc. Default 0.9 (not 0.8) because the refine pass already
    gets us much closer to the true ceiling than raw doubling did.

    Model and optimizer state are restored afterward so this doesn't
    corrupt the pretrained weights before real training starts.
    """
    if device.type != "cuda":
        print("[AUTOTUNE] Not running on CUDA -- skipping batch size probe, "
              f"using --batch_size as given ({start_batch}).")
        return start_batch

    amp_tag = "with AMP" if use_amp else "without AMP"
    print(f"[AUTOTUNE] Probing max batch size that fits in GPU memory ({amp_tag}), "
          f"ceiling={max_batch:,} ...")
    model_state = copy.deepcopy(unwrap_model(model).state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    scaler = GradScaler("cuda", enabled=use_amp)

    def try_batch(batch):
        """Runs one real forward+backward+step at this batch size. True = fit OK."""
        try:
            torch.cuda.empty_cache()
            dummy_inputs = torch.randn(batch, 3, 224, 224, device=device)
            dummy_labels = torch.randint(0, num_classes, (batch,), device=device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                outputs = model(dummy_inputs)
                loss = criterion(outputs, dummy_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return False
            raise

    last_good = start_batch
    first_bad = None
    try:
        model.train()

        # --- Pass 1: coarse doubling to find SOME working batch and SOME failing batch ---
        batch = start_batch
        while batch <= max_batch:
            if try_batch(batch):
                print(f"[AUTOTUNE] batch_size={batch} OK")
                last_good = batch
                if batch == max_batch:
                    break
                batch = min(batch * growth_factor, max_batch)
            else:
                print(f"[AUTOTUNE] batch_size={batch} ran out of memory -- stopping coarse search")
                first_bad = batch
                break

        # --- Pass 2: binary search the gap between last_good and first_bad ---
        if first_bad is not None:
            lo, hi = last_good, first_bad
            print(f"[AUTOTUNE] Refining between {lo} (OK) and {hi} (OOM) ...")
            for _ in range(refine_steps):
                if hi - lo <= 1:
                    break
                mid = (lo + hi) // 2
                if try_batch(mid):
                    print(f"[AUTOTUNE] batch_size={mid} OK")
                    lo = mid
                else:
                    print(f"[AUTOTUNE] batch_size={mid} ran out of memory")
                    hi = mid
            last_good = lo
    finally:
        unwrap_model(model).load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        torch.cuda.empty_cache()

    safe_batch = max(start_batch, int(last_good * safety_margin))
    print(f"[AUTOTUNE] Largest batch size that fit: {last_good}. "
          f"Using batch size with safety margin: {safe_batch}")
    return safe_batch


def main():
    parser = argparse.ArgumentParser(description="Train ResNet50 on the augmented skin cancer dataset.")
    parser.add_argument("--meta_csv", default=DEFAULT_META_CSV,
                         help='Path to meta.csv / lowmeta.csv produced by pump_data_augment.py '
                              'or lowpump_data_augment.py. Defaults to DEFAULT_META_CSV set near '
                              'the top of this file -- override here to use a different dataset '
                              'without editing that constant.')
    parser.add_argument("--metadata_csv", default=None,
                         help="Optional id,label CSV (e.g. HAM10000_metadata.csv) used to fill in "
                              "labels if meta.csv's label column came out empty")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--val_fraction", type=float, default=0.15,
                         help="Fraction of SOURCE IMAGES (not rows) held out for validation")
    parser.add_argument("--test_fraction", type=float, default=0.10,
                         help="Fraction of SOURCE IMAGES (not rows) held out for test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=8,
                         help="Epochs with no improvement in val macro recall before stopping. "
                              "Raised from 5 -- macro recall is noisier epoch-to-epoch than "
                              "val_loss was, so a short patience stops training prematurely.")
    parser.add_argument("--best_model_path", default="skin_cancer_best_resnet_100K.pth")
    parser.add_argument("--checkpoint_path", default="train_checkpoint_100K.pth")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from --checkpoint_path if it exists")
    parser.add_argument("--gpus", default=None,
                         help='Comma-separated GPU ids to use, e.g. "0,1". '
                              "Default: use all visible GPUs.")
    parser.add_argument("--auto_batch_size", action="store_true",
                         help="Before training, probe the largest batch size that fits in "
                              "GPU memory and use that instead of --batch_size.")
    parser.add_argument("--probe_max_batch", type=int, default=2048,
                         help="Upper bound to try during --auto_batch_size probing. Raise this "
                              "to actually search past small batch sizes -- the probe won't try "
                              "anything above this ceiling no matter how much VRAM is free.")
    parser.add_argument("--log_every", type=int, default=50,
                         help="Print training progress every N steps within an epoch")
    parser.add_argument("--no_amp", action="store_true",
                         help="Disable mixed-precision (AMP) training. AMP is ON by default on "
                              "CUDA -- it roughly halves activation memory and speeds up training "
                              "on Tensor Core GPUs like the A4000, letting you fit a larger batch.")
    parser.add_argument("--loss_type", choices=["ce", "focal"], default="focal",
                         help="'ce' = (class-weighted) CrossEntropy. 'focal' = focal loss, which "
                              "additionally concentrates gradient on hard-to-classify examples "
                              "(e.g. mel images that look like nv) instead of just scaling by "
                              "class frequency. Default 'focal' -- tends to help recall on rare, "
                              "hard classes more than weighted CE alone.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                         help="Focal loss focusing parameter (only used with --loss_type focal). "
                              "Higher = more focus on hard examples, 0 = equivalent to weighted CE.")
    parser.add_argument("--label_smoothing", type=float, default=0.05,
                         help="Label smoothing for the loss (0.0 = off). Train accuracy was "
                              "hitting 99%% while val/test lagged far behind -- smoothing keeps "
                              "the model from getting overconfident on training examples.")
    parser.add_argument("--class_weight_power", type=float, default=0.5,
                         help="Exponent applied to inverse-frequency class weights: 1.0 = full "
                              "inverse frequency (can over-correct and tank overall accuracy on "
                              "a heavily imbalanced dataset like HAM10000), 0.5 = softened "
                              "(sqrt), 0.0 = no weighting at all. Default 0.5 is a middle ground.")
    parser.add_argument("--sampler_weight_power", type=float, default=0.75,
                         help="Exponent applied to inverse-frequency weights used by the "
                              "oversampling SAMPLER (separate from --class_weight_power, which "
                              "only affects the loss). Higher than the loss power is often better: "
                              "it lets the model physically SEE rare classes more often per epoch "
                              "without making the loss itself noisier/more unstable to optimize.")
    parser.add_argument("--no_class_weighted_loss", action="store_true",
                         help="Disable inverse-frequency class weighting in the loss. ON by "
                              "default: HAM10000 is heavily imbalanced (nv is the majority "
                              "class) and unweighted loss tends to sacrifice recall on rare "
                              "classes like mel/akiec to optimize overall accuracy on nv.")
    parser.add_argument("--no_oversample_minority", action="store_true",
                         help="Disable oversampling of minority classes during training. ON by "
                              "default: pairs with class-weighted loss to make sure the model "
                              "actually SEES rare classes (mel, akiec, bcc, df, vasc) often "
                              "enough per epoch, not just penalizes mistakes on them more.")
    args = parser.parse_args()

    set_seed(args.seed)

    # --- 1. Transforms ---
    # The 100K dataset only bakes in 10 static combos/image (2 flips x 5
    # rotations, no color/tone variation) -- the model sees the SAME 10
    # pixel-exact images every single epoch, which is easy to memorize.
    # Live augmentation applies a DIFFERENT random transform each time an
    # image is loaded, so the model never sees quite the same pixels twice
    # across epochs. This is genuine regularization the static files can't
    # provide on their own, however many combos are baked in.
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- 2. Load meta.csv and split at the source-image level (75/15/10) ---
    train_rows, val_rows, test_rows, class_to_idx = load_meta_and_split(
        args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
        metadata_csv=args.metadata_csv, id_col=args.id_col, label_col=args.label_col,
    )
    num_classes = len(class_to_idx)
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # Inverse-frequency class weights, computed from the TRAIN split only.
    # class_counts[i] = how many train rows have label i. A class with few
    # rows gets a bigger weight so the loss doesn't just get minimized by
    # nailing the majority class (nv) and ignoring rare ones (mel, akiec).
    # loss_weights and sampler_weights are DELIBERATELY separate: the sampler
    # can oversample rare classes harder than the loss penalizes them, which
    # affects how often the model sees rare examples without directly
    # destabilizing the loss landscape the way a very high loss weight would.
    class_counts = np.bincount([label for _, label in train_rows], minlength=num_classes)
    raw_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    loss_weights = raw_weights ** args.class_weight_power
    sampler_weights = raw_weights ** args.sampler_weight_power
    class_weights_tensor = torch.tensor(loss_weights, dtype=torch.float32)
    print(f"[CLASS BALANCE] train-set counts per class "
          f"(loss_power={args.class_weight_power}, sampler_power={args.sampler_weight_power}):")
    for i in range(num_classes):
        print(f"    {idx_to_class[i]:>6}: {class_counts[i]:>8,} rows  "
              f"(loss weight: {loss_weights[i]:.3f}, sampler weight: {sampler_weights[i]:.3f})")

    dataset_train = AugmentedSkinDataset(train_rows, transform=train_transforms)
    dataset_val = AugmentedSkinDataset(val_rows, transform=val_transforms)
    dataset_test = AugmentedSkinDataset(test_rows, transform=val_transforms)

    print_gpu_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"[AMP] Mixed precision training: {'ON' if use_amp else 'OFF'}")

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

    # --- Multi-GPU data-parallel wrap ---
    if device.type == "cuda":
        if args.gpus:
            gpu_ids = [int(x) for x in args.gpus.split(",")]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))

        if len(gpu_ids) > 1:
            print(f"[GPU] Using DataParallel across GPUs: {gpu_ids}")
            model = nn.DataParallel(model, device_ids=gpu_ids)
        else:
            print(f"[GPU] Single GPU in use: {gpu_ids[0]}")

    alpha = None if args.no_class_weighted_loss else class_weights_tensor.to(device)

    if args.loss_type == "focal":
        criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma,
                               label_smoothing=args.label_smoothing).to(device)
        print(f"[LOSS] Focal loss (gamma={args.focal_gamma}), "
              f"class weighting: {'ON' if alpha is not None else 'OFF'}")
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=args.label_smoothing)
        print(f"[LOSS] CrossEntropy, class weighting: {'ON' if alpha is not None else 'OFF'}")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if args.auto_batch_size:
        args.batch_size = find_max_batch_size(
            model, optimizer, criterion, device, num_classes,
            start_batch=args.batch_size, max_batch=args.probe_max_batch,
            use_amp=use_amp,
        )
        print(f"[AUTOTUNE] Training will use batch_size={args.batch_size}")

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(args.seed)

    if args.no_oversample_minority:
        train_sampler = None
        print("[CLASS BALANCE] Minority oversampling: OFF")
    else:
        # Give each row a sampling weight = inverse frequency of its class,
        # so a WeightedRandomSampler draws rare-class rows (mel, akiec, ...)
        # proportionally more often per epoch instead of the natural
        # imbalanced frequency. replacement=True since minority classes need
        # to be drawn more times than they physically have rows.
        per_row_weights = np.array([sampler_weights[label] for _, label in train_rows],
                                    dtype=np.float64)
        train_sampler = WeightedRandomSampler(
            weights=per_row_weights, num_samples=len(train_rows),
            replacement=True, generator=dataloader_generator,
        )
        print("[CLASS BALANCE] Minority oversampling: ON (WeightedRandomSampler)")

    train_loader = DataLoader(
        dataset_train, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker, generator=dataloader_generator,
    )
    val_loader = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker,
    )

    start_epoch = 0
    resume_step = 0
    best_val_metric = 0.0  # macro recall on val set; higher is better
    counter = 0
    scaler = GradScaler("cuda", enabled=use_amp)

    if args.resume and os.path.exists(args.checkpoint_path):
        start_epoch, resume_step, best_val_metric, counter, saved_class_to_idx = load_checkpoint(
            args.checkpoint_path, model, optimizer, scheduler, scaler=scaler
        )
        if saved_class_to_idx != class_to_idx:
            print("[WARN] class_to_idx from checkpoint differs from current data. "
                  "Proceeding with current data's class_to_idx.")
        print(f"[RESUME] Resuming from epoch {start_epoch}, step {resume_step}, "
              f"best_val_macro_recall={best_val_metric:.4f}, patience_counter={counter}")

    print("Starting training...")
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        skip_until = resume_step if epoch == start_epoch else 0
        num_steps = len(train_loader)
        epoch_start_time = time.time()

        print(f"[EPOCH {epoch+1:02d}/{args.epochs}] starting -- {num_steps:,} steps "
              f"this epoch (batch_size={args.batch_size})", flush=True)

        for step, (inputs, labels) in enumerate(train_loader):
            if step < skip_until:
                continue  # fast-skip batches already done before the crash

            step_start_time = time.time()

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            steps_done = step + 1 - skip_until
            if steps_done > 0 and (steps_done % args.log_every == 0 or step + 1 == num_steps):
                elapsed = time.time() - epoch_start_time
                steps_per_sec = steps_done / max(elapsed, 1e-6)
                remaining_steps = num_steps - (step + 1)
                eta_sec = remaining_steps / max(steps_per_sec, 1e-6)
                cur_loss = running_loss / steps_done
                cur_acc = 100 * correct / max(1, total)
                print(f"[EPOCH {epoch+1:02d}] step {step + 1:,}/{num_steps:,} "
                      f"({100 * (step + 1) / num_steps:.1f}%) | "
                      f"loss: {cur_loss:.4f} acc: {cur_acc:.2f}% | "
                      f"{steps_per_sec:.2f} it/s | "
                      f"elapsed: {elapsed/60:.1f}m ETA: {eta_sec/60:.1f}m", flush=True)

            if (step + 1) % CHECKPOINT_EVERY_STEPS == 0:
                save_checkpoint(args.checkpoint_path, model, optimizer, scheduler,
                                 epoch, step + 1, best_val_metric, counter, class_to_idx,
                                 scaler=scaler)
                print(f"[CHECKPOINT] saved at epoch {epoch+1}, step {step + 1}", flush=True)

        train_loss = running_loss / max(1, (len(train_loader) - skip_until))
        train_acc = 100 * correct / max(1, total)

        print(f"[EPOCH {epoch+1:02d}] training done in {(time.time()-epoch_start_time)/60:.1f}m "
              f"-- running validation...", flush=True)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_all_preds, val_all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast("cuda", enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_all_preds.extend(predicted.cpu().numpy().tolist())
                val_all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        # Macro recall = average of each class's own recall, unweighted by how
        # common that class is. This is what early stopping/model selection
        # should track on an imbalanced dataset -- val_loss here is computed
        # with the same class-weighted criterion as training, so its absolute
        # scale isn't a clean signal of "is this model actually better", and
        # plain val_acc rewards being good at the majority class (nv) same as
        # before. Macro recall treats mel/akiec/df/vasc as equally important
        # as nv, matching what we actually care about for a cancer screen.
        val_macro_recall = recall_score(val_all_labels, val_all_preds,
                                         average="macro", zero_division=0)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% Macro Recall: {val_macro_recall:.4f}",
              flush=True)

        scheduler.step()

        # end-of-epoch checkpoint (step resets to 0 for the next epoch)
        save_checkpoint(args.checkpoint_path, model, optimizer, scheduler,
                         epoch + 1, 0, best_val_metric, counter, class_to_idx,
                         scaler=scaler)

        if val_macro_recall > best_val_metric:
            best_val_metric = val_macro_recall
            counter = 0
            torch.save(unwrap_model(model).state_dict(), args.best_model_path)
        else:
            counter += 1
            if counter >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Total time: {(time.time() - start_time)/60:.2f} minutes")

    # --- Final evaluation on the held-out TEST set (never seen during train/val) ---
    print("Evaluating best model on test set...")
    unwrap_model(model).load_state_dict(torch.load(args.best_model_path, map_location=device))
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * test_correct / test_total
    print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")


if __name__ == "__main__":
    main()