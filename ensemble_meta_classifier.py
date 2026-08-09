"""
Meta-Learning Ensemble (DAME-style Level-0 / Level-1 stacking)

Loads three independently trained backbones:
  - ResNet50   (skin_cancer_best_resnet_1M.pth)
  - DenseNet121 (skin_cancer_best_densenet_1M.pth)
  - ViT-B/16    (skin_cancer_best_vit_1M.pth)

Level 0: each backbone produces a 7-class softmax vector for every image.
Level 1: the three 7-dim vectors are concatenated into one 21-dim feature
         vector, and a Logistic Regression meta-classifier is trained on
         top of THAT to produce the final prediction.

IMPORTANT METHODOLOGY NOTE: the meta-classifier is trained on the VAL split
predictions, NOT the train split. Backbones were fit directly on the train
split and are near-memorized on it (we saw train acc hit 96-99% earlier),
so train-split softmax outputs are overconfident and would teach the
meta-classifier a distorted picture. The val split is something none of the
three backbones were trained on, so its predictions reflect genuine
generalization behavior -- this is standard stacked-ensemble practice.

The TEST split (never touched by backbone training OR meta-classifier
training) is used only for final, honest evaluation.

Usage (defaults match the 1M dataset / checkpoint naming used throughout):
    python3 ensemble_meta_classifier.py

Expects all three best_model_path files described above to already exist
(i.e. all three train_*_1M.py scripts have finished running).

--------------------------------------------------------------------------
ADDED ON TOP OF THE ORIGINAL SCRIPT (all original logic untouched above):
  1. Saves the fitted meta-classifier + a config describing the ensemble,
     so it can be reloaded for inference without re-fitting.
  2. Saves a confusion-matrix heatmap PNG (if matplotlib/seaborn installed).
  3. Optional --error_analysis: dumps misclassified filenames for the main
     confusion pairs (mel/nv, mel/bkl, akiec/bkl) to CSV for manual review.
  4. Optional --gradcam_per_pair N: saves N Grad-CAM overlay images per
     backbone per confusion pair (requires `pip install grad-cam`).
  5. MULTI-GPU: the three backbones now run their val/test inference
     CONCURRENTLY, each pinned to its own device -- ResNet50 + DenseNet121
     share one GPU, ViT (the slowest backbone) gets a dedicated GPU. Falls
     back gracefully to a single GPU or CPU if only one device is available.
     Override with --resnet_device / --densenet_device / --vit_device
     (e.g. "cuda:0", "cuda:1", "cpu") if the auto-assignment isn't right
     for your machine.
  6. Fixes a "Too many open files" DataLoader crash on the large val split
     via torch.multiprocessing.set_sharing_strategy('file_system').
--------------------------------------------------------------------------
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn

# Fixes "RuntimeError: Too many open files" crashes from DataLoader workers
# on large splits (val set here is 200k+ rows) -- must be set before any
# DataLoader with num_workers > 0 is created.
torch.multiprocessing.set_sharing_strategy('file_system')

from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CANCER_RELATED_CLASSES = {"mel", "bcc", "akiec"}

# Optional plotting deps -- mirrors the guarded-import pattern used in
# evaluate_confusion_matrix.py ("[INFO] matplotlib not installed -- skipping...")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Optional Grad-CAM deps
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False


# ---------------------------------------------------------------------------
# Shared dataset / split logic (same as train_resnet_1M.py / evaluate_confusion_matrix.py)
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

    classes = sorted(df["label"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    per_image = df.groupby("original_image", observed=True)["label"].first().reset_index()
    train_images, val_images, test_images = three_way_split(
        per_image, val_fraction, test_fraction, seed
    )

    val_df = df[df["original_image"].isin(val_images)]
    test_df = df[df["original_image"].isin(test_images)]

    val_rows = list(zip(val_df["filename"].tolist(),
                         val_df["label"].map(class_to_idx).tolist()))
    test_rows = list(zip(test_df["filename"].tolist(),
                          test_df["label"].map(class_to_idx).tolist()))
    return val_rows, test_rows, class_to_idx


# ---------------------------------------------------------------------------
# Backbone construction (must match each train_*.py script's architecture exactly)
# ---------------------------------------------------------------------------

def build_resnet50(num_classes, device):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Dropout(0.8), nn.Linear(128, num_classes),
    )
    return model.to(device)


def build_densenet121(num_classes, device):
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Dropout(0.8), nn.Linear(128, num_classes),
    )
    return model.to(device)


def build_vit(num_classes, device):
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Dropout(0.8), nn.Linear(128, num_classes),
    )
    return model.to(device)


BACKBONE_BUILDERS = {
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "vit": build_vit,
}


@torch.no_grad()
def get_softmax_outputs(model, loader, device):
    """Runs inference over a loader, returns (N, num_classes) softmax probs and (N,) labels."""
    model.eval()
    all_probs, all_labels = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# ADDED: multi-GPU pipeline -- each backbone gets its own device and its own
# DataLoader instances, and the three backbones run concurrently via a
# thread pool (CUDA ops release the GIL during kernel execution/waits, so
# separate devices genuinely overlap).
# ---------------------------------------------------------------------------

def pick_devices(args):
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        default_cnn_device, default_vit_device = "cuda:0", "cuda:1"
    elif num_gpus == 1:
        default_cnn_device = default_vit_device = "cuda:0"
    else:
        default_cnn_device = default_vit_device = "cpu"

    return {
        "resnet50": args.resnet_device or default_cnn_device,
        "densenet121": args.densenet_device or default_cnn_device,
        "vit": args.vit_device or default_vit_device,
    }


def run_backbone_pipeline(name, path, device_str, num_classes,
                           val_rows, test_rows, eval_transforms,
                           batch_size, workers):
    """Loads one backbone onto its own device and runs val+test inference.
    Each backbone gets its OWN DataLoader instances (safe for concurrent
    use across threads/devices); shuffle=False on both guarantees row
    order matches test_rows/val_rows exactly, same as the original script."""
    device = torch.device(device_str)
    print(f"[{name}] loading weights from {path} (device={device_str}) ...")
    model = BACKBONE_BUILDERS[name](num_classes, device)
    model.load_state_dict(torch.load(path, map_location=device))

    val_loader = DataLoader(AugmentedSkinDataset(val_rows, eval_transforms),
                             batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(AugmentedSkinDataset(test_rows, eval_transforms),
                              batch_size=batch_size, shuffle=False, num_workers=workers)

    print(f"[{name}] running inference on val set ...")
    val_probs, val_labels = get_softmax_outputs(model, val_loader, device)
    print(f"[{name}] running inference on test set ...")
    test_probs, test_labels = get_softmax_outputs(model, test_loader, device)

    return name, model, val_probs, val_labels, test_probs, test_labels


# ---------------------------------------------------------------------------
# ADDED: artifact saving, heatmap, error analysis, Grad-CAM
# ---------------------------------------------------------------------------

def save_ensemble_artifacts(meta_clf, args, classes, ensemble_acc,
                             ensemble_macro_recall, out_dir="."):
    """Persist the fitted meta-classifier + a config describing how the
    ensemble is built, so future inference doesn't require re-fitting it."""
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, "ensemble_meta_classifier.joblib")
    joblib.dump(meta_clf, meta_path)

    config = {
        "resnet_path": args.resnet_path,
        "densenet_path": args.densenet_path,
        "vit_path": args.vit_path,
        "meta_classifier_path": meta_path,
        "classes": classes,
        "test_accuracy": float(ensemble_acc),
        "test_macro_recall": float(ensemble_macro_recall),
    }
    config_path = os.path.join(out_dir, "ensemble_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[artifacts] saved meta-classifier to {meta_path}")
    print(f"[artifacts] saved config to {config_path}")


def save_confusion_heatmap(cm, classes, out_path="ensemble_confusion_heatmap.png"):
    if not HAS_PLOTTING:
        print("[INFO] matplotlib/seaborn not installed -- skipping the PNG "
              "heatmap (the printed confusion matrix above still has "
              "everything). Install with: pip install matplotlib seaborn")
        return
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[artifacts] saved confusion matrix heatmap to {out_path}")


def save_misclassified_examples(test_rows, y_true, y_pred, class_to_idx,
                                 error_pairs, out_dir="error_analysis"):
    """Dumps one CSV per (true_class, predicted_class) pair, listing the
    filenames of every test-set image in that specific confusion bucket --
    for manual visual review. Relies on shuffle=False on the test_loader so
    test_rows order matches y_true/y_pred order exactly."""
    os.makedirs(out_dir, exist_ok=True)
    filenames = [r[0] for r in test_rows]

    for true_c, pred_c in error_pairs:
        true_idx = class_to_idx[true_c]
        pred_idx = class_to_idx[pred_c]
        mask = (y_true == true_idx) & (y_pred == pred_idx)
        rows = [filenames[i] for i in np.where(mask)[0]]
        out_path = os.path.join(out_dir, f"errors_{true_c}_as_{pred_c}.csv")
        pd.DataFrame({"filename": rows}).to_csv(out_path, index=False)
        print(f"[error-analysis] {true_c} -> {pred_c}: {len(rows)} cases "
              f"saved to {out_path}")


def _vit_reshape_transform(tensor, height=14, width=14):
    """height*width must equal the number of patch tokens for the ViT config
    in use (224/16 = 14x14 for vit_b_16 at 224px input)."""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def _get_target_layers(name, model):
    if name == "resnet50":
        return [model.layer4[-1]], None
    if name == "densenet121":
        return [model.features.denseblock4], None
    if name == "vit":
        return [model.encoder.layers[-1].ln_1], _vit_reshape_transform
    raise ValueError(f"No Grad-CAM target layer configured for backbone '{name}'")


def save_gradcam_examples(backbones, eval_transforms, test_rows,
                           y_true, y_pred, class_to_idx, error_pairs,
                           n_per_pair=5, out_dir="gradcam_examples"):
    """For each backbone and each (true_class, predicted_class) error pair,
    saves up to n_per_pair Grad-CAM overlay images so the model's attention
    can be visually inspected on cases it got wrong. Each backbone's device
    is read directly off its parameters, since backbones may now live on
    different GPUs (multi-GPU pipeline)."""
    if not HAS_GRADCAM:
        print("[INFO] pytorch-grad-cam not installed -- skipping Grad-CAM. "
              "Install with: pip install grad-cam")
        return

    os.makedirs(out_dir, exist_ok=True)
    filenames = [r[0] for r in test_rows]

    for backbone_name, (model, _path) in backbones.items():
        device = next(model.parameters()).device
        target_layers, reshape_transform = _get_target_layers(backbone_name, model)
        cam = GradCAM(model=model, target_layers=target_layers,
                      reshape_transform=reshape_transform)

        for true_c, pred_c in error_pairs:
            true_idx = class_to_idx[true_c]
            pred_idx = class_to_idx[pred_c]
            mask = (y_true == true_idx) & (y_pred == pred_idx)
            indices = np.where(mask)[0][:n_per_pair]

            pair_dir = os.path.join(out_dir, backbone_name, f"{true_c}_as_{pred_c}")
            os.makedirs(pair_dir, exist_ok=True)

            for i in indices:
                path = filenames[i]
                pil_img = Image.open(path).convert("RGB").resize((224, 224))
                input_tensor = eval_transforms(pil_img).unsqueeze(0).to(device)

                targets = [ClassifierOutputTarget(pred_idx)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

                rgb_img = np.array(pil_img) / 255.0
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                out_name = os.path.basename(path)
                Image.fromarray(visualization).save(os.path.join(pair_dir, out_name))

            print(f"[grad-cam] {backbone_name} {true_c}->{pred_c}: "
                  f"saved {len(indices)} overlays to {pair_dir}")


def main():
    parser = argparse.ArgumentParser(description="DAME-style meta-learning ensemble.")
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
    parser.add_argument("--meta_max_iter", type=int, default=1000,
                         help="max_iter for the Logistic Regression meta-classifier "
                              "(1000, matching the DAME paper's reported setting)")
    # --- ADDED ARGS ---
    parser.add_argument("--artifacts_dir", default=".",
                         help="Where to save ensemble_meta_classifier.joblib + ensemble_config.json")
    parser.add_argument("--error_analysis", action="store_true",
                         help="If set, dump misclassified filenames for the main "
                              "confusion pairs (mel/nv, mel/bkl, akiec/bkl) to CSV.")
    parser.add_argument("--error_analysis_dir", default="error_analysis")
    parser.add_argument("--gradcam_per_pair", type=int, default=0,
                         help="If > 0 (and --error_analysis is set), save this many "
                              "Grad-CAM overlay images per backbone per confusion pair.")
    parser.add_argument("--gradcam_dir", default="gradcam_examples")
    # --- ADDED: multi-GPU device overrides ---
    parser.add_argument("--resnet_device", default=None,
                         help="e.g. cuda:0 (default: auto -- shares a GPU with densenet121)")
    parser.add_argument("--densenet_device", default=None,
                         help="e.g. cuda:0 (default: auto -- shares a GPU with resnet50)")
    parser.add_argument("--vit_device", default=None,
                         help="e.g. cuda:1 (default: auto -- gets its own GPU if 2+ available)")
    args = parser.parse_args()

    for path, label in [(args.resnet_path, "ResNet50"),
                         (args.densenet_path, "DenseNet121"),
                         (args.vit_path, "ViT")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[MISSING] {label} weights not found at '{path}'. "
                f"All three backbones must finish training before running the ensemble."
            )

    print(f"Loading {args.meta_csv} and rebuilding val/test splits "
          f"(seed={args.seed}) ...")
    val_rows, test_rows, class_to_idx = load_meta_and_split(
        args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
        metadata_csv=args.metadata_csv, id_col=args.id_col, label_col=args.label_col,
    )
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(classes)
    print(f"Val set: {len(val_rows):,} rows | Test set: {len(test_rows):,} rows | "
          f"Classes: {classes}")

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- MULTI-GPU: assign each backbone a device, run all three concurrently ---
    devices = pick_devices(args)
    print(f"Device assignment: {devices}")
    backbone_paths = {
        "resnet50": args.resnet_path,
        "densenet121": args.densenet_path,
        "vit": args.vit_path,
    }
    # Reduce per-loader workers a bit when running concurrently, so three
    # backbones don't collectively spawn 3x args.workers processes at once.
    per_backbone_workers = max(1, args.workers // len(backbone_paths))

    results = {}
    with ThreadPoolExecutor(max_workers=len(backbone_paths)) as executor:
        futures = {
            executor.submit(
                run_backbone_pipeline, name, path, devices[name], num_classes,
                val_rows, test_rows, eval_transforms, args.batch_size,
                per_backbone_workers,
            ): name
            for name, path in backbone_paths.items()
        }
        for future in as_completed(futures):
            name, model, val_probs, val_labels, test_probs, test_labels = future.result()
            results[name] = {
                "model": model,
                "val_probs": val_probs, "val_labels": val_labels,
                "test_probs": test_probs, "test_labels": test_labels,
            }

    # Reassemble in a fixed order (resnet50, densenet121, vit) regardless of
    # which finished first, and re-run the same order-consistency check the
    # original script had.
    order = ["resnet50", "densenet121", "vit"]
    val_labels_ref = results[order[0]]["val_labels"]
    test_labels_ref = results[order[0]]["test_labels"]
    val_feature_blocks, test_feature_blocks = [], []
    backbones = {}
    for name in order:
        r = results[name]
        assert np.array_equal(r["val_labels"], val_labels_ref), \
            f"[{name}] val label order mismatch -- dataloader ordering is inconsistent!"
        assert np.array_equal(r["test_labels"], test_labels_ref), \
            f"[{name}] test label order mismatch -- dataloader ordering is inconsistent!"
        val_feature_blocks.append(r["val_probs"])
        test_feature_blocks.append(r["test_probs"])
        backbones[name] = (r["model"], backbone_paths[name])

    val_features = np.concatenate(val_feature_blocks, axis=1)   # (N_val, 21)
    test_features = np.concatenate(test_feature_blocks, axis=1)  # (N_test, 21)
    print(f"Stacked feature shape -- val: {val_features.shape}, test: {test_features.shape}")

    print(f"Training Logistic Regression meta-classifier "
          f"(max_iter={args.meta_max_iter}) on VAL-set predictions ...")
    meta_clf = LogisticRegression(max_iter=args.meta_max_iter)
    meta_clf.fit(val_features, val_labels_ref)

    test_preds = meta_clf.predict(test_features)

    print("\n" + "=" * 70)
    print("INDIVIDUAL BACKBONE ACCURACY ON TEST SET (for comparison)")
    print("=" * 70)
    for i, name in enumerate(order):
        block = test_feature_blocks[i]
        preds = block.argmax(axis=1)
        acc = (preds == test_labels_ref).mean()
        macro_rec = recall_score(test_labels_ref, preds, average="macro", zero_division=0)
        print(f"  {name:>12}: accuracy={acc:.4f}  macro_recall={macro_rec:.4f}")

    ensemble_acc = (test_preds == test_labels_ref).mean()
    ensemble_macro_recall = recall_score(test_labels_ref, test_preds, average="macro", zero_division=0)
    print(f"  {'ENSEMBLE':>12}: accuracy={ensemble_acc:.4f}  macro_recall={ensemble_macro_recall:.4f}")

    print("\n" + "=" * 70)
    print("PER-CLASS REPORT -- ENSEMBLE (precision / recall / f1-score / support)")
    print("=" * 70)
    print(classification_report(test_labels_ref, test_preds, target_names=classes, digits=3))

    print("=" * 70)
    print("CONFUSION MATRIX -- ENSEMBLE (rows = true label, columns = predicted label)")
    print("=" * 70)
    cm = confusion_matrix(test_labels_ref, test_preds, labels=list(range(num_classes)))
    header = "        " + "".join(f"{c:>8}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        print(f"{classes[i]:>8}" + "".join(f"{v:>8}" for v in row))

    # --- ADDED: heatmap PNG ---
    save_confusion_heatmap(cm, classes, out_path="ensemble_confusion_heatmap.png")

    print("\n" + "=" * 70)
    print("RECALL ON CANCER-RELATED CLASSES (mel / bcc / akiec) -- ENSEMBLE")
    print("=" * 70)
    for i, cls in enumerate(classes):
        if cls in CANCER_RELATED_CLASSES:
            true_count = int((test_labels_ref == i).sum())
            correct_count = int(((test_labels_ref == i) & (test_preds == i)).sum())
            recall = correct_count / true_count if true_count > 0 else float("nan")
            print(f"  {cls:>6}: recall = {recall:.3f}  ({correct_count}/{true_count} caught)")

    # --- ADDED: save artifacts ---
    save_ensemble_artifacts(meta_clf, args, classes, ensemble_acc,
                             ensemble_macro_recall, out_dir=args.artifacts_dir)

    # --- ADDED: error analysis + Grad-CAM (opt-in via flags) ---
    if args.error_analysis:
        error_pairs = [("mel", "nv"), ("mel", "bkl"), ("akiec", "bkl")]
        save_misclassified_examples(test_rows, test_labels_ref, test_preds,
                                     class_to_idx, error_pairs,
                                     out_dir=args.error_analysis_dir)

        if args.gradcam_per_pair > 0:
            save_gradcam_examples(backbones, eval_transforms, test_rows,
                                   test_labels_ref, test_preds, class_to_idx,
                                   error_pairs, n_per_pair=args.gradcam_per_pair,
                                   out_dir=args.gradcam_dir)


if __name__ == "__main__":
    main()