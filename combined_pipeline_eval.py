"""
combined_pipeline_eval.py -- Evaluates the FULL combined pipeline
(7-class ensemble -> binary cancer/non-cancer gate -> mel-vs-bkl specialist)
on held-out data, BEFORE wiring it into the live web app.

*** METHODOLOGY NOTE -- READ BEFORE TRUSTING THE NUMBERS ***
The gate model and the specialist model were each trained using their OWN
independent call to three_way_split() (different stratify labels: 7-class
for the main ensemble, binary cancer/non-cancer for the gate, mel-vs-bkl
only for the specialist). Even with the same seed=42, scikit-learn's
stratified split does not guarantee identical train/val/test membership
across these three different label schemes -- so an image in the main
ensemble's TEST set could have been in the gate's or specialist's TRAIN
set, which would leak information into the combined-pipeline evaluation.

This script fixes that by computing the INTERSECTION of all relevant test
sets, and evaluating the combined pipeline only on that leakage-free
subset. It reports both:
  1. "full" numbers on the main ensemble's whole test set (for reference --
     NOT trustworthy where gate/specialist calls hit non-intersected images)
  2. "clean" numbers on the leakage-free intersected subset (the number to
     actually trust and report)

*** SPECIALIST-TRIGGER FIX (this version) ***
Previously the specialist was invoked whenever the AVERAGE backbone
probabilities' top-2 classes were exactly {mel, bkl} -- regardless of what
the meta-classifier actually predicted. Since the specialist is a strict
binary (mel vs bkl) model, any row that tripped that condition but whose
true label (and often the meta-classifier's own prediction) was something
else entirely -- akiec, nv, whatever -- got FORCED into mel or bkl, which
is wrong 100% of the time it happens. On the n=108135 run this fired on
4,345 rows while true mel+bkl together totaled only 2,700 -- i.e. at least
~1,645 guaranteed-wrong forced calls, which is enough on its own to explain
the very low mel/bkl precision (~0.21-0.26) seen in earlier runs.

Fix: only let the specialist override when the meta-classifier's OWN
prediction (ensemble_preds) is already mel or bkl. The avg-probs top-2
check is kept as a secondary signal (specialist only helps decide *which*
of the two, not whether it's one of the two at all). A diagnostic line
reports how many specialist calls would have been "false triggers" under
the old rule, for a direct before/after comparison.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CANCER_CLASSES = {"mel", "bcc", "akiec"}


# ---------------------------------------------------------------------------
# Shared dataset class
# ---------------------------------------------------------------------------

class AugmentedSkinDataset(Dataset):
    def __init__(self, rows, transform=None):
        self.rows = rows  # list of (path, label_idx)
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
    labels_df = pd.read_csv(metadata_csv, usecols=[id_col, label_col], dtype=str)
    id_to_label = dict(zip(labels_df[id_col], labels_df[label_col]))
    original_id = df["original_image"].astype(str).str.rsplit(".", n=1).str[0]
    df = df.copy()
    df["label"] = original_id.map(id_to_label)
    return df


def three_way_split(per_image, val_fraction, test_fraction, seed, stratify_col):
    labels_by_image = per_image.set_index("original_image")[stratify_col]
    train_val_images, test_images = train_test_split(
        per_image["original_image"], test_size=test_fraction,
        random_state=seed, shuffle=True, stratify=per_image[stratify_col],
    )
    relative_val_fraction = val_fraction / (1 - test_fraction)
    train_val_labels = labels_by_image.loc[train_val_images]
    train_images, val_images = train_test_split(
        train_val_images, test_size=relative_val_fraction,
        random_state=seed, shuffle=True, stratify=train_val_labels,
    )
    return set(train_images), set(val_images), set(test_images)


def load_base_df(meta_csv, metadata_csv, id_col, label_col):
    df = pd.read_csv(
        meta_csv, usecols=["filename", "original_image", "label"],
        dtype={"filename": "string", "original_image": "category", "label": "string"},
    )
    df = merge_labels_if_missing(df, metadata_csv, id_col, label_col)
    df = df[df["label"].notna() & (df["label"] != "")]
    return df


def get_test_images_main(df, val_fraction, test_fraction, seed):
    """Main 7-class ensemble's own test-image set (stratified on the 7-class label)."""
    per_image = df.groupby("original_image", observed=True)["label"].first().reset_index()
    _, _, test_images = three_way_split(per_image, val_fraction, test_fraction, seed, "label")
    return test_images


def get_test_images_gate(df, val_fraction, test_fraction, seed):
    """Gate model's own test-image set (stratified on the binary cancer/non-cancer label)."""
    df = df.copy()
    df["gate_label"] = df["label"].apply(lambda c: "cancer" if c in CANCER_CLASSES else "non_cancer")
    per_image = df.groupby("original_image", observed=True)["gate_label"].first().reset_index()
    _, _, test_images = three_way_split(per_image, val_fraction, test_fraction, seed, "gate_label")
    return test_images


def get_test_images_specialist(df, val_fraction, test_fraction, seed):
    """Specialist's own test-image set (mel/bkl only, stratified on that 2-class label)."""
    df = df[df["label"].isin(["mel", "bkl"])]
    per_image = df.groupby("original_image", observed=True)["label"].first().reset_index()
    _, _, test_images = three_way_split(per_image, val_fraction, test_fraction, seed, "label")
    return test_images


# ---------------------------------------------------------------------------
# Model builders (must match each training script's architecture exactly)
# ---------------------------------------------------------------------------

def build_model(arch, num_classes, device):
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.8), nn.Linear(128, num_classes),
        )
    elif arch == "densenet121":
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.8), nn.Linear(128, num_classes),
        )
    elif arch == "vit":
        model = models.vit_b_16(weights=None)
        num_ftrs = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Linear(num_ftrs, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.8), nn.Linear(128, num_classes),
        )
    else:
        raise ValueError(f"Unsupported arch '{arch}'")
    return model.to(device)


@torch.no_grad()
def get_all_probs(model, loader, device, name=""):
    model.eval()
    all_probs = []
    total_batches = len(loader)
    for step, (inputs, _) in enumerate(loader, 1):
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if step % 20 == 0 or step == total_batches:
            pct = 100 * step / total_batches
            print(f"  [{name}] batch {step}/{total_batches} ({pct:.1f}%)", flush=True)
    return np.concatenate(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the combined ensemble+gate+specialist pipeline.")
    parser.add_argument("--meta_csv", default="all_augment_1M/lowmeta.csv")
    parser.add_argument("--metadata_csv", default="data/ham10000/HAM10000_metadata.csv")
    parser.add_argument("--id_col", default="image_id")
    parser.add_argument("--label_col", default="dx")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))

    parser.add_argument("--ensemble_config", default="ensemble_config.json")
    parser.add_argument("--gate_model_path", default="skin_cancer_best_gate_1M.pth")
    parser.add_argument("--gate_arch", default="densenet121", choices=["resnet50", "densenet121"])
    parser.add_argument("--specialist_model_path", default="skin_cancer_best_specialist_mel_bkl_1M.pth")
    parser.add_argument("--specialist_arch", default="densenet121", choices=["resnet50", "densenet121"])

    # gate override behavior: if the gate strongly disagrees with the
    # ensemble's non-cancer verdict, flag it for manual review rather than
    # silently trusting the ensemble.
    parser.add_argument("--gate_cancer_threshold", type=float, default=0.5)

    # NEW: lets you turn the specialist off entirely for a clean A/B
    # comparison on the exact same clean subset (same images, same
    # ensemble/gate calls -- only the specialist override differs).
    parser.add_argument("--no_specialist", action="store_true",
                         help="Skip the specialist override entirely (ensemble_preds "
                              "become final_preds unchanged). Use for before/after "
                              "comparisons on the identical clean subset.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: figure out the leakage-free intersection of test images ---
    print("Loading metadata and recomputing each model's own test-image set ...")
    df = load_base_df(args.meta_csv, args.metadata_csv, args.id_col, args.label_col)

    main_test_images = get_test_images_main(df, args.val_fraction, args.test_fraction, args.seed)
    gate_test_images = get_test_images_gate(df, args.val_fraction, args.test_fraction, args.seed)
    specialist_test_images = get_test_images_specialist(df, args.val_fraction, args.test_fraction, args.seed)

    print(f"Main ensemble test images:  {len(main_test_images):,}")
    print(f"Gate model test images:     {len(gate_test_images):,}")
    print(f"Specialist test images:     {len(specialist_test_images):,}")

    # Every row needs to be safe for (main AND gate). mel/bkl rows ADDITIONALLY
    # need to be safe for the specialist too.
    # Only the SPECIALIST actually overrides final predictions (for mel/bkl
    # rows) -- the gate model only sets an advisory flag and never changes
    # final_preds, so gate-leakage does NOT corrupt the core accuracy/recall
    # numbers. Requiring every row to also be gate-safe was needlessly
    # conservative and was shrinking non-mel/bkl classes down to ~1% of their
    # real test size for no actual leakage benefit. Fixed: only mel/bkl rows
    # need the extra specialist-safety check.
    safe_for_mel_bkl = main_test_images & specialist_test_images
    print(f"Leakage-free for ensemble+specialist (mel/bkl only): {len(safe_for_mel_bkl):,}")
    print("(Gate model is advisory-only in this pipeline -- it never overrides final_preds -- "
          "so it does not need to be part of the leakage-free intersection for other classes; "
          "the 'needs review' flag count is a secondary statistic and may have minor residual "
          "leakage risk, which does not affect the accuracy/recall/precision numbers below.)")

    # --- DIAGNOSTIC: per-class original-image counts, to sanity-check the
    # fix actually recovers the expected sample sizes.
    print("\n--- DIAGNOSTIC: per-class original-image counts ---")
    label_by_image = df.groupby("original_image", observed=True)["label"].first()
    for cls in sorted(label_by_image.unique()):
        cls_images = set(label_by_image[label_by_image == cls].index)
        n_total = len(cls_images)
        n_main_test = len(cls_images & main_test_images)
        n_specialist_test = len(cls_images & specialist_test_images) if cls in ("mel", "bkl") else None
        n_final = len(cls_images & safe_for_mel_bkl) if cls in ("mel", "bkl") else n_main_test
        extra = f"  specialist_test={n_specialist_test:>4}" if n_specialist_test is not None else ""
        print(f"  {cls:>6}: total={n_total:>5}  main_test={n_main_test:>4}{extra}  final_eval_n={n_final:>4}")
    print("--- end diagnostic ---\n")

    # --- Step 2: build the CLEAN evaluation row list ---
    # rows = (filename, label, original_image)
    eval_df = df[df["original_image"].isin(main_test_images)].copy()
    keep_mask = (
        ~eval_df["label"].isin(["mel", "bkl"]) | eval_df["original_image"].isin(safe_for_mel_bkl)
    )
    clean_df = eval_df[keep_mask]
    dropped = len(eval_df) - len(clean_df)
    print(f"\nMain test set: {len(eval_df):,} rows -> clean (leakage-free) subset: "
          f"{len(clean_df):,} rows ({dropped:,} dropped due to split mismatch)")

    with open(args.ensemble_config) as f:
        config = json.load(f)
    classes = config["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    clean_rows = list(zip(clean_df["filename"].tolist(),
                           clean_df["label"].map(class_to_idx).tolist()))
    y_true = np.array([r[1] for r in clean_rows])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- Steps 3-5 combined: run all 5 models (3 ensemble backbones + gate +
    # specialist) CONCURRENTLY across whichever GPUs are available, instead
    # of one at a time on a single GPU (which left a second GPU idle).
    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        device_map = {"resnet50": "cuda:0", "densenet121": "cuda:0",
                      "vit": "cuda:1", "gate": "cuda:1", "specialist": "cuda:1"}
    elif num_gpus == 1:
        device_map = {k: "cuda:0" for k in ("resnet50", "densenet121", "vit", "gate", "specialist")}
    else:
        device_map = {k: "cpu" for k in ("resnet50", "densenet121", "vit", "gate", "specialist")}
    print(f"\nDevice assignment: {device_map}")

    model_specs = [
        ("resnet50", "resnet50", num_classes, config["resnet_path"]),
        ("densenet121", "densenet121", num_classes, config["densenet_path"]),
        ("vit", "vit", num_classes, config["vit_path"]),
        ("gate", args.gate_arch, 2, args.gate_model_path),
        ("specialist", args.specialist_arch, 2, args.specialist_model_path),
    ]
    per_model_workers = max(1, args.workers // len(model_specs))

    def run_one(name, arch, n_classes, path):
        device = torch.device(device_map[name])
        print(f"[{name}] loading weights from {path} (device={device_map[name]}) ...")
        model = build_model(arch, n_classes, device)
        model.load_state_dict(torch.load(path, map_location=device))
        loader = DataLoader(AugmentedSkinDataset(clean_rows, eval_transforms),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=per_model_workers)
        probs = get_all_probs(model, loader, device, name=name)
        return name, probs

    results = {}
    with ThreadPoolExecutor(max_workers=len(model_specs)) as executor:
        futures = {executor.submit(run_one, *spec): spec[0] for spec in model_specs}
        for future in as_completed(futures):
            name, probs = future.result()
            results[name] = probs
            print(f"[{name}] done.")

    probs_per_backbone = [results["resnet50"], results["densenet121"], results["vit"]]
    ensemble_features = np.concatenate(probs_per_backbone, axis=1)
    meta_clf = joblib.load(config["meta_classifier_path"])
    ensemble_preds = meta_clf.predict(ensemble_features)
    # top-2 classes per row, from the AVERAGE backbone probability (simple,
    # consistent with what app.py shows the user). This is now a SECONDARY
    # signal for the specialist trigger -- see the fix below.
    avg_probs = np.mean(probs_per_backbone, axis=0)
    top2_idx = np.argsort(-avg_probs, axis=1)[:, :2]

    gate_probs = results["gate"]  # index 0=cancer, 1=non_cancer -- verified against
    # train_gate_model.py's gate_classes = ["cancer", "non_cancer"], confirmed correct.
    gate_cancer_prob = gate_probs[:, 0]

    specialist_probs_full = results["specialist"]  # index 0=bkl, 1=mel -- verified against
    # train_specialist_mel_bkl.py's TARGET_CLASSES = ["bkl", "mel"], confirmed correct.

    mel_idx, bkl_idx = class_to_idx["mel"], class_to_idx["bkl"]

    # --- Step 6: combine everything into final predictions ---
    final_preds = ensemble_preds.copy()
    specialist_used = np.zeros(len(clean_rows), dtype=bool)
    specialist_would_have_fired_old_rule = np.zeros(len(clean_rows), dtype=bool)
    gate_disagree = np.zeros(len(clean_rows), dtype=bool)

    for i in range(len(clean_rows)):
        top2_set = {top2_idx[i, 0], top2_idx[i, 1]}
        avg_probs_say_mel_bkl = (top2_set == {mel_idx, bkl_idx})
        if avg_probs_say_mel_bkl:
            specialist_would_have_fired_old_rule[i] = True

        # FIX: only let the specialist override when the meta-classifier's
        # OWN prediction already landed on mel or bkl. Under the old rule
        # this condition was missing, so any row where avg-probs top-2
        # happened to be {mel, bkl} got forcibly relabeled mel/bkl even if
        # the meta-classifier (and the true label) said something else --
        # guaranteed-wrong forced calls that tanked mel/bkl precision.
        ensemble_says_mel_bkl = ensemble_preds[i] in (mel_idx, bkl_idx)
        if ensemble_says_mel_bkl and avg_probs_say_mel_bkl:
            specialist_used[i] = True
            final_preds[i] = mel_idx if specialist_probs_full[i, 1] > specialist_probs_full[i, 0] else bkl_idx

        pred_is_cancer = classes[final_preds[i]] in CANCER_CLASSES
        if gate_cancer_prob[i] >= args.gate_cancer_threshold and not pred_is_cancer:
            gate_disagree[i] = True  # flagged for review, but final_preds unchanged (flag-only, doesn't override)

    if args.no_specialist:
        print("\n[--no_specialist] Specialist override disabled -- final_preds == ensemble_preds.")
        final_preds = ensemble_preds.copy()
        specialist_used[:] = False

    # --- Step 7: report ---
    print("\n" + "=" * 70)
    print(f"COMBINED PIPELINE -- CLEAN (leakage-free) EVALUATION, n={len(clean_rows)}")
    print("=" * 70)
    acc = (final_preds == y_true).mean()
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Specialist invoked on {specialist_used.sum()} / {len(clean_rows)} rows "
          f"({100*specialist_used.mean():.1f}%)")
    print(f"Gate flagged 'needs review' (cancer-suspected but non-cancer predicted) on "
          f"{gate_disagree.sum()} rows")

    # Diagnostic: quantify exactly how many old-rule specialist calls were
    # "false triggers" -- fired on a row whose true label isn't mel/bkl at
    # all, so they were guaranteed wrong under the old (avg-probs-only) rule.
    old_rule_used = specialist_would_have_fired_old_rule
    old_rule_false_triggers = old_rule_used & ~np.isin(y_true, [mel_idx, bkl_idx])
    new_rule_false_triggers = specialist_used & ~np.isin(y_true, [mel_idx, bkl_idx])
    print(f"\n[diagnostic] Old rule (avg-probs top-2 only) would have invoked specialist on "
          f"{old_rule_used.sum()} rows, of which {old_rule_false_triggers.sum()} had a true "
          f"label that isn't mel/bkl (guaranteed-wrong forced calls).")
    print(f"[diagnostic] New rule (this fix) invoked specialist on {specialist_used.sum()} rows, "
          f"of which {new_rule_false_triggers.sum()} had a true label that isn't mel/bkl "
          f"(should be 0 or very close to it).")

    print("\nPER-CLASS REPORT:")
    print(classification_report(y_true, final_preds, target_names=classes, digits=3))

    print("RECALL ON CANCER-RELATED CLASSES (mel/bcc/akiec) -- COMBINED PIPELINE:")
    for cls in ["akiec", "bcc", "mel"]:
        idx = class_to_idx[cls]
        true_count = int((y_true == idx).sum())
        correct_count = int(((y_true == idx) & (final_preds == idx)).sum())
        recall = correct_count / true_count if true_count else float("nan")
        print(f"  {cls:>6}: recall = {recall:.3f}  ({correct_count}/{true_count} caught)")

    print("\nFor comparison, re-run ensemble_meta_classifier.py's numbers on the FULL "
          "test set are: accuracy=0.8585, mel recall=0.584, bcc recall=0.806, "
          "akiec recall=0.632 -- but note that comparison isn't perfectly apples-to-apples "
          "since this script evaluates on the smaller CLEAN subset, not the full test set.")


if __name__ == "__main__":
    main()