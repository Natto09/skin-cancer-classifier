"""
Evaluates the FULL combined pipeline (7-class ensemble -> binary
cancer/non-cancer gate -> mel-vs-bkl specialist) on held-out data. Replaces
combined_pipeline_eval.py.

*** METHODOLOGY NOTE -- READ BEFORE TRUSTING THE NUMBERS ***
The gate model and the specialist model were each trained using their OWN
independent split call (different stratify labels: 7-class for the main
ensemble, binary cancer/non-cancer for the gate, mel-vs-bkl only for the
specialist). Even with the same seed=42, scikit-learn's stratified split
does not guarantee identical train/val/test membership across these three
different label schemes -- so an image in the main ensemble's TEST set
could have been in the gate's or specialist's TRAIN set, which would leak
information into the combined-pipeline evaluation.

This script computes the INTERSECTION of the relevant test sets and
evaluates only on that leakage-free subset. Only the SPECIALIST actually
overrides final predictions (for mel/bkl rows) -- the gate model is
advisory-only (it only sets a "needs review" flag) -- so only ensemble+
specialist test-set membership needs to be intersected; gate-leakage does
not corrupt the accuracy/recall/precision numbers.

Usage:
    python -m src.evaluate.combined_pipeline_eval \\
        --ensemble_config ensemble_config.json \\
        --gate_model_path skin_cancer_best_gate_1M.pth --gate_arch densenet121 \\
        --specialist_model_path skin_cancer_best_specialist_mel_bkl_1M.pth --specialist_arch densenet121
"""

import argparse
import os

import numpy as np
from sklearn.metrics import classification_report

from ..data.dataset import AugmentedSkinDataset
from ..data.splits import load_base_df, get_test_images
from ..models.classifier import SkinLesionClassifier
from ..models.config import ModelConfig
from ..models.ensemble import EnsembleClassifier
from ..models.pipeline import CombinedPipeline

CANCER_CLASSES = {"mel", "bcc", "akiec"}


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
    parser.add_argument("--gate_cancer_threshold", type=float, default=0.5)
    parser.add_argument("--no_specialist", action="store_true",
                         help="Skip the specialist override entirely, for a before/after A/B comparison "
                              "on the identical clean subset.")
    args = parser.parse_args()

    # --- Step 1: leakage-free intersection of test images ------------------
    print("Loading metadata and recomputing each model's own test-image set ...")
    df = load_base_df(args.meta_csv, args.metadata_csv, args.id_col, args.label_col)

    main_test_images = get_test_images(args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
                                        mode="full", metadata_csv=args.metadata_csv,
                                        id_col=args.id_col, label_col=args.label_col)
    specialist_test_images = get_test_images(args.meta_csv, args.val_fraction, args.test_fraction, args.seed,
                                               mode="filtered", target_classes=["mel", "bkl"],
                                               metadata_csv=args.metadata_csv,
                                               id_col=args.id_col, label_col=args.label_col)
    print(f"Main ensemble test images:  {len(main_test_images):,}")
    print(f"Specialist test images:     {len(specialist_test_images):,}")

    safe_for_mel_bkl = main_test_images & specialist_test_images
    print(f"Leakage-free for ensemble+specialist (mel/bkl only): {len(safe_for_mel_bkl):,}")
    print("(Gate model is advisory-only -- it never overrides final_preds -- so its own "
          "test-set membership does not need to be part of the leakage-free intersection.)")

    # --- Step 2: build the CLEAN evaluation row list ------------------------
    eval_df = df[df["original_image"].isin(main_test_images)].copy()
    keep_mask = (~eval_df["label"].isin(["mel", "bkl"])) | eval_df["original_image"].isin(safe_for_mel_bkl)
    clean_df = eval_df[keep_mask]
    print(f"\nMain test set: {len(eval_df):,} rows -> clean (leakage-free) subset: "
          f"{len(clean_df):,} rows ({len(eval_df) - len(clean_df):,} dropped due to split mismatch)")

    import json
    with open(args.ensemble_config) as f:
        config = json.load(f)
    classes = config["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    clean_rows = list(zip(clean_df["filename"].tolist(), clean_df["label"].map(class_to_idx).tolist()))

    from torch.utils.data import DataLoader
    from torchvision import transforms
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(AugmentedSkinDataset(clean_rows, eval_tf), batch_size=args.batch_size,
                         shuffle=False, num_workers=args.workers)

    # --- Steps 3-5: load ensemble + gate + specialist ------------------------
    model_configs = {
        "resnet50": ModelConfig(backbone="resnet50", num_classes=num_classes, pretrained=False),
        "densenet121": ModelConfig(backbone="densenet121", num_classes=num_classes, pretrained=False),
        "vit_b16": ModelConfig(backbone="vit_b16", num_classes=num_classes, pretrained=False),
    }
    ensemble = EnsembleClassifier.load(args.ensemble_config, model_configs)

    gate_cfg = ModelConfig(backbone=args.gate_arch, num_classes=2, pretrained=False)
    gate = SkinLesionClassifier.from_checkpoint(args.gate_model_path, gate_cfg)

    specialist_cfg = ModelConfig(backbone=args.specialist_arch, num_classes=2, pretrained=False)
    specialist = SkinLesionClassifier.from_checkpoint(args.specialist_model_path, specialist_cfg)

    pipeline = CombinedPipeline(
        ensemble, gate, specialist,
        cancer_classes=CANCER_CLASSES, specialist_classes=("bkl", "mel"),
        gate_class_to_idx={"cancer": 0, "non_cancer": 1},
        gate_cancer_threshold=args.gate_cancer_threshold,
    )
    result = pipeline.predict(loader, no_specialist=args.no_specialist)

    # --- Step 7: report -------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"COMBINED PIPELINE -- CLEAN (leakage-free) EVALUATION, n={len(clean_rows)}")
    print("=" * 70)
    acc = (result.final_preds == result.true_labels).mean()
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Specialist invoked on {result.specialist_used.sum()} / {len(clean_rows)} rows "
          f"({100*result.specialist_used.mean():.1f}%)")
    print(f"Gate flagged 'needs review' on {result.needs_review.sum()} rows")

    print("\nPER-CLASS REPORT:")
    print(classification_report(result.true_labels, result.final_preds, target_names=classes, digits=3))

    print("RECALL ON CANCER-RELATED CLASSES (mel/bcc/akiec) -- COMBINED PIPELINE:")
    for cls in ["akiec", "bcc", "mel"]:
        idx = class_to_idx[cls]
        true_count = int((result.true_labels == idx).sum())
        correct_count = int(((result.true_labels == idx) & (result.final_preds == idx)).sum())
        recall = correct_count / true_count if true_count else float("nan")
        print(f"  {cls:>6}: recall = {recall:.3f}  ({correct_count}/{true_count} caught)")


if __name__ == "__main__":
    main()
