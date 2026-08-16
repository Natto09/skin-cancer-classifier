# skin-cancer-classifier

AI skin lesion screening: a 7-class ensemble (ResNet50 + DenseNet121 +
ViT-B/16) plus a binary cancer/non-cancer gate model and a mel-vs-bkl
specialist, combined into one pipeline.

This repo was reorganized from 20 standalone scripts (7,400+ lines, much of
it copy-pasted between files) into a `src/` package: reusable classes for
data loading, models, training, and evaluation, configured through JSON
presets instead of hardcoded values or long argparse flag lists.

## Layout

```
src/
  data/       AugmentedSkinDataset, train/val/test splitting, augmentation, prep
  models/     ModelConfig, backbone registry, SkinLesionClassifier,
              EnsembleClassifier, CombinedPipeline
  train/      TrainConfig, Trainer, CLI entrypoint
  evaluate/   confusion matrix, ensemble fit+eval, combined pipeline eval,
              single-image inference
  utils/      GPU/device helpers, training-log plotting
configs/
  model/      ModelConfig presets (backbone, head shape, freeze strategy)
  train/      TrainConfig presets (hyperparameters) -- see configs/README.md
legacy/       original scripts, preserved for reference (see legacy/README.md)
assets/       reference images, logs, Grad-CAM examples
```

## Quick start

```bash
pip install -r requirements.txt

# train a model from a preset
python -m src.train.cli --preset gate_1m

# same preset, one hyperparameter changed
python -m src.train.cli --preset gate_1m --set lr=2e-4

# same preset, different backbone
python -m src.train.cli --preset specialist_1m --model-set backbone=resnet50

# evaluate a checkpoint
python -m src.evaluate.confusion_matrix --model-preset resnet_1m \
    --checkpoint skin_cancer_best_resnet_1M.pth --meta_csv all_augment_1M/lowmeta.csv

# fit + evaluate the 3-backbone ensemble
python -m src.evaluate.ensemble_eval \
    --resnet_path skin_cancer_best_resnet_1M.pth \
    --densenet_path skin_cancer_best_densenet_1M.pth \
    --vit_path skin_cancer_best_vit_1M.pth

# evaluate the full ensemble+gate+specialist pipeline
python -m src.evaluate.combined_pipeline_eval \
    --gate_model_path skin_cancer_best_gate_1M.pth \
    --specialist_model_path skin_cancer_best_specialist_mel_bkl_1M.pth

# single image
python -m src.evaluate.predict_image --model-preset resnet_1m \
    --checkpoint skin_cancer_best_resnet_1M.pth --image assets/test_image.jpg
```

Your **already-trained checkpoints and `ensemble_config.json` load as-is** --
`EnsembleClassifier.load()` reads the existing flat
`resnet_path`/`densenet_path`/`vit_path` format, no retraining needed.

## Using the model classes directly

```python
from src.models.config import ModelConfig
from src.models.classifier import SkinLesionClassifier

# pick a backbone, adjust settings one at a time
cfg = ModelConfig(backbone="densenet121", num_classes=2, dropout=0.5)
clf = SkinLesionClassifier(cfg)
clf.load_weights("skin_cancer_best_gate_1M.pth")

predicted_class, confidences = clf.predict_image("some_image.jpg", transform)
```

## Where everything went

| old file | new location |
|---|---|
| `train_resnet.py` | `legacy/train_resnet_v1_base.py` (see `configs/README.md` for why) |
| `train_resnet_100K.py` | `src/train/trainer.py` + `configs/{model,train}/resnet_100k.json` |
| `train_resnet_1M.py` | `src/train/trainer.py` + `configs/{model,train}/resnet_1m.json` |
| `train_resnet_6M.py` | `src/train/trainer.py` + `configs/{model,train}/resnet_6m.json` |
| `train_densenet_1M.py` | `src/train/trainer.py` + `configs/{model,train}/densenet_1m.json` |
| `train_vit_1M.py` | `src/train/trainer.py` + `configs/{model,train}/vit_1m.json` |
| `train_gate_model.py` | `src/train/trainer.py` + `configs/{model,train}/gate_1m.json` |
| `train_specialist_mel_bkl.py` | `src/train/trainer.py` + `configs/{model,train}/specialist_1m.json` |
| `train_model.py` (custom CNN) | `legacy/train_model_custom_cnn.py` + `src/models/custom_cnn.py` (fixed) |
| `ensemble_meta_classifier.py` | `src/models/ensemble.py` + `src/evaluate/ensemble_eval.py` |
| `evaluate_confusion_matrix.py` | `src/evaluate/confusion_matrix.py` |
| `combined_pipeline_eval.py` | `src/models/pipeline.py` + `src/evaluate/combined_pipeline_eval.py` |
| `evaluate_model.py` | `legacy/evaluate_model_v1.py` (old/unused head shape -- see `legacy/README.md`) |
| `test_model.py` | `legacy/test_model_v1.py` + `src/evaluate/predict_image.py` (modern replacement) |
| `prepare_data.py` | `src/data/prepare.py` (`balance_test_set`) |
| `pump_data.py` | `src/data/prepare.py` (`split_and_balance_train`) |
| `pump_data_augment.py` | `src/data/augment.py` (`preset="full_6m"`) |
| `lowpump_data_augment.py` | `src/data/augment.py` (`preset="low_1m"`) |
| `lowestpump_data_augment.py` | `src/data/augment.py` (`preset="lowest_100k"`) |
| `plot_history.py` | `src/utils/plot_history.py` (also fixes a regex bug -- see file docstring) |

The full original scripts are still available, unmodified, in
`legacy/superseded_scripts/` and `legacy/` for reference -- see
`legacy/README.md`.

## What actually changed vs. what's just moved

Every original script's logic was ported faithfully -- same splits, same
loss functions, same class weighting, same freeze strategies. Two real
(intentional) changes:

1. **`src/models/custom_cnn.py`**: fixed a structural bug where
   `DermascanCNN`'s `Flatten()` lived inside the classifier head, which
   would have broken if the head were ever swapped out generically (as the
   new backbone registry does for every architecture). Doesn't affect the
   original script (never wired downstream anyway -- see `legacy/README.md`).
2. **`src/utils/plot_history.py`**: fixed a regex bug where the original
   looked for `"Accuracy:"` in log text, but every trainer actually prints
   `"Acc:"` -- so the original script's accuracy list was always empty.

Everything else, including the `resnet_6m` checkpoint-selection fidelity
caveat, is documented in `configs/README.md`.
