"""
EnsembleClassifier -- DAME-style Level-0/Level-1 stacking ensemble.

Level 0: each backbone (ResNet50, DenseNet121, ViT-B/16 by default, but any
         set of SkinLesionClassifiers works) produces a softmax vector per
         image.
Level 1: the concatenated softmax vectors are fed to a Logistic Regression
         meta-classifier, which produces the final prediction.

METHODOLOGY: fit the meta-classifier on VAL-split predictions, never TRAIN-
split -- the backbones are close to memorized on their own training data,
so train-split softmax outputs are overconfident and would teach the
meta-classifier a distorted picture. This class enforces that by naming the
fit method fit_meta_classifier(val_loader, ...) -- pass it the val loader.
"""

import json
import os

import numpy as np

from .classifier import SkinLesionClassifier
from .config import ModelConfig

DEFAULT_ORDER = ["resnet50", "densenet121", "vit_b16"]


class EnsembleClassifier:
    def __init__(self, backbones: dict, order=None):
        """backbones: {name: SkinLesionClassifier}. order fixes the column
        order features get concatenated in (must be consistent between
        fitting and inference, and between separately-run backbones)."""
        self.backbones = backbones
        self.order = order or [n for n in DEFAULT_ORDER if n in backbones] or list(backbones)
        self.meta_clf = None
        self.class_names = None

    @classmethod
    def from_configs(cls, configs: dict, weight_paths: dict, device=None, order=None):
        """configs: {name: ModelConfig}, weight_paths: {name: checkpoint_path}."""
        backbones = {}
        for name, cfg in configs.items():
            backbones[name] = SkinLesionClassifier.from_checkpoint(
                weight_paths[name], cfg, device=device
            )
        return cls(backbones, order=order)

    def _stacked_features(self, loader):
        blocks, labels_ref = [], None
        for name in self.order:
            probs, labels = self.backbones[name].predict_proba(loader)
            if labels_ref is None:
                labels_ref = labels
            elif not np.array_equal(labels, labels_ref):
                raise RuntimeError(
                    f"[{name}] label order mismatch -- dataloader ordering is "
                    f"inconsistent between backbones (make sure shuffle=False)."
                )
            blocks.append(probs)
        return np.concatenate(blocks, axis=1), labels_ref, blocks

    def fit_meta_classifier(self, val_loader, class_names, max_iter=1000):
        from sklearn.linear_model import LogisticRegression
        features, labels, _ = self._stacked_features(val_loader)
        self.meta_clf = LogisticRegression(max_iter=max_iter)
        self.meta_clf.fit(features, labels)
        self.class_names = class_names
        return self

    def predict(self, loader):
        """Returns (final_preds, true_labels, per_backbone_prob_blocks).
        per_backbone_prob_blocks is a list (same order as self.order) of
        (N, num_classes) arrays -- handy for e.g. the specialist-trigger
        top-2-average-probability check in src.models.pipeline."""
        if self.meta_clf is None:
            raise RuntimeError("Call fit_meta_classifier(val_loader, ...) first, "
                                "or load() a previously-fitted ensemble.")
        features, labels, blocks = self._stacked_features(loader)
        preds = self.meta_clf.predict(features)
        return preds, labels, blocks

    # -- persistence ------------------------------------------------------

    # Maps this project's original flat ensemble_config.json keys
    # (resnet_path/densenet_path/vit_path) to the backbone names used here,
    # so the ALREADY-TRAINED ensemble_config.json checked into this repo
    # (test_accuracy 0.858) loads without retraining anything.
    _LEGACY_KEY_TO_NAME = {"resnet_path": "resnet50", "densenet_path": "densenet121", "vit_path": "vit_b16"}

    def save(self, out_dir, weight_paths: dict, test_accuracy=None, test_macro_recall=None):
        """Saves the fitted meta-classifier (.joblib) + a config JSON. Writes
        BOTH the new structured "weight_paths" dict AND the original flat
        resnet_path/densenet_path/vit_path keys, so this file stays a
        drop-in replacement for the original ensemble_config.json."""
        import joblib
        os.makedirs(out_dir, exist_ok=True)
        meta_path = os.path.join(out_dir, "ensemble_meta_classifier.joblib")
        joblib.dump(self.meta_clf, meta_path)

        config = {
            "backbone_order": self.order,
            "weight_paths": weight_paths,
            "meta_classifier_path": meta_path,
            "classes": self.class_names,
            "test_accuracy": float(test_accuracy) if test_accuracy is not None else None,
            "test_macro_recall": float(test_macro_recall) if test_macro_recall is not None else None,
        }
        for legacy_key, name in self._LEGACY_KEY_TO_NAME.items():
            if name in weight_paths:
                config[legacy_key] = weight_paths[name]
        config_path = os.path.join(out_dir, "ensemble_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[artifacts] saved meta-classifier to {meta_path}")
        print(f"[artifacts] saved config to {config_path}")
        return config_path

    @classmethod
    def load(cls, config_path, model_configs: dict, device=None):
        """model_configs: {name: ModelConfig} -- the backbone architectures
        must be supplied (they aren't stored in the JSON), matching what
        each weight file was trained with. Reads either this project's new
        "weight_paths" format or the original flat
        resnet_path/densenet_path/vit_path format (e.g. the
        ensemble_config.json already checked into this repo)."""
        import joblib
        with open(config_path) as f:
            config = json.load(f)

        if "weight_paths" in config:
            weight_paths = config["weight_paths"]
            order = config.get("backbone_order", list(weight_paths))
        else:
            weight_paths = {name: config[key] for key, name in cls._LEGACY_KEY_TO_NAME.items() if key in config}
            order = [n for n in DEFAULT_ORDER if n in weight_paths]

        backbones = {
            name: SkinLesionClassifier.from_checkpoint(path, model_configs[name], device=device)
            for name, path in weight_paths.items()
        }
        ens = cls(backbones, order=order)
        ens.meta_clf = joblib.load(config["meta_classifier_path"])
        ens.class_names = config["classes"]
        return ens
