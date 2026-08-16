"""
CombinedPipeline -- wraps the three-stage routing logic (7-class ensemble ->
binary cancer/non-cancer gate -> mel-vs-bkl specialist) as a reusable class,
instead of the inline steps 3-7 inside combined_pipeline_eval.py's main().

Routing rule (SPECIALIST-TRIGGER FIX -- see original combined_pipeline_eval.py
docstring for the full history of why): the specialist only overrides the
ensemble's prediction when BOTH of these hold for a given image:
  1. the ensemble's OWN prediction is already mel or bkl, AND
  2. the average backbone probability's top-2 classes are exactly {mel, bkl}
An earlier version used condition 2 alone, which forced mel/bkl onto rows
whose true (and often predicted) class was something else entirely --
guaranteed-wrong forced calls that tanked mel/bkl precision. Condition 1
fixes that; condition 2 is kept as a secondary signal (the specialist only
helps decide *which* of the two, not whether it's one of the two at all).

The gate model is ADVISORY ONLY here: when it strongly suspects cancer but
the pipeline's final prediction is non-cancer, that row is flagged
"needs_review" -- but the final prediction is never changed by the gate.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PipelineResult:
    final_preds: np.ndarray
    true_labels: np.ndarray
    specialist_used: np.ndarray       # bool mask
    needs_review: np.ndarray          # bool mask (gate flagged cancer-suspected but non-cancer predicted)
    ensemble_preds: np.ndarray
    gate_cancer_prob: np.ndarray


class CombinedPipeline:
    def __init__(self, ensemble, gate, specialist,
                 cancer_classes=("mel", "bcc", "akiec"),
                 specialist_classes=("bkl", "mel"),
                 gate_class_to_idx=None,
                 gate_cancer_threshold=0.5):
        """
        ensemble:   a fitted src.models.ensemble.EnsembleClassifier
        gate:       a src.models.classifier.SkinLesionClassifier (binary
                    cancer/non_cancer)
        specialist: a src.models.classifier.SkinLesionClassifier (binary,
                    over `specialist_classes`)
        gate_class_to_idx: mapping used by the gate model, e.g.
                    {"cancer": 0, "non_cancer": 1}. Defaults to that if not given.
        """
        self.ensemble = ensemble
        self.gate = gate
        self.specialist = specialist
        self.cancer_classes = set(cancer_classes)
        self.specialist_classes = list(specialist_classes)
        self.gate_class_to_idx = gate_class_to_idx or {"cancer": 0, "non_cancer": 1}
        self.gate_cancer_threshold = gate_cancer_threshold

    def predict(self, loader, no_specialist=False) -> PipelineResult:
        classes = self.ensemble.class_names
        class_to_idx = {c: i for i, c in enumerate(classes)}
        mel_idx, bkl_idx = class_to_idx["mel"], class_to_idx["bkl"]

        ensemble_preds, y_true, backbone_blocks = self.ensemble.predict(loader)
        avg_probs = np.mean(backbone_blocks, axis=0)
        top2_idx = np.argsort(-avg_probs, axis=1)[:, :2]

        gate_probs, _ = self.gate.predict_proba(loader)
        gate_cancer_prob = gate_probs[:, self.gate_class_to_idx["cancer"]]

        specialist_probs, _ = self.specialist.predict_proba(loader)
        spec_bkl_col = self.specialist_classes.index("bkl")
        spec_mel_col = self.specialist_classes.index("mel")

        n = len(ensemble_preds)
        final_preds = ensemble_preds.copy()
        specialist_used = np.zeros(n, dtype=bool)
        needs_review = np.zeros(n, dtype=bool)

        for i in range(n):
            top2_set = {top2_idx[i, 0], top2_idx[i, 1]}
            avg_probs_say_mel_bkl = (top2_set == {mel_idx, bkl_idx})

            ensemble_says_mel_bkl = ensemble_preds[i] in (mel_idx, bkl_idx)
            if (not no_specialist) and ensemble_says_mel_bkl and avg_probs_say_mel_bkl:
                specialist_used[i] = True
                final_preds[i] = (mel_idx if specialist_probs[i, spec_mel_col] > specialist_probs[i, spec_bkl_col]
                                   else bkl_idx)

            pred_is_cancer = classes[final_preds[i]] in self.cancer_classes
            if gate_cancer_prob[i] >= self.gate_cancer_threshold and not pred_is_cancer:
                needs_review[i] = True  # flag only -- never overrides final_preds

        return PipelineResult(
            final_preds=final_preds, true_labels=y_true,
            specialist_used=specialist_used, needs_review=needs_review,
            ensemble_preds=ensemble_preds, gate_cancer_prob=gate_cancer_prob,
        )
