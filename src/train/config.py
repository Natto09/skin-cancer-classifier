"""
TrainConfig -- one dataclass covering every training hyperparameter that
used to be scattered across argparse flags in 9 separate scripts
(train_resnet.py/_100K/_1M/_6M, train_densenet_1M.py, train_vit_1M.py,
train_gate_model.py, train_specialist_mel_bkl.py). Two "families" of
scripts existed with different feature sets (the resnet/densenet/vit
scripts had focal loss + AMP + plateau LR + multi-GPU autotuning; the
gate/specialist scripts had neither, but added a single-target-class extra
loss weight). This config is a superset -- every field from both families,
so either behavior is reachable by setting the right fields (see
configs/train/*.json for presets reproducing each original script's exact
defaults).

Adjust ONE thing at a time:

    from src.train.config import TrainConfig
    cfg = TrainConfig.from_json("configs/train/gate_1m.json")
    cfg.dropout_note = None          # (model shape lives in ModelConfig, not here)
    cfg.lr = 5e-5                    # tweak just the learning rate
    cfg.extra_class_weight = {"cancer": 4.0}   # tweak just the cancer weight
"""

from dataclasses import dataclass, field, asdict
import json


@dataclass
class TrainConfig:
    # --- data / split ---------------------------------------------------
    meta_csv: str = "all_augment_1M/lowmeta.csv"
    metadata_csv: str = "data/ham10000/HAM10000_metadata.csv"
    id_col: str = "image_id"
    label_col: str = "dx"
    val_fraction: float = 0.15
    test_fraction: float = 0.10
    seed: int = 42

    # split_mode: "full" (all 7 classes, e.g. the ensemble backbones),
    # "gate" (binary cancer/non_cancer), or "filtered" (subset of classes,
    # e.g. ["bkl", "mel"] for the specialist) -- see src/data/splits.py
    split_mode: str = "full"
    cancer_classes: list = field(default_factory=lambda: ["mel", "bcc", "akiec"])
    target_classes: list = field(default_factory=list)   # only used when split_mode == "filtered"

    # --- optimization ----------------------------------------------------
    batch_size: int = 128
    epochs: int = 50
    workers: int = 8
    lr: float = 5e-5
    weight_decay: float = 1e-2
    optimizer_type: str = "adam"          # "adam" | "adamw"
    patience: int = 8

    lr_schedule: str = "plateau"          # "plateau" | "steplr" | "none"
    lr_step_size: int = 5

    # "full" = RandomResizedCrop + rotation + color jitter (used by the
    # resnet/densenet/vit backbone trainers). "basic" = plain resize + flips
    # only (used by the gate/specialist trainers).
    augmentation_level: str = "full"

    # --- loss --------------------------------------------------------------
    loss_type: str = "focal"              # "focal" | "ce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    class_weight_power: float = 0.5       # exponent on inverse-frequency weights
    sampler_weight_power: float = 0.75    # separate exponent for the oversampling sampler
    class_weighted_loss: bool = True
    oversample_minority: bool = True
    # Extra multiplier applied to specific classes' loss+sampler weight ON
    # TOP OF the inverse-frequency weighting -- e.g. {"cancer": 3.0} for the
    # gate model, {"mel": 2.0} for the specialist. Empty = no extra boost.
    extra_class_weight: dict = field(default_factory=dict)

    # --- model selection / early stopping --------------------------------
    # Classes to average recall over when deciding "is this the best
    # checkpoint" and for early stopping -- e.g. ["mel","bcc","akiec"] for
    # the main ensemble backbones, ["cancer"] for the gate, ["mel"] for the
    # specialist. Falls back to macro recall (all classes) if empty.
    priority_classes: list = field(default_factory=lambda: ["mel", "bcc", "akiec"])

    # --- performance / hardware -------------------------------------------
    use_amp: bool = True
    auto_batch_size: bool = False
    probe_max_batch: int = 2048
    gpu_ids: list = None                  # None = use every visible GPU
    multi_gpu: bool = True                # wrap in DataParallel if len(gpu_ids) > 1

    # --- checkpointing -----------------------------------------------------
    best_model_path: str = "best_model.pth"
    checkpoint_path: str = "train_checkpoint.pth"
    resume: bool = False
    checkpoint_every_steps: int = 2000
    log_every: int = 50

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
