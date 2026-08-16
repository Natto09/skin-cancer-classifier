"""
Train any model in this project from a named preset, overriding individual
settings from the command line.

Examples:
    # exactly reproduces the old `python3 train_resnet_1M.py` defaults
    python -m src.train.cli --preset resnet_1m

    # same, but with a different learning rate and no focal loss --
    # everything else stays at the preset's defaults
    python -m src.train.cli --preset resnet_1m --set lr=1e-4 --set loss_type=ce

    # gate model, bumping the cancer-class weight from 3.0 to 4.0
    python -m src.train.cli --preset gate_1m --set extra_class_weight='{"cancer": 4.0}'

    # swap the gate model's backbone from densenet121 to resnet50
    python -m src.train.cli --preset gate_1m --model-set backbone=resnet50

Presets live in configs/train/*.json (TrainConfig fields) and
configs/model/*.json (ModelConfig fields) -- see configs/README.md for the
full list and what each one reproduces.
"""

import argparse
import ast
import os

from ..models.config import ModelConfig
from .config import TrainConfig
from .trainer import Trainer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "train")
MODEL_CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "model")


def _parse_value(raw):
    """--set lr=0.0001 -> 0.0001 (float). --set extra_class_weight='{"cancer": 3.0}' -> dict.
    Falls back to the raw string if it isn't valid Python literal syntax."""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _apply_overrides(config, overrides):
    for item in overrides or []:
        key, _, raw_value = item.partition("=")
        if not hasattr(config, key):
            raise ValueError(f"Unknown field '{key}' for {type(config).__name__}")
        setattr(config, key, _parse_value(raw_value))
    return config


def main():
    parser = argparse.ArgumentParser(description="Train a model using a named preset.")
    parser.add_argument("--preset", required=True,
                         help="Name of a JSON file in configs/train/ (without .json). "
                              "The matching configs/model/<preset>.json is loaded automatically "
                              "if present, else pass --model-preset explicitly.")
    parser.add_argument("--model-preset", default=None,
                         help="Name of a JSON file in configs/model/ (without .json). "
                              "Defaults to the same name as --preset.")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                         metavar="field=value",
                         help="Override one TrainConfig field. Repeatable.")
    parser.add_argument("--model-set", dest="model_overrides", action="append", default=[],
                         metavar="field=value",
                         help="Override one ModelConfig field. Repeatable.")
    args = parser.parse_args()

    train_cfg = TrainConfig.from_json(os.path.join(TRAIN_CONFIG_DIR, f"{args.preset}.json"))
    _apply_overrides(train_cfg, args.overrides)

    model_preset = args.model_preset or args.preset
    model_cfg_path = os.path.join(MODEL_CONFIG_DIR, f"{model_preset}.json")
    if not os.path.exists(model_cfg_path):
        raise FileNotFoundError(
            f"No configs/model/{model_preset}.json found. Pass --model-preset explicitly, "
            f"or create that file (see configs/model/ for examples)."
        )
    model_cfg = ModelConfig.from_json(model_cfg_path)
    _apply_overrides(model_cfg, args.model_overrides)

    print(f"[CONFIG] train preset: {args.preset}  model preset: {model_preset}")
    print(f"[CONFIG] model: {model_cfg.to_dict()}")
    print(f"[CONFIG] train: {train_cfg.to_dict()}")

    Trainer(model_cfg, train_cfg).run()


if __name__ == "__main__":
    main()
