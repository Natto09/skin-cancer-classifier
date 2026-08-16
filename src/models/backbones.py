"""
Backbone registry.

This is the piece that used to force a NEW training script into existence
every time someone wanted to try a different architecture
(train_resnet_1M.py, train_densenet_1M.py, train_vit_1M.py were ~95%
identical files that only differed in this section). Now it's one registry:
pick a backbone by name, everything else (data loading, loss, checkpointing,
the training loop) is shared.

Each backbone keeps the EXACT freeze strategy the original script used for
it, since that was tuned per-architecture (freezing by top-level named
child for ResNet, by parameter position for DenseNet's deeply-nested dense
blocks, by encoder-layer index for ViT):

  resnet50:     unfreeze only "layer4" + the head (rest of the pretrained
                backbone stays frozen)
  densenet121:  unfreeze only the last 10 named parameters of
                model.features (unfreezing by top-level child doesn't work
                well here -- DenseNet's blocks are deeply nested)
  vit_b16:      unfreeze only the last 2 transformer encoder blocks + the
                head (12 encoder blocks total)
  custom_cnn:   DermascanCNN, a small from-scratch CNN (2 conv blocks, 28x28
                input). This was an early prototype (see
                legacy/train_model_custom_cnn.py) -- confirmed never wired
                into the ensemble/gate/specialist pipeline that's actually
                used. Included here for completeness / if you want to
                revisit it, not because anything downstream depends on it.
"""

import torch.nn as nn

from .heads import build_head


# ---------------------------------------------------------------------------
# Individual backbone builders. Each returns (model, num_features, head_attr)
# where head_attr is the attribute name to overwrite with the classifier
# head ("fc" for ResNet, "classifier" for DenseNet, "heads" for ViT).
# ---------------------------------------------------------------------------

def _build_resnet50(pretrained=True):
    from torchvision import models
    model = models.resnet50(weights="DEFAULT" if pretrained else None)
    return model, model.fc.in_features, "fc"


def _build_densenet121(pretrained=True):
    from torchvision import models
    model = models.densenet121(weights="DEFAULT" if pretrained else None)
    return model, model.classifier.in_features, "classifier"


def _build_vit_b16(pretrained=True):
    from torchvision import models
    model = models.vit_b_16(weights="DEFAULT" if pretrained else None)
    return model, model.heads.head.in_features, "heads"


def _build_custom_cnn(pretrained=True):
    """DermascanCNN -- see legacy/train_model_custom_cnn.py. No pretrained
    weights exist for this architecture (`pretrained` is accepted for
    interface consistency but ignored). Expects 28x28 input -- see the
    docstring in custom_cnn.py before using this with src.train.Trainer,
    which assumes 224x224 elsewhere."""
    from .custom_cnn import DermascanCNN
    model = DermascanCNN(num_classes=7)  # head is replaced below like any other backbone
    return model, model.num_features, "classifier"


# ---------------------------------------------------------------------------
# Freeze strategies -- applied AFTER the backbone is built, BEFORE the head
# is attached (head is always left trainable regardless of freeze mode).
# ---------------------------------------------------------------------------

def _freeze_resnet50(model, mode):
    if mode == "none":
        return
    for name, child in model.named_children():
        if mode == "head_only":
            unfreeze = name == "fc"
        else:  # "default" -- matches the original train_resnet_*.py scripts
            unfreeze = name in ["layer4", "fc"]
        for param in child.parameters():
            param.requires_grad = unfreeze


def _freeze_densenet121(model, mode):
    if mode == "none":
        return
    if mode == "head_only":
        for param in model.features.parameters():
            param.requires_grad = False
        return
    all_params = list(model.features.named_parameters())
    unfreeze_names = {name for name, _ in all_params[-10:]}
    for name, param in model.features.named_parameters():
        param.requires_grad = name in unfreeze_names


def _freeze_vit_b16(model, mode):
    if mode == "none":
        return
    for param in model.parameters():
        param.requires_grad = False
    if mode != "head_only":
        # ViT-B/16 has 12 encoder blocks (model.encoder.layers); unfreeze the last 2
        for layer in model.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True


def _freeze_custom_cnn(model, mode):
    pass  # trained fully from scratch either way -- nothing to freeze


BACKBONE_REGISTRY = {
    "resnet50": {"build": _build_resnet50, "freeze": _freeze_resnet50},
    "densenet121": {"build": _build_densenet121, "freeze": _freeze_densenet121},
    "vit_b16": {"build": _build_vit_b16, "freeze": _freeze_vit_b16},
    "custom_cnn": {"build": _build_custom_cnn, "freeze": _freeze_custom_cnn},
}


def build_backbone(name, num_classes, pretrained=True, freeze_mode="default",
                    hidden_dim=128, dropout=0.8, use_batchnorm=True):
    """
    Builds a full model: backbone (with the freeze strategy applied) + a
    fresh classifier head of the right size.

    name:        one of BACKBONE_REGISTRY.keys()
    freeze_mode: "default" (matches the original per-backbone strategy
                 described above), "none" (train every parameter),
                 or "head_only" (freeze the whole pretrained backbone,
                 train only the new head).
    """
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choices: {list(BACKBONE_REGISTRY)}")
    spec = BACKBONE_REGISTRY[name]

    model, num_features, head_attr = spec["build"](pretrained=pretrained)
    spec["freeze"](model, freeze_mode)

    head = build_head(num_features, num_classes, hidden_dim=hidden_dim,
                       dropout=dropout, use_batchnorm=use_batchnorm)
    setattr(model, head_attr, head)
    for param in getattr(model, head_attr).parameters():
        param.requires_grad = True

    return model
