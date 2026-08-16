"""
Every backbone in this project (ResNet50, DenseNet121, ViT-B/16) replaces
its final layer with the SAME small head:
    Linear(in_features, hidden_dim) -> BatchNorm1d -> ReLU -> Dropout -> Linear(hidden_dim, num_classes)
with hidden_dim=128 and dropout=0.8 hardcoded identically in every training
script. Pulled out here as one function so hidden_dim/dropout become a
setting you adjust in one place (a ModelConfig field) instead of a literal
you'd have to find-and-replace across every training script.
"""

import torch.nn as nn


def build_head(in_features, num_classes, hidden_dim=128, dropout=0.8, use_batchnorm=True):
    """Builds the classifier head. Set hidden_dim=None to skip the hidden
    layer entirely and go straight in_features -> num_classes."""
    if hidden_dim is None:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    layers = [nn.Linear(in_features, hidden_dim)]
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes)]
    return nn.Sequential(*layers)
