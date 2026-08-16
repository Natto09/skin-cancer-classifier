"""
DermascanCNN -- the small from-scratch CNN originally defined in
train_model.py (see legacy/train_model_custom_cnn.py for the original
script). Confirmed (by grep across the whole codebase) to never be loaded
or referenced by the ensemble, gate, specialist, or combined pipeline --
this was an early prototype, kept here for completeness.

IMPORTANT: unlike the other backbones (resnet50/densenet121/vit_b16), this
architecture expects 28x28 input (matching the original script), not the
224x224 the rest of this project's pipeline uses -- the feature map size
(7x7 after two 2x2 max-pools) is baked into num_features below. Plugging
this into src.train.Trainer as-is (which hardcodes 224x224 transforms)
will not work without also overriding the transforms and re-deriving
num_features for a 224x224 input. There's no configs/train/custom_cnn.json
preset for this reason -- see configs/README.md.

Structural note vs. the original train_model.py: the original had
`nn.Flatten()` as the first layer INSIDE self.classifier. Moved into
self.features here instead, so self.classifier is just the linear head --
that's what lets src.models.backbones swap it out generically like every
other backbone's head, the same way build_head() does for resnet/densenet/vit.
"""

import torch.nn as nn


class DermascanCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.num_features = 64 * 7 * 7  # assumes 28x28 input
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
