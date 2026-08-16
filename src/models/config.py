"""
ModelConfig -- a plain Python dataclass describing ONE model choice: which
backbone, how many classes, head shape, freeze strategy.

Why a dataclass instead of scattering these as arguments through a training
script (the original pattern): every field has a name, a type, and a
default you can see in one place, autocomplete works, and you can adjust
ONE field at a time --

    cfg = ModelConfig(backbone="densenet121", num_classes=2)
    cfg.dropout = 0.5          # tweak one thing
    cfg.hidden_dim = 256       # tweak another

-- without touching anything else. It also round-trips to/from JSON so a
specific configuration (e.g. "the exact setup that produced gate v2") can
be saved as a file in configs/model/ and reloaded later instead of having
to remember which flags were passed on the command line.
"""

from dataclasses import dataclass, asdict, field
import json


@dataclass
class ModelConfig:
    backbone: str = "resnet50"          # one of: resnet50, densenet121, vit_b16, custom_cnn
    num_classes: int = 7
    class_names: list = field(default_factory=lambda: ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
    pretrained: bool = True
    freeze_mode: str = "default"        # "default" | "none" | "head_only" -- see src/models/backbones.py
    hidden_dim: int = 128               # head's hidden layer width (None = no hidden layer)
    dropout: float = 0.8
    use_batchnorm: bool = True

    def build(self):
        """Constructs the actual torch.nn.Module for this config."""
        from .backbones import build_backbone
        return build_backbone(
            self.backbone, self.num_classes,
            pretrained=self.pretrained, freeze_mode=self.freeze_mode,
            hidden_dim=self.hidden_dim, dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
        )

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
