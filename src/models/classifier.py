"""
SkinLesionClassifier -- wraps a ModelConfig + the built torch model into one
object with a small, consistent interface (load/save/predict), instead of
every script re-writing its own "build_resnet50(num_classes, device)" /
"build_densenet121(...)" / "build_vit(...)" functions (this exact
duplication existed in ensemble_meta_classifier.py, evaluate_confusion_matrix.py,
combined_pipeline_eval.py, train_gate_model.py, and train_specialist_mel_bkl.py
-- five copies of the same three functions).

Typical usage -- swap backbone or tweak one setting at a time:

    from src.models.config import ModelConfig
    from src.models.classifier import SkinLesionClassifier

    cfg = ModelConfig(backbone="densenet121", num_classes=2, dropout=0.5)
    clf = SkinLesionClassifier(cfg)              # builds the model
    clf.load_weights("skin_cancer_best_gate_1M.pth")
    probs, preds = clf.predict(some_dataloader)

Or load straight from a saved checkpoint + its config:

    clf = SkinLesionClassifier.from_checkpoint(
        "skin_cancer_best_resnet_1M.pth",
        ModelConfig(backbone="resnet50", num_classes=7),
    )
"""

import torch

from .config import ModelConfig
from ..utils.gpu import pick_device, unwrap_model, maybe_data_parallel


class SkinLesionClassifier:
    def __init__(self, config: ModelConfig, device=None, gpu_ids=None):
        self.config = config
        if device is not None:
            self.device = device
            self.gpu_ids = gpu_ids or []
        else:
            self.device, self.gpu_ids = pick_device(gpu_ids)
        self.model = config.build().to(self.device)

    # -- construction shortcuts -------------------------------------------

    @classmethod
    def from_checkpoint(cls, checkpoint_path, config: ModelConfig, device=None, gpu_ids=None):
        clf = cls(config, device=device, gpu_ids=gpu_ids)
        clf.load_weights(checkpoint_path)
        return clf

    @classmethod
    def from_config_file(cls, config_path, checkpoint_path=None, device=None, gpu_ids=None):
        config = ModelConfig.from_json(config_path)
        clf = cls(config, device=device, gpu_ids=gpu_ids)
        if checkpoint_path:
            clf.load_weights(checkpoint_path)
        return clf

    # -- weights ------------------------------------------------------------

    def load_weights(self, path):
        state = torch.load(path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(state)
        return self

    def save_weights(self, path):
        torch.save(unwrap_model(self.model).state_dict(), path)
        return self

    # -- mode / multi-gpu ----------------------------------------------------

    def train_mode(self):
        self.model.train()
        return self

    def eval_mode(self):
        self.model.eval()
        return self

    def enable_multi_gpu(self):
        self.model = maybe_data_parallel(self.model, self.gpu_ids)
        return self

    # -- inference ------------------------------------------------------------

    @torch.no_grad()
    def predict_proba(self, loader):
        """Runs inference over a DataLoader of (image, label) pairs. Returns
        (probs, labels) as numpy arrays -- probs shape (N, num_classes)."""
        import numpy as np
        self.eval_mode()
        all_probs, all_labels = [], []
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
        return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

    @torch.no_grad()
    def predict_image(self, image_path, transform):
        """Single-image inference. Returns (predicted_class_name, {class_name: pct, ...}).
        Modern replacement for the old test_model.py, using this classifier's
        own config.class_names instead of a hardcoded list."""
        from PIL import Image
        import torch.nn.functional as F

        self.eval_mode()
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0] * 100
        names = self.config.class_names
        result = {names[i]: float(probs[i]) for i in range(len(names))}
        predicted = names[int(torch.argmax(probs))]
        return predicted, result
