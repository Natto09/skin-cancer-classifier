import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Multi-class focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Unlike plain class-weighted CrossEntropy (which scales every mistake on
    a given class by a fixed amount regardless of how confidently wrong the
    model was), focal loss additionally down-weights EASY examples (already
    classified with high confidence) and concentrates gradient on HARD ones
    -- e.g. a mel image that looks a lot like nv, which is exactly the kind
    of mistake that matters for a cancer screen. gamma controls how strongly
    easy examples get down-weighted (0 = reduces to weighted CrossEntropy).
    """

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=-1)
        if self.label_smoothing > 0:
            num_classes = logits.size(-1)
            smooth_targets = torch.full_like(log_probs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            ce = -(smooth_targets * log_probs).sum(dim=-1)
            pt = torch.exp(-ce)
        else:
            ce = torch.nn.functional.nll_loss(log_probs, targets, reduction="none")
            pt = torch.exp(-ce)

        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()
