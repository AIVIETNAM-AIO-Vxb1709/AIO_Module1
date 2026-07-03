"""Backbone construction through timm.

Any timm model id can be used as a backbone, so the backbone is a config string
rather than a fixed set. The classifier head is created by timm to match the
dataset's class count, and its name is read from the model metadata instead of
being hardcoded per architecture.
"""

import timm
import torch.nn as nn

from ..utils.config import ModelConfig

TRANSFORMER = "transformer"
CONV = "conv"


def build_backbone(model: ModelConfig, n_classes: int) -> nn.Module:
    """Create a timm backbone with a fresh head sized to ``n_classes``.

    ``dynamic_img_size`` is requested so transformer backbones pretrained at a
    larger resolution accept the project's 64x64 inputs; backbones that do not
    accept the argument are created without it.
    """
    if n_classes <= 0:
        raise ValueError("n_classes must be greater than zero.")

    if not timm.is_model(model.backbone):
        raise ValueError(
            f"Unknown timm backbone '{model.backbone}'. "
            "Use a valid timm model id (see timm.list_models())."
        )
    kwargs = {"pretrained": model.pretrained, "num_classes": n_classes}
    try:
        return timm.create_model(model.backbone, dynamic_img_size=True, **kwargs)
    except TypeError:
        return timm.create_model(model.backbone, **kwargs)


def get_head_name(model: nn.Module) -> str:
    """Return the classifier head module name from the model's default config."""
    name = getattr(model, "default_cfg", {}).get("classifier")
    if not name:
        raise ValueError("Backbone does not expose a classifier head name.")
    return name


def detect_family(model: nn.Module) -> str:
    """Classify a backbone as transformer or conv by its attention modules."""
    for name, _ in model.named_modules():
        if name.endswith("attn.qkv"):
            return TRANSFORMER
    return CONV
