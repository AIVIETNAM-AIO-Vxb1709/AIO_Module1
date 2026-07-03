import gc

import pytest
import torch

from src.models.backbone import CONV, TRANSFORMER, build_backbone, detect_family
from src.models.transfer import (
    _conv_target_names,
    build_transfer_model,
    count_trainable_params,
)
from src.utils.config import LoraConfig, ModelConfig
from src.utils.constants import TransferStrategy

_N_CLASSES = 3
# A small CNN and the smallest standard ViT keep the CPU memory footprint low
# while still covering both model families; the conv-LoRA, family detection, and
# parameter-ordering behaviour is identical on larger CNNs used in real runs.
_BACKBONES = ["resnet18", "vit_small_patch16_224"]


@pytest.fixture(autouse=True)
def _free_memory():
    yield
    gc.collect()


def _model_config(backbone, strategy, lora=None):
    return ModelConfig(
        backbone=backbone,
        transfer=strategy,
        pretrained=False,
        lora=lora or LoraConfig(),
    )


@pytest.mark.parametrize("backbone", _BACKBONES)
@pytest.mark.parametrize("strategy", list(TransferStrategy))
def test_each_arm_outputs_class_logits(backbone, strategy):
    model = build_transfer_model(_model_config(backbone, strategy), _N_CLASSES)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))
    assert tuple(output.shape) == (2, _N_CLASSES)


@pytest.mark.parametrize("backbone", _BACKBONES)
def test_trainable_param_ordering(backbone):
    counts = {}
    for strategy in TransferStrategy:
        model = build_transfer_model(_model_config(backbone, strategy), _N_CLASSES)
        counts[strategy] = count_trainable_params(model)
    assert counts[TransferStrategy.LP] < counts[TransferStrategy.LORA]
    assert counts[TransferStrategy.LORA] < counts[TransferStrategy.FT]


def test_family_detection():
    resnet = build_backbone(_model_config("resnet18", TransferStrategy.FT), _N_CLASSES)
    vit = build_backbone(
        _model_config("vit_small_patch16_224", TransferStrategy.FT), _N_CLASSES
    )
    assert detect_family(resnet) == CONV
    assert detect_family(vit) == TRANSFORMER


def test_conv_target_kernel3_is_subset_of_all():
    resnet = build_backbone(_model_config("resnet18", TransferStrategy.FT), _N_CLASSES)
    all_convs = _conv_target_names(resnet, "all")
    kernel3 = _conv_target_names(resnet, "kernel3")
    assert set(kernel3).issubset(set(all_convs))
    assert 0 < len(kernel3) < len(all_convs)


def test_invalid_backbone_is_rejected():
    with pytest.raises(ValueError):
        build_backbone(_model_config("not_a_real_model", TransferStrategy.FT), _N_CLASSES)


@pytest.mark.parametrize("n_classes", [0, -1])
def test_invalid_class_count_is_rejected(n_classes):
    config = _model_config("resnet18", TransferStrategy.FT)

    with pytest.raises(ValueError, match="n_classes must be greater than zero"):
        build_backbone(config, n_classes)
