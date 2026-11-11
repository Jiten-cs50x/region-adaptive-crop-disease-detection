import pytest
import yaml

from crop_adaptive.models.efficientnet import build_model, build_model_from_config


def _torchvision_or_skip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    return torch


def test_efficientnet_factory_replaces_classifier_for_requested_classes():
    torch = _torchvision_or_skip()
    model = build_model(num_classes=6, pretrained=False)
    assert model.classifier[1].out_features == 6

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert tuple(output.shape) == (1, 6)


def test_model_factory_freezes_features_but_not_classifier():
    _torchvision_or_skip()
    model = build_model(num_classes=3, pretrained=False, freeze_backbone=True)
    assert all(not parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_config_factory_uses_taxonomy_class_count():
    _torchvision_or_skip()
    config = {
        "task": {"backbone": "efficientnet_b0"},
        "model": {"pretrained": False, "freeze_backbone": False, "dropout": 0.2},
        "taxonomy": {"classes": ["healthy", "rust", "mosaic"]},
    }
    model = build_model_from_config(config)
    assert model.classifier[1].out_features == 3


def test_config_factory_rejects_unknown_backbone():
    config = {
        "task": {"backbone": "resnet50"},
        "taxonomy": {"classes": ["healthy", "rust"]},
    }
    with pytest.raises(ValueError, match="Unsupported backbone"):
        build_model_from_config(config)


def test_config_factory_reads_yaml_class_count():
    _torchvision_or_skip()
    with open("configs/sugarcane_karnataka.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model = build_model_from_config(config, pretrained_override=False)
    assert model.classifier[1].out_features == len(config["taxonomy"]["classes"])
