from __future__ import annotations

from typing import Any


def build_model(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float | None = None,
):
    """Create a torchvision EfficientNet-B0 classifier.

    PyTorch and torchvision are optional training dependencies. Install them
    separately for the local CPU/CUDA environment before calling this factory.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if dropout is not None and not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1)")

    try:
        import torch.nn as nn
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "EfficientNet-B0 requires compatible PyTorch and torchvision installs. "
            "Install PyTorch for your CPU/CUDA environment, then install a matching torchvision version."
        ) from exc

    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    if dropout is not None:
        model.classifier[0] = nn.Dropout(p=dropout)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def build_model_from_config(
    config: dict[str, Any],
    pretrained_override: bool | None = None,
):
    """Build EfficientNet-B0 using the project YAML configuration."""
    task = config.get("task", {})
    model_config = config.get("model", {})
    backbone = model_config.get("name", task.get("backbone", "efficientnet_b0"))
    if backbone != "efficientnet_b0":
        raise ValueError(
            f"Unsupported backbone '{backbone}'. This release supports efficientnet_b0 only."
        )

    classes = config.get("taxonomy", {}).get("classes", [])
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("Configuration must define at least two taxonomy.classes entries")
    if len(set(classes)) != len(classes):
        raise ValueError("taxonomy.classes contains duplicate class names")

    pretrained = model_config.get("pretrained", True)
    if pretrained_override is not None:
        pretrained = pretrained_override

    return build_model(
        num_classes=len(classes),
        pretrained=bool(pretrained),
        freeze_backbone=bool(model_config.get("freeze_backbone", False)),
        dropout=model_config.get("dropout", 0.2),
    )
