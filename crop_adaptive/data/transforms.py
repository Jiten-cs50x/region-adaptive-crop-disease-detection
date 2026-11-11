from __future__ import annotations

from typing import Any

from PIL import ImageOps


def _torchvision_transforms():
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "Image transforms require compatible PyTorch and torchvision installs. "
            "Install matching versions for your CPU/CUDA environment."
        ) from exc
    return transforms, InterpolationMode


def _validate_transform_config(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("model", {})
    transform_config = config.get("transforms", {})
    input_size = int(transform_config.get("input_size", model_config.get("input_size", 224)))
    resize_size = int(transform_config.get("resize_size", round(input_size * 1.14)))
    if input_size < 32:
        raise ValueError("transforms.input_size must be at least 32")
    if resize_size < input_size:
        raise ValueError("transforms.resize_size must be greater than or equal to input_size")

    mean = transform_config.get("mean", [0.485, 0.456, 0.406])
    std = transform_config.get("std", [0.229, 0.224, 0.225])
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("transforms.mean and transforms.std must each contain three RGB values")
    if any(float(value) <= 0 for value in std):
        raise ValueError("transforms.std values must be greater than zero")

    augmentation = transform_config.get("augmentation", {})
    scale = augmentation.get("random_resized_crop_scale", [0.85, 1.0])
    if len(scale) != 2 or not 0 < float(scale[0]) <= float(scale[1]) <= 1:
        raise ValueError("random_resized_crop_scale must be two values satisfying 0 < min <= max <= 1")
    rotation = float(augmentation.get("rotation_degrees", 8.0))
    if not 0 <= rotation <= 45:
        raise ValueError("rotation_degrees must be between 0 and 45")
    return {
        "input_size": input_size,
        "resize_size": resize_size,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "augmentation": augmentation,
        "scale": (float(scale[0]), float(scale[1])),
        "rotation": rotation,
    }


def build_transforms(config: dict[str, Any], split: str = "train"):
    """Build an EfficientNet-compatible transform for train, validation, or test."""
    if split not in {"train", "validation", "test", "predict"}:
        raise ValueError("split must be one of: train, validation, test, predict")

    values = _validate_transform_config(config)
    transforms, interpolation = _torchvision_transforms()
    size = values["input_size"]
    operations = [
        transforms.Lambda(lambda image: ImageOps.exif_transpose(image).convert("RGB"))
    ]
    if split == "train":
        aug = values["augmentation"]
        operations.extend([
            transforms.RandomResizedCrop(
                size,
                scale=values["scale"],
                ratio=tuple(aug.get("random_resized_crop_ratio", [0.9, 1.1])),
                interpolation=interpolation.BILINEAR,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=float(aug.get("horizontal_flip_probability", 0.5))),
        ])
        if values["rotation"] > 0:
            operations.append(
                transforms.RandomRotation(values["rotation"], interpolation=interpolation.BILINEAR)
            )
        brightness = float(aug.get("brightness", 0.12))
        contrast = float(aug.get("contrast", 0.12))
        saturation = float(aug.get("saturation", 0.08))
        hue = float(aug.get("hue", 0.01))
        if any(value > 0 for value in (brightness, contrast, saturation, hue)):
            operations.append(transforms.ColorJitter(
                brightness=brightness, contrast=contrast,
                saturation=saturation, hue=hue,
            ))
    else:
        operations.extend([
            transforms.Resize(values["resize_size"], interpolation=interpolation.BILINEAR, antialias=True),
            transforms.CenterCrop(size),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=values["mean"], std=values["std"]),
    ])
    return transforms.Compose(operations)


def build_transforms_from_config(config: dict[str, Any]):
    """Return named transforms for train, validation, test, and prediction."""
    return {
        split: build_transforms(config, split)
        for split in ("train", "validation", "test", "predict")
    }
