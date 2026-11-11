import pytest
from PIL import Image

from crop_adaptive.data.transforms import build_transforms, build_transforms_from_config


def _torchvision_or_skip():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    return torch


def _config():
    return {
        "model": {"input_size": 224},
        "transforms": {
            "resize_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "augmentation": {
                "random_resized_crop_scale": [0.85, 1.0],
                "random_resized_crop_ratio": [0.9, 1.1],
                "rotation_degrees": 8,
                "horizontal_flip_probability": 0.5,
                "brightness": 0.12,
                "contrast": 0.12,
                "saturation": 0.08,
                "hue": 0.01,
            },
        },
    }


def test_eval_transform_returns_normalized_rgb_tensor_with_expected_shape():
    torch = _torchvision_or_skip()
    image = Image.new("RGB", (320, 240), color=(120, 100, 80))
    result = build_transforms(_config(), "validation")(image)
    assert isinstance(result, torch.Tensor)
    assert tuple(result.shape) == (3, 224, 224)
    expected = (torch.tensor([120, 100, 80], dtype=torch.float32) / 255.0 -
                torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(result[:, 100, 100], expected, atol=1e-4)


def test_validation_and_test_transforms_are_deterministic():
    _torchvision_or_skip()
    image = Image.new("RGB", (300, 260), color=(90, 130, 170))
    validation = build_transforms(_config(), "validation")
    test = build_transforms(_config(), "test")
    assert (validation(image) == validation(image)).all()
    assert (test(image) == test(image)).all()


def test_train_transform_returns_expected_shape():
    torch = _torchvision_or_skip()
    image = Image.new("RGB", (300, 260), color=(100, 140, 180))
    result = build_transforms(_config(), "train")(image)
    assert tuple(result.shape) == (3, 224, 224)
    assert result.dtype == torch.float32


def test_transform_factory_returns_all_splits():
    _torchvision_or_skip()
    result = build_transforms_from_config(_config())
    assert set(result) == {"train", "validation", "test", "predict"}


def test_invalid_split_is_rejected():
    with pytest.raises(ValueError, match="split must be"):
        build_transforms(_config(), "holdout")


def test_invalid_normalization_or_crop_values_are_rejected():
    config = _config()
    config["transforms"]["std"] = [0.2, 0.0, 0.2]
    with pytest.raises(ValueError, match="std values"):
        build_transforms(config, "test")
    config = _config()
    config["transforms"]["augmentation"]["random_resized_crop_scale"] = [1.1, 1.2]
    with pytest.raises(ValueError, match="random_resized_crop_scale"):
        build_transforms(config, "train")
