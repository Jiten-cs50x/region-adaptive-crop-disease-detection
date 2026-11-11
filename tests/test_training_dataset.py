import csv

import pytest
from PIL import Image


def test_manifest_dataset_reads_image_and_label(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from crop_adaptive.training.engine import ManifestImageDataset
    from crop_adaptive.data.transforms import build_transforms

    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (48, 48), (80, 120, 160)).save(root / "sample.jpg")
    split = tmp_path / "train.csv"
    with split.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "disease_label"])
        writer.writeheader()
        writer.writerow({"image_path": "sample.jpg", "disease_label": "healthy"})

    transform = build_transforms({
        "model": {"input_size": 224},
        "transforms": {"resize_size": 256},
    }, "validation")
    dataset = ManifestImageDataset(root, split, {"healthy": 0, "rust": 1}, transform)
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert tuple(image.shape) == (3, 224, 224)
    assert label == 0


def test_manifest_dataset_rejects_unknown_label_and_path_escape(tmp_path):
    pytest.importorskip("torch")
    from crop_adaptive.training.engine import ManifestImageDataset

    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (32, 32)).save(root / "sample.jpg")
    split = tmp_path / "train.csv"
    with split.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "disease_label"])
        writer.writeheader()
        writer.writerow({"image_path": "sample.jpg", "disease_label": "not_in_taxonomy"})
    with pytest.raises(ValueError, match="not in taxonomy"):
        ManifestImageDataset(root, split, {"healthy": 0, "rust": 1}, None)

    with split.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "disease_label"])
        writer.writeheader()
        writer.writerow({"image_path": "../outside.jpg", "disease_label": "healthy"})
    with pytest.raises(ValueError, match="escapes dataset root"):
        ManifestImageDataset(root, split, {"healthy": 0, "rust": 1}, None)
