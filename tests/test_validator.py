from PIL import Image

from crop_adaptive.data.validator import validate_dataset


def test_validator_counts_images(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "healthy").mkdir(parents=True)
    Image.new("RGB", (128, 128)).save(dataset / "healthy" / "sample.jpg")
    report = validate_dataset(dataset, tmp_path / "reports", {})
    assert report["total_images"] == 1
    assert report["valid_images"] == 1
    assert report["class_counts"]["healthy"] == 1
