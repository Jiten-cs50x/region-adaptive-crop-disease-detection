import csv
import json

import pytest
from PIL import Image

from crop_adaptive.data.acquisition import audit_acquisition_patterns


def _write_manifest(path, rows):
    fields = ["image_path", "disease_label", "source_dataset", "image_type", "region"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_acquisition_audit_reports_class_capture_profiles_without_modifying_images(tmp_path):
    root = tmp_path / "dataset"
    (root / "healthy").mkdir(parents=True)
    (root / "rust").mkdir()
    first, second = root / "healthy" / "leaf.jpg", root / "rust" / "leaf.jpg"
    Image.new("RGB", (640, 480), (30, 150, 40)).save(first)
    Image.new("RGB", (300, 600), (210, 190, 170)).save(second)
    before = (first.read_bytes(), second.read_bytes())
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"image_path": "healthy/leaf.jpg", "disease_label": "Healthy", "source_dataset": "AgML", "image_type": "leaf", "region": ""},
        {"image_path": "rust/leaf.jpg", "disease_label": "Rust", "source_dataset": "AgML", "image_type": "leaf", "region": "Karnataka"},
    ])
    output = tmp_path / "reports" / "acquisition_audit.json"
    result = audit_acquisition_patterns(root, manifest, output)
    assert result["images_analyzed"] == 2
    assert result["class_count"] == 2
    assert result["class_capture_counts"]["Healthy"]["orientation"] == {"landscape": 1}
    assert result["class_capture_counts"]["Rust"]["orientation"] == {"portrait": 1}
    assert result["metadata_completeness"]["region"]["missing"] == 1
    assert (first.read_bytes(), second.read_bytes()) == before
    assert output.is_file() and output.with_suffix(".csv").is_file()
    assert json.loads(output.read_text())["images_analyzed"] == 2


def test_acquisition_audit_rejects_path_traversal(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [{"image_path": "../outside.jpg", "disease_label": "Healthy",
                                "source_dataset": "AgML", "image_type": "leaf", "region": ""}])
    with pytest.raises(ValueError, match="escapes dataset root"):
        audit_acquisition_patterns(root, manifest, tmp_path / "report.json")


def test_acquisition_audit_rejects_duplicate_manifest_paths(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = tmp_path / "manifest.csv"
    row = {"image_path": "missing.jpg", "disease_label": "Healthy",
           "source_dataset": "AgML", "image_type": "leaf", "region": ""}
    _write_manifest(manifest, [row, {**row, "disease_label": "Rust"}])
    with pytest.raises(ValueError, match="Duplicate image_path"):
        audit_acquisition_patterns(root, manifest, tmp_path / "report.json")
