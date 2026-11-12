import csv
import json

import pytest

from crop_adaptive.data.bias import analyze_acquisition_bias


def _write_features(path, rows):
    fields = [
        "image_path", "disease_label", "width", "height", "aspect_ratio",
        "brightness_mean", "contrast_stddev", "edge_variance_sharpness",
        "mean_red", "mean_green", "mean_blue", "orientation",
        "resolution_tier", "brightness_band",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_bias_audit_reports_effect_sizes_and_preserves_input(tmp_path):
    source = tmp_path / "features.csv"
    rows = [
        {"image_path": "a.jpg", "disease_label": "A", "brightness_mean": "50", "orientation": "portrait"},
        {"image_path": "b.jpg", "disease_label": "A", "brightness_mean": "55", "orientation": "portrait"},
        {"image_path": "c.jpg", "disease_label": "B", "brightness_mean": "180", "orientation": "landscape"},
        {"image_path": "d.jpg", "disease_label": "B", "brightness_mean": "190", "orientation": "landscape"},
    ]
    _write_features(source, rows)
    before = source.read_bytes()
    output = tmp_path / "reports" / "bias.json"
    report = analyze_acquisition_bias(source, output)
    assert report["image_count"] == 4
    assert report["class_count"] == 2
    assert report["numeric_class_associations"]["brightness_mean"]["eta_squared"] > 0.9
    assert report["categorical_class_associations"]["orientation"]["bias_corrected_cramers_v"] > 0
    assert source.read_bytes() == before
    assert json.loads(output.read_text())["image_count"] == 4


def test_bias_audit_rejects_missing_label(tmp_path):
    source = tmp_path / "features.csv"
    _write_features(source, [{"image_path": "a.jpg", "disease_label": ""}])
    with pytest.raises(ValueError, match="disease_label"):
        analyze_acquisition_bias(source, tmp_path / "out.json")


def test_bias_audit_rejects_empty_csv(tmp_path):
    source = tmp_path / "features.csv"
    _write_features(source, [])
    with pytest.raises(ValueError, match="no image records"):
        analyze_acquisition_bias(source, tmp_path / "out.json")
