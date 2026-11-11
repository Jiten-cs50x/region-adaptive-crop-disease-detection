import json

from PIL import Image, ImageDraw

from crop_adaptive.data.quality import analyze_image_quality


def test_quality_report_flags_low_resolution_and_preserves_source(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image_path = dataset / "small.png"
    Image.new("RGB", (32, 24), "white").save(image_path)
    before = image_path.read_bytes()

    report_path = tmp_path / "reports" / "quality.json"
    report = analyze_image_quality(dataset, report_path)

    assert report["image_count_analyzed"] == 1
    assert "low_resolution" in report["images"][0]["quality_flags"]
    assert image_path.read_bytes() == before
    assert report_path.exists()
    assert report_path.with_suffix(".csv").exists()
    assert json.loads(report_path.read_text())["flagged_image_count"] == 1


def test_quality_metrics_distinguish_flat_and_textured_images(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (256, 256), "gray").save(dataset / "flat.png")
    textured = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(textured)
    for x in range(0, 256, 8):
        draw.line((x, 0, x, 255), fill="black", width=3)
    textured.save(dataset / "textured.png")

    report = analyze_image_quality(
        dataset,
        tmp_path / "quality.json",
        min_width=64,
        min_height=64,
        min_brightness=0,
        max_brightness=255,
        min_contrast=0,
        min_sharpness=0,
    )
    rows = {row["image_path"]: row for row in report["images"]}
    assert rows["textured.png"]["contrast_stddev"] > rows["flat.png"]["contrast_stddev"]
    assert rows["textured.png"]["edge_variance_sharpness"] > rows["flat.png"]["edge_variance_sharpness"]


def test_invalid_quality_thresholds_are_rejected(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    try:
        analyze_image_quality(dataset, tmp_path / "quality.json", min_brightness=230, max_brightness=20)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected threshold validation error")
