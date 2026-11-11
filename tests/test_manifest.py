import csv

from PIL import Image

from crop_adaptive.data.manifest import create_manifest_template, validate_manifest


def test_create_manifest_template_has_required_columns(tmp_path):
    path = create_manifest_template(tmp_path / "manifest.csv")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        headers = next(csv.reader(stream))
    assert "image_path" in headers
    assert "district" in headers
    assert "label_status" in headers


def test_validate_manifest_accepts_verified_image(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (128, 128)).save(dataset / "leaf.jpg")
    manifest = tmp_path / "manifest.csv"
    fields = [
        "image_path", "crop", "disease_label", "image_type",
        "source_dataset", "license", "label_status", "verified_by",
        "collection_date", "district",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "image_path": "leaf.jpg",
            "crop": "sugarcane",
            "disease_label": "healthy",
            "image_type": "leaf",
            "source_dataset": "local_collection",
            "license": "owner_permission",
            "label_status": "verified",
            "verified_by": "agronomist",
            "collection_date": "2026-11-01",
            "district": "Uttara Kannada",
        })
    report = validate_manifest(dataset, manifest, tmp_path / "reports" / "manifest.json")
    assert report["total_rows"] == 1
    assert report["valid_rows"] == 1
    assert report["rows_requiring_review"] == 0


def test_validate_manifest_flags_missing_image_and_unverified_label(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "manifest.csv"
    fields = [
        "image_path", "crop", "disease_label", "image_type",
        "source_dataset", "license", "label_status",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "image_path": "missing.jpg",
            "crop": "sugarcane",
            "disease_label": "rust",
            "image_type": "leaf",
            "source_dataset": "test",
            "license": "unknown",
            "label_status": "unverified",
        })
    report = validate_manifest(dataset, manifest, tmp_path / "report.json")
    assert report["valid_rows"] == 0
    assert "image_file_not_found" in report["issues"][0]["issues"]
