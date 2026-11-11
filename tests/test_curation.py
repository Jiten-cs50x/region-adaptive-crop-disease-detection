import csv
import hashlib
import json

from crop_adaptive.data.curation import curate_exact_duplicates


def _write_manifest(root, rows, path):
    fields = ["image_path", "crop", "disease_label", "image_type",
              "source_dataset", "license", "label_status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_curation_keeps_one_same_class_copy_and_excludes_cross_class(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    payload_a = b"same-class-image-bytes"
    payload_b = b"cross-class-image-bytes"
    payload_c = b"unique-image-bytes"

    for name, payload in [
        ("healthy_a.jpg", payload_a), ("healthy_b.jpg", payload_a),
        ("rust_a.jpg", payload_b), ("yellow_a.jpg", payload_b),
        ("spot.jpg", payload_c),
    ]:
        (root / name).write_bytes(payload)

    manifest = tmp_path / "manifest.csv"
    _write_manifest(root, [
        {"image_path": "healthy_b.jpg", "crop": "sugarcane", "disease_label": "healthy",
         "image_type": "leaf", "source_dataset": "test", "license": "CC-BY-4.0", "label_status": "unverified"},
        {"image_path": "healthy_a.jpg", "crop": "sugarcane", "disease_label": "healthy",
         "image_type": "leaf", "source_dataset": "test", "license": "CC-BY-4.0", "label_status": "unverified"},
        {"image_path": "rust_a.jpg", "crop": "sugarcane", "disease_label": "rust",
         "image_type": "leaf", "source_dataset": "test", "license": "CC-BY-4.0", "label_status": "unverified"},
        {"image_path": "yellow_a.jpg", "crop": "sugarcane", "disease_label": "yellow",
         "image_type": "leaf", "source_dataset": "test", "license": "CC-BY-4.0", "label_status": "unverified"},
        {"image_path": "spot.jpg", "crop": "sugarcane", "disease_label": "spot",
         "image_type": "leaf", "source_dataset": "test", "license": "CC-BY-4.0", "label_status": "unverified"},
    ], manifest)

    output = tmp_path / "curated"
    report = curate_exact_duplicates(root, manifest, output)

    assert report["source_rows"] == 5
    assert report["same_class_duplicate_groups"] == 1
    assert report["cross_class_duplicate_groups"] == 1
    assert report["same_class_duplicate_copies_removed"] == 1
    assert report["cross_class_duplicate_images_excluded_pending_review"] == 2
    assert report["curated_manifest_rows"] == 2

    with (output / "curated_manifest.csv").open(newline="", encoding="utf-8-sig") as stream:
        curated = list(csv.DictReader(stream))
    assert {row["image_path"] for row in curated} == {"healthy_a.jpg", "spot.jpg"}
    assert all(row["label_status"] == "unverified" for row in curated)

    with (output / "cross_class_duplicate_review.csv").open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        review = list(csv.DictReader(stream))
    assert len(review) == 2
    assert all(row["review_status"] == "pending" for row in review)

    # Source files and source manifest remain unchanged.
    assert (root / "healthy_b.jpg").read_bytes() == payload_a
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_curation_rejects_path_traversal(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    manifest = tmp_path / "manifest.csv"
    _write_manifest(root, [{
        "image_path": "../outside.jpg", "crop": "sugarcane", "disease_label": "rust",
        "image_type": "leaf", "source_dataset": "test", "license": "unknown",
        "label_status": "unverified",
    }], manifest)

    report = curate_exact_duplicates(root, manifest, tmp_path / "curated")
    assert report["invalid_rows_excluded"] == 1
    assert report["curated_manifest_rows"] == 0
    assert report["invalid_rows"][0]["reason"] == "image_path_outside_dataset_root"
