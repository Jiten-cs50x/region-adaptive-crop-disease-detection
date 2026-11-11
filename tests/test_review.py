import csv
import json
import pytest
from PIL import Image
from crop_adaptive.data.review import create_misclassification_review

def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "true_label", "predicted_label", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)

def test_review_queue_and_contact_sheet_are_created_without_changing_source(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "crop").mkdir()
    image_path = dataset / "crop" / "leaf.jpg"
    Image.new("RGB", (80, 60), "green").save(image_path)
    source = tmp_path / "misclassified.csv"
    _write_csv(source, [{"image_path": "crop/leaf.jpg", "true_label": "Healthy",
                         "predicted_label": "Viral Disease", "confidence": "0.99", "correct": "False"}])
    original_bytes = image_path.read_bytes()
    result = create_misclassification_review(dataset, source, tmp_path / "review")
    assert result["review_items"] == 1
    assert result["high_confidence_errors_at_or_above_0_90"] == 1
    assert (tmp_path / "review" / "contact_sheet_001.jpg").is_file()
    assert (tmp_path / "review" / "REVIEW_GUIDE.md").is_file()
    with (tmp_path / "review" / "review_queue.csv").open(encoding="utf-8") as stream:
        review_row = next(csv.DictReader(stream))
    assert review_row["review_status"] == ""
    assert review_row["corrected_label"] == ""
    assert image_path.read_bytes() == original_bytes
    assert json.loads((tmp_path / "review" / "review_summary.json").read_text())["review_items"] == 1

def test_review_rejects_path_traversal(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = tmp_path / "misclassified.csv"
    _write_csv(source, [{"image_path": "../outside.jpg", "true_label": "Healthy",
                         "predicted_label": "Viral Disease", "confidence": "0.8", "correct": "False"}])
    with pytest.raises(ValueError, match="escapes dataset root"):
        create_misclassification_review(dataset, source, tmp_path / "review")

def test_review_rejects_invalid_confidence(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = tmp_path / "misclassified.csv"
    _write_csv(source, [{"image_path": "leaf.jpg", "true_label": "Healthy",
                         "predicted_label": "Viral Disease", "confidence": "1.2", "correct": "False"}])
    with pytest.raises(ValueError, match="between 0 and 1"):
        create_misclassification_review(dataset, source, tmp_path / "review")
