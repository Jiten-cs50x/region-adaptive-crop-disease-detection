import csv
import json

import pytest

from crop_adaptive.data.provenance import (
    apply_approved_metadata_recovery,
    audit_metadata_provenance,
)

FIELDS = [
    "image_path", "crop", "disease_label", "image_type", "source_dataset",
    "license", "label_status", "source_url", "country", "region", "state",
    "district", "collection_date", "variety", "growth_stage", "plant_id",
    "field_id", "capture_session_id", "verified_by", "notes",
]


def _write(path, rows, fields=FIELDS):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _approved_proposal():
    return {
        "manifest_row_number": "2", "image_path": "a.jpg", "disease_label": "Healthy",
        "proposed_country": "India", "proposed_state": "Karnataka",
        "review_status": "approved", "evidence_type": "dataset_documentation",
        "evidence_reference": "https://example.org/dataset-card", "reviewer": "Reviewer A",
        "review_date": "2026-11-12", "review_notes": "Explicitly stated in source documentation",
    }


def test_metadata_audit_writes_blank_evidence_gated_queue_without_changing_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write(manifest, [
        {"image_path": "a.jpg", "crop": "sugarcane", "disease_label": "Healthy",
         "image_type": "leaf", "source_dataset": "Project-AgML", "license": "CC BY 4.0",
         "label_status": "unverified"},
    ])
    original = manifest.read_bytes()
    report_path = tmp_path / "reports" / "metadata_audit.json"
    report = audit_metadata_provenance(manifest, report_path)
    assert report["manifest_rows"] == 1
    assert report["rows_in_recovery_queue"] == 1
    assert report["recovery_policy"]["explicit_approval_required"] is True
    assert manifest.read_bytes() == original
    queue = report_path.with_name("metadata_audit_recovery_queue.csv")
    with queue.open(newline="", encoding="utf-8-sig") as stream:
        row = next(csv.DictReader(stream))
    assert row["proposed_country"] == ""
    assert row["review_status"] == ""


def test_metadata_audit_rejects_missing_columns_and_empty_manifest(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("image_path,label\na.jpg,Healthy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        audit_metadata_provenance(bad, tmp_path / "bad.json")
    empty = tmp_path / "empty.csv"
    _write(empty, [])
    with pytest.raises(ValueError, match="no data rows"):
        audit_metadata_provenance(empty, tmp_path / "empty.json")


def test_metadata_audit_reports_duplicate_paths_without_mutating_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    row = {"image_path": "same.jpg", "disease_label": "Healthy", "source_dataset": "AgML"}
    _write(manifest, [row, row])
    original = manifest.read_bytes()
    report = audit_metadata_provenance(manifest, tmp_path / "report.json")
    assert len(report["duplicate_manifest_paths"]) == 1
    assert manifest.read_bytes() == original


def test_recovery_applies_only_approved_evidence_backed_values_to_new_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    source_row = {
        "image_path": "a.jpg", "crop": "sugarcane", "disease_label": "Healthy",
        "image_type": "leaf", "source_dataset": "Project-AgML", "license": "CC BY 4.0",
        "label_status": "unverified",
    }
    _write(manifest, [source_row])
    queue = tmp_path / "queue.csv"
    proposal = _approved_proposal()
    queue_fields = list(proposal)
    _write(queue, [proposal], queue_fields)
    source_before, queue_before = manifest.read_bytes(), queue.read_bytes()
    output = tmp_path / "versioned_manifest.csv"
    result = apply_approved_metadata_recovery(manifest, queue, output)
    assert result["approved_rows_applied"] == 1
    assert result["metadata_values_applied"] == 2
    with output.open(newline="", encoding="utf-8-sig") as stream:
        recovered = next(csv.DictReader(stream))
    assert recovered["country"] == "India"
    assert recovered["state"] == "Karnataka"
    assert manifest.read_bytes() == source_before
    assert queue.read_bytes() == queue_before


@pytest.mark.parametrize("change, message", [
    ({"review_status": "needs_more_evidence"}, "review_status=approved"),
    ({"evidence_reference": ""}, "evidence_reference"),
    ({"review_date": "12-11-2026"}, "YYYY-MM-DD"),
])
def test_recovery_rejects_unapproved_or_incomplete_evidence(tmp_path, change, message):
    manifest = tmp_path / "manifest.csv"
    _write(manifest, [{"image_path": "a.jpg", "disease_label": "Healthy"}])
    queue = tmp_path / "queue.csv"
    proposal = _approved_proposal()
    proposal.update(change)
    _write(queue, [proposal], list(proposal))
    with pytest.raises(ValueError, match=message):
        apply_approved_metadata_recovery(manifest, queue, tmp_path / "new.csv")
    assert not (tmp_path / "new.csv").exists()


def test_recovery_rejects_stale_queue_identity_and_existing_output(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write(manifest, [{"image_path": "a.jpg", "disease_label": "Healthy"}])
    queue = tmp_path / "queue.csv"
    proposal = _approved_proposal()
    proposal["image_path"] = "different.jpg"
    _write(queue, [proposal], list(proposal))
    with pytest.raises(ValueError, match="does not match source manifest"):
        apply_approved_metadata_recovery(manifest, queue, tmp_path / "new.csv")
    existing = tmp_path / "existing.csv"
    existing.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError):
        apply_approved_metadata_recovery(manifest, queue, existing)
    assert existing.read_text(encoding="utf-8") == "do not overwrite"


def test_recovery_refuses_to_overwrite_existing_metadata(tmp_path):
    manifest = tmp_path / "manifest.csv"
    _write(manifest, [{"image_path": "a.jpg", "disease_label": "Healthy", "country": "India"}])
    queue = tmp_path / "queue.csv"
    proposal = _approved_proposal()
    _write(queue, [proposal], list(proposal))
    with pytest.raises(ValueError, match="Refusing to overwrite existing country"):
        apply_approved_metadata_recovery(manifest, queue, tmp_path / "new.csv")
