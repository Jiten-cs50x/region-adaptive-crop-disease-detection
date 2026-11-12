import csv, json
import pytest
from crop_adaptive.data.provenance import audit_metadata_provenance
FIELDS = ["image_path","crop","disease_label","image_type","source_dataset","license","label_status","source_url","country","region","state","district","collection_date","variety","growth_stage","plant_id","field_id","capture_session_id","verified_by","notes"]
def _write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
def test_metadata_audit_writes_blank_evidence_led_queue_without_changing_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"image_path":"a.jpg","crop":"sugarcane","disease_label":"Healthy","image_type":"leaf","source_dataset":"Project-AgML","license":"CC BY 4.0","label_status":"unverified"},
        {"image_path":"b.jpg","crop":"sugarcane","disease_label":"Rust","image_type":"leaf","source_dataset":"Project-AgML","license":"CC BY 4.0","label_status":"unverified","country":"India","state":"Karnataka"},
    ]
    _write(manifest, rows); original = manifest.read_bytes()
    output = tmp_path / "reports" / "metadata_audit.json"
    report = audit_metadata_provenance(manifest, output)
    assert report["manifest_rows"] == 2
    assert report["metadata_completeness"]["country"]["missing"] == 1
    assert report["metadata_completeness"]["state"]["coverage_percent"] == 50.0
    assert report["rows_in_recovery_queue"] == 2
    assert report["recovery_policy"]["automatic_metadata_inference"] is False
    assert manifest.read_bytes() == original
    queue = output.with_name("metadata_audit_recovery_queue.csv")
    with queue.open(newline="", encoding="utf-8-sig") as stream: recovered = list(csv.DictReader(stream))
    assert recovered[0]["proposed_country"] == "" and recovered[0]["proposed_state"] == ""
    assert recovered[0]["evidence_reference"] == "" and recovered[0]["disease_label"] == "Healthy"
    assert json.loads(output.read_text(encoding="utf-8"))["manifest_rows"] == 2
def test_metadata_audit_rejects_missing_required_columns(tmp_path):
    manifest = tmp_path / "bad.csv"; manifest.write_text("image_path,label\na.jpg,Healthy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"): audit_metadata_provenance(manifest, tmp_path / "report.json")
def test_metadata_audit_rejects_empty_manifest(tmp_path):
    manifest = tmp_path / "empty.csv"; _write(manifest, [])
    with pytest.raises(ValueError, match="no data rows"): audit_metadata_provenance(manifest, tmp_path / "report.json")
def test_metadata_audit_reports_duplicate_paths_without_mutating_manifest(tmp_path):
    manifest = tmp_path / "manifest.csv"; row = {"image_path":"same.jpg","disease_label":"Healthy","source_dataset":"AgML"}
    _write(manifest, [row,row]); original = manifest.read_bytes()
    report = audit_metadata_provenance(manifest, tmp_path / "report.json")
    assert len(report["duplicate_manifest_paths"]) == 1 and manifest.read_bytes() == original
