"""Non-destructive provenance and metadata recovery audit for dataset manifests."""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = ("image_path", "disease_label")
PROVENANCE_FIELDS = ("source_dataset", "source_url", "license")
RECOVERY_FIELDS = ("country", "region", "state", "district", "collection_date", "variety", "growth_stage", "plant_id", "field_id", "capture_session_id")
OPTIONAL_METADATA_FIELDS = ("source_url", "country", "region", "state", "district", "collection_date", "variety", "growth_stage", "plant_id", "field_id", "capture_session_id", "verified_by", "notes")
EVIDENCE_FIELDS = ("evidence_type", "evidence_reference", "reviewer", "review_date", "review_notes")

def audit_metadata_provenance(manifest_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Summarize metadata gaps and write a separate evidence-led recovery worksheet."""
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        absent = sorted(set(REQUIRED_COLUMNS) - set(headers))
        if absent:
            raise ValueError(f"Manifest is missing required columns: {absent}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest contains no data rows")
    fields = tuple(dict.fromkeys((*PROVENANCE_FIELDS, *OPTIONAL_METADATA_FIELDS)))
    completeness = {}
    for field in fields:
        values = [(row.get(field) or "").strip() for row in rows]
        nonempty = [value for value in values if value]
        completeness[field] = {"present": len(nonempty), "missing": len(rows)-len(nonempty), "coverage_percent": round(100*len(nonempty)/len(rows), 2), "distinct_nonempty_values": len(set(nonempty))}
    labels = sorted({(r.get("disease_label") or "").strip() or "(missing)" for r in rows})
    class_profiles = {}
    for label in labels:
        selected = [r for r in rows if ((r.get("disease_label") or "").strip() or "(missing)") == label]
        class_profiles[label] = {field: {"present": sum(bool((r.get(field) or "").strip()) for r in selected), "missing": sum(not bool((r.get(field) or "").strip()) for r in selected)} for field in ("source_dataset", "license", *RECOVERY_FIELDS)}
    source_counts = Counter((r.get("source_dataset") or "").strip() or "(missing)" for r in rows)
    license_counts = Counter((r.get("license") or "").strip() or "(missing)" for r in rows)
    original_columns = list(headers)
    worksheet_fields = [*original_columns, "manifest_row_number", "missing_metadata_fields", *[f"proposed_{f}" for f in RECOVERY_FIELDS], *EVIDENCE_FIELDS]
    seen, duplicates, recovery_rows = set(), [], []
    for row_number, row in enumerate(rows, start=2):
        image_path = (row.get("image_path") or "").strip()
        if image_path in seen:
            duplicates.append({"manifest_row_number": row_number, "image_path": image_path})
        seen.add(image_path)
        missing = [f for f in ("source_dataset", "source_url", "license", *RECOVERY_FIELDS) if not (row.get(f) or "").strip()]
        if missing:
            item = {c: (row.get(c) or "") for c in original_columns}
            item["manifest_row_number"] = row_number
            item["missing_metadata_fields"] = ";".join(missing)
            item.update({f"proposed_{f}": "" for f in RECOVERY_FIELDS})
            item.update({f: "" for f in EVIDENCE_FIELDS})
            recovery_rows.append(item)
    queue = output.with_name(output.stem + "_recovery_queue.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with queue.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=worksheet_fields)
        writer.writeheader()
        writer.writerows(recovery_rows)
    report = {
        "manifest_path": str(manifest), "manifest_rows": len(rows), "class_count": len(labels),
        "source_dataset_counts": dict(sorted(source_counts.items())), "license_counts": dict(sorted(license_counts.items())),
        "metadata_completeness": completeness, "class_metadata_completeness": class_profiles,
        "rows_in_recovery_queue": len(recovery_rows), "duplicate_manifest_paths": duplicates,
        "recovery_queue_csv": str(queue),
        "recovery_policy": {"automatic_metadata_inference": False, "source_manifest_modified": False, "missing_values_representation": "empty; unknown is not equivalent to a negative value", "evidence_required_for_recovery": True, "approved_values_should_be_added_to_a_new_manifest_version": True},
        "notes": [
            "This audit measures recorded metadata coverage; it does not establish the real-world origin of any image.",
            "A dataset name or repository location is not evidence that images were captured in Karnataka or any other region.",
            "Do not infer location, date, variety, growth stage, plant, field, or capture session from image appearance or folder names.",
            "Enter proposed values only when supported by traceable source documentation or qualified human review; record evidence and reviewer details.",
            "The recovery queue is separate. Proposed fields are blank by design and do not alter the source manifest.",
            "No images, labels, splits, checkpoints, training runs, or held-out evaluation results are modified.",
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
