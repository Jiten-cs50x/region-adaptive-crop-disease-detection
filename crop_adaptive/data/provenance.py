"""Evidence-gated metadata recovery for dataset manifests."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = ("image_path", "disease_label")
RECOVERY_FIELDS = (
    "source_url", "country", "region", "state", "district",
    "collection_date", "variety", "growth_stage", "plant_id",
    "field_id", "capture_session_id",
)
EVIDENCE_FIELDS = ("evidence_type", "evidence_reference", "reviewer", "review_date", "review_notes")
REVIEW_STATUSES = {"approved", "rejected", "needs_more_evidence"}


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

    fields = tuple(dict.fromkeys(("source_dataset", "license", *RECOVERY_FIELDS, "verified_by", "notes")))
    completeness = {}
    for field in fields:
        values = [(row.get(field) or "").strip() for row in rows]
        nonempty = [value for value in values if value]
        completeness[field] = {
            "present": len(nonempty), "missing": len(rows) - len(nonempty),
            "coverage_percent": round(100 * len(nonempty) / len(rows), 2),
            "distinct_nonempty_values": len(set(nonempty)),
        }
    labels = sorted({(r.get("disease_label") or "").strip() or "(missing)" for r in rows})
    class_profiles = {}
    for label in labels:
        selected = [r for r in rows if ((r.get("disease_label") or "").strip() or "(missing)") == label]
        class_profiles[label] = {
            field: {
                "present": sum(bool((r.get(field) or "").strip()) for r in selected),
                "missing": sum(not bool((r.get(field) or "").strip()) for r in selected),
            }
            for field in ("source_dataset", "license", *RECOVERY_FIELDS)
        }
    source_counts = Counter((r.get("source_dataset") or "").strip() or "(missing)" for r in rows)
    license_counts = Counter((r.get("license") or "").strip() or "(missing)" for r in rows)

    original_columns = list(headers)
    worksheet_fields = [
        *original_columns, "manifest_row_number", "missing_metadata_fields",
        *[f"proposed_{f}" for f in RECOVERY_FIELDS], "review_status", *EVIDENCE_FIELDS,
    ]
    seen, duplicates, recovery_rows = set(), [], []
    for row_number, row in enumerate(rows, start=2):
        image_path = (row.get("image_path") or "").strip()
        if image_path in seen:
            duplicates.append({"manifest_row_number": row_number, "image_path": image_path})
        seen.add(image_path)
        missing = [f for f in ("source_dataset", "license", *RECOVERY_FIELDS) if not (row.get(f) or "").strip()]
        if missing:
            item = {c: (row.get(c) or "") for c in original_columns}
            item["manifest_row_number"] = row_number
            item["missing_metadata_fields"] = ";".join(missing)
            item.update({f"proposed_{f}": "" for f in RECOVERY_FIELDS})
            item["review_status"] = ""
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
        "source_dataset_counts": dict(sorted(source_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "metadata_completeness": completeness, "class_metadata_completeness": class_profiles,
        "rows_in_recovery_queue": len(recovery_rows), "duplicate_manifest_paths": duplicates,
        "recovery_queue_csv": str(queue),
        "recovery_policy": {
            "automatic_metadata_inference": False, "source_manifest_modified": False,
            "missing_values_representation": "empty; unknown is not equivalent to a negative value",
            "evidence_required_for_recovery": True, "explicit_approval_required": True,
            "approved_values_should_be_added_to_a_new_manifest_version": True,
        },
        "notes": [
            "This audit measures recorded metadata coverage; it does not establish the real-world origin of any image.",
            "A dataset name or repository location is not evidence that images were captured in Karnataka or any other region.",
            "Do not infer location, date, variety, growth stage, plant, field, or capture session from image appearance or folder names.",
            "Every proposed value requires traceable evidence, a reviewer, a review date, and explicit approved status.",
            "The recovery queue is separate. Proposed values are blank by design and do not alter the source manifest.",
            "No images, labels, splits, checkpoints, training runs, or held-out evaluation results are modified.",
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def apply_approved_metadata_recovery(
    manifest_path: str | Path, recovery_queue_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Create a new manifest using only explicitly approved, evidence-backed proposals.

    The source manifest and recovery worksheet are read-only. Existing non-empty
    metadata is never overwritten, and output creation is exclusive.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    queue_path = Path(recovery_queue_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    for path, label in ((manifest, "Manifest"), (queue_path, "Recovery queue")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new versioned path: {output}")

    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if not set(REQUIRED_COLUMNS).issubset(columns):
            raise ValueError(f"Manifest must contain columns: {REQUIRED_COLUMNS}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("Manifest contains no data rows")

    with queue_path.open("r", newline="", encoding="utf-8-sig") as stream:
        queue_reader = csv.DictReader(stream)
        queue_columns = set(queue_reader.fieldnames or [])
        required_queue = {
            "manifest_row_number", "image_path", "disease_label", "review_status",
            *EVIDENCE_FIELDS, *[f"proposed_{field}" for field in RECOVERY_FIELDS],
        }
        missing = sorted(required_queue - queue_columns)
        if missing:
            raise ValueError(f"Recovery queue is missing required columns: {missing}")
        proposals = list(queue_reader)

    updated_rows = [dict(row) for row in source_rows]
    applied_fields = 0
    approved_rows = 0
    seen_row_numbers: set[int] = set()
    for proposal in proposals:
        values = {field: (proposal.get(f"proposed_{field}") or "").strip() for field in RECOVERY_FIELDS}
        proposed = {field: value for field, value in values.items() if value}
        if not proposed:
            continue
        try:
            row_number = int((proposal.get("manifest_row_number") or "").strip())
        except ValueError as exc:
            raise ValueError("Every proposal must contain a valid manifest_row_number") from exc
        if row_number < 2 or row_number > len(source_rows) + 1:
            raise ValueError(f"Manifest row number is out of range: {row_number}")
        if row_number in seen_row_numbers:
            raise ValueError(f"Duplicate recovery proposal for manifest row {row_number}")
        seen_row_numbers.add(row_number)
        target = updated_rows[row_number - 2]
        for identity_field in REQUIRED_COLUMNS:
            if (proposal.get(identity_field) or "").strip() != (target.get(identity_field) or "").strip():
                raise ValueError(
                    f"Recovery row {row_number} does not match source manifest {identity_field}; "
                    "regenerate the worksheet from the current manifest"
                )
        status = (proposal.get("review_status") or "").strip().lower()
        if status != "approved":
            raise ValueError(f"Proposed metadata on manifest row {row_number} requires review_status=approved")
        for field in EVIDENCE_FIELDS:
            if not (proposal.get(field) or "").strip():
                raise ValueError(f"Approved proposal on row {row_number} is missing {field}")
        try:
            date.fromisoformat((proposal.get("review_date") or "").strip())
        except ValueError as exc:
            raise ValueError(f"review_date on row {row_number} must use YYYY-MM-DD") from exc
        for field, value in proposed.items():
            if (target.get(field) or "").strip():
                raise ValueError(f"Refusing to overwrite existing {field} on manifest row {row_number}")
            if field == "collection_date":
                try:
                    date.fromisoformat(value)
                except ValueError as exc:
                    raise ValueError(f"proposed collection_date on row {row_number} must use YYYY-MM-DD") from exc
            target[field] = value
            applied_fields += 1
        approved_rows += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output_columns = list(columns)
    for field in RECOVERY_FIELDS:
        if field not in output_columns:
            output_columns.append(field)
    with output.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(updated_rows)
    return {
        "source_manifest": str(manifest), "recovery_queue": str(queue_path),
        "output_manifest": str(output), "source_rows": len(source_rows),
        "approved_rows_applied": approved_rows, "metadata_values_applied": applied_fields,
        "source_manifest_modified": False, "recovery_queue_modified": False,
        "notes": [
            "Only rows explicitly marked approved with complete evidence and reviewer metadata were applied.",
            "Existing non-empty metadata was never overwritten.",
            "This output is a new manifest version; rerun validation, curation, and splitting as a new experiment before training.",
        ],
    }
