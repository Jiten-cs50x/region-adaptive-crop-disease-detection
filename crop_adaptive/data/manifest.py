from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = [
    "image_path", "crop", "disease_label", "image_type",
    "source_dataset", "license", "label_status",
]
OPTIONAL_COLUMNS = [
    "source_url", "country", "region", "state", "district",
    "collection_date", "variety", "growth_stage", "plant_id",
    "field_id", "capture_session_id", "verified_by", "notes",
]
ALLOWED_IMAGE_TYPES = {"leaf", "whole_plant", "stalk", "root", "fruit", "other"}
ALLOWED_LABEL_STATUSES = {"verified", "unverified", "rejected"}
MANIFEST_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def create_manifest_template(output_path: str | Path) -> Path:
    """Create an empty CSV manifest with the supported metadata columns."""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
    return path


def validate_manifest(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate manifest structure, metadata values, and image references.

    Paths are interpreted relative to dataset_dir. This function never edits
    images or the source manifest.
    """
    root = Path(dataset_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in headers]
        rows = list(reader)

    issues: list[dict[str, Any]] = []
    if missing_columns:
        issues.append({
            "row": 1,
            "image_path": "",
            "issues": [f"missing_required_column:{name}" for name in missing_columns],
        })
        report = {
            "manifest": str(manifest),
            "total_rows": len(rows),
            "valid_rows": 0,
            "rows_requiring_review": len(rows),
            "missing_required_columns": missing_columns,
            "issues": issues,
            "notes": ["No source files or manifest rows were modified."],
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    seen_paths: set[str] = set()
    valid_rows = 0
    for line_number, row in enumerate(rows, start=2):
        row_issues: list[str] = []
        image_path = (row.get("image_path") or "").strip()
        if not image_path:
            row_issues.append("missing_image_path")
        else:
            normalized = image_path.replace("\\", "/")
            if normalized in seen_paths:
                row_issues.append("duplicate_manifest_image_path")
            seen_paths.add(normalized)
            candidate = (root / normalized).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                row_issues.append("image_path_outside_dataset_root")
            else:
                if not candidate.is_file():
                    row_issues.append("image_file_not_found")

        for column in REQUIRED_COLUMNS:
            if not (row.get(column) or "").strip():
                row_issues.append(f"missing_value:{column}")

        image_type = (row.get("image_type") or "").strip().lower()
        if image_type and image_type not in ALLOWED_IMAGE_TYPES:
            row_issues.append("invalid_image_type")

        label_status = (row.get("label_status") or "").strip().lower()
        if label_status and label_status not in ALLOWED_LABEL_STATUSES:
            row_issues.append("invalid_label_status")
        if label_status == "verified" and not (row.get("verified_by") or "").strip():
            row_issues.append("verified_label_requires_verified_by")

        collection_date = (row.get("collection_date") or "").strip()
        if collection_date:
            try:
                date.fromisoformat(collection_date)
            except ValueError:
                row_issues.append("collection_date_must_be_YYYY_MM_DD")

        if row_issues:
            issues.append({
                "row": line_number,
                "image_path": image_path,
                "issues": row_issues,
            })
        else:
            valid_rows += 1

    report = {
        "manifest": str(manifest),
        "dataset_root": str(root),
        "total_rows": len(rows),
        "valid_rows": valid_rows,
        "rows_requiring_review": len(issues),
        "missing_required_columns": [],
        "issues": issues,
        "label_status_counts": {
            status: sum((r.get("label_status") or "").strip().lower() == status for r in rows)
            for status in sorted(ALLOWED_LABEL_STATUSES)
        },
        "notes": [
            "A structurally valid row does not prove that its disease label is biologically correct.",
            "Only rows with label_status=verified and a verified_by value should be considered for a verified-label training subset.",
            "Record region, district, variety, growth stage, plant_id, field_id, and capture_session_id whenever known.",
            "Do not invent missing provenance or collection metadata.",
            "No source files or manifest rows were modified.",
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
