from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = {"image_path", "disease_label", "label_status"}


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curate_exact_duplicates(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create a conservative, non-destructive manifest using exact SHA-256 groups.

    For exact duplicates with one consistent class label, keep the
    lexicographically first path as the representative. For exact duplicates
    carrying multiple labels, exclude every member from the curated manifest
    and place every member in a pending human-review CSV.

    Source images and the source manifest are never modified. Label statuses
    are preserved; this function never upgrades unverified labels.
    """
    root = Path(dataset_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest}")

    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - set(fields))
        if missing:
            raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
        rows = list(reader)

    # Validate paths and hash each unique manifest path only once.
    valid: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    hash_to_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row_number, original in enumerate(rows, start=2):
        row = {key: (value or "") for key, value in original.items()}
        rel = row.get("image_path", "").strip().replace("\\", "/")
        label = row.get("disease_label", "").strip()
        if not rel or not label:
            invalid.append({
                "row_number": str(row_number),
                "image_path": rel,
                "reason": "missing_image_path_or_disease_label",
            })
            continue
        if rel in seen_paths:
            invalid.append({
                "row_number": str(row_number),
                "image_path": rel,
                "reason": "duplicate_manifest_path",
            })
            continue
        seen_paths.add(rel)

        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            invalid.append({
                "row_number": str(row_number),
                "image_path": rel,
                "reason": "image_path_outside_dataset_root",
            })
            continue
        if not candidate.is_file():
            invalid.append({
                "row_number": str(row_number),
                "image_path": rel,
                "reason": "image_file_not_found",
            })
            continue

        row["image_path"] = rel
        try:
            digest = _sha256(candidate)
        except OSError as exc:
            invalid.append({
                "row_number": str(row_number),
                "image_path": rel,
                "reason": f"sha256_read_error:{type(exc).__name__}",
            })
            continue
        row["_sha256"] = digest
        valid.append(row)
        hash_to_rows[digest].append(row)

    output.mkdir(parents=True, exist_ok=True)
    curated_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    duplicate_groups = 0
    same_class_groups = 0
    cross_class_groups = 0
    same_class_removed = 0
    cross_class_excluded = 0

    # Process by hash for deterministic group identifiers and output ordering.
    for digest in sorted(hash_to_rows):
        members = sorted(hash_to_rows[digest], key=lambda item: item["image_path"])
        labels = sorted({member["disease_label"].strip() for member in members})
        is_duplicate = len(members) > 1
        is_cross_class = len(labels) > 1
        group_id = f"sha256-{digest[:16]}" if is_duplicate else ""

        if is_duplicate:
            duplicate_groups += 1
            if is_cross_class:
                cross_class_groups += 1
                cross_class_excluded += len(members)
                for member in members:
                    review_rows.append({
                        "group_id": group_id,
                        "sha256": digest,
                        "image_path": member["image_path"],
                        "disease_label": member["disease_label"],
                        "group_size": str(len(members)),
                        "group_labels": "|".join(labels),
                        "review_status": "pending",
                        "review_decision": "",
                        "reviewed_by": "",
                        "review_notes": "",
                    })
                    audit_rows.append({
                        "image_path": member["image_path"],
                        "sha256": digest,
                        "group_id": group_id,
                        "disease_label": member["disease_label"],
                        "disposition": "excluded",
                        "reason": "exact_duplicate_cross_class_label_conflict",
                    })
                continue

            same_class_groups += 1
            representative = members[0]
            curated_rows.append({key: value for key, value in representative.items() if key != "_sha256"})
            audit_rows.append({
                "image_path": representative["image_path"],
                "sha256": digest,
                "group_id": group_id,
                "disease_label": representative["disease_label"],
                "disposition": "included",
                "reason": "same_class_exact_duplicate_representative",
            })
            for duplicate in members[1:]:
                same_class_removed += 1
                audit_rows.append({
                    "image_path": duplicate["image_path"],
                    "sha256": digest,
                    "group_id": group_id,
                    "disease_label": duplicate["disease_label"],
                    "disposition": "excluded",
                    "reason": "same_class_exact_duplicate_of_representative",
                })
            continue

        member = members[0]
        curated_rows.append({key: value for key, value in member.items() if key != "_sha256"})
        audit_rows.append({
            "image_path": member["image_path"],
            "sha256": digest,
            "group_id": "",
            "disease_label": member["disease_label"],
            "disposition": "included",
            "reason": "unique_sha256",
        })

    curated_path = output / "curated_manifest.csv"
    review_path = output / "cross_class_duplicate_review.csv"
    audit_path = output / "curation_audit.csv"
    summary_path = output / "curation_summary.json"

    def write_csv(path: Path, columns: list[str], records: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

    write_csv(curated_path, fields, curated_rows)
    write_csv(review_path, [
        "group_id", "sha256", "image_path", "disease_label", "group_size",
        "group_labels", "review_status", "review_decision", "reviewed_by", "review_notes",
    ], review_rows)
    write_csv(audit_path, [
        "image_path", "sha256", "group_id", "disease_label", "disposition", "reason",
    ], audit_rows)

    class_counts = Counter(row["disease_label"] for row in curated_rows)
    report = {
        "dataset_root": str(root),
        "source_manifest": str(manifest),
        "source_rows": len(rows),
        "valid_source_images_hashed": len(valid),
        "invalid_rows_excluded": len(invalid),
        "unique_sha256_hashes": len(hash_to_rows),
        "exact_duplicate_groups": duplicate_groups,
        "same_class_duplicate_groups": same_class_groups,
        "cross_class_duplicate_groups": cross_class_groups,
        "same_class_duplicate_copies_removed": same_class_removed,
        "cross_class_duplicate_images_excluded_pending_review": cross_class_excluded,
        "curated_manifest_rows": len(curated_rows),
        "curated_class_counts": dict(sorted(class_counts.items())),
        "label_status_counts": dict(sorted(Counter(
            row.get("label_status", "").strip().lower() for row in curated_rows
        ).items())),
        "outputs": {
            "curated_manifest": str(curated_path),
            "cross_class_review": str(review_path),
            "audit": str(audit_path),
            "summary": str(summary_path),
        },
        "invalid_rows": invalid,
        "notes": [
            "Exact duplicates are identified by SHA-256 byte equality, not perceptual similarity.",
            "For same-class exact duplicates, the lexicographically first image_path is retained as the representative.",
            "Every member of a cross-class exact-duplicate group is excluded pending human label review.",
            "No source image or source manifest is modified or deleted.",
            "Original label_status values are preserved; unverified labels are never promoted to verified.",
            "Review decisions in the cross-class CSV are blank and require a human reviewer; do not fill them by guessing.",
            "Near-duplicate perceptual-hash candidates are not automatically excluded by this command.",
        ],
    }
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
