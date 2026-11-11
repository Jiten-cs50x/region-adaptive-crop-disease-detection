from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_dir: str | Path, output_dir: str | Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Audit class-folder images without modifying source files."""
    config = config or {}
    dataset_cfg = config.get("dataset", {})
    validation_cfg = config.get("validation", {})
    root = Path(dataset_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    output.mkdir(parents=True, exist_ok=True)

    extensions = {x.lower() for x in dataset_cfg.get("image_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"])}
    min_width = int(dataset_cfg.get("min_width", 64))
    min_height = int(dataset_cfg.get("min_height", 64))
    max_bytes = float(dataset_cfg.get("max_file_size_mb", 30)) * 1024 * 1024
    hash_files = bool(validation_cfg.get("compute_sha256", True))
    rows = []
    class_counts: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    skipped = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            skipped += 1
            continue
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        class_name = parts[0] if len(parts) > 1 else "UNLABELLED_FLAT_FOLDER"
        class_counts[class_name] += 1
        row = {"image_path": relative, "class_name": class_name, "status": "valid", "width": "", "height": "", "format": "", "file_size_bytes": path.stat().st_size, "sha256": "", "issues": ""}
        issues = []
        if row["file_size_bytes"] > max_bytes:
            issues.append("file_exceeds_max_size")
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                row["width"], row["height"] = image.size
                row["format"] = image.format or ""
            if row["width"] < min_width or row["height"] < min_height:
                issues.append("below_min_dimensions")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            issues.append(f"corrupt_or_unreadable:{type(exc).__name__}")
        if hash_files:
            try:
                row["sha256"] = _sha256(path)
                hashes[row["sha256"]].append(relative)
            except OSError:
                issues.append("hash_read_error")
        if issues:
            row["status"] = "review"
            row["issues"] = ";".join(issues)
        rows.append(row)

    duplicate_groups = [items for items in hashes.values() if len(items) > 1]
    valid_count = sum(row["status"] == "valid" for row in rows)
    report = {"dataset_root": str(root), "total_images": len(rows), "valid_images": valid_count, "images_requiring_review": len(rows) - valid_count, "skipped_non_image_files": skipped, "class_counts": dict(sorted(class_counts.items())), "exact_duplicate_groups": duplicate_groups, "exact_duplicate_extra_files": sum(len(group) - 1 for group in duplicate_groups), "notes": ["File validation does not verify disease labels or biological diagnoses.", "SHA-256 detects byte-identical files only.", "Original dataset files are not modified or deleted.", "Class names are inferred from the first directory below the dataset root."]}
    (output / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["image_path", "class_name", "status", "width", "height", "format", "file_size_bytes", "sha256", "issues"]
    with (output / "image_inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return report
