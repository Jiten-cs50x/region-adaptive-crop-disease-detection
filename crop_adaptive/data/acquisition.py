"""Non-destructive audit of image acquisition patterns across dataset labels."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from crop_adaptive.data.near_duplicates import IMAGE_EXTENSIONS

REQUIRED_COLUMNS = {"image_path", "disease_label"}
METADATA_FIELDS = (
    "source_dataset", "image_type", "country", "region", "state", "district",
    "collection_date", "variety", "growth_stage", "plant_id", "field_id",
    "capture_session_id",
)


def _safe_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes dataset root: {relative}") from exc
    return candidate


def _image_features(path: Path) -> dict[str, Any]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        gray = image.convert("L")
        gray_stats = ImageStat.Stat(gray)
        rgb_stats = ImageStat.Stat(image)
        edge_variance = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).var[0]
        brightness = float(gray_stats.mean[0])
        contrast = float(gray_stats.stddev[0])
        mean_rgb = [round(float(value), 3) for value in rgb_stats.mean]
        aspect = width / height
        orientation = "landscape" if aspect > 1.05 else "portrait" if aspect < 0.95 else "square"
        min_side = min(width, height)
        resolution_tier = "small_<224" if min_side < 224 else "medium_224_to_511" if min_side < 512 else "large_>=512"
        brightness_band = "dark_<85" if brightness < 85 else "mid_85_to_170" if brightness < 170 else "bright_>=170"
        return {
            "width": width, "height": height, "aspect_ratio": round(aspect, 5),
            "orientation": orientation, "resolution_tier": resolution_tier,
            "brightness_mean": round(brightness, 3), "brightness_band": brightness_band,
            "contrast_stddev": round(contrast, 3),
            "edge_variance_sharpness": round(float(edge_variance), 3),
            "mean_red": mean_rgb[0], "mean_green": mean_rgb[1], "mean_blue": mean_rgb[2],
        }


def _median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 3) if values else None


def _summarize(rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(group_field) or "(missing)"].append(row)
    output = []
    numeric_fields = ("width", "height", "aspect_ratio", "brightness_mean",
                      "contrast_stddev", "edge_variance_sharpness")
    for group, items in sorted(groups.items()):
        record: dict[str, Any] = {"group": group, "image_count": len(items)}
        for field in numeric_fields:
            record[f"median_{field}"] = _median([float(item[field]) for item in items])
        for field in ("orientation", "resolution_tier", "brightness_band"):
            record[f"{field}_counts"] = dict(sorted(Counter(item[field] for item in items).items()))
        output.append(record)
    return output


def audit_acquisition_patterns(
    dataset_dir: str | Path, manifest_path: str | Path, output_path: str | Path,
) -> dict[str, Any]:
    """Report image acquisition characteristics using manifest labels, without modifying data."""
    root = Path(dataset_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest}")

    rows: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    missing: list[str] = []
    seen: set[str] = set()
    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        absent = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"Manifest is missing required columns: {sorted(absent)}")
        for line_number, raw in enumerate(reader, start=2):
            relative = (raw.get("image_path") or "").strip()
            label = (raw.get("disease_label") or "").strip()
            if not relative or not label:
                raise ValueError(f"Missing image_path or disease_label in manifest row {line_number}")
            if relative in seen:
                raise ValueError(f"Duplicate image_path in manifest: {relative}")
            seen.add(relative)
            image_path = _safe_path(root, relative)
            if not image_path.is_file():
                missing.append(relative)
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                unreadable.append({"image_path": relative, "error": "unsupported_extension"})
                continue
            try:
                features = _image_features(image_path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                unreadable.append({"image_path": relative, "error": type(exc).__name__})
                continue
            rows.append({
                "image_path": relative, "disease_label": label,
                **{field: (raw.get(field) or "").strip() for field in METADATA_FIELDS},
                **features,
            })

    if not rows:
        raise ValueError("No readable manifest images were available for acquisition audit")

    metadata_completeness = {
        field: {
            "present": sum(bool(row.get(field)) for row in rows),
            "missing": sum(not bool(row.get(field)) for row in rows),
            "distinct_nonempty_values": len({row[field] for row in rows if row.get(field)}),
        } for field in METADATA_FIELDS
    }
    class_capture_counts = {}
    for label in sorted({row["disease_label"] for row in rows}):
        selected = [row for row in rows if row["disease_label"] == label]
        class_capture_counts[label] = {
            field: dict(sorted(Counter(row[field] for row in selected).items()))
            for field in ("orientation", "resolution_tier", "brightness_band")
        }

    report = {
        "dataset_root": str(root), "manifest_path": str(manifest),
        "images_analyzed": len(rows), "missing_image_count": len(missing),
        "unreadable_image_count": len(unreadable),
        "class_count": len({row["disease_label"] for row in rows}),
        "class_capture_profile": _summarize(rows, "disease_label"),
        "source_dataset_profile": _summarize(rows, "source_dataset"),
        "image_type_profile": _summarize(rows, "image_type"),
        "class_capture_counts": class_capture_counts,
        "metadata_completeness": metadata_completeness,
        "missing_images": missing, "unreadable_images": unreadable,
        "feature_definitions": {
            "brightness_mean": "Mean grayscale intensity on a 0-255 scale.",
            "contrast_stddev": "Standard deviation of grayscale intensity.",
            "edge_variance_sharpness": "Variance of FIND_EDGES response; a rough sharpness proxy, not calibrated blur.",
            "orientation": "Aspect-ratio bucket: portrait (<0.95), square (0.95-1.05), landscape (>1.05).",
            "resolution_tier": "Bucket based on shorter image dimension before resizing.",
            "brightness_band": "Coarse brightness bucket; descriptive, not a quality threshold.",
            "mean_rgb": "Mean RGB values over the whole image; not a background classifier.",
        },
        "notes": [
            "Labels and metadata are read from the manifest; labels are never inferred from folder names.",
            "Class-wise differences can indicate acquisition imbalance but do not prove model shortcut learning.",
            "Mean RGB and brightness summarize the whole image; they do not isolate or identify backgrounds.",
            "These statistics are not disease diagnosis, label verification, or image exclusion criteria.",
            "No source image, manifest, split, checkpoint, or evaluation metric is modified.",
            "For causal shortcut testing, use expert-reviewed data and controlled capture-condition experiments.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fields = [
        "image_path", "disease_label", *METADATA_FIELDS, "width", "height", "aspect_ratio",
        "orientation", "resolution_tier", "brightness_mean", "brightness_band",
        "contrast_stddev", "edge_variance_sharpness", "mean_red", "mean_green", "mean_blue",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report["per_image_csv"] = str(csv_path)
    return report
