from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from crop_adaptive.data.near_duplicates import IMAGE_EXTENSIONS


def analyze_image_quality(
    dataset_dir: str | Path,
    output_path: str | Path,
    min_width: int = 128,
    min_height: int = 128,
    min_brightness: float = 35.0,
    max_brightness: float = 220.0,
    min_contrast: float = 20.0,
    min_sharpness: float = 10.0,
    analysis_size: int = 256,
) -> dict[str, Any]:
    """Measure basic image-quality signals and flag images for human review.

    Sharpness is approximated by the variance of a FIND_EDGES response.
    Metrics are screening signals, not universal acceptance criteria.
    Source images are never modified.
    """
    root = Path(dataset_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if min_width < 1 or min_height < 1 or analysis_size < 16:
        raise ValueError("Image dimensions and analysis_size must be positive (analysis_size >= 16)")
    if not 0 <= min_brightness < max_brightness <= 255:
        raise ValueError("Brightness thresholds must satisfy 0 <= min < max <= 255")
    if min_contrast < 0 or min_sharpness < 0:
        raise ValueError("Contrast and sharpness thresholds cannot be negative")

    thresholds = {
        "min_width": min_width,
        "min_height": min_height,
        "min_brightness": min_brightness,
        "max_brightness": max_brightness,
        "min_contrast": min_contrast,
        "min_sharpness": min_sharpness,
        "analysis_size": analysis_size,
    }
    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                image.thumbnail((analysis_size, analysis_size), Image.Resampling.LANCZOS)
                gray = image.convert("L")
                stats = ImageStat.Stat(gray)
                brightness = round(float(stats.mean[0]), 3)
                contrast = round(float(stats.stddev[0]), 3)
                edge_stats = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES))
                sharpness = round(float(edge_stats.var[0]), 3)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append({"image_path": relative, "error": type(exc).__name__})
            continue

        flags: list[str] = []
        if width < min_width or height < min_height:
            flags.append("low_resolution")
        if brightness < min_brightness:
            flags.append("too_dark")
        elif brightness > max_brightness:
            flags.append("too_bright")
        if contrast < min_contrast:
            flags.append("low_contrast")
        if sharpness < min_sharpness:
            flags.append("possibly_blurry")

        records.append({
            "image_path": relative,
            "width": width,
            "height": height,
            "brightness_mean": brightness,
            "contrast_stddev": contrast,
            "edge_variance_sharpness": sharpness,
            "quality_flags": flags,
            "manual_review_recommended": bool(flags),
        })

    flagged_count = sum(bool(row["quality_flags"]) for row in records)
    report = {
        "dataset_root": str(root),
        "image_count_analyzed": len(records),
        "unreadable_image_count": len(unreadable),
        "flagged_image_count": flagged_count,
        "thresholds": thresholds,
        "metrics": {
            "brightness_mean": "Mean grayscale intensity, range 0-255.",
            "contrast_stddev": "Grayscale intensity standard deviation; low values indicate limited tonal variation.",
            "edge_variance_sharpness": "Variance of Pillow FIND_EDGES response; a rough sharpness proxy, not a calibrated blur score.",
        },
        "images": records,
        "unreadable_images": unreadable,
        "notes": [
            "Flags are screening signals for manual review, not proof that an image is unusable.",
            "Natural lighting, symptoms, camera distance, and crop stage can affect these metrics.",
            "Thresholds should be calibrated on representative images from the target crop and capture conditions.",
            "No image is deleted, modified, relabeled, or automatically excluded from training.",
            "Sharpness is an approximate edge-variance measure and should not be compared across unrelated pipelines as a calibrated score.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image_path", "width", "height", "brightness_mean",
                "contrast_stddev", "edge_variance_sharpness",
                "quality_flags", "manual_review_recommended",
            ],
        )
        writer.writeheader()
        for row in records:
            csv_row = dict(row)
            csv_row["quality_flags"] = ";".join(row["quality_flags"])
            writer.writerow(csv_row)
    report["csv_report"] = str(csv_path)
    return report
