"""Human review queue and contact sheets for held-out model errors."""
from __future__ import annotations
import csv, json, math
from collections import Counter
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw, ImageFont

REVIEW_STATUSES = ("label_correct", "label_suspect", "ambiguous", "poor_quality", "needs_expert_review")
REVIEW_COLUMNS = ["image_path", "true_label", "predicted_label", "confidence", "correct",
                  "review_status", "corrected_label", "reviewer", "review_notes"]

def _safe_image_path(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as exc:
        raise ValueError(f"Image path escapes dataset root: {relative_path}") from exc
    return candidate

def _read_errors(csv_path: Path) -> list[dict[str, str]]:
    required = {"image_path", "true_label", "predicted_label", "confidence"}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Misclassification CSV is missing columns: {sorted(missing)}")
        rows, seen = [], set()
        for line_number, raw in enumerate(reader, start=2):
            path = (raw.get("image_path") or "").strip()
            true_label = (raw.get("true_label") or "").strip()
            predicted_label = (raw.get("predicted_label") or "").strip()
            if not path or not true_label or not predicted_label:
                raise ValueError(f"Missing required value in CSV row {line_number}")
            if path in seen:
                raise ValueError(f"Duplicate image_path in misclassification CSV: {path}")
            seen.add(path)
            try:
                confidence = float(raw["confidence"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid confidence in CSV row {line_number}") from exc
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0 and 1 in CSV row {line_number}")
            correct_text = (raw.get("correct") or "false").strip().lower()
            if correct_text in {"true", "1", "yes"}:
                continue
            if correct_text not in {"false", "0", "no", ""}:
                raise ValueError(f"Invalid correct value in CSV row {line_number}: {correct_text}")
            rows.append({"image_path": path, "true_label": true_label, "predicted_label": predicted_label,
                         "confidence": f"{confidence:.8f}", "correct": "false", "review_status": "",
                         "corrected_label": "", "reviewer": "", "review_notes": ""})
    return rows

def _font(size: int = 16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

def _make_contact_sheets(rows, dataset_root, output, *, per_sheet=12, columns=3):
    if per_sheet < 1 or columns < 1:
        raise ValueError("per_sheet and columns must be positive")
    sheet_names, missing_images = [], []
    cell_width, cell_height, thumb_width, thumb_height = 420, 350, 390, 260
    font, small_font = _font(15), _font(13)
    for start in range(0, len(rows), per_sheet):
        batch = rows[start:start + per_sheet]
        sheet_number = start // per_sheet + 1
        sheet = Image.new("RGB", (columns * cell_width, math.ceil(len(batch) / columns) * cell_height), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, item in enumerate(batch):
            col, row_index = offset % columns, offset // columns
            x, y = col * cell_width, row_index * cell_height
            draw.rectangle((x + 4, y + 4, x + cell_width - 4, y + cell_height - 4), outline="#999999", width=2)
            try:
                image_path = _safe_image_path(dataset_root, item["image_path"])
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((thumb_width, thumb_height))
                sheet.paste(image, (x + (cell_width - image.width) // 2,
                                    y + 10 + (thumb_height - image.height) // 2))
            except (FileNotFoundError, OSError, ValueError):
                missing_images.append(item["image_path"])
                draw.rectangle((x + 15, y + 15, x + cell_width - 15, y + 10 + thumb_height), fill="#eeeeee")
                draw.text((x + 30, y + 125), "IMAGE MISSING / UNREADABLE", fill="#990000", font=font)
            caption_y = y + 280
            captions = [f'{start + offset + 1}. {item["image_path"]}',
                        f'True: {item["true_label"]} | Predicted: {item["predicted_label"]}',
                        f'Confidence: {float(item["confidence"]):.2%}']
            for line_index, caption in enumerate(captions):
                draw.text((x + 12, caption_y + line_index * 20), caption[:58],
                          fill="black", font=font if line_index < 2 else small_font)
        name = f"contact_sheet_{sheet_number:03d}.jpg"
        sheet.save(output / name, quality=90)
        sheet_names.append(name)
    return sheet_names, missing_images

def create_misclassification_review(dataset_dir: str | Path, misclassified_csv: str | Path,
                                    output_dir: str | Path, *, per_sheet: int = 12,
                                    columns: int = 3) -> dict[str, Any]:
    """Create a blank human-review CSV, contact sheets, and an audit summary."""
    dataset_root = Path(dataset_dir).expanduser().resolve()
    source_csv = Path(misclassified_csv).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {dataset_root}")
    if not source_csv.is_file():
        raise FileNotFoundError(f"Misclassification CSV does not exist: {source_csv}")
    rows = _read_errors(source_csv)
    for item in rows:
        _safe_image_path(dataset_root, item["image_path"])
    output.mkdir(parents=True, exist_ok=True)
    review_csv = output / "review_queue.csv"
    with review_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    sheets, missing = _make_contact_sheets(rows, dataset_root, output, per_sheet=per_sheet, columns=columns)
    confusion_counts = Counter((row["true_label"], row["predicted_label"]) for row in rows)
    summary = {
        "review_items": len(rows),
        "high_confidence_errors_at_or_above_0_90": sum(float(row["confidence"]) >= 0.90 for row in rows),
        "true_to_predicted_counts": [{"true_label": a, "predicted_label": b, "count": n}
                                     for (a, b), n in sorted(confusion_counts.items())],
        "missing_or_unreadable_images": missing,
        "review_status_options": list(REVIEW_STATUSES),
        "review_queue_csv": str(review_csv), "contact_sheets": sheets,
        "important": ["Review decisions are blank; no labels are changed automatically.",
                      "A high-confidence error is a review priority, not evidence that the source label is wrong.",
                      "Do not tune on the held-out test set and then describe it as an untouched test."]
    }
    (output / "review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    guide = """# Misclassification review guide

Review each image in the contact sheets and record decisions in review_queue.csv.

Allowed review_status values:
- label_correct: supplied label appears supported; model prediction is an error.
- label_suspect: supplied label may be incorrect; record evidence and seek verification.
- ambiguous: image does not support a confident visual decision.
- poor_quality: image is unsuitable for reliable assessment.
- needs_expert_review: an agricultural/domain expert is needed.

Only fill corrected_label when a reviewer has sufficient evidence and records the basis in
review_notes. Record the reviewer. Never copy predicted_label into corrected_label merely
because the model was confident.

This review output does not modify source images, source manifest, split CSVs, or evaluation
metrics. If reviewed samples are later used for training or model selection, create a new
experiment and keep a separate, untouched final test set.
"""
    (output / "REVIEW_GUIDE.md").write_text(guide, encoding="utf-8")
    return summary
