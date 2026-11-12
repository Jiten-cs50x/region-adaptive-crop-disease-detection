"""Association analysis for acquisition features versus source labels.

This is a descriptive bias-screening tool, not a causal shortcut detector.
It never changes labels, images, splits, checkpoints, or evaluation results.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NUMERIC_FIELDS = (
    "width", "height", "aspect_ratio", "brightness_mean",
    "contrast_stddev", "edge_variance_sharpness",
    "mean_red", "mean_green", "mean_blue",
)
CATEGORICAL_FIELDS = ("orientation", "resolution_tier", "brightness_band")


def _eta_squared(groups: dict[str, list[float]]) -> float | None:
    values = [value for group in groups.values() for value in group]
    if len(values) < 2 or len(groups) < 2:
        return None
    overall = sum(values) / len(values)
    total = sum((value - overall) ** 2 for value in values)
    if total <= 0:
        return 0.0
    between = sum(len(group) * (sum(group) / len(group) - overall) ** 2
                  for group in groups.values() if group)
    return round(max(0.0, min(1.0, between / total)), 6)


def _cramers_v(rows: list[dict[str, str]], field: str) -> float | None:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    classes = set()
    categories = set()
    for row in rows:
        label, value = row.get("disease_label", "").strip(), row.get(field, "").strip()
        if not label or not value:
            continue
        table[label][value] += 1
        classes.add(label)
        categories.add(value)
    n, r, k = sum(sum(counts.values()) for counts in table.values()), len(classes), len(categories)
    if n == 0 or min(r, k) < 2:
        return None
    row_totals = {label: sum(counts.values()) for label, counts in table.items()}
    col_totals = Counter()
    for counts in table.values():
        col_totals.update(counts)
    chi2 = 0.0
    for label in classes:
        for category in categories:
            observed = table[label][category]
            expected = row_totals[label] * col_totals[category] / n
            if expected:
                chi2 += (observed - expected) ** 2 / expected
    # Bias-corrected Cramer's V (Bergsma correction).
    phi2 = chi2 / n
    correction = ((k - 1) * (r - 1)) / (n - 1) if n > 1 else 0.0
    phi2_corrected = max(0.0, phi2 - correction)
    r_corrected = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    k_corrected = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denominator = min(k_corrected - 1, r_corrected - 1)
    if denominator <= 0:
        return 0.0
    return round(math.sqrt(phi2_corrected / denominator), 6)


def analyze_acquisition_bias(features_csv: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Measure descriptive association between class labels and acquisition features."""
    source = Path(features_csv).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Per-image acquisition CSV does not exist: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = set(reader.fieldnames or [])
        required = {"image_path", "disease_label"}
        if not required.issubset(headers):
            raise ValueError(f"Feature CSV must contain columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Feature CSV contains no image records")
    labels = [row.get("disease_label", "").strip() for row in rows]
    if any(not label for label in labels):
        raise ValueError("Every feature row must have a disease_label")

    numeric_results = {}
    for field in NUMERIC_FIELDS:
        if field not in headers:
            continue
        groups: dict[str, list[float]] = defaultdict(list)
        invalid = 0
        for row in rows:
            raw = (row.get(field) or "").strip()
            if not raw:
                continue
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError
            except ValueError:
                invalid += 1
                continue
            groups[row["disease_label"].strip()].append(value)
        numeric_results[field] = {
            "eta_squared": _eta_squared(groups),
            "valid_count": sum(map(len, groups.values())),
            "invalid_or_missing_count": len(rows) - sum(map(len, groups.values())),
            "invalid_numeric_count": invalid,
            "class_medians": {
                label: round(sorted(values)[len(values) // 2], 4)
                if len(values) % 2 else round((sorted(values)[len(values)//2 - 1] + sorted(values)[len(values)//2]) / 2, 4)
                for label, values in sorted(groups.items()) if values
            },
        }

    categorical_results = {}
    for field in CATEGORICAL_FIELDS:
        if field in headers:
            categorical_results[field] = {
                "bias_corrected_cramers_v": _cramers_v(rows, field),
                "value_counts_by_class": {
                    label: dict(sorted(Counter(
                        (row.get(field) or "").strip() or "(missing)"
                        for row in rows if row["disease_label"].strip() == label
                    ).items()))
                    for label in sorted(set(labels))
                },
            }

    report = {
        "features_csv": str(source),
        "image_count": len(rows),
        "class_count": len(set(labels)),
        "class_counts": dict(sorted(Counter(labels).items())),
        "numeric_class_associations": numeric_results,
        "categorical_class_associations": categorical_results,
        "interpretation": {
            "eta_squared": "Descriptive proportion of numeric-feature variance associated with class membership; not causal.",
            "cramers_v": "Bias-corrected association strength between a categorical feature and class; not causal.",
            "warning": "Associations can reflect disease morphology, acquisition differences, class imbalance, or confounding. They do not prove shortcut learning.",
            "background": "Whole-image mean RGB is not a background measurement. Background-specific claims require segmentation or a separately defined border/foreground analysis.",
        },
        "notes": [
            "No hypothesis-test p-values are reported; effect sizes are descriptive screening statistics.",
            "No images, labels, manifests, splits, checkpoints, or model metrics are modified.",
            "Use expert-reviewed samples and controlled interventions for causal shortcut testing.",
            "Do not use the held-out test set to choose model changes or thresholds.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
