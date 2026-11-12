"""Controlled, non-destructive robustness evaluation for image classifiers."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter

from crop_adaptive.training.evaluation import classification_metrics

CONDITIONS = {
    "brightness": (0.65, 0.80, 1.20, 1.40),
    "contrast": (0.60, 0.80, 1.20, 1.40),
    "blur": (0.6, 1.2, 2.0),
    "resolution": (0.75, 0.50, 0.25),
    "jpeg": (50, 25, 10),
    "rotation": (90, 180, 270),
}


def perturb_image(image: Image.Image, condition: str, severity: float | int) -> Image.Image:
    """Return a degraded copy; never mutate the caller's source image."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported robustness condition: {condition}")
    if severity not in CONDITIONS[condition]:
        raise ValueError(f"Unsupported severity {severity!r} for {condition}")
    original = image.convert("RGB")
    if condition == "brightness":
        return ImageEnhance.Brightness(original).enhance(float(severity))
    if condition == "contrast":
        return ImageEnhance.Contrast(original).enhance(float(severity))
    if condition == "blur":
        return original.filter(ImageFilter.GaussianBlur(radius=float(severity)))
    if condition == "resolution":
        width, height = original.size
        reduced = (max(1, round(width * float(severity))), max(1, round(height * float(severity))))
        small = original.resize(reduced, Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.BICUBIC)
    if condition == "jpeg":
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=int(severity), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()
    if condition == "rotation":
        return original.rotate(int(severity), resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    raise AssertionError("unreachable")


class _PerturbedDataset:
    """Apply a deterministic PIL perturbation before the normal test transform."""
    def __init__(self, base_dataset, transform, condition=None, severity=None):
        self.base = base_dataset
        self.transform = transform
        self.condition = condition
        self.severity = severity
        self.samples = base_dataset.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        from PIL import Image
        path, target = self.samples[index]
        with Image.open(path) as source:
            image = source.copy()
        if self.condition is not None:
            image = perturb_image(image, self.condition, self.severity)
        return self.transform(image), target


def _evaluate_loader(model, loader, device, classes):
    import torch
    confusion = [[0 for _ in classes] for _ in classes]
    prediction_rows = []
    sample_index = 0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            for true_idx, pred_idx, confidence in zip(
                targets.tolist(), predictions.cpu().tolist(), confidences.cpu().tolist()
            ):
                confusion[true_idx][pred_idx] += 1
                prediction_rows.append({
                    "sample_index": sample_index,
                    "true_label": classes[true_idx],
                    "predicted_label": classes[pred_idx],
                    "confidence": float(confidence),
                    "correct": true_idx == pred_idx,
                })
                sample_index += 1
    return classification_metrics(confusion, classes), prediction_rows


def evaluate_robustness_from_config(
    config: dict[str, Any],
    dataset_dir: str | Path,
    splits_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate clean and controlled degraded copies of the unchanged test split."""
    import torch
    from torch.utils.data import DataLoader
    from crop_adaptive.data.transforms import build_transforms
    from crop_adaptive.models.efficientnet import build_model_from_config
    from crop_adaptive.training.engine import ManifestImageDataset, _read_split_metadata

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    root = Path(dataset_dir).expanduser().resolve()
    splits = Path(splits_dir).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    test_csv = splits / "test.csv"
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    if not test_csv.is_file():
        raise FileNotFoundError(f"Expected test.csv in split directory: {splits}")

    test_paths, test_groups = _read_split_metadata(test_csv, root)
    for split_name in ("train", "validation"):
        split_csv = splits / f"{split_name}.csv"
        if not split_csv.is_file():
            raise FileNotFoundError(f"Expected {split_name}.csv in split directory: {splits}")
        other_paths, other_groups = _read_split_metadata(split_csv, root)
        if test_paths & other_paths:
            raise ValueError(f"Test/{split_name} image leakage detected: {len(test_paths & other_paths)} shared paths")
        if test_groups & other_groups:
            raise ValueError(f"Test/{split_name} group leakage detected: {len(test_groups & other_groups)} shared IDs")

    if device_name:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location=device)
    classes = config.get("taxonomy", {}).get("classes", [])
    expected_mapping = {name: index for index, name in enumerate(classes)}
    if checkpoint.get("classes") != classes or checkpoint.get("class_to_idx") != expected_mapping:
        raise ValueError("Checkpoint class order/mapping does not match config taxonomy.classes")

    base = ManifestImageDataset(root, test_csv, expected_mapping, transform=None)
    transform = build_transforms(config, "test")
    loader_options = {"batch_size": batch_size, "shuffle": False, "num_workers": num_workers,
                      "pin_memory": device.type == "cuda"}
    model = build_model_from_config(config, pretrained_override=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    results = []
    prediction_details = []
    clean_loader = DataLoader(_PerturbedDataset(base, transform), **loader_options)
    clean_metrics, clean_predictions = _evaluate_loader(model, clean_loader, device, classes)
    results.append({"condition": "clean", "severity": "none", **clean_metrics})
    prediction_details.extend({"condition": "clean", **row} for row in clean_predictions)
    baseline_accuracy = clean_metrics["accuracy"]
    baseline_macro_f1 = clean_metrics["macro_f1"]

    for condition, severities in CONDITIONS.items():
        for severity in severities:
            dataset = _PerturbedDataset(base, transform, condition, severity)
            loader = DataLoader(dataset, **loader_options)
            metrics, predictions = _evaluate_loader(model, loader, device, classes)
            results.append({
                "condition": condition, "severity": severity, **metrics,
                "accuracy_drop": baseline_accuracy - metrics["accuracy"],
                "macro_f1_drop": baseline_macro_f1 - metrics["macro_f1"],
            })
            prediction_details.extend(
                {"condition": condition, "severity": severity, **row} for row in predictions
            )

    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_file),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "test_split": str(test_csv),
        "test_sample_count": len(base),
        "class_count": len(classes),
        "baseline": {
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1,
            "weighted_f1": clean_metrics["weighted_f1"],
        },
        "conditions": {name: list(levels) for name, levels in CONDITIONS.items()},
        "results": results,
        "methodology": [
            "Every condition is evaluated on the same unchanged test.csv samples and labels.",
            "Perturbations are generated in memory; source images, manifests, splits, and checkpoints are not modified.",
            "The clean result uses the same deterministic test transform as the standard evaluator.",
            "Accuracy and macro-F1 drops are absolute differences from the clean baseline.",
            "Rotation is a geometric stress test; black corner fill is introduced by fixed-canvas rotation.",
            "Resolution severity is the fraction of original width and height before resizing back to original dimensions.",
            "JPEG severity is encoder quality (lower means stronger compression).",
            "These controlled corruptions measure sensitivity, not real-world generalization or regional performance.",
            "Do not use this held-out test report to tune model weights or choose thresholds.",
        ],
        "files": {
            "metrics_csv": str(output / "robustness_metrics.csv"),
            "predictions_csv": str(output / "robustness_predictions.csv"),
            "report_json": str(output / "robustness_report.json"),
        },
    }
    metric_fields = [
        "condition", "severity", "sample_count", "accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "accuracy_drop", "macro_f1_drop",
    ]
    with (output / "robustness_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    with (output / "robustness_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["condition", "severity", "sample_index", "true_label", "predicted_label", "confidence", "correct"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_details)
    (output / "robustness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
