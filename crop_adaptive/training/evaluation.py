from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from crop_adaptive.data.transforms import build_transforms
from crop_adaptive.models.efficientnet import build_model_from_config

def classification_metrics(confusion_matrix, classes: list[str]) -> dict[str, Any]:
    """Calculate accuracy and macro/weighted classification metrics.

    Rows in the matrix are true classes; columns are predicted classes.
    Zero-division precision/recall/F1 values are reported as 0.
    """
    size = len(classes)
    if size < 2 or len(set(classes)) != size:
        raise ValueError("classes must contain at least two unique labels")
    if len(confusion_matrix) != size or any(len(row) != size for row in confusion_matrix):
        raise ValueError("confusion matrix dimensions must match the number of classes")
    matrix = []
    for row in confusion_matrix:
        converted = []
        for value in row:
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError("confusion matrix values must be non-negative integers")
            converted.append(int(value))
        matrix.append(converted)
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("confusion matrix is empty (zero samples)")

    per_class = {}
    weighted_precision = weighted_recall = weighted_f1 = 0.0
    precisions, recalls, f1_scores = [], [], []
    for index, name in enumerate(classes):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(size))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision, "recall": recall, "f1_score": f1,
            "support": support, "predicted_count": predicted,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    return {
        "sample_count": total,
        "accuracy": sum(matrix[i][i] for i in range(size)) / total,
        "macro_precision": sum(precisions) / size,
        "macro_recall": sum(recalls) / size,
        "macro_f1": sum(f1_scores) / size,
        "weighted_precision": weighted_precision / total,
        "weighted_recall": weighted_recall / total,
        "weighted_f1": weighted_f1 / total,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": classes,
        "confusion_matrix_orientation": "rows=true labels, columns=predicted labels",
    }


def evaluate_from_config(
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
    """Evaluate a trusted local checkpoint once against the held-out test split."""
    import torch
    from torch.utils.data import DataLoader

    from crop_adaptive.training.engine import ManifestImageDataset, _read_split_metadata

    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    root = Path(dataset_dir).expanduser().resolve()
    splits = Path(splits_dir).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    test_csv = splits / "test.csv"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    if not test_csv.is_file():
        raise FileNotFoundError(f"Expected test.csv in split directory: {splits}")

    # Test paths and provenance groups must be disjoint from train and validation.
    test_paths, test_groups = _read_split_metadata(test_csv, root)
    for split_name in ("train", "validation"):
        split_csv = splits / f"{split_name}.csv"
        if not split_csv.is_file():
            raise FileNotFoundError(f"Expected {split_name}.csv in split directory: {splits}")
        other_paths, other_groups = _read_split_metadata(split_csv, root)
        shared_paths = test_paths & other_paths
        shared_groups = test_groups & other_groups
        if shared_paths:
            raise ValueError(f"Test/{split_name} image leakage detected: {len(shared_paths)} shared paths")
        if shared_groups:
            raise ValueError(
                f"Test/{split_name} group leakage detected: {len(shared_groups)} shared plant/field/session IDs"
            )

    if device_name:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Checkpoints are Python pickle containers: only load checkpoints produced locally
    # by this project or another trusted source.
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with PyTorch versions predating weights_only.
        checkpoint = torch.load(checkpoint_file, map_location=device)
    classes = config.get("taxonomy", {}).get("classes", [])
    if checkpoint.get("classes") != classes:
        raise ValueError("Checkpoint class order does not match config taxonomy.classes")
    expected_mapping = {name: index for index, name in enumerate(classes)}
    if checkpoint.get("class_to_idx") != expected_mapping:
        raise ValueError("Checkpoint class_to_idx does not match config taxonomy order")

    dataset = ManifestImageDataset(root, test_csv, expected_mapping, build_transforms(config, "test"))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model_from_config(config, pretrained_override=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    confusion = [[0 for _ in classes] for _ in classes]
    prediction_rows = []
    misclassified_rows = []
    sample_index = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            for true_index, predicted_index, confidence in zip(
                targets.tolist(), predictions.cpu().tolist(), confidences.cpu().tolist()
            ):
                image_path = str(dataset.samples[sample_index][0].relative_to(root)).replace("\\", "/")
                true_label = classes[true_index]
                predicted_label = classes[predicted_index]
                row = {
                    "image_path": image_path, "true_label": true_label,
                    "predicted_label": predicted_label, "confidence": float(confidence),
                    "correct": true_index == predicted_index,
                }
                prediction_rows.append(row)
                if true_index != predicted_index:
                    misclassified_rows.append(row)
                confusion[true_index][predicted_index] += 1
                sample_index += 1

    metrics = classification_metrics(confusion, classes)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = {
        "checkpoint": str(checkpoint_file),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "test_samples": len(dataset),
        "misclassified_samples": len(misclassified_rows),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "metrics_file": str(output / "evaluation_metrics.json"),
        "predictions_file": str(output / "predictions.csv"),
        "misclassified_file": str(output / "misclassified_images.csv"),
        "confusion_matrix_file": str(output / "confusion_matrix.csv"),
        "notes": [
            "Only the held-out test.csv was evaluated; it was not used during training or checkpoint selection.",
            "Metrics describe this dataset split and do not establish performance on independent Karnataka field data.",
            "Dataset labels are source-provided and remain unverified unless independently reviewed.",
        ],
    }
    for filename, rows in (
        ("predictions.csv", prediction_rows),
        ("misclassified_images.csv", misclassified_rows),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["image_path", "true_label", "predicted_label", "confidence", "correct"]
            )
            writer.writeheader()
            writer.writerows(rows)
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label/predicted_label", *classes])
        for label, row in zip(classes, confusion):
            writer.writerow([label, *row])
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
