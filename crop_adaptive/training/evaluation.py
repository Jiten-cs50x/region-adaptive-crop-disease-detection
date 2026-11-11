from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any
from crop_adaptive.training.engine import ManifestImageDataset, _read_split_metadata
from crop_adaptive.data.transforms import build_transforms
from crop_adaptive.models.efficientnet import build_model_from_config

def classification_metrics(confusion: list[list[int]], classes: list[str]) -> dict[str, Any]:
    """Calculate accuracy and one-vs-rest precision/recall/F1 without sklearn."""
    if len(confusion) != len(classes) or any(len(row) != len(classes) for row in confusion):
        raise ValueError("Confusion matrix dimensions must match the class list.")
    total = sum(sum(row) for row in confusion)
    if total == 0:
        raise ValueError("Cannot calculate metrics for an empty confusion matrix.")
    per_class = {}
    precisions, recalls, f1s = [], [], []
    weighted_precision = weighted_recall = weighted_f1 = 0.0
    correct = sum(confusion[i][i] for i in range(len(classes)))
    for index, name in enumerate(classes):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(confusion[row][index] for row in range(len(classes)))
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1_score": f1,
                           "support": support, "predicted_count": predicted}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support
    return {
        "sample_count": total, "correct": correct, "incorrect": total - correct,
        "accuracy": correct / total,
        "macro_precision": sum(precisions) / len(classes),
        "macro_recall": sum(recalls) / len(classes),
        "macro_f1": sum(f1s) / len(classes),
        "weighted_precision": weighted_precision / total,
        "weighted_recall": weighted_recall / total,
        "weighted_f1": weighted_f1 / total,
        "per_class": per_class,
    }

def _assert_no_test_leakage(dataset_root: Path, splits_dir: Path) -> dict[str, int]:
    metadata = {}
    for name in ("train", "validation", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Expected {name}.csv in split directory: {splits_dir}")
        metadata[name] = _read_split_metadata(path, dataset_root)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        paths = metadata[left][0] & metadata[right][0]
        groups = metadata[left][1] & metadata[right][1]
        if paths:
            raise ValueError(f"{left}/{right} image leakage detected: {len(paths)} shared image paths.")
        if groups:
            raise ValueError(f"{left}/{right} group leakage detected: {len(groups)} shared plant/field/session IDs.")
    return {f"{name}_images": len(metadata[name][0]) for name in ("train", "validation", "test")}

def evaluate_checkpoint(config: dict[str, Any], dataset_dir: str | Path,
                        splits_dir: str | Path, checkpoint_path: str | Path,
                        output_dir: str | Path, *, batch_size: int | None = None,
                        num_workers: int | None = None, device_name: str | None = None) -> dict[str, Any]:
    """Evaluate a trusted local training checkpoint exactly once on test.csv."""
    import torch
    from torch.utils.data import DataLoader
    classes = config.get("taxonomy", {}).get("classes", [])
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Configuration must define at least two unique taxonomy.classes")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    root = Path(dataset_dir).expanduser().resolve()
    splits = Path(splits_dir).expanduser().resolve()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    split_counts = _assert_no_test_leakage(root, splits)
    test_csv = splits / "test.csv"
    train_cfg = config.get("training", {})
    batch_size = int(batch_size if batch_size is not None else train_cfg.get("batch_size", 16))
    num_workers = int(num_workers if num_workers is not None else train_cfg.get("num_workers", 0))
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if device_name:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Checkpoints use pickle metadata; load only files from this project or another trusted source.
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is not a supported crop-adaptive training checkpoint.")
    if checkpoint.get("class_to_idx") != class_to_idx:
        raise ValueError("Checkpoint class_to_idx does not match config taxonomy.classes. Use the exact training config.")
    if checkpoint.get("classes") is not None and checkpoint["classes"] != classes:
        raise ValueError("Checkpoint class order does not match config taxonomy.classes.")
    dataset = ManifestImageDataset(root, test_csv, class_to_idx, build_transforms(config, "test"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == "cuda")
    model = build_model_from_config(config, pretrained_override=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    confusion = [[0 for _ in classes] for _ in classes]
    predictions = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device, non_blocking=True))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)
            for target, pred, score in zip(targets.tolist(), predicted.cpu().tolist(), confidence.cpu().tolist()):
                confusion[target][pred] += 1
                predictions.append((target, pred, float(score)))
    if len(predictions) != len(dataset):
        raise RuntimeError("Evaluation did not produce exactly one prediction per test image.")
    metrics = classification_metrics(confusion, classes)
    output.mkdir(parents=True, exist_ok=True)
    confusion_json = {"class_order": classes, "rows_true_labels_columns_predicted_labels": confusion}
    (output / "confusion_matrix.json").write_text(json.dumps(confusion_json, indent=2), encoding="utf-8")
    with (output / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label/predicted_label", *classes])
        for name, row in zip(classes, confusion):
            writer.writerow([name, *row])
    misclassified_path = output / "misclassified.csv"
    with misclassified_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_path", "true_label", "predicted_label", "confidence"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for (image_path, _), (target, pred, score) in zip(dataset.samples, predictions):
            if target != pred:
                writer.writerow({"image_path": image_path.relative_to(root).as_posix(),
                                 "true_label": classes[target], "predicted_label": classes[pred],
                                 "confidence": f"{score:.8f}"})
    summary = {
        "checkpoint": str(checkpoint_file), "dataset_root": str(root), "test_split": str(test_csv),
        "device": str(device), "batch_size": batch_size, "class_to_idx": class_to_idx,
        "split_image_counts_checked": split_counts, "metrics": metrics,
        "artifacts": {"summary": str(output / "evaluation_summary.json"),
                      "confusion_matrix_json": str(output / "confusion_matrix.json"),
                      "confusion_matrix_csv": str(output / "confusion_matrix.csv"),
                      "misclassified_csv": str(misclassified_path)},
        "notes": [
            "Only test.csv was used for metric calculation; train.csv and validation.csv were read only for leakage checks.",
            "No model weights, thresholds, or hyperparameters were changed during evaluation.",
            "Metrics measure agreement with source-provided labels, which may be unverified.",
            "This test set is not a substitute for an independent Karnataka field-collected regional holdout.",
            "Checkpoint loading uses pickle metadata; evaluate only trusted checkpoint files."
        ],
    }
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
