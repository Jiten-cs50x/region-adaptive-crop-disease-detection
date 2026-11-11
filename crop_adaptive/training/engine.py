from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from crop_adaptive.data.transforms import build_transforms
from crop_adaptive.models.efficientnet import build_model_from_config


class ManifestImageDataset(Dataset):
    """Read image samples from a generated split CSV without moving source files."""

    def __init__(self, dataset_dir, split_csv, class_to_idx, transform):
        self.root = Path(dataset_dir).expanduser().resolve()
        self.split_csv = Path(split_csv).expanduser().resolve()
        self.class_to_idx = class_to_idx
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split CSV does not exist: {self.split_csv}")

        with self.split_csv.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"image_path", "disease_label"}
            missing = sorted(required - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"Split CSV missing required columns: {', '.join(missing)}")
            rows = list(reader)

        self.samples = []
        seen = set()
        for line_number, row in enumerate(rows, start=2):
            relative = (row.get("image_path") or "").strip().replace("\\", "/")
            label = (row.get("disease_label") or "").strip()
            if not relative:
                raise ValueError(f"{self.split_csv}:{line_number}: image_path is empty")
            if label not in self.class_to_idx:
                raise ValueError(
                    f"{self.split_csv}:{line_number}: label '{label}' is not in taxonomy.classes"
                )
            image_path = (self.root / relative).resolve()
            try:
                image_path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError(
                    f"{self.split_csv}:{line_number}: image path escapes dataset root"
                ) from exc
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"{self.split_csv}:{line_number}: image does not exist: {image_path}"
                )
            key = str(image_path).casefold()
            if key in seen:
                raise ValueError(f"Duplicate image path in split CSV: {relative}")
            seen.add(key)
            self.samples.append((image_path, self.class_to_idx[label]))

        if not self.samples:
            raise ValueError(f"Split CSV contains no usable rows: {self.split_csv}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, target = self.samples[index]
        with Image.open(image_path) as source:
            image = source.copy()
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_split_metadata(split_csv: Path, dataset_root: Path):
    with split_csv.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        if "image_path" not in fields:
            raise ValueError(f"Split CSV has no image_path column: {split_csv}")
        paths, groups = set(), set()
        for row in reader:
            relative = (row.get("image_path") or "").strip().replace("\\", "/")
            if relative:
                resolved = (dataset_root / relative).resolve()
                try:
                    resolved.relative_to(dataset_root)
                except ValueError as exc:
                    raise ValueError(
                        f"Image path escapes dataset root in {split_csv}: {relative}"
                    ) from exc
                paths.add(str(resolved).casefold())
            for column in ("plant_id", "field_id", "capture_session_id"):
                value = (row.get(column) or "").strip()
                if value:
                    groups.add((column, value))
        return paths, groups


def _run_epoch(model, loader, criterion, optimizer, device, training: bool):
    import torch

    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; stop training and inspect the dataset.")
            if training:
                loss.backward()
                optimizer.step()
            batch_size = targets.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += batch_size

    if total == 0:
        raise RuntimeError("A DataLoader produced no samples.")
    return {"loss": total_loss / total, "accuracy": correct / total, "samples": total}


def train_from_config(
    config: dict[str, Any],
    dataset_dir: str | Path,
    splits_dir: str | Path,
    output_dir: str | Path,
    *,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    num_workers: int | None = None,
    device_name: str | None = None,
    pretrained_override: bool | None = None,
) -> dict[str, Any]:
    """Train EfficientNet-B0 and select the best checkpoint by validation loss."""
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    classes = config.get("taxonomy", {}).get("classes", [])
    if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("Configuration must define at least two unique taxonomy.classes")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 30))
    batch_size = int(batch_size if batch_size is not None else train_cfg.get("batch_size", 16))
    learning_rate = float(
        learning_rate if learning_rate is not None else train_cfg.get("learning_rate", 0.0003)
    )
    num_workers = int(
        num_workers if num_workers is not None else train_cfg.get("num_workers", 0)
    )
    patience = int(train_cfg.get("early_stopping_patience", 7))
    weight_decay = float(train_cfg.get("weight_decay", 0.0001))
    min_delta = float(train_cfg.get("min_delta", 0.0001))
    if epochs < 1 or batch_size < 1 or num_workers < 0 or patience < 1:
        raise ValueError("epochs/batch_size/patience must be positive and num_workers non-negative")
    if learning_rate <= 0 or weight_decay < 0 or min_delta < 0:
        raise ValueError("learning_rate must be positive; weight_decay and min_delta non-negative")

    root = Path(dataset_dir).expanduser().resolve()
    splits = Path(splits_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    train_csv, validation_csv = splits / "train.csv", splits / "validation.csv"
    if not train_csv.is_file() or not validation_csv.is_file():
        raise FileNotFoundError(
            f"Expected train.csv and validation.csv in {splits}. Generate group-aware splits first."
        )
    train_paths, train_groups = _read_split_metadata(train_csv, root)
    validation_paths, validation_groups = _read_split_metadata(validation_csv, root)
    path_overlap = train_paths & validation_paths
    group_overlap = train_groups & validation_groups
    if path_overlap:
        raise ValueError(f"Train/validation image leakage detected: {len(path_overlap)} shared paths.")
    if group_overlap:
        raise ValueError(
            f"Train/validation group leakage detected: {len(group_overlap)} shared plant/field/session IDs."
        )

    _set_seed(seed)
    if device_name:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch installation.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = ManifestImageDataset(
        root, train_csv, class_to_idx, build_transforms(config, "train")
    )
    validation_dataset = ManifestImageDataset(
        root, validation_csv, class_to_idx, build_transforms(config, "validation")
    )
    train_label_counts = {
        classes[index]: count
        for index, count in enumerate(
            [sum(target == i for _, target in train_dataset.samples) for i in range(len(classes))]
        )
        if count
    }
    missing_training_classes = [name for name in classes if name not in train_label_counts]
    if missing_training_classes:
        raise ValueError(
            "Training split has no samples for taxonomy classes: "
            + ", ".join(missing_training_classes)
            + ". Review labels or regenerate group-aware splits."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = build_model_from_config(config, pretrained_override=pretrained_override)
    model.to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable model parameters. Check model.freeze_backbone configuration.")
    optimizer = AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("scheduler_factor", 0.5)),
        patience=int(train_cfg.get("scheduler_patience", 2)),
    )
    criterion = nn.CrossEntropyLoss()

    output.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output / "best_model.pt", output / "last_checkpoint.pt"
    history_path = output / "training_history.json"
    history = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        validation_metrics = _run_epoch(
            model, validation_loader, criterion, optimizer, device, False
        )
        scheduler.step(validation_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train": train_metrics,
            "validation": validation_metrics,
        })
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_to_idx": class_to_idx,
            "classes": classes,
            "config": config,
            "validation_loss": validation_metrics["loss"],
            "seed": seed,
        }
        torch.save(checkpoint, last_path)
        if validation_metrics["loss"] < best_validation_loss - min_delta:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improvement += 1

        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f}, acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f}, acc {validation_metrics['accuracy']:.4f} | "
            f"lr {current_lr:.2e}"
        )
        if epochs_without_improvement >= patience:
            print(f"Early stopping: validation loss did not improve for {patience} epochs.")
            break

    summary = {
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": str(history_path),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "train_class_counts": train_label_counts,
        "class_to_idx": class_to_idx,
        "notes": [
            "The test split was not loaded or used during training.",
            "The best checkpoint is selected by minimum validation loss.",
            "Training metrics are not a substitute for independent regional evaluation.",
        ],
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
