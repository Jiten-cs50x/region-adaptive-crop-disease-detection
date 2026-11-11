# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-verified feedback. Current components cover dataset auditing, metadata integrity, leakage-aware splitting, duplicate and quality screening, EfficientNet-B0 setup and image transforms, and a generic training loop.

## Current status: v0.8.0
- Dataset inventory, metadata manifest validation, duplicate and quality screening
- Deterministic group-aware train/validation/test splits
- EfficientNet-B0 model factory and configurable image transforms
- Manifest-backed PyTorch Dataset and DataLoaders
- Reproducible training, validation monitoring, AdamW, ReduceLROnPlateau, early stopping, and checkpoints
- Test split is deliberately not loaded by the training command

## Install (Windows)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .

Install compatible PyTorch and torchvision versions using https://pytorch.org/get-started/locally/ (select CPU or CUDA wheels), then install development tools:

    pip install -e ".[dev]"

## Prepare data

The manifest must contain the required metadata fields, including image_path, disease_label, image_type, source_dataset, license, and label_status. Verified rows should include verified_by. Keep source images in the dataset directory and paths in the manifest relative to that directory.

    crop-adaptive manifest-template --output datasets/sugarcane/dataset_manifest.csv
    crop-adaptive validate-manifest --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/reports/manifest_validation.json
    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/splits --seed 42

The split command defaults to verified labels only and keeps shared plant, field, and capture-session IDs together. Review split_report.json and ensure each configured class has adequate representation in training. Images without group identifiers cannot be protected against unknown group leakage.

## Train EfficientNet-B0

    crop-adaptive train --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --output runs/sugarcane_efficientnet_b0

Optional overrides:

    crop-adaptive train --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --output runs/experiment_01 --epochs 40 --batch-size 8 --learning-rate 0.0001 --device cuda

Use --no-pretrained to avoid downloading ImageNet weights on first use. The default device is CUDA when available, otherwise CPU. On Windows, begin with num_workers: 0; increase only after confirming stable DataLoader behavior.

Training outputs:
- best_model.pt: checkpoint with the lowest validation loss
- last_checkpoint.pt: most recent epoch checkpoint
- training_history.json: per-epoch train/validation loss, accuracy, sample counts, and learning rate
- training_summary.json: device, sample counts, class mapping, best epoch, and output paths

The training command checks image path and plant/field/session group separation between train and validation. It fails if configured taxonomy classes are absent from the training split or if labels are outside the taxonomy. It does not read test.csv.

## Model and image transforms

EfficientNet-B0 receives 224 x 224 RGB tensors normalized with ImageNet statistics. Training uses conservative spatial and colour augmentation. Validation and test preprocessing is deterministic. Inspect transformed samples and calibrate augmentation against real sugarcane images before training.

The six classes in the example configuration are provisional foliar categories. Confirm the actual disease taxonomy and image modalities with reliable dataset documentation and domain expertise. A leaf-only model cannot be assumed to diagnose diseases requiring stalk, root, or whole-plant evidence.

## Research integrity and limitations

- Dataset split files and reports are inputs to the training workflow; do not edit their assignments casually.
- Validation metrics are for model selection, not final research claims.
- Keep test data untouched until the separate evaluation milestone.
- For region-adaptation research, preserve an independent Karnataka regional holdout; do not use it for training, tuning, or early stopping.
- Group-aware splitting depends on complete and accurate plant, field, and capture-session metadata.
- Keep datasets and model weights out of Git; check third-party dataset licenses before redistribution.

## Roadmap

1. Dataset auditing, metadata, leakage-aware splits, duplicate and quality screening
2. EfficientNet-B0 model factory and image transforms
3. Generic training pipeline
4. Evaluation metrics, confusion matrix, per-class analysis, and error review
5. Independent Karnataka regional holdout and regional fine-tuning
6. Human-reviewed correction workflow and controlled retraining
7. Python API and package release
