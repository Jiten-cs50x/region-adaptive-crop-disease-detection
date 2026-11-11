# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-reviewed feedback. Current components cover dataset auditing, metadata integrity, duplicate screening, conservative curation, leakage-aware splitting, EfficientNet-B0 training, and held-out test evaluation.

## Current status: v1.0.0
- Test-only checkpoint evaluation with leakage checks, accuracy, macro/weighted precision/recall/F1, per-class metrics, confusion matrix, and misclassification CSV.
- AgML sugarcane configuration aligned to all 11 source folder labels.
- Downloaded images, generated runs, checkpoints, and local reports remain local and are not committed.

## Install (Windows)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .
    pip install -e ".[dev]"

Install compatible PyTorch and torchvision wheels using https://pytorch.org/get-started/locally/.

## Dataset curation and splits

The AgML dataset contains 6,748 leaf images in 11 source folders. The reported curation retained 5,389 images after keeping one representative from same-class exact-duplicate groups and excluding cross-class exact-duplicate groups pending review.

    crop-adaptive curate --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/curated

The source labels remain unverified. For exploratory training, create splits from the curated manifest with an explicit opt-in:

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/splits --include-unverified --seed 42

## Train

    crop-adaptive train --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --output runs/sugarcane_efficientnet_b0 --device cuda

Training reads train.csv and validation.csv only. It does not load test.csv. The best checkpoint is selected by validation loss.

## Evaluate the untouched test split

Run evaluation once after training. The command reads train.csv and validation.csv only for leakage checks; metric calculation uses only test.csv.

    crop-adaptive evaluate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --checkpoint runs/sugarcane_efficientnet_b0/best_model.pt --output runs/sugarcane_efficientnet_b0/evaluation --device cuda

Optional controls include --batch-size 8, --num-workers 0, and --device cpu.

Outputs:
- evaluation_summary.json: accuracy, macro/weighted precision, recall and F1, and per-class metrics
- confusion_matrix.csv and confusion_matrix.json
- misclassified.csv: relative image path, true label, predicted label, and confidence

The checkpoint class order must match the supplied YAML config. Only evaluate trusted local checkpoint files because PyTorch checkpoint metadata is loaded with pickle support.

## Research integrity

- All AgML manifest labels remain unverified; --include-unverified is exploratory use, not label verification.
- Preserve an independent Karnataka regional holdout; do not use it for training, tuning, or early stopping.
- A leaf-only classifier cannot be assumed to diagnose conditions requiring stalk, root, or whole-plant evidence.
- Check third-party dataset licenses before redistribution.
