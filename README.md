# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-verified feedback. The current implementation covers dataset auditing, metadata integrity, leakage-aware splitting, perceptual duplicate screening, image-quality analysis, and an EfficientNet-B0 model factory. Training and regional adaptation are planned, not yet implemented.

## Current status: v0.6.0
- YAML-based project and dataset configuration
- Class-folder image inventory and image readability checks
- SHA-256 exact duplicate detection
- CSV metadata manifest template and validation
- Deterministic group-aware train/validation/test split generation
- Perceptual near-duplicate candidate detection using difference hash
- Image-quality metrics: resolution, brightness, contrast, and approximate sharpness
- Configurable EfficientNet-B0 factory with ImageNet weights, classifier replacement, and optional frozen feature extractor
- CLI model inspection using the configured crop taxonomy

## Quick start (Windows)

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -e .
    
Install a compatible PyTorch and torchvision pair for your machine. Follow the official PyTorch selector at https://pytorch.org/get-started/locally/ to choose CPU or CUDA wheels. The package exposes a convenience extra, but installing PyTorch using the official selector is recommended to avoid mismatched CUDA builds:

    pip install -e ".[dev]"

### Configure EfficientNet-B0

The sugarcane configuration sets:
- backbone: efficientnet_b0
- ImageNet pretrained weights: enabled
- input size: 224 x 224
- classifier output: derived from taxonomy.classes (currently six provisional foliar labels)
- feature extractor freezing: disabled
- classifier dropout: 0.2

The class list is provisional. Confirm the dataset's actual disease labels, image modalities, and expert-verified taxonomy before training. A leaf-only model should not be expected to diagnose diseases that require stalk, root, or whole-plant evidence.

### Inspect the configured model

With pretrained weights (downloads ImageNet weights on first use):

    crop-adaptive model-info --config configs/sugarcane_karnataka.yaml

To construct the model with random initialization and avoid a weight download:

    crop-adaptive model-info --config configs/sugarcane_karnataka.yaml --no-pretrained

### Use the model factory in Python

    import yaml
    from crop_adaptive.models import build_model_from_config

    with open("configs/sugarcane_karnataka.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    model = build_model_from_config(config)

For a scratch-initialized model, pass pretrained_override=False. The factory returns a torchvision model that produces raw class logits; preprocessing, training, evaluation, checkpointing, and inference pipelines are separate upcoming milestones.

## Dataset workflow

### 1. Audit raw image folders

    crop-adaptive validate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports

### 2. Create and complete a metadata manifest

    crop-adaptive manifest-template --output datasets/sugarcane/dataset_manifest.csv

Fill one row per image. Use paths relative to the dataset root. Record source, license, disease label, and label verification accurately. Add plant_id, field_id, and capture_session_id wherever known.

### 3. Validate the manifest

    crop-adaptive validate-manifest --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/reports/manifest_validation.json

### 4. Find perceptual near-duplicate candidates

    crop-adaptive near-duplicates --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports/near_duplicates.json --threshold 6

Inspect candidate pairs manually; do not automatically remove them.

### 5. Analyze image quality

    crop-adaptive quality --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports/image_quality.json

This writes JSON and CSV reports. Flags are not automatic exclusions; calibrate thresholds on representative sugarcane images.

### 6. Generate group-aware splits

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/splits --seed 42

Defaults: 70% train, 15% validation, 15% test. Only verified labels are included by default. Plant, field, and capture-session links are kept in one split.

## Research integrity and limitations

- Quality flags and perceptual similarity are screening signals; manually review candidates.
- Group-aware splitting reduces leakage only when identifiers are accurate and sufficiently complete.
- Images without group identifiers are treated as independent samples; unknown relationships may still leak.
- Keep a separately collected Karnataka regional holdout untouched until final evaluation.
- The six sugarcane foliar classes are provisional and must match the final expert-reviewed dataset.
- Keep datasets and trained weights out of Git.

## Roadmap

1. Dataset audit, metadata manifest, group-aware splits, duplicate and quality screening
2. EfficientNet-B0 model factory
3. Image preprocessing and augmentation pipeline
4. Generic training loop, checkpointing, and evaluation metrics
5. Independent Karnataka regional holdout workflow and regional fine-tuning
6. Human-reviewed correction queue and controlled retraining
7. Python API and package release
