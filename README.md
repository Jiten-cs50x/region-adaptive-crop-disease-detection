# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-verified feedback. The current implementation covers dataset auditing, metadata integrity, leakage-aware splitting, duplicate and quality screening, EfficientNet-B0 model setup, and image transforms. Training and regional adaptation are planned, not yet implemented.

## Current status: v0.7.0
- YAML-based project and dataset configuration
- Class-folder image inventory, readability checks, and exact duplicate detection
- CSV metadata manifest template and validation
- Deterministic group-aware train/validation/test split generation
- Perceptual near-duplicate candidate detection
- Image-quality screening: resolution, brightness, contrast, and approximate sharpness
- Configurable EfficientNet-B0 factory
- Configurable train-time augmentation and deterministic validation/test/prediction transforms
- EXIF orientation correction, RGB conversion, tensor conversion, and ImageNet normalization

## Quick start (Windows)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .

Install compatible PyTorch and torchvision versions for your machine. Follow https://pytorch.org/get-started/locally/ to choose CPU or CUDA wheels. Then install development tools:

    pip install -e ".[dev]"

## Image preprocessing and augmentation

The sugarcane configuration uses 224 x 224 model inputs and ImageNet normalization (mean [0.485, 0.456, 0.406], standard deviation [0.229, 0.224, 0.225]).

- Training: EXIF correction, RGB conversion, RandomResizedCrop with a conservative 0.85–1.0 area scale, horizontal flip, up to 8-degree rotation, mild brightness/contrast/saturation jitter, tensor conversion, and normalization.
- Validation/test/prediction: EXIF correction, RGB conversion, resize shorter side to 256, center crop to 224, tensor conversion, and normalization. These transforms are deterministic.
- Vertical flipping and aggressive hue/colour changes are omitted because leaf orientation and symptom colour can carry diagnostic information.

Use the configured transforms:

    import yaml
    from crop_adaptive.data.transforms import build_transforms, build_transforms_from_config

    with open("configs/sugarcane_karnataka.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    train_transform = build_transforms(config, split="train")
    validation_transform = build_transforms(config, split="validation")
    all_transforms = build_transforms_from_config(config)

The functions accept PIL images and return normalized PyTorch tensors. PyTorch and torchvision are optional dependencies and must be installed as a compatible pair. Inspect transformed samples and calibrate augmentation strength against real sugarcane images before training.

## Configure EfficientNet-B0

The sugarcane configuration enables ImageNet pretrained weights, uses a 224 x 224 input, derives classifier outputs from taxonomy.classes, and sets classifier dropout to 0.2. The six foliar classes are provisional. Confirm actual labels, image modalities, and the expert-reviewed taxonomy before training. A leaf-only model should not be expected to diagnose diseases requiring stalk, root, or whole-plant evidence.

Inspect the model:

    crop-adaptive model-info --config configs/sugarcane_karnataka.yaml

Avoid downloading pretrained weights:

    crop-adaptive model-info --config configs/sugarcane_karnataka.yaml --no-pretrained

## Dataset workflow

    crop-adaptive validate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports
    crop-adaptive manifest-template --output datasets/sugarcane/dataset_manifest.csv
    crop-adaptive validate-manifest --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/reports/manifest_validation.json
    crop-adaptive near-duplicates --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports/near_duplicates.json --threshold 6
    crop-adaptive quality --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports/image_quality.json
    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/splits --seed 42

Splits default to 70% train, 15% validation, and 15% test. Only verified labels are included by default. Plant, field, and capture-session links are kept in one split. Keep a separately collected Karnataka regional holdout untouched until final evaluation.

## Research integrity and limitations

- Quality flags and perceptual similarity are screening signals; manually review candidates.
- Group-aware splitting reduces leakage only when identifiers are accurate and sufficiently complete.
- Images without group identifiers are treated as independent samples; unknown relationships may still leak.
- Keep datasets and trained weights out of Git.
- Check third-party dataset licenses before redistribution.
- The six sugarcane foliar classes are provisional; validate them with dataset evidence and domain experts.
- A leaf-only classifier cannot be assumed to detect diseases requiring stalk, root, or whole-plant evidence.

## Roadmap

1. Dataset auditing, metadata, leakage-aware splits, duplicate and quality screening
2. EfficientNet-B0 model factory and image transforms
3. Generic training loop, checkpointing, and evaluation metrics
4. Independent Karnataka regional holdout and regional fine-tuning
5. Human-reviewed correction queue and controlled retraining
6. Python API and package release
