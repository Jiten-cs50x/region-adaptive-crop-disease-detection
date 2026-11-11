# Project status

## Implemented in v0.1.0
- Initial package skeleton and CLI
- YAML project configuration
- Non-destructive image validation
- File inventory and exact duplicate report

## Implemented in v0.2.0
- CSV metadata manifest template and validation
- Image path boundary checks and metadata validation

## Implemented in v0.3.0
- Deterministic train/validation/test split CSV generation
- Group linking across plant_id, field_id, and capture_session_id
- Verified-label-only splitting by default
- Split class-count and grouping-coverage report

## Implemented in v0.4.0
- 64-bit difference-hash perceptual similarity screening
- BK-tree search for candidate pairs within a Hamming-distance threshold
- Candidate-pair and connected-review-group JSON report
- Unreadable image reporting; no source image modification

## Implemented in v0.5.0
- Image dimensions, grayscale brightness, contrast, and approximate sharpness analysis
- Configurable thresholds and analysis resize limit
- JSON and CSV quality reports with per-image review flags
- Non-destructive operation; no automatic deletion or dataset exclusion

## Implemented in v0.6.0
- EfficientNet-B0 model factory using torchvision
- Dynamic classifier output size from taxonomy.classes
- Optional ImageNet pretrained weights and feature-extractor freezing
- Configurable classifier dropout
- CLI model-info command and model-factory tests

## Planned
- Image preprocessing and augmentation
- Generic training loop, checkpointing, and evaluation metrics
- Independent Karnataka regional holdout workflow
- Regional adaptation
- Human-reviewed correction queue and controlled retraining
- Package distribution

## Research integrity
- Preserve original source labels and provenance.
- Never infer missing region, variety, or diagnosis metadata.
- The current sugarcane class taxonomy is provisional; verify it against the dataset and domain experts.
- A leaf-only classifier cannot be assumed to detect diseases requiring stalk, root, or whole-plant evidence.
- Keep regional test data separate from training, validation, and adaptation.
- Check third-party dataset licenses before redistribution.

## Testing status
Tests have been added for EfficientNet-B0 classifier dimensions, forward output shape, feature freezing, and configuration-driven class count. They have not been executed in the GitHub environment; install compatible PyTorch/torchvision and run pytest -q locally.
