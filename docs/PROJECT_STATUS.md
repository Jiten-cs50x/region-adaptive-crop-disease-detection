# Project status

## Implemented in v0.1.0
- Initial package skeleton, CLI, YAML configuration, non-destructive image validation, inventory, and exact duplicate report.

## Implemented in v0.2.0
- CSV metadata manifest template and validation.
- Image path boundary checks and metadata validation.

## Implemented in v0.3.0
- Deterministic train/validation/test split CSV generation.
- Group linking across plant_id, field_id, and capture_session_id.
- Verified-label-only splitting by default.

## Implemented in v0.4.0
- Difference-hash perceptual similarity screening and candidate review groups.
- Unreadable image reporting; source images are not modified.

## Implemented in v0.5.0
- Image dimensions, grayscale brightness, contrast, and approximate sharpness analysis.
- Configurable thresholds and JSON/CSV reports; no automatic image exclusion.

## Implemented in v0.6.0
- EfficientNet-B0 model factory using torchvision.
- Taxonomy-driven classifier output size, optional pretrained weights, feature freezing, and dropout.
- CLI model-info command and model-factory tests.

## Implemented in v0.7.0
- EXIF orientation correction and RGB conversion in the transform pipeline.
- Configurable EfficientNet-compatible image size and ImageNet normalization.
- Conservative training augmentation: random resized crop, horizontal flip, small rotation, and mild colour jitter.
- Deterministic resize/center-crop preprocessing for validation, test, and prediction.
- Transform tests for tensor shape, normalization, deterministic evaluation, and invalid configuration.

## Next
- Dataset/Dataloader integration using the manifest and split CSVs.
- Generic training loop with reproducible seeds, checkpointing, early stopping, and validation monitoring.
- Evaluation metrics and per-class error analysis.
- Independent Karnataka regional holdout workflow and regional fine-tuning.
- Human-reviewed correction queue and controlled retraining.

## Research integrity
- Preserve original source labels and provenance; never infer missing region, variety, or diagnosis metadata.
- The sugarcane taxonomy is provisional and must be verified against the dataset and domain experts.
- A leaf-only classifier cannot be assumed to detect diseases requiring stalk, root, or whole-plant evidence.
- Keep regional test data separate from training, validation, and adaptation.
- Check third-party dataset licenses before redistribution.

## Testing status
Tests have been added for transforms, but they have not been executed in this GitHub workflow. Install compatible PyTorch/torchvision and run pytest -q locally.
