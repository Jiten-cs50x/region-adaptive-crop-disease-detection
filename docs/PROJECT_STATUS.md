# Project status

## Implemented in v0.1.0
- Package skeleton, CLI, YAML configuration, non-destructive image validation, inventory, and exact duplicate report.

## Implemented in v0.2.0
- CSV metadata manifest template and validation.
- Image path boundary checks and metadata validation.

## Implemented in v0.3.0
- Deterministic train/validation/test split CSV generation.
- Group linking across plant_id, field_id, and capture_session_id.
- Verified-label-only splitting by default.

## Implemented in v0.4.0
- Difference-hash perceptual similarity screening and candidate review groups.

## Implemented in v0.5.0
- Image dimensions, brightness, contrast, and approximate sharpness screening.

## Implemented in v0.6.0
- EfficientNet-B0 model factory, taxonomy-driven classifier, optional pretrained weights, feature freezing, and dropout.

## Implemented in v0.7.0
- EXIF correction, RGB conversion, ImageNet normalization, conservative train augmentation, deterministic validation/test/prediction transforms.

## Implemented in v0.8.0
- Manifest-backed PyTorch Dataset and train/validation DataLoaders.
- Configurable EfficientNet-B0 training loop with AdamW.
- Reproducible random seeds and CUDA/CPU device selection.
- Validation loss/accuracy monitoring and ReduceLROnPlateau scheduling.
- Best-validation-loss and latest-epoch checkpoints.
- Early stopping, training history, and summary JSON outputs.
- Image path and plant/field/session group leakage checks between train and validation.
- Test split is not read by the training command.

## Next
- Independent evaluation using test.csv: accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, and error review.
- Compare generic and regional model performance on an untouched Karnataka regional holdout.
- Regional fine-tuning using training/adaptation data only.
- Human-reviewed correction queue and controlled retraining.

## Research integrity
- Preserve source labels and provenance; never infer missing region, variety, or diagnosis metadata.
- The example sugarcane taxonomy is provisional and must be verified against the actual dataset and domain experts.
- A leaf-only classifier cannot be assumed to detect diseases requiring stalk, root, or whole-plant evidence.
- Keep regional test data separate from training, validation, and adaptation.
- Check third-party dataset licenses before redistribution.

## Testing status
Training dataset tests have been added but have not been executed in this GitHub workflow. Install compatible PyTorch/torchvision and run pytest -q locally before using the training command.
