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
- Validation loss/accuracy monitoring, scheduler, checkpoints, early stopping, history, and summary outputs.
- Test split is not read by the training command.

## Implemented in v0.9.0
- Exact SHA-256 duplicate grouping during manifest curation.
- Same-class exact duplicates reduced to one deterministic representative.
- Cross-class exact duplicate groups excluded from the curated manifest pending human review.
- Human-review CSV, per-image curation audit CSV, and curation summary JSON.
- Source images and source manifest remain unchanged; label statuses are preserved.
- Group-aware splitting can optionally connect rows using an exact-duplicate review CSV.
- Explicit --include-unverified opt-in remains available for source-labelled datasets; no automatic label verification.

## Next
- Finalize AgML taxonomy and label provenance policy before training.
- Test-set evaluation: accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, and error review.
- Compare generic and regional model performance on an untouched Karnataka regional holdout.
- Regional fine-tuning using training/adaptation data only.
- Human-reviewed correction queue and controlled retraining.

## Research integrity
- SHA-256 equality identifies byte-identical files but cannot determine which conflicting disease label is correct.
- Perceptual hash matches are screening candidates, not confirmed duplicates.
- Preserve source labels and provenance; never infer missing region, variety, or diagnosis metadata.
- Unverified source labels must not be described as independently verified.
- A leaf-only classifier cannot be assumed to detect diseases requiring stalk, root, or whole-plant evidence.
- Keep regional test data separate from training, validation, and adaptation.
- Check third-party dataset licenses before redistribution.

## Testing status
- Curation and duplicate-aware split tests have been added in v0.9.0.
- Tests have not been executed in this GitHub editing workflow. Run pytest -q locally before training.
