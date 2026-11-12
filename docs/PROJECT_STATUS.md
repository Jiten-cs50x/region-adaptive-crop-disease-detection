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

## Implemented in v0.9.1
- Sugarcane config updated from six provisional classes to the 11 exact AgML source folder labels.
- AgML source and unverified label status recorded explicitly in config.
- EfficientNet-B0 GPU starting defaults set to batch size 8 and num_workers 0 for the user's RTX 3050 6 GB.
- Added tests for exact taxonomy/configuration consistency.

## Implemented in v0.10.0
- Human-review queue generated from misclassified_images.csv with blank decision fields; no automatic relabeling.
- Numbered image contact sheets, confidence/error-pair summary, missing-image reporting, and reviewer guide.
- Dataset-root path traversal protection; source images and evaluation artifacts remain unchanged.

## Next
- Regenerate splits from the final curated manifest and inspect split_report.json.
- Run a short GPU smoke test before the full training run.
- Test-set evaluation: accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, and error review.
- Compare generic and regional model performance on an untouched Karnataka regional holdout.
- Regional fine-tuning using training/adaptation data only.
- Apply only explicitly approved human label corrections to a new manifest version, preserving a full audit trail.
- Cross-split perceptual near-duplicate audit and independent regional holdout evaluation.

## Research integrity
- SHA-256 equality identifies byte-identical files but cannot determine which conflicting disease label is correct.
- Perceptual hash matches are screening candidates, not confirmed duplicates.
- Preserve source labels and provenance; never infer missing region, variety, or diagnosis metadata.
- Unverified source labels must not be described as independently verified.
- A leaf-only classifier cannot be assumed to detect diseases requiring stalk, root, or whole-plant evidence.
- Keep regional test data separate from training, validation, and adaptation.
- Check third-party dataset licenses before redistribution.

## Testing status
- Curation, duplicate-aware split, model factory, transforms, and training dataset tests are present.
- The user reports all tests executed locally after installing CUDA-enabled PyTorch; exact pass/skip/failure counts should be retained from their final pytest summary.


## Implemented in v0.11.0
- Non-destructive acquisition-pattern audit using manifest-provided labels.
- Per-image dimensions, aspect ratio, orientation, coarse resolution/brightness bands, brightness, contrast, approximate sharpness, and whole-image mean RGB features.
- Class-wise, source-dataset-wise, and image-type-wise descriptive profiles plus metadata completeness counts.
- Missing/unreadable image reporting and path traversal/duplicate manifest-path protection.
- No automatic label changes, image exclusion, test-set tuning, or source-file modification.

## Next
- Run acquisition-audit locally on the curated AgML manifest and inspect class-wise capture profiles.
- Manually compare representative examples from classes with strongly different capture profiles.
- Review source labels with an agricultural expert before making any label corrections.
- Plan controlled capture-condition experiments and an independent Karnataka regional holdout.

## Testing note
- v0.11.0 tests were added for profile reporting, source-image immutability, path traversal, and duplicate manifest paths. They have not been executed in the GitHub environment; run the local pytest suite after pulling the commit.

## Implemented in v0.12.0
- Read-only provenance and metadata completeness audit for source dataset, source URL, license, country, region, state, district, collection date, variety, growth stage, plant, field, and capture-session identifiers.
- Class-wise metadata coverage and source/license attribution summaries.
- Separate evidence-led recovery worksheet preserving original manifest values, with blank proposed metadata and reviewer/evidence fields.
- Duplicate manifest path reporting.
- No automatic metadata inference, source-manifest edits, relabeling, image changes, split changes, checkpoint changes, or held-out evaluation changes.

## Next
- Run metadata-audit locally and review the recovery worksheet against original Project-AgML documentation and traceable source records.
- Recover only documented values; leave unsupported fields blank.
- If approved metadata is added, save a new manifest version and regenerate duplicate-aware group splits as a new experiment.
- Obtain expert-reviewed Karnataka field images with variety, growth stage, date, and field/session provenance for independent regional evaluation.
