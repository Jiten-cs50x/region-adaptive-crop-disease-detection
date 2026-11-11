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

## Planned
- Duplicate/quality review-decision workflow
- Model factory and training/evaluation
- Independent Karnataka regional holdout workflow
- Regional adaptation
- Human-reviewed correction queue and controlled retraining
- Package distribution

## Research integrity
- Preserve original source labels and provenance.
- Never infer missing region, variety, or diagnosis metadata.
- Quality metrics are screening signals and require crop/capture-specific threshold calibration.
- Perceptual similarity is not proof that images are duplicates; manually review candidates.
- Group-aware splitting reduces leakage only to the extent that grouping metadata is complete and accurate.
- Keep regional test data separate from training, validation, and adaptation.
- Check third-party dataset licenses before redistribution.

## Testing status
Tests have been added for quality flags, metric behavior, and source-file preservation. They have not been executed in the GitHub environment; run pytest -q locally after installing the package.
