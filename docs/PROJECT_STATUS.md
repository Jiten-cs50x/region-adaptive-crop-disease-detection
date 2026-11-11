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

## Planned
- Duplicate review/decision workflow
- Image-quality analysis
- Model factory and training/evaluation
- Independent Karnataka regional holdout workflow
- Regional adaptation
- Human-reviewed correction queue and controlled retraining
- Package distribution

## Research integrity
- Preserve original source labels and provenance.
- Never infer missing region, variety, or diagnosis metadata.
- Perceptual similarity is not proof that images are duplicates; manually review candidates.
- Group-aware splits reduce leakage only to the extent that grouping metadata is complete and accurate.
- Keep regional test data separate from training, validation, and adaptation.
- Check third-party dataset licenses before redistribution.

## Testing status
Tests have been added for perceptual hash similarity, candidate reporting, and source-file preservation. They have not been executed in the GitHub environment; run pytest -q locally after installing the package.
