# Project status

## Implemented in v0.1.0
- Initial package skeleton and CLI
- YAML project configuration
- Non-destructive image validation
- File inventory and exact duplicate report

## Implemented in v0.2.0
- CSV metadata manifest template
- Manifest structure and required-field validation
- Image path existence and dataset-root boundary checks
- Image type, label status, verifier, and ISO collection-date checks
- JSON manifest validation report

## Implemented in v0.3.0
- Deterministic train/validation/test split CSV generation
- Group linking across plant_id, field_id, and capture_session_id
- Verified-label-only splitting by default
- Split class-count and grouping-coverage report
- Configurable split ratios and random seed

## Planned
- Perceptual near-duplicate detection
- Model factory and training/evaluation
- Independent Karnataka regional holdout workflow
- Regional adaptation
- Human-reviewed correction queue and controlled retraining
- Package distribution

## Research integrity
- Preserve original source labels and provenance.
- Never infer missing region, variety, or diagnosis metadata.
- Keep regional test data separate from training, validation, and adaptation.
- Require human verification before feedback is used for retraining.
- Check third-party dataset licenses before redistribution.
- Manifest validation checks metadata structure and file references, not biological correctness.
- Group-aware splits reduce leakage only to the extent that plant, field, and capture-session metadata are accurate and complete.

## Testing status
Tests have been added for split determinism, group isolation, and verified-label filtering. They have not been executed in the GitHub environment; run pytest -q locally after installing the package.
