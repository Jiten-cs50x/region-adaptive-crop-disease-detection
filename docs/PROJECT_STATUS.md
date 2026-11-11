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

## Planned
- Perceptual near-duplicate detection
- Group-aware train/validation/test splitting
- Model factory and training/evaluation
- Regional adaptation
- Human-reviewed correction queue and controlled retraining
- Package distribution

## Research integrity
- Preserve original source labels and provenance.
- Never infer missing region, variety, or diagnosis metadata.
- Keep regional holdout data separate from training and adaptation data.
- Require human verification before feedback is used for retraining.
- Check third-party dataset licenses before redistribution.
- Manifest validation checks metadata structure and file references, not the biological correctness of disease labels.

## Testing status
Tests have been added for manifest template creation and validation cases. They have not been executed in the GitHub environment; run pytest -q locally after installing the package.
