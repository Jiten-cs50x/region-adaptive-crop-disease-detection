# Project status

## Implemented in v0.1.0
- Initial package skeleton and CLI
- YAML project configuration
- Non-destructive image validation
- File inventory and exact duplicate report

## Planned
- Metadata manifest validation
- Perceptual near-duplicate detection
- Group-aware dataset splits
- Model training, evaluation, and regional adaptation
- Human-reviewed correction queue and controlled retraining

## Research integrity
- Preserve original source labels and provenance.
- Never infer missing region, variety, or diagnosis metadata.
- Keep regional holdout data separate from training and adaptation data.
- Require human verification before feedback is used for retraining.
- Check third-party dataset licenses before redistribution.
