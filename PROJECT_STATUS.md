# Project status

## Current release
v0.13.0 — evidence-gated metadata recovery.

## Completed
- Dataset validation, manifest generation and validation.
- Exact duplicate curation and cross-class duplicate review.
- Near-duplicate screening, image quality analysis, and group-aware splitting.
- EfficientNet-B0 model setup, transforms, training, and held-out evaluation.
- Misclassification review queue and acquisition-pattern audit.
- Metadata provenance audit and evidence-gated recovery into a new manifest.

## Current dataset facts
- Project-AgML source: 6,748 images across 11 source labels.
- Curated manifest: 5,389 leaf images.
- Recorded source/license values: Project-AgML / CC BY 4.0 (per supplied manifest).
- Geographic origin, collection date, variety, growth stage, plant, field, and capture-session metadata are absent in the current manifest.
- Therefore this dataset must not be represented as Karnataka-collected regional data.
- The current model's test metrics are internal results on the existing dataset split, not evidence of regional generalization.

## Next research milestone
Recover metadata only from traceable evidence. Separately acquire expert-reviewed Karnataka sugarcane images with capture metadata, define a region-held-out test set, and compare generic versus region-adapted models without leakage.
