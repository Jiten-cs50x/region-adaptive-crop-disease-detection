# Project status

## Implemented through v0.9.1
- Dataset validation, manifest validation, duplicate screening, curation, group-aware splits, EfficientNet-B0 model factory, transforms, and training.
- AgML sugarcane configuration aligned to 11 source folder labels.
- Source label status remains unverified.

## Implemented in v1.0.0
- Checkpoint evaluation calculates metrics using test.csv only.
- Train and validation CSVs are read only for image-path and plant/field/session leakage checks.
- Reports accuracy, macro and weighted precision/recall/F1, per-class metrics, and sample counts.
- Exports confusion matrix as CSV and JSON.
- Exports misclassified-image CSV with true/predicted labels and confidence.
- Enforces checkpoint/config class-index and class-order compatibility.
- Adds unit tests for metric calculations.

## Next
- Run the v1.0.0 test suite locally.
- Evaluate the completed sugarcane checkpoint on the held-out test split exactly once.
- Review misclassifications and label quality with an agronomy-informed reviewer.
- Build an independent Karnataka regional holdout and keep it separate from training and adaptation.
- Implement regional fine-tuning and a human-reviewed correction queue.

## Research integrity
- Exact SHA-256 duplicates do not identify the correct conflicting disease label.
- Perceptual hash matches are review candidates, not confirmed duplicates.
- Preserve source labels and provenance; never infer missing region, variety, or diagnosis metadata.
- Unverified source labels must not be described as independently verified.
- A leaf-only classifier cannot be assumed to detect diseases requiring stalk, root, or whole-plant evidence.
