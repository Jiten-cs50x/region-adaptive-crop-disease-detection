# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-reviewed feedback. Current components cover dataset auditing, metadata integrity, exact/perceptual duplicate screening, image-quality analysis, conservative dataset curation, leakage-aware splitting, EfficientNet-B0 setup and training.

## Current status: v0.9.0
- Dataset inventory and metadata manifest validation
- SHA-256 exact duplicate grouping and curation reports
- Cross-class exact-duplicate review queue; conflicting groups excluded from the curated manifest pending human review
- Same-class exact duplicates reduced to one deterministic representative
- Perceptual near-duplicate screening remains candidate-only
- Group-aware train/validation/test splits across plant, field, capture-session, and optional duplicate-group identifiers
- EfficientNet-B0 model factory, configurable transforms, and training pipeline
- Training command does not read test.csv

## Install (Windows)

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -e .
    pip install -e ".[dev]"

Install compatible PyTorch and torchvision versions using https://pytorch.org/get-started/locally/ (select CPU or CUDA wheels) before training.

## Prepare and curate data

The manifest must contain image_path, crop, disease_label, image_type, source_dataset, license, and label_status. Paths are relative to the dataset root. Verified rows should include verified_by.

First validate the source manifest:

    crop-adaptive validate-manifest --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/reports/manifest_validation.json

Create a conservative curated manifest using exact SHA-256 equality:

    crop-adaptive curate --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/curated

Curation outputs:
- curated_manifest.csv: unique images plus one representative per same-class exact-duplicate group
- cross_class_duplicate_review.csv: all members of cross-class exact-duplicate groups, with review_status=pending
- curation_audit.csv: included/excluded decision and reason for each source row, including invalid rows
- curation_summary.json: counts, class distribution, paths, and limitations

Curation never modifies or deletes source images or the source manifest. SHA-256 equality proves byte-identical files, not that a disease label is biologically correct. Same-class duplicate groups keep the lexicographically first image_path. Every member of a cross-class exact-duplicate group is excluded from the curated manifest pending human review. The original label_status is preserved; curation never upgrades unverified labels. Near-duplicate perceptual-hash candidates are not automatically removed.

## Create splits

For the AgML source labels that remain unverified, use the explicit opt-in below only after reviewing the dataset's annotation documentation and accepting that labels are source-provided rather than independently verified:

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/splits --include-unverified --seed 42

For other datasets, verified labels remain required by default. Optional exact-duplicate group linking is available when splitting a manifest that intentionally retains duplicate members after label conflicts have been resolved:

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/reviewed_manifest.csv --output datasets/sugarcane/reviewed_splits --include-unverified --duplicate-groups datasets/sugarcane/curated/curation_audit.csv --seed 42

Do not use the unreviewed source manifest as a training manifest while cross-class exact-duplicate conflicts remain unresolved. The duplicate-group option only prevents group members from crossing splits; it does not resolve label conflicts.

The split command keeps shared plant_id, field_id, capture_session_id, and supplied exact-duplicate group IDs together. It references source images and does not copy them. Review split_report.json. Missing group/provenance metadata means unknown relationships cannot be ruled out. Keep regional evaluation data independent.

## Train EfficientNet-B0

After confirming that the configured taxonomy exactly matches the curated manifest classes:

    crop-adaptive train --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --output runs/sugarcane_efficientnet_b0

Use --no-pretrained to avoid downloading ImageNet weights on first use. On Windows, begin with num_workers: 0.

The current example taxonomy is provisional and still needs to be updated to the final reviewed AgML class mapping before training. Do not train until the taxonomy and label policy are settled.

## Research integrity and limitations

- Keep the original AgML download unchanged and out of Git.
- A cross-class exact duplicate is a label conflict requiring review; SHA-256 cannot determine the correct disease.
- A perceptual near-duplicate is only a candidate, not proof of duplication.
- Do not promote unverified source labels to verified without independent evidence and a recorded reviewer.
- Validation metrics are for model selection; keep test data untouched until evaluation.
- Preserve an independent Karnataka regional holdout; do not use it for training, tuning, or early stopping.
- A leaf-only classifier cannot be assumed to diagnose diseases requiring stalk, root, or whole-plant evidence.
- Check third-party dataset licenses before redistribution.

## Roadmap

1. Dataset curation, metadata, exact/perceptual duplicate screening, and leakage-aware splits
2. EfficientNet-B0 training
3. Test-set evaluation, confusion matrix, per-class metrics, and error review
4. Independent Karnataka regional holdout and regional fine-tuning
5. Human-reviewed correction workflow and controlled retraining
6. Python API and package release
