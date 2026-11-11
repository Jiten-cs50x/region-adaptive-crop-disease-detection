# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-verified feedback. The current implementation covers dataset auditing, metadata integrity, leakage-aware splitting, and perceptual duplicate screening; model training and regional adaptation are planned, not yet implemented.

## Current status: v0.4.0
- YAML-based project and dataset configuration
- Class-folder image inventory and image readability checks
- SHA-256 exact duplicate detection
- CSV metadata manifest template and validation
- Deterministic group-aware train/validation/test split generation
- Perceptual near-duplicate candidate detection using difference hash
- JSON reports with candidate pairs and connected review groups
- Installable local Python package and CLI

## Quick start (Windows)

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -e .

### 1. Audit raw image folders

    crop-adaptive validate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports

### 2. Create and complete a metadata manifest

    crop-adaptive manifest-template --output datasets/sugarcane/dataset_manifest.csv

Fill one row per image. Use paths relative to the dataset root. Record source, license, disease label, and label verification accurately. Add plant_id, field_id, and capture_session_id wherever known.

### 3. Validate the manifest

    crop-adaptive validate-manifest --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/reports/manifest_validation.json

### 4. Find perceptual near-duplicate candidates

    crop-adaptive near-duplicates --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports/near_duplicates.json --threshold 6

The command uses a 64-bit difference hash by default. Lower thresholds are stricter; higher thresholds return more candidates. Inspect the report manually. Do not automatically remove similar images: different leaves can look alike, and the hash can miss crops, rotations, or substantial edits.

### 5. Generate group-aware splits

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/splits --seed 42

Defaults: 70% train, 15% validation, 15% test. Only verified labels are included by default. Plant, field, and capture-session links are kept in one split.

## Manifest fields

Required: image_path, crop, disease_label, image_type, source_dataset, license, label_status.

Recommended: source_url, country, region, state, district, collection_date (YYYY-MM-DD), variety, growth_stage, plant_id, field_id, capture_session_id, verified_by, notes.

## Outputs

- validation_report.json and image_inventory.csv: raw image audit
- manifest_validation.json: metadata and path validation
- near_duplicates.json: candidate pairs, hash distances, and connected review groups
- train.csv, validation.csv, test.csv: split assignments referencing original images
- split_report.json: class counts, exclusions, grouping coverage, and seed

## Research integrity and limitations

- Perceptual similarity is a screening signal, not proof of duplicate content or identical plants.
- Difference hash is useful for resized/recompressed copies but can miss crops, rotations, and substantial edits.
- Review candidate pairs before excluding any sample; preserve source files and record review decisions.
- Group-aware splitting reduces leakage only when identifiers are accurate and sufficiently complete.
- Images without group identifiers are treated as independent samples; unknown relationships may still leak.
- Keep a separately collected Karnataka regional holdout untouched until final evaluation.
- Keep datasets and trained weights out of Git.

## Roadmap

1. Dataset audit, metadata manifest, group-aware splits, and near-duplicate screening
2. Image-quality analysis and duplicate-review workflow
3. Configurable PyTorch model factory (EfficientNet-B0, ResNet)
4. Generic training and evaluation
5. Regional fine-tuning and domain evaluation
6. Human-reviewed correction queue and controlled retraining
7. Python API and package release
