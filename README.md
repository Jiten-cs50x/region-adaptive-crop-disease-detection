# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-verified feedback. The current implementation focuses on dataset auditing, metadata integrity, and leakage-aware split generation; model training and regional adaptation are planned, not yet implemented.

## Current status: v0.3.0
- YAML-based project and dataset configuration
- Class-folder image inventory and image readability checks
- SHA-256 exact duplicate detection
- CSV metadata manifest template and validation
- Deterministic group-aware train/validation/test split generation
- Verified-label-only splitting by default
- Installable local Python package and CLI

## Quick start (Windows)

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -e .

### 1. Audit the raw image folders

    crop-adaptive validate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports

### 2. Create and complete a metadata manifest

    crop-adaptive manifest-template --output datasets/sugarcane/dataset_manifest.csv

Fill one row per image. Use paths relative to the dataset root. Record the source, license, disease label, and label verification accurately. Add plant_id, field_id, and capture_session_id wherever known.

### 3. Validate the manifest

    crop-adaptive validate-manifest --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/reports/manifest_validation.json

### 4. Generate group-aware splits

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/splits --seed 42

Defaults: 70% train, 15% validation, 15% test. Change with --train-ratio, --validation-ratio, and --test-ratio. Only label_status=verified rows are included by default. Use --include-unverified only when you deliberately want them in the split files; they should not silently enter a verified training set.

The splitter connects rows sharing plant_id, field_id, or capture_session_id and keeps each connected group in one split. Missing identifiers are treated as individual images and are counted in split_report.json. Therefore, populate grouping metadata whenever possible.

## Manifest fields

Required: image_path, crop, disease_label, image_type, source_dataset, license, label_status.

Recommended: source_url, country, region, state, district, collection_date (YYYY-MM-DD), variety, growth_stage, plant_id, field_id, capture_session_id, verified_by, notes.

A structurally valid row does not prove that a diagnosis is biologically correct. For label_status=verified, verified_by is required by manifest validation.

## Outputs

- validation_report.json and image_inventory.csv: raw image audit
- manifest_validation.json: metadata and path validation
- train.csv, validation.csv, test.csv: split assignments referencing original images
- split_report.json: class and image counts, exclusions, grouping coverage, and reproducibility seed

## Research integrity and limitations

- Split files contain paths and metadata, not copied image files.
- Group-aware assignment reduces leakage only when identifiers are accurate and sufficiently complete.
- Images without group identifiers are split as independent samples; unknown relationships may still leak.
- The split algorithm is deterministic for a fixed manifest, ratios, and seed, but class balance is approximate when groups are large.
- This command does not create a Karnataka regional holdout. Keep a separately collected regional holdout untouched until final evaluation; do not use it for training, tuning, or adaptation.
- Source images and manifests are never modified.
- Keep datasets and trained weights out of Git.

## Roadmap

1. Dataset audit, metadata manifest, and group-aware splits
2. Perceptual near-duplicate detection and image-quality analysis
3. Configurable PyTorch model factory (EfficientNet-B0, ResNet)
4. Generic training and evaluation
5. Regional fine-tuning and domain evaluation
6. Human-reviewed correction queue and controlled retraining
7. Python API and package release
