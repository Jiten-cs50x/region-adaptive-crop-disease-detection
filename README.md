# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-verified feedback. The first implementation milestones focus on dataset auditing and metadata integrity; model training and regional adaptation are planned, not yet implemented.

## Current status: v0.2.0
- YAML-based project and dataset configuration
- Class-folder image inventory and image readability checks
- SHA-256 exact duplicate detection
- JSON and CSV audit reports
- CSV metadata manifest template
- Manifest validation for image references, required provenance fields, image type, label status, and collection date
- Installable local Python package and CLI

## Quick start (Windows)

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -e .

### 1. Audit the raw image folders

    crop-adaptive validate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports

Expected class-folder layout:

    datasets/sugarcane/raw/
      healthy/image001.jpg
      rust/image002.jpg
      yellow_leaf_disease/image003.jpg

### 2. Create a metadata manifest

    crop-adaptive manifest-template --output datasets/sugarcane/dataset_manifest.csv

Fill one row per image. Use paths relative to the dataset root, for example healthy/image001.jpg. Record provenance and label status accurately; do not guess missing metadata.

### 3. Validate the manifest

    crop-adaptive validate-manifest --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/reports/manifest_validation.json

## Manifest fields

Required columns:
- image_path: relative path from the dataset root
- crop: crop name, e.g. sugarcane
- disease_label: approved disease/healthy class label
- image_type: leaf, whole_plant, stalk, root, fruit, or other
- source_dataset: original dataset or collection identifier
- license: license or permission basis
- label_status: verified, unverified, or rejected

Optional but strongly recommended columns:
- source_url, country, region, state, district
- collection_date (YYYY-MM-DD), variety, growth_stage
- plant_id, field_id, capture_session_id
- verified_by, notes

For label_status=verified, verified_by is required. A valid manifest row means its metadata and file reference pass structural checks; it does not prove the diagnosis is biologically correct. Keep leaf, whole-plant, and stalk images distinguishable because some sugarcane diseases cannot be diagnosed reliably from leaf photographs alone.

## Reports
- validation_report.json: image counts, class distribution, exact duplicate groups, and notes
- image_inventory.csv: file-level image audit
- manifest_validation.json: manifest row counts and row-level issues

## Limitations and research integrity
- File checks do not confirm disease diagnoses or label correctness.
- Exact duplicates are byte-identical files, not necessarily duplicate plants or collection events.
- Regional metadata must reflect actual collection provenance; never infer a district or region from the dataset name.
- Group-aware splits should use plant, field, or capture-session identifiers to reduce leakage.
- Keep the regional holdout set separate from training and adaptation.
- Check dataset provenance and license compatibility before combining or redistributing datasets.
- Source images and manifest files are never modified by validation commands.
- Keep datasets and trained weights out of Git.

## Roadmap
1. Dataset audit and metadata manifest validation
2. Near-duplicate detection and image-quality analysis
3. Group-aware train/validation/test splitting
4. Configurable PyTorch model factory (EfficientNet-B0, ResNet)
5. Generic training and evaluation
6. Regional fine-tuning and domain evaluation
7. Human-reviewed correction queue and controlled retraining
8. Python API and package release
