# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-verified feedback. The first implementation milestone focuses on dataset auditing; model training and adaptation are planned, not yet implemented.

## Current status: v0.1.0
- YAML-based project and dataset configuration
- Class-folder image inventory
- Image readability and minimum-dimension checks
- SHA-256 exact duplicate detection
- JSON and CSV audit reports
- Installable local Python package and CLI

## Quick start (Windows)

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -e .
    crop-adaptive validate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --output datasets/sugarcane/reports

## Expected dataset layout

    datasets/sugarcane/raw/
      healthy/image001.jpg
      rust/image002.jpg
      yellow_leaf_disease/image003.jpg

The class name is inferred from the first directory under the dataset root. Flat folders are inventoried as UNLABELLED_FLAT_FOLDER and should not be used for supervised training without a manifest.

## Reports
- validation_report.json: counts, class distribution, duplicate groups, and notes
- image_inventory.csv: file-level audit results

## Limitations
- File checks do not confirm disease diagnoses or label correctness.
- Exact duplicates are byte-identical files, not necessarily duplicate plants or collection events.
- Source files are never changed or deleted.
- Verify dataset provenance, label quality, and license compatibility before combining datasets.
- Keep image datasets and trained weights out of Git.

## Roadmap
1. Dataset audit and metadata manifest validation
2. Near-duplicate detection and image-quality analysis
3. Group-aware train/validation/test splitting
4. Configurable PyTorch model factory (EfficientNet-B0, ResNet)
5. Generic training and evaluation
6. Regional fine-tuning and domain evaluation
7. Human-reviewed feedback and controlled retraining
8. Python API and package release
