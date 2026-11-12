# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-reviewed feedback. Current components cover dataset auditing, metadata integrity, exact/perceptual duplicate screening, image-quality analysis, conservative dataset curation, leakage-aware splitting, EfficientNet-B0 setup, training, held-out test evaluation, and evidence-gated metadata recovery.

## Current status: v0.13.0
- Dataset inventory and metadata manifest validation
- SHA-256 exact duplicate grouping and conservative curation
- Cross-class exact-duplicate review queue; conflicting groups excluded pending human review
- Perceptual near-duplicate screening remains candidate-only
- Group-aware train/validation/test splits
- EfficientNet-B0 model factory, configurable transforms, and training pipeline
- Test evaluation with accuracy, macro/weighted precision/recall/F1, per-class metrics, confusion matrix, predictions, and misclassification CSV
- Test/train/validation path and provenance-group leakage checks
- Human misclassification review queue and image contact sheets
- Manifest-driven acquisition-pattern audit
- Provenance gap audit and evidence-gated recovery into a new manifest version
- AgML sugarcane configuration aligned to all 11 source folder labels

## Install (Windows)

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -e .
    pip install -e ".[dev]"

Install compatible PyTorch and torchvision wheels using https://pytorch.org/get-started/locally/. Confirm CUDA with:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

## Dataset location and curation

Keep downloaded Project-AgML images locally under `datasets/sugarcane/raw/project_agml/`; do not commit dataset images to Git.

The source dataset contained 6,748 leaf images in 11 source folders. The curation run retained 5,389 images after keeping one representative from same-class exact-duplicate groups and excluding cross-class exact-duplicate groups pending review. Source images and original manifest are not modified.

    crop-adaptive curate --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/curated

Review `datasets/sugarcane/curated/cross_class_duplicate_review.csv` before reconsidering excluded conflicting-label samples.

## Create or regenerate splits

The current AgML manifest rows are marked unverified. The following command is an explicit opt-in to use source-provided labels for exploratory training; it does not make them independently verified:

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/splits --include-unverified --seed 42

Run this after curation and whenever the curated manifest changes. Check `split_report.json`. Missing plant/field/session identifiers mean unknown relationships cannot be ruled out.

## AgML 11-class taxonomy

The source-provided classes are Banded Chlorosis, Brown Spot, BrownRust, Dried, Grassy shoot, Healthy, Pokkah Boeng, Sett Rot, Smut, Viral Disease, and Yellow Leaf. Confirm definitions and annotation quality against dataset documentation and expert review. Do not assume every folder is a clearly defined disease category.

## GPU training: RTX 3050 6 GB

The configuration uses EfficientNet-B0, 224-pixel input, batch size 8, and `num_workers: 0`. If CUDA runs out of memory, retry with batch size 4.

    crop-adaptive model-info --config configs/sugarcane_karnataka.yaml --no-pretrained
    crop-adaptive train --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --output runs/sugarcane_smoke_test --epochs 2 --batch-size 4 --device cuda --no-pretrained

Training reads train.csv and validation.csv only, not test.csv. The best checkpoint is selected by validation loss.

## Evaluate the held-out test split

    crop-adaptive evaluate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --checkpoint runs/sugarcane_efficientnet_b0/best_model.pt --output runs/sugarcane_efficientnet_b0/evaluation --device cuda

The evaluator writes metrics, confusion matrix, predictions, and misclassified-image reports. Do not tune on this test report and continue describing it as untouched.

## Human review of model errors

    crop-adaptive review-errors --dataset datasets/sugarcane/raw --misclassified runs/sugarcane_efficientnet_b0/evaluation/misclassified_images.csv --output runs/sugarcane_efficientnet_b0/error_review

A high-confidence error is a review priority, not proof that the source label is wrong. Never automatically replace labels with model predictions.

## Acquisition and provenance audits

    crop-adaptive acquisition-audit --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/reports/acquisition_audit.json
    crop-adaptive metadata-audit --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/reports/metadata_provenance_audit.json

The acquisition audit describes image characteristics; it does not prove shortcut learning. The metadata audit measures recorded metadata coverage and creates a separate recovery worksheet. A dataset name or repository location is not evidence that images were captured in Karnataka or elsewhere.

## Evidence-gated metadata recovery (v0.13.0)

1. Generate a fresh worksheet from the exact source manifest you intend to update:

       crop-adaptive metadata-audit --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/reports/metadata_provenance_audit.json

2. Open `datasets/sugarcane/reports/metadata_provenance_audit_recovery_queue.csv`. Enter proposed values only when supported by traceable dataset documentation or qualified review. For each row with proposed values, set:
   - `review_status` to `approved`
   - `evidence_type` and `evidence_reference` (a traceable document, dataset card, publication, or other source)
   - `reviewer`, `review_date` (YYYY-MM-DD), and `review_notes`

   Rows without proposals may remain blank. Values marked `rejected` or `needs_more_evidence` are not applied; if a row contains proposals but is not approved, the command stops rather than silently ignoring them.

3. Create a new manifest file; choose a new output path that does not already exist:

       crop-adaptive metadata-recover --manifest datasets/sugarcane/curated/curated_manifest.csv --recovery-queue datasets/sugarcane/reports/metadata_provenance_audit_recovery_queue.csv --output datasets/sugarcane/curated/curated_manifest_v2.csv

The recovery command checks row number plus image path and disease label to detect stale worksheets, requires complete evidence and explicit approval, validates ISO dates, refuses to overwrite existing metadata, and never overwrites an existing output file. The source manifest and recovery worksheet remain unchanged.

After recovery, run `validate-manifest`, then create a new curation/split/training experiment as appropriate. Do not retroactively replace metadata in old splits or checkpoints. Missing geographic metadata must remain unknown: do not infer Karnataka from the project's target region, image appearance, folder names, or model output.

## Research integrity and limitations

- Current Project-AgML labels are source-provided and remain unverified unless independently reviewed.
- The current 5,389-image curated dataset has no recorded country/state/district, collection date, variety, growth stage, plant ID, field ID, or capture-session ID in the supplied manifest. It is not a Karnataka regional dataset.
- Keep an independent Karnataka regional holdout separate from training, tuning, and early stopping.
- A leaf-only classifier cannot be assumed to diagnose conditions requiring stalk, root, or whole-plant evidence.
- Check third-party dataset licenses before redistribution.
- Keep downloaded images, generated runs, checkpoints, and local reports out of Git unless intentionally publishing a permitted artifact.

## Roadmap

1. Dataset curation, metadata, duplicate screening, and leakage-aware splits
2. EfficientNet-B0 training and held-out test evaluation
3. Evidence-backed metadata recovery and source documentation
4. Independent Karnataka regional data collection and holdout
5. Regional fine-tuning and controlled baseline comparison
6. Human-reviewed correction queue and Python package/API release
