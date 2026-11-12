# Region-Adaptive Crop Disease Detection

A reusable, configuration-driven pipeline for crop disease image classification, regional adaptation, and human-reviewed feedback. Current components cover dataset auditing, metadata integrity, exact/perceptual duplicate screening, image-quality analysis, conservative dataset curation, leakage-aware splitting, EfficientNet-B0 setup, training, and held-out test evaluation.

## Current status: v0.12.0
- Dataset inventory and metadata manifest validation
- SHA-256 exact duplicate grouping and conservative curation
- Cross-class exact-duplicate review queue; conflicting groups excluded pending human review
- Perceptual near-duplicate screening remains candidate-only
- Group-aware train/validation/test splits
- EfficientNet-B0 model factory, configurable transforms, and training pipeline
- Test evaluation with accuracy, macro/weighted precision/recall/F1, per-class metrics, confusion matrix, predictions, and misclassification CSV
- Test/train/validation path and provenance-group leakage checks
- Human misclassification review queue, blank reviewer decision fields, confidence-prioritized summary, and image contact sheets
- Manifest-driven acquisition-pattern audit with class/source/image-type profiles and metadata completeness reporting
- Provenance/metadata gap audit with an evidence-led recovery worksheet; no automatic metadata inference
- AgML sugarcane configuration aligned to all 11 source folder labels

## Install (Windows)

    python -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -e .
    pip install -e ".[dev]"

Install compatible PyTorch and torchvision wheels using https://pytorch.org/get-started/locally/. Confirm CUDA with:

    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

## Dataset location and curation

Keep the downloaded Project-AgML images locally under `datasets/sugarcane/raw/project_agml/`; do not commit dataset images to Git.

The current source dataset contains 6,748 leaf images in 11 source folders. The curation run reported 5,389 retained images after keeping one representative from same-class exact-duplicate groups and excluding cross-class exact-duplicate groups pending review. The source images and original manifest are not modified.

    crop-adaptive curate --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/dataset_manifest.csv --output datasets/sugarcane/curated

Review `datasets/sugarcane/curated/cross_class_duplicate_review.csv` before reconsidering any excluded conflicting-label samples. Exact byte equality identifies identical files; it does not identify the correct disease label.

## Create or regenerate splits

The current AgML manifest rows are marked unverified. The following command is an explicit opt-in to use source-provided labels for exploratory training; it does not make them independently verified:

    crop-adaptive split --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/splits --include-unverified --seed 42

Run this after curation and whenever the curated manifest changes. Check `split_report.json` to confirm the split class counts. The split command keeps shared plant, field, capture-session, and supplied duplicate-group IDs together. Missing provenance identifiers mean unknown relationships cannot be ruled out.

## AgML 11-class taxonomy

The config `configs/sugarcane_karnataka.yaml` uses the exact folder labels reported by the downloaded dataset:

1. Banded Chlorosis
2. Brown Spot
3. BrownRust
4. Dried
5. Grassy shoot
6. Healthy
7. Pokkah Boeng
8. Sett Rot
9. Smut
10. Viral Disease
11. Yellow Leaf

The class names are source-provided categories. Confirm their definitions and annotation quality against the dataset documentation and expert review before making disease-diagnosis claims. In particular, do not assume every folder is a clearly defined disease category.

## GPU training: RTX 3050 6 GB

The configuration uses EfficientNet-B0, 224-pixel input, batch size 8, and `num_workers: 0`. This is a conservative starting point, not a guarantee against out-of-memory errors. If CUDA runs out of memory, retry with batch size 4.

First run a model configuration check:

    crop-adaptive model-info --config configs/sugarcane_karnataka.yaml --no-pretrained

Then run a short GPU smoke test (two epochs, batch size 4, no pretrained download):

    crop-adaptive train --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --output runs/sugarcane_smoke_test --epochs 2 --batch-size 4 --device cuda --no-pretrained

If the smoke test completes and produces `best_model.pt`, `last_checkpoint.pt`, `training_history.json`, and `training_summary.json`, start the full run:

    crop-adaptive train --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --output runs/sugarcane_efficientnet_b0 --device cuda

Training reads train.csv and validation.csv only. It does not load test.csv. The best checkpoint is selected by validation loss.

## Evaluate the held-out test split

After training, evaluate the best checkpoint exactly once for the current experiment:

    crop-adaptive evaluate --config configs/sugarcane_karnataka.yaml --dataset datasets/sugarcane/raw --splits datasets/sugarcane/splits --checkpoint runs/sugarcane_efficientnet_b0/best_model.pt --output runs/sugarcane_efficientnet_b0/evaluation --device cuda

The evaluator checks that test images and available plant/field/capture-session identifiers do not overlap train or validation. It writes:
- `evaluation_summary.json` and `evaluation_metrics.json`
- `confusion_matrix.csv`
- `predictions.csv`
- `misclassified_images.csv`

The confusion matrix uses true labels as rows and predicted labels as columns. Do not tune the model or thresholds on this test report and then continue to describe it as an untouched test result. If you make changes based on errors, create a new experiment and retain a separate final holdout.

## Review misclassified images

Create a human-review queue and contact sheets from the evaluation CSV (the command does not modify source images, labels, manifests, splits, or metrics):

    crop-adaptive review-errors --dataset datasets/sugarcane/raw --misclassified runs/sugarcane_efficientnet_b0/evaluation/misclassified_images.csv --output runs/sugarcane_efficientnet_b0/error_review

The output contains review_queue.csv, review_summary.json, REVIEW_GUIDE.md, and numbered contact sheets. Fill review_status, corrected_label, reviewer, and review_notes manually. A high-confidence error is a review priority, not proof that the source label is wrong. Do not automatically replace source labels with model predictions. Preserve the original held-out test report; if reviewed samples are used for training or model selection, use a new experiment and a separate final holdout.

## Audit image acquisition patterns (v0.11.0)

Compare image characteristics across the source labels and available manifest metadata:

    crop-adaptive acquisition-audit --dataset datasets/sugarcane/raw --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/reports/acquisition_audit.json

The command writes a JSON summary and a sibling per-image CSV. It reports dimensions, aspect ratio/orientation, coarse resolution and brightness bands, brightness, contrast, approximate edge-variance sharpness, whole-image mean RGB, and metadata completeness. Summaries are grouped by disease label, source dataset, and image type where those metadata fields exist.

This is a descriptive audit, not a background detector or proof of shortcut learning. Whole-image color statistics do not isolate backgrounds. Class-wise capture differences are signals to investigate with expert-reviewed examples and controlled experiments. The command reads labels from the manifest, never infers them from folder names, and does not modify images, labels, splits, checkpoints, or evaluation results.

## Audit provenance and prepare metadata recovery (v0.12.0)

Run a read-only metadata provenance audit against the curated manifest:

    crop-adaptive metadata-audit --manifest datasets/sugarcane/curated/curated_manifest.csv --output datasets/sugarcane/reports/metadata_provenance_audit.json

The command writes a JSON report and a sibling metadata_provenance_audit_recovery_queue.csv. The report summarizes source-dataset/license attribution and completeness of location, collection date, variety, growth-stage, plant, field, and capture-session metadata, including class-wise coverage. The separate worksheet preserves each original manifest row and adds blank proposed_* fields plus evidence/reviewer columns.

Do not fill proposed values from visual guesses, folder names, model predictions, or the fact that this project targets Karnataka. Enter a value only when supported by traceable dataset documentation or qualified review, and record the evidence. Keep the original manifest unchanged; after approval, create a new versioned manifest and rerun validation/curation/splitting as a new experiment. This command does not modify images, labels, existing splits, checkpoints, or held-out evaluation reports.

## Research integrity and limitations

- All current AgML manifest labels remain `unverified`; using `--include-unverified` is an explicit exploratory-training decision, not verification.
- Do not invent missing provenance, region, district, variety, growth stage, plant, field, or capture-session metadata.
- Keep a separate regional Karnataka holdout independent; do not use it for training, tuning, or early stopping.
- A leaf-only classifier cannot be assumed to diagnose conditions requiring stalk, root, or whole-plant evidence.
- Check third-party dataset licenses before redistribution.
- Keep downloaded images, generated runs, checkpoints, and local reports out of Git unless intentionally publishing a permitted artifact.

## Roadmap

1. Dataset curation, metadata, exact/perceptual duplicate screening, and leakage-aware splits
2. EfficientNet-B0 training
3. Held-out test evaluation and error review
4. Independent Karnataka regional holdout and regional fine-tuning
5. Human-reviewed correction queue and controlled retraining
6. Python API and package release
