import argparse
import json
from pathlib import Path

import yaml

from crop_adaptive.data.curation import curate_exact_duplicates
from crop_adaptive.data.manifest import create_manifest_template, validate_manifest
from crop_adaptive.data.near_duplicates import find_near_duplicates
from crop_adaptive.data.quality import analyze_image_quality
from crop_adaptive.data.splitting import create_group_splits
from crop_adaptive.data.validator import validate_dataset
from crop_adaptive.models.efficientnet import build_model_from_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="crop-adaptive", description="Reusable crop disease pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Audit a class-folder image dataset")
    validate.add_argument("--config", required=True)
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--output", default="reports/validation")
    template = commands.add_parser("manifest-template", help="Create an empty dataset metadata CSV")
    template.add_argument("--output", default="dataset_manifest.csv")
    manifest = commands.add_parser("validate-manifest", help="Validate metadata and image paths in a CSV manifest")
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--output", default="reports/manifest_validation.json")
    curate = commands.add_parser("curate", help="Create a conservative manifest after exact SHA-256 duplicate analysis")
    curate.add_argument("--dataset", required=True)
    curate.add_argument("--manifest", required=True)
    curate.add_argument("--output", default="datasets/sugarcane/curated")
    split = commands.add_parser("split", help="Create deterministic group-aware train/validation/test CSVs")
    split.add_argument("--dataset", required=True)
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", default="datasets/splits")
    split.add_argument("--train-ratio", type=float, default=0.70)
    split.add_argument("--validation-ratio", type=float, default=0.15)
    split.add_argument("--test-ratio", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--include-unverified", action="store_true")
    split.add_argument("--duplicate-groups", help="Optional cross_class_duplicate_review.csv to keep exact-duplicate members in one split")
    near = commands.add_parser("near-duplicates", help="Find perceptually similar images for manual review")
    near.add_argument("--dataset", required=True)
    near.add_argument("--output", default="reports/near_duplicates.json")
    near.add_argument("--threshold", type=int, default=6)
    near.add_argument("--hash-size", type=int, default=8)
    quality = commands.add_parser("quality", help="Measure image quality and flag samples for manual review")
    quality.add_argument("--dataset", required=True)
    quality.add_argument("--output", default="reports/image_quality.json")
    quality.add_argument("--min-width", type=int, default=128)
    quality.add_argument("--min-height", type=int, default=128)
    quality.add_argument("--min-brightness", type=float, default=35.0)
    quality.add_argument("--max-brightness", type=float, default=220.0)
    quality.add_argument("--min-contrast", type=float, default=20.0)
    quality.add_argument("--min-sharpness", type=float, default=10.0)
    quality.add_argument("--analysis-size", type=int, default=256)
    model_info = commands.add_parser("model-info", help="Build EfficientNet-B0 and report its classifier configuration")
    model_info.add_argument("--config", required=True)
    model_info.add_argument("--no-pretrained", action="store_true")
    train = commands.add_parser("train", help="Train EfficientNet-B0 using prepared train/validation split CSVs")
    train.add_argument("--config", required=True)
    train.add_argument("--dataset", required=True)
    train.add_argument("--splits", required=True)
    train.add_argument("--output", default="runs/sugarcane_efficientnet_b0")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--device", help="Optional torch device, for example cpu or cuda")
    train.add_argument("--no-pretrained", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Evaluate a trained checkpoint on the held-out test split")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--splits", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", default="runs/evaluation")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--device", help="Optional torch device, for example cpu or cuda")
    metadata_audit = commands.add_parser("metadata-audit", help="Audit provenance gaps and create an evidence-led recovery worksheet")
    metadata_audit.add_argument("--manifest", required=True)
    metadata_audit.add_argument("--output", default="reports/metadata_provenance_audit.json")
    metadata_recover = commands.add_parser("metadata-recover", help="Create a new manifest from explicitly approved, evidence-backed metadata proposals")
    metadata_recover.add_argument("--manifest", required=True, help="Unmodified source manifest used to create the recovery worksheet")
    metadata_recover.add_argument("--recovery-queue", required=True, help="Completed recovery worksheet CSV")
    metadata_recover.add_argument("--output", required=True, help="New manifest path; existing files are never overwritten")
    acquisition = commands.add_parser("acquisition-audit", help="Compare image acquisition characteristics across manifest labels")
    acquisition.add_argument("--dataset", required=True)
    acquisition.add_argument("--manifest", required=True)
    acquisition.add_argument("--output", default="reports/acquisition_audit.json")
    bias = commands.add_parser("bias-audit", help="Measure descriptive association between acquisition features and class labels")
    bias.add_argument("--features", required=True, help="Per-image CSV produced by acquisition-audit")
    bias.add_argument("--output", default="reports/acquisition_bias_audit.json")
    review_errors = commands.add_parser("review-errors", help="Create a human-review CSV and image contact sheets for model errors")
    review_errors.add_argument("--dataset", required=True)
    review_errors.add_argument("--misclassified", required=True)
    review_errors.add_argument("--output", default="runs/error_review")
    review_errors.add_argument("--per-sheet", type=int, default=12)
    review_errors.add_argument("--columns", type=int, default=3)

    args = parser.parse_args()
    if args.command == "validate":
        with Path(args.config).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        report = validate_dataset(args.dataset, args.output, config)
        print(json.dumps(report, indent=2))
        print(f"\nReports saved to: {Path(args.output).resolve()}")
    elif args.command == "manifest-template":
        path = create_manifest_template(args.output)
        print(f"Manifest template created: {path.resolve()}")
    elif args.command == "validate-manifest":
        report = validate_manifest(args.dataset, args.manifest, args.output)
        print(json.dumps(report, indent=2))
        print(f"\nReport saved to: {Path(args.output).resolve()}")
    elif args.command == "curate":
        report = curate_exact_duplicates(args.dataset, args.manifest, args.output)
        print(json.dumps(report, indent=2))
        print(f"\nCuration reports saved to: {Path(args.output).resolve()}")
    elif args.command == "split":
        report = create_group_splits(
            args.dataset, args.manifest, args.output,
            ratios={"train": args.train_ratio, "validation": args.validation_ratio, "test": args.test_ratio},
            seed=args.seed, verified_only=not args.include_unverified,
            duplicate_groups_path=args.duplicate_groups,
        )
        print(json.dumps(report, indent=2))
        print(f"\nSplit files saved to: {Path(args.output).resolve()}")
    elif args.command == "near-duplicates":
        report = find_near_duplicates(args.dataset, args.output, args.threshold, args.hash_size)
        print(json.dumps(report, indent=2))
        print(f"\nNear-duplicate report saved to: {Path(args.output).resolve()}")
    elif args.command == "quality":
        report = analyze_image_quality(
            args.dataset, args.output, min_width=args.min_width, min_height=args.min_height,
            min_brightness=args.min_brightness, max_brightness=args.max_brightness,
            min_contrast=args.min_contrast, min_sharpness=args.min_sharpness,
            analysis_size=args.analysis_size,
        )
        print(json.dumps(report, indent=2))
        print(f"\nQuality reports saved to: {Path(args.output).with_suffix('.json').resolve()} and {Path(args.output).with_suffix('.csv').resolve()}")
    elif args.command == "model-info":
        with Path(args.config).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        model = build_model_from_config(config, pretrained_override=False if args.no_pretrained else None)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        classes = config.get("taxonomy", {}).get("classes", [])
        print(json.dumps({
            "backbone": "efficientnet_b0", "num_classes": len(classes), "classes": classes,
            "input_size": config.get("model", {}).get("input_size", 224),
            "total_parameters": total, "trainable_parameters": trainable,
            "pretrained_weights": bool(config.get("model", {}).get("pretrained", True)) and not args.no_pretrained,
        }, indent=2))
    elif args.command == "train":
        from crop_adaptive.training.engine import train_from_config
        with Path(args.config).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        summary = train_from_config(
            config, args.dataset, args.splits, args.output,
            epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, num_workers=args.num_workers,
            device_name=args.device,
            pretrained_override=False if args.no_pretrained else None,
        )
        print(json.dumps(summary, indent=2))
    elif args.command == "metadata-audit":
        from crop_adaptive.data.provenance import audit_metadata_provenance
        report = audit_metadata_provenance(args.manifest, args.output)
        print(json.dumps(report, indent=2))
        print(f"\nMetadata provenance audit saved to: {Path(args.output).resolve()}")
        queue = Path(args.output).with_name(Path(args.output).stem + "_recovery_queue.csv")
        print(f"Recovery worksheet saved to: {queue.resolve()}")
    elif args.command == "metadata-recover":
        from crop_adaptive.data.provenance import apply_approved_metadata_recovery
        report = apply_approved_metadata_recovery(args.manifest, args.recovery_queue, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "acquisition-audit":
        from crop_adaptive.data.acquisition import audit_acquisition_patterns
        report = audit_acquisition_patterns(args.dataset, args.manifest, args.output)
        print(json.dumps(report, indent=2))
        print(f"\nAcquisition audit saved to: {Path(args.output).resolve()}")
        print(f"Per-image features saved to: {Path(args.output).with_suffix('.csv').resolve()}")
    elif args.command == "bias-audit":
        from crop_adaptive.data.bias import analyze_acquisition_bias
        report = analyze_acquisition_bias(args.features, args.output)
        print(json.dumps(report, indent=2))
        print(f"\\nAcquisition bias report saved to: {Path(args.output).resolve()}")
    elif args.command == "review-errors":
        from crop_adaptive.data.review import create_misclassification_review
        summary = create_misclassification_review(args.dataset, args.misclassified, args.output,
                                                  per_sheet=args.per_sheet, columns=args.columns)
        print(json.dumps(summary, indent=2))
        print(f"\nHuman review files saved to: {Path(args.output).resolve()}")
    elif args.command == "evaluate":
        from crop_adaptive.training.evaluation import evaluate_from_config
        with Path(args.config).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        summary = evaluate_from_config(
            config, args.dataset, args.splits, args.checkpoint, args.output,
            batch_size=args.batch_size, num_workers=args.num_workers,
            device_name=args.device,
        )
        print(json.dumps(summary, indent=2))
        print(f"\nEvaluation reports saved to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
