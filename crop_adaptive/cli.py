import argparse
import json
from pathlib import Path

import yaml

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

    split = commands.add_parser("split", help="Create deterministic group-aware train/validation/test CSVs")
    split.add_argument("--dataset", required=True)
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", default="datasets/splits")
    split.add_argument("--train-ratio", type=float, default=0.70)
    split.add_argument("--validation-ratio", type=float, default=0.15)
    split.add_argument("--test-ratio", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--include-unverified", action="store_true")

    near = commands.add_parser("near-duplicates", help="Find perceptually similar images for manual review")
    near.add_argument("--dataset", required=True)
    near.add_argument("--output", default="reports/near_duplicates.json")
    near.add_argument("--threshold", type=int, default=6, help="Maximum 64-bit hash Hamming distance")
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
    model_info.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet pretrained weights")

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
    elif args.command == "split":
        report = create_group_splits(
            args.dataset,
            args.manifest,
            args.output,
            ratios={
                "train": args.train_ratio,
                "validation": args.validation_ratio,
                "test": args.test_ratio,
            },
            seed=args.seed,
            verified_only=not args.include_unverified,
        )
        print(json.dumps(report, indent=2))
        print(f"\nSplit files saved to: {Path(args.output).resolve()}")
    elif args.command == "near-duplicates":
        report = find_near_duplicates(args.dataset, args.output, args.threshold, args.hash_size)
        print(json.dumps(report, indent=2))
        print(f"\nNear-duplicate report saved to: {Path(args.output).resolve()}")
    elif args.command == "quality":
        report = analyze_image_quality(
            args.dataset,
            args.output,
            min_width=args.min_width,
            min_height=args.min_height,
            min_brightness=args.min_brightness,
            max_brightness=args.max_brightness,
            min_contrast=args.min_contrast,
            min_sharpness=args.min_sharpness,
            analysis_size=args.analysis_size,
        )
        print(json.dumps(report, indent=2))
        print(f"\nQuality reports saved to: {Path(args.output).resolve()} and {Path(args.output).with_suffix('.csv').resolve()}")
    elif args.command == "model-info":
        with Path(args.config).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        model = build_model_from_config(
            config,
            pretrained_override=False if args.no_pretrained else None,
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        classes = config.get("taxonomy", {}).get("classes", [])
        print(json.dumps({
            "backbone": "efficientnet_b0",
            "num_classes": len(classes),
            "classes": classes,
            "input_size": config.get("model", {}).get("input_size", 224),
            "total_parameters": total,
            "trainable_parameters": trainable,
            "pretrained_weights": bool(config.get("model", {}).get("pretrained", True)) and not args.no_pretrained,
        }, indent=2))


if __name__ == "__main__":
    main()
