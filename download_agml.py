
from datasets import load_dataset
from pathlib import Path
import csv

DATASET_ID = "Project-AgML/sugarcane_leaf_disease_classification"

OUTPUT_DIR = Path("datasets/sugarcane/raw/project_agml")
MANIFEST_PATH = Path("datasets/sugarcane/dataset_manifest.csv")

print("Downloading AgML sugarcane dataset...")
dataset = load_dataset(DATASET_ID, split="train")

class_names = dataset.features["label"].names

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

fields = [
    "image_path",
    "crop",
    "disease_label",
    "image_type",
    "source_dataset",
    "license",
    "label_status",
]

with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for index, sample in enumerate(dataset):
        label = class_names[sample["label"]]

        class_dir = OUTPUT_DIR / label.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)

        filename = f"image_{index:05d}.jpg"
        image_path = class_dir / filename

        sample["image"].convert("RGB").save(
            image_path,
            format="JPEG",
            quality=95
        )

        relative_path = image_path.as_posix().replace(
            "datasets/sugarcane/raw/", ""
        )

        writer.writerow({
            "image_path": relative_path,
            "crop": "sugarcane",
            "disease_label": label,
            "image_type": "leaf",
            "source_dataset": "Project-AgML",
            "license": "CC-BY-4.0",
            "label_status": "unverified",
        })

        if (index + 1) % 500 == 0:
            print(f"Exported {index + 1}/{len(dataset)} images")

print("\nDownload and extraction complete!")
print(f"Images: {len(dataset)}")
print(f"Classes: {class_names}")
print(f"Manifest: {MANIFEST_PATH}")
