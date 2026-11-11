import csv

from PIL import Image

from crop_adaptive.data.splitting import create_group_splits


def test_group_ids_never_cross_splits_and_seed_is_reproducible(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "manifest.csv"
    fields = [
        "image_path", "disease_label", "label_status",
        "plant_id", "field_id", "capture_session_id",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(12):
            name = f"leaf_{i}.jpg"
            Image.new("RGB", (32, 32)).save(dataset / name)
            writer.writerow({
                "image_path": name,
                "disease_label": "healthy" if i % 2 == 0 else "rust",
                "label_status": "verified",
                "plant_id": f"plant_{i // 2}",
                "field_id": f"field_{i // 4}",
                "capture_session_id": f"session_{i // 2}",
            })

    first = create_group_splits(dataset, manifest, tmp_path / "out1", seed=17)
    second = create_group_splits(dataset, manifest, tmp_path / "out2", seed=17)
    assert first["split_image_counts"] == second["split_image_counts"]

    memberships = {}
    for split in ("train", "validation", "test"):
        with (tmp_path / "out1" / f"{split}.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                for key in ("plant_id", "field_id", "capture_session_id"):
                    value = row[key]
                    if value:
                        memberships.setdefault((key, value), set()).add(split)
    assert all(len(splits) == 1 for splits in memberships.values())


def test_unverified_rows_excluded_by_default(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = tmp_path / "manifest.csv"
    fields = ["image_path", "disease_label", "label_status"]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, status in enumerate(["verified", "verified", "unverified", "verified"]):
            name = f"{i}.jpg"
            Image.new("RGB", (32, 32)).save(dataset / name)
            writer.writerow({"image_path": name, "disease_label": "healthy", "label_status": status})
    report = create_group_splits(dataset, manifest, tmp_path / "splits")
    assert report["eligible_images"] == 3
    assert report["excluded_unverified"] == 1
