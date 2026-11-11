from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GROUP_COLUMNS = ("plant_id", "field_id", "capture_session_id")
SPLITS = ("train", "validation", "test")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _load_duplicate_membership(path: str | Path | None) -> dict[str, str]:
    """Load image_path -> duplicate group ID from the curation review CSV."""
    if path is None:
        return {}
    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Duplicate review CSV does not exist: {report_path}")
    membership: dict[str, str] = {}
    with report_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "group_id"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Duplicate review CSV missing columns: {', '.join(missing)}")
        for row in reader:
            image_path = (row.get("image_path") or "").strip().replace("\\", "/")
            group_id = (row.get("group_id") or "").strip()
            if image_path and group_id:
                previous = membership.get(image_path)
                if previous and previous != group_id:
                    raise ValueError(f"Image appears in multiple duplicate groups: {image_path}")
                membership[image_path] = group_id
    return membership


def create_group_splits(
    dataset_dir: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    verified_only: bool = True,
    duplicate_groups_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create deterministic group-aware CSV splits without copying images.

    Rows sharing plant, field, capture-session, or supplied exact-duplicate
    group identifiers are connected and kept in one split. Missing identifiers
    are treated as independent images and counted in the report.
    """
    root = Path(dataset_dir).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest}")

    duplicate_membership = _load_duplicate_membership(duplicate_groups_path)
    ratios = ratios or {"train": 0.70, "validation": 0.15, "test": 0.15}
    if set(ratios) != set(SPLITS) or any(value <= 0 for value in ratios.values()):
        raise ValueError("Ratios must contain positive train, validation, and test values.")
    total_ratio = sum(ratios.values())
    ratios = {name: value / total_ratio for name, value in ratios.items()}

    with manifest.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"image_path", "disease_label", "label_status"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
        rows = list(reader)

    eligible: list[dict[str, str]] = []
    excluded_unverified = 0
    excluded_invalid = 0
    seen_paths: set[str] = set()
    for row in rows:
        rel = (row.get("image_path") or "").strip().replace("\\", "/")
        label = (row.get("disease_label") or "").strip()
        status = (row.get("label_status") or "").strip().lower()
        candidate = (root / rel).resolve() if rel else root
        try:
            candidate.relative_to(root)
            safe = bool(rel) and candidate.is_file()
        except ValueError:
            safe = False
        if not safe or not label or rel in seen_paths:
            excluded_invalid += 1
            continue
        seen_paths.add(rel)
        if verified_only and status != "verified":
            excluded_unverified += 1
            continue
        eligible.append(row)

    if len(eligible) < 3:
        raise ValueError("At least three eligible images are required to create three splits.")

    uf = _UnionFind(len(eligible))
    owner: dict[tuple[str, str], int] = {}
    linked_by_duplicate_group = 0
    for index, row in enumerate(eligible):
        rel = (row.get("image_path") or "").strip().replace("\\", "/")
        keys = [
            (column, (row.get(column) or "").strip())
            for column in GROUP_COLUMNS
            if (row.get(column) or "").strip()
        ]
        duplicate_group = duplicate_membership.get(rel, "")
        if duplicate_group:
            keys.append(("exact_duplicate_group", duplicate_group))
            linked_by_duplicate_group += 1
        for key in keys:
            if key in owner:
                uf.union(index, owner[key])
            else:
                owner[key] = index

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(eligible)):
        groups[uf.find(index)].append(index)

    rng = random.Random(seed)
    group_items = list(groups.values())
    rng.shuffle(group_items)
    group_items.sort(key=len, reverse=True)

    labels = sorted({(row.get("disease_label") or "").strip() for row in eligible})
    total_by_label = Counter((row.get("disease_label") or "").strip() for row in eligible)
    targets = {
        split: {label: total_by_label[label] * ratios[split] for label in labels}
        for split in SPLITS
    }
    assigned: dict[str, list[int]] = {split: [] for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    sizes = Counter()

    for group in group_items:
        group_counts = Counter((eligible[i].get("disease_label") or "").strip() for i in group)
        best_split = None
        best_score = None
        for split in SPLITS:
            score = 0.0
            for label in labels:
                after = counts[split][label] + group_counts[label]
                target = targets[split][label]
                score += ((after - target) / max(target, 1.0)) ** 2
            size_target = len(eligible) * ratios[split]
            score += 0.25 * (((sizes[split] + len(group) - size_target) / max(size_target, 1.0)) ** 2)
            score += rng.random() * 1e-9
            if best_score is None or score < best_score:
                best_score, best_split = score, split
        assert best_split is not None
        assigned[best_split].extend(group)
        counts[best_split].update(group_counts)
        sizes[best_split] += len(group)

    output.mkdir(parents=True, exist_ok=True)
    assignment = {}
    for split, indices in assigned.items():
        split_rows = []
        for index in indices:
            row = dict(eligible[index])
            row["split"] = split
            split_rows.append(row)
            assignment[index] = split
        fields = list(eligible[0].keys()) + ["split"]
        with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows)

    for group in group_items:
        split_names = {assignment[index] for index in group}
        if len(split_names) != 1:
            raise RuntimeError("Internal error: a linked group crossed split boundaries.")

    report = {
        "dataset_root": str(root),
        "manifest": str(manifest),
        "seed": seed,
        "ratios": ratios,
        "verified_only": verified_only,
        "duplicate_groups_path": str(Path(duplicate_groups_path).expanduser().resolve()) if duplicate_groups_path else None,
        "eligible_images": len(eligible),
        "excluded_unverified": excluded_unverified,
        "excluded_invalid_or_duplicate_paths": excluded_invalid,
        "groups": len(groups),
        "images_linked_by_exact_duplicate_group": linked_by_duplicate_group,
        "images_without_group_metadata": sum(
            not any((row.get(column) or "").strip() for column in GROUP_COLUMNS)
            and (row.get("image_path") or "").strip().replace("\\", "/") not in duplicate_membership
            for row in eligible
        ),
        "split_image_counts": {split: len(assigned[split]) for split in SPLITS},
        "split_class_counts": {split: dict(sorted(counts[split].items())) for split in SPLITS},
        "notes": [
            "Images connected by plant_id, field_id, capture_session_id, or a supplied exact-duplicate group are assigned to one split.",
            "Rows without group identifiers are treated as individual images; unknown relationships cannot be ruled out.",
            "Only verified labels are included by default; --include-unverified is an explicit opt-in for source-labelled data requiring later review.",
            "Splits reference original image paths; images are not copied or modified.",
            "Keep any regional adaptation holdout independent of these generic splits.",
        ],
    }
    (output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
