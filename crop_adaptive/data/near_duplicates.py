from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def difference_hash(path: str | Path, hash_size: int = 8) -> int:
    """Return a horizontal difference hash as an integer."""
    if hash_size < 2:
        raise ValueError("hash_size must be at least 2")
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("L")
        image = image.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())
    bits = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


class _BKTree:
    """Metric index for finding hashes within a Hamming-distance threshold."""

    def __init__(self) -> None:
        self.root: tuple[int, list[str], dict[int, Any]] | None = None

    def add(self, value: int, path: str) -> None:
        if self.root is None:
            self.root = (value, [path], {})
            return
        node = self.root
        while True:
            distance = hamming_distance(value, node[0])
            if distance == 0:
                node[1].append(path)
                return
            child = node[2].get(distance)
            if child is None:
                node[2][distance] = (value, [path], {})
                return
            node = child

    def search(self, value: int, threshold: int) -> list[tuple[int, str]]:
        found: list[tuple[int, str]] = []
        if self.root is None:
            return found
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node[0])
            if distance <= threshold:
                found.extend((distance, path) for path in node[1])
            low, high = distance - threshold, distance + threshold
            stack.extend(child for edge, child in node[2].items() if low <= edge <= high)
        return found


def find_near_duplicates(
    dataset_dir: str | Path,
    output_path: str | Path,
    threshold: int = 6,
    hash_size: int = 8,
) -> dict[str, Any]:
    """Find perceptually similar image pairs without changing source images.

    A difference hash is a screening signal, not proof that two images depict
    the same leaf, plant, or disease. Review reported pairs before exclusion.
    """
    root = Path(dataset_dir).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    max_distance = hash_size * hash_size
    if threshold < 0 or threshold > max_distance:
        raise ValueError(f"threshold must be between 0 and {max_distance}")

    index = _BKTree()
    pairs: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    image_count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            image_hash = difference_hash(path, hash_size)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            unreadable.append({"image_path": relative, "error": type(exc).__name__})
            continue
        for distance, previous in index.search(image_hash, threshold):
            pairs.append({
                "image_a": previous,
                "image_b": relative,
                "hamming_distance": distance,
                "review_recommended": True,
            })
        index.add(image_hash, relative)
        image_count += 1

    # Build connected components from candidate pairs for convenient manual review.
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for pair in pairs:
        union(pair["image_a"], pair["image_b"])
    components: dict[str, list[str]] = defaultdict(list)
    for item in parent:
        components[find(item)].append(item)
    groups = [sorted(items) for items in components.values() if len(items) > 1]

    report = {
        "dataset_root": str(root),
        "image_count_hashed": image_count,
        "unreadable_images": unreadable,
        "hash_algorithm": "difference_hash",
        "hash_size": hash_size,
        "maximum_hamming_distance": threshold,
        "candidate_pair_count": len(pairs),
        "candidate_groups": sorted(groups),
        "candidate_pairs": pairs,
        "notes": [
            "Perceptual similarity is a review signal, not a verified duplicate diagnosis.",
            "Difference hash is useful for resized or recompressed copies but may miss crops, rotations, and substantial edits.",
            "Similar-looking leaves from different plants can be legitimate distinct samples.",
            "Do not automatically delete or exclude candidates; review them with provenance and group metadata.",
            "Original images are never modified."
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
