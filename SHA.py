
import csv
import json
import hashlib
from pathlib import Path
from collections import Counter

ROOT = Path("datasets/sugarcane/raw/project_agml")
REPORT = Path("datasets/sugarcane/reports/near_duplicates.json")

with open(REPORT, encoding="utf-8") as f:
    report = json.load(f)

pairs = report["candidate_pairs"]

# 1. Cross-class candidate distribution
cross_class_by_distance = Counter()
strict_cross_class = []

for pair in pairs:
    a = pair["image_a"]
    b = pair["image_b"]
    distance = pair["hamming_distance"]

    class_a = a.split("/")[0]
    class_b = b.split("/")[0]

    if class_a != class_b:
        cross_class_by_distance[distance] += 1

        if distance <= 2:
            strict_cross_class.append({
                "image_a": a,
                "image_b": b,
                "class_a": class_a,
                "class_b": class_b,
                "hamming_distance": distance
            })

# 2. Exact file comparison using SHA-256
hash_groups = {}

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hash_groups.setdefault(digest, []).append(
            path.relative_to(ROOT).as_posix()
        )
    except OSError:
        continue

exact_duplicate_groups = [
    paths for paths in hash_groups.values()
    if len(paths) > 1
]

# 3. Save results
output_dir = Path("datasets/sugarcane/reports")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "cross_class_strict_candidates.csv",
          "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_a", "image_b", "class_a",
            "class_b", "hamming_distance"
        ]
    )
    writer.writeheader()
    writer.writerows(strict_cross_class)

with open(output_dir / "exact_duplicate_groups.json",
          "w", encoding="utf-8") as f:
    json.dump(exact_duplicate_groups, f, indent=2)

print("CROSS-CLASS PAIRS BY DISTANCE:")
for distance, count in sorted(cross_class_by_distance.items()):
    print(f"{distance}: {count}")

print("\nSTRICT CROSS-CLASS CANDIDATES:", len(strict_cross_class))
print("EXACT DUPLICATE GROUPS:", len(exact_duplicate_groups))

print("\nEXACT DUPLICATE EXAMPLES:")
for group in exact_duplicate_groups[:20]:
    print(group)

print("\nReports saved:")
print("cross_class_strict_candidates.csv")
print("exact_duplicate_groups.json")
