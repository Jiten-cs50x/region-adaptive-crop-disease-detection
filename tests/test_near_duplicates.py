from PIL import Image, ImageDraw

from crop_adaptive.data.near_duplicates import (
    difference_hash,
    find_near_duplicates,
    hamming_distance,
)


def test_difference_hash_is_identical_after_resize_and_reencode(tmp_path):
    original = Image.new("RGB", (160, 120), "white")
    draw = ImageDraw.Draw(original)
    draw.ellipse((25, 15, 120, 105), fill="green")
    draw.line((70, 20, 70, 110), fill="black", width=4)
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    original.save(first)
    original.resize((320, 240)).save(second, quality=88)

    assert hamming_distance(difference_hash(first), difference_hash(second)) <= 6


def test_near_duplicate_report_contains_pair_and_does_not_modify_images(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 80, 80), fill="darkgreen")
    image.save(dataset / "a.png")
    image.resize((200, 200)).save(dataset / "b.png")

    before = (dataset / "a.png").read_bytes()
    report = find_near_duplicates(dataset, tmp_path / "report.json", threshold=6)

    assert report["candidate_pair_count"] >= 1
    assert report["candidate_groups"]
    assert (dataset / "a.png").read_bytes() == before


def test_invalid_threshold_is_rejected(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    try:
        find_near_duplicates(dataset, tmp_path / "report.json", threshold=65)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected threshold validation error")
