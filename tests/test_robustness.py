from PIL import Image, ImageChops
import pytest

from crop_adaptive.training.robustness import CONDITIONS, perturb_image


@pytest.mark.parametrize("condition,severity", [
    (condition, severity)
    for condition, severities in CONDITIONS.items()
    for severity in severities
])
def test_perturbation_returns_same_size_rgb_copy(condition, severity):
    image = Image.new("RGB", (128, 96), (120, 140, 160))
    original = image.copy()
    result = perturb_image(image, condition, severity)
    assert result.mode == "RGB"
    assert result.size == image.size
    assert ImageChops.difference(image, original).getbbox() is None
    assert result is not image


def test_perturbation_rejects_unknown_condition_and_severity():
    image = Image.new("RGB", (32, 32), "green")
    with pytest.raises(ValueError, match="Unsupported robustness condition"):
        perturb_image(image, "background_swap", 1)
    with pytest.raises(ValueError, match="Unsupported severity"):
        perturb_image(image, "blur", 99)
