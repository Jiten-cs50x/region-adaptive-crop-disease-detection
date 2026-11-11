import pytest
from crop_adaptive.training.evaluation import classification_metrics

def test_classification_metrics_perfect_predictions():
    result = classification_metrics([[3, 0], [0, 2]], ["healthy", "rust"])
    assert result["sample_count"] == 5
    assert result["accuracy"] == 1.0
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["per_class"]["rust"]["support"] == 2

def test_classification_metrics_handles_missing_predictions_for_a_class():
    result = classification_metrics([[4, 0], [2, 0]], ["healthy", "rust"])
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["per_class"]["rust"]["precision"] == 0.0
    assert result["per_class"]["rust"]["recall"] == 0.0
    assert result["per_class"]["rust"]["f1_score"] == 0.0
    assert result["macro_f1"] == pytest.approx(0.4)

def test_classification_metrics_rejects_empty_or_mismatched_matrix():
    with pytest.raises(ValueError, match="empty"):
        classification_metrics([[0, 0], [0, 0]], ["healthy", "rust"])
    with pytest.raises(ValueError, match="dimensions"):
        classification_metrics([[1]], ["healthy", "rust"])
