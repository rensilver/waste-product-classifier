import pytest

from waste_product_classifier.inference import classify_score


def test_classify_score_below_threshold_maps_to_first_class():
    label, confidence = classify_score(0.2, ["organic", "recyclable"])

    assert label == "organic"
    assert confidence == pytest.approx(0.8)


def test_classify_score_at_or_above_threshold_maps_to_second_class():
    label, confidence = classify_score(0.7, ["organic", "recyclable"])

    assert label == "recyclable"
    assert confidence == pytest.approx(0.7)


def test_classify_score_at_exact_threshold_maps_to_second_class():
    label, confidence = classify_score(0.5, ["organic", "recyclable"])

    assert label == "recyclable"
    assert confidence == pytest.approx(0.5)


def test_classify_score_accepts_a_custom_threshold():
    label, confidence = classify_score(0.6, ["organic", "recyclable"], threshold=0.7)

    assert label == "organic"
    assert confidence == pytest.approx(0.4)
