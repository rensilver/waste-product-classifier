import numpy as np
from PIL import Image

from waste_product_classifier.evaluation import evaluation


class FakeModel:
    def __init__(self, scores):
        self._scores = scores

    def predict(self, images, verbose=0):
        return np.array(self._scores).reshape(-1, 1)


def test_predict_labels_maps_scores_below_half_to_first_class():
    labels = evaluation.predict_labels(FakeModel([0.1, 0.4]), np.zeros((2, 1)), ["organic", "recyclable"])

    assert labels == ["organic", "organic"]


def test_predict_labels_maps_scores_at_or_above_half_to_second_class():
    labels = evaluation.predict_labels(FakeModel([0.5, 0.9]), np.zeros((2, 1)), ["organic", "recyclable"])

    assert labels == ["recyclable", "recyclable"]


def test_load_test_set_reads_every_image_with_matching_true_label(config):
    for label, count in [("organic", 2), ("recyclable", 1)]:
        for i in range(count):
            path = config.test_dir / label / f"{label}{i}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10)).save(path)

    images, true_labels, class_names = evaluation.load_test_set(config)

    assert class_names == ["organic", "recyclable"]
    assert images.shape[0] == 3
    assert true_labels.count("organic") == 2
    assert true_labels.count("recyclable") == 1


def test_evaluate_models_reports_metrics_for_both_checkpoints(config, monkeypatch):
    for label in ("organic", "recyclable"):
        path = config.test_dir / label / f"{label}0.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (10, 10)).save(path)

    monkeypatch.setattr(evaluation, "load_model", lambda path: FakeModel([0.1, 0.9]))

    results = evaluation.evaluate_models(config)

    assert set(results.keys()) == {"Feature Extraction", "Fine-tuning"}
    assert "accuracy" in results["Feature Extraction"]
