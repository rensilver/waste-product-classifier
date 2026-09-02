import numpy as np
import pytest
import requests
from PIL import Image

from waste_product_classifier.benchmark import benchmark


def _write_fake_jpg(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="white").save(path)


@pytest.fixture
def populated_config(config):
    for label, count in [("organic", 2), ("recyclable", 2)]:
        for i in range(count):
            _write_fake_jpg(config.test_dir / label / f"img{i}.jpg")
    return config


class FakeModel:
    """Always predicts the second class with high confidence."""

    def predict(self, img_array, verbose=0):
        return np.array([[0.9]])


def test_check_ollama_available_raises_helpful_error_when_unreachable(monkeypatch):
    def _raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("boom")

    monkeypatch.setattr(benchmark.requests, "get", _raise_connection_error)

    with pytest.raises(RuntimeError, match="docker compose up -d"):
        benchmark.check_ollama_available()


def test_load_test_paths_pairs_each_image_with_its_class_label(populated_config):
    rows = benchmark.load_test_paths(populated_config.test_dir, ["organic", "recyclable"])

    assert len(rows) == 4
    assert {row["true_label"] for row in rows} == {"organic", "recyclable"}


def test_load_test_paths_caps_images_per_class(populated_config, monkeypatch):
    monkeypatch.setattr(benchmark, "MAX_IMAGES_PER_CLASS", 1)

    rows = benchmark.load_test_paths(populated_config.test_dir, ["organic", "recyclable"])

    assert len(rows) == 2


def test_predict_cnn_labels_high_score_as_second_class(populated_config):
    image_path = next((populated_config.test_dir / "organic").glob("*.jpg"))

    label, confidence, latency = benchmark.predict_cnn(
        FakeModel(), image_path, populated_config.target_size, ["organic", "recyclable"]
    )

    assert label == "recyclable"
    assert confidence == pytest.approx(0.9)
    assert latency >= 0


def test_run_benchmark_covers_every_image_in_the_test_set(populated_config, monkeypatch):
    """Regression test: run_benchmark used to return after the first image because
    the DataFrame/CSV/return logic was indented inside the per-image loop."""
    monkeypatch.setattr(benchmark, "check_ollama_available", lambda: None)
    monkeypatch.setattr(benchmark, "load_model", lambda path: FakeModel())
    monkeypatch.setattr(benchmark, "load_config", lambda: populated_config)

    vlm_calls = []

    def fake_classify_with_vlm(path, model_name):
        vlm_calls.append(model_name)
        return {"label": "organic", "confidence": 0.5, "latency_s": 0.01}

    monkeypatch.setattr(benchmark, "classify_with_vlm", fake_classify_with_vlm)

    df = benchmark.run_benchmark()

    assert len(df) == 4
    assert len(vlm_calls) == 4
    assert all(model_name == "qwen2.5vl:3b" for model_name in vlm_calls)
    assert (populated_config.artifacts_dir / benchmark.RESULTS_FILENAME).exists()
