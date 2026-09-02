from pathlib import Path

from waste_product_classifier.config import Config, load_config


def test_config_derives_split_dirs_from_data_dir():
    config = Config(data_dir=Path("/data"), artifacts_dir=Path("/artifacts"))

    assert config.train_dir == Path("/data/train")
    assert config.valid_dir == Path("/data/valid")
    assert config.test_dir == Path("/data/test")


def test_config_derives_artifact_paths_from_artifacts_dir():
    config = Config(data_dir=Path("/data"), artifacts_dir=Path("/artifacts"))

    assert config.model_path == Path("/artifacts/vgg16_waste_product_classifier.keras")
    assert config.metrics_path == Path("/artifacts/metrics.json")
    assert config.accuracy_curve_path == Path("/artifacts/accuracy_curve.png")
    assert config.loss_curve_path == Path("/artifacts/loss_curve.png")


def test_config_target_size_and_input_shape_derive_from_rows_and_cols():
    config = Config(data_dir=Path("/data"), artifacts_dir=Path("/artifacts"), img_rows=100, img_cols=120)

    assert config.target_size == (100, 120)
    assert config.input_shape == (100, 120, 3)


def test_load_config_uses_defaults_when_no_env_vars_set(monkeypatch):
    monkeypatch.delenv("ADC_DATA_DIR", raising=False)
    monkeypatch.delenv("ADC_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    config = load_config()

    assert config.data_dir.name == "o-vs-r-split"
    assert config.artifacts_dir.name == "artifacts"
    assert config.ollama_host == "http://localhost:11434"
    assert config.ollama_model == "qwen2.5vl:3b"


def test_load_config_reads_overrides_from_env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("ADC_DATA_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("ADC_ARTIFACTS_DIR", str(tmp_path / "custom-artifacts"))
    monkeypatch.setenv("OLLAMA_HOST", "http://example.com:1234")
    monkeypatch.setenv("OLLAMA_MODEL", "custom-model")

    config = load_config()

    assert config.data_dir == tmp_path / "custom-data"
    assert config.artifacts_dir == tmp_path / "custom-artifacts"
    assert config.ollama_host == "http://example.com:1234"
    assert config.ollama_model == "custom-model"
