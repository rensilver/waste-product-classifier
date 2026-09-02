import pytest

from waste_product_classifier.config import Config


@pytest.fixture
def config(tmp_path) -> Config:
    """A Config pointed at an isolated tmp_path, so tests never touch real data/artifacts."""
    return Config(data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts")
