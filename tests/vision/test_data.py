import numpy as np
import pytest
from keras import layers
from PIL import Image

from waste_product_classifier.vision.data import build_augmentation, count_images, get_datasets, validate_dirs


def _write_split_images(split_dir, class_names, images_per_class=2):
    for label in class_names:
        for i in range(images_per_class):
            path = split_dir / label / f"{label}{i}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), color=(i * 10, 0, 0)).save(path)


def test_count_images_only_counts_recognized_image_extensions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"fake")
    (tmp_path / "b.png").write_bytes(b"fake")
    (tmp_path / "c.txt").write_bytes(b"fake")

    assert count_images(tmp_path) == 2


def test_count_images_recurses_into_subdirectories(tmp_path):
    nested = tmp_path / "organic"
    nested.mkdir()
    (nested / "a.jpg").write_bytes(b"fake")
    (nested / "b.jpeg").write_bytes(b"fake")

    assert count_images(tmp_path) == 2


def test_count_images_returns_zero_for_empty_directory(tmp_path):
    assert count_images(tmp_path) == 0


def test_validate_dirs_raises_when_a_split_is_missing(config):
    config.train_dir.mkdir(parents=True)
    config.valid_dir.mkdir(parents=True)
    # test_dir intentionally not created

    with pytest.raises(FileNotFoundError, match="test_dir not found"):
        validate_dirs(config)


def test_validate_dirs_passes_when_all_splits_exist(config):
    config.train_dir.mkdir(parents=True)
    config.valid_dir.mkdir(parents=True)
    config.test_dir.mkdir(parents=True)

    validate_dirs(config)


def test_build_augmentation_contains_flip_and_translation_layers():
    augmentation = build_augmentation()

    layer_types = [type(layer) for layer in augmentation.layers]
    assert layers.RandomFlip in layer_types
    assert layers.RandomTranslation in layer_types


def test_get_datasets_returns_rescaled_batches_and_sorted_class_names(config):
    for split_dir in (config.train_dir, config.valid_dir, config.test_dir):
        _write_split_images(split_dir, ["organic", "recyclable"])

    train_ds, val_ds, test_ds, class_names = get_datasets(config)

    assert class_names == ["organic", "recyclable"]

    images, labels = next(iter(test_ds))
    assert images.numpy().min() >= 0.0
    assert images.numpy().max() <= 1.0
    assert labels.shape[-1] == 1
