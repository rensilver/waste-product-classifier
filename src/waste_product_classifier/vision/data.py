from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf
from keras import Sequential
from keras.utils import image_dataset_from_directory
from keras import layers

from waste_product_classifier.config import Config, load_config

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def count_images(directory: Path) -> int:
    """Counts image files under a directory (recursively), matching the
    extensions Keras's image_dataset_from_directory recognizes."""
    return sum(1 for p in Path(directory).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

def validate_dirs(config: Config) -> None:
    for name, path in [
        ("train_dir", config.train_dir),
        ("valid_dir", config.valid_dir),
        ("test_dir", config.test_dir)
    ]:
        if not Path(path).is_dir():
            raise FileNotFoundError(f"{name} não encontrado: {path}")

def load_dataset(directory: Path, config: Config, *, shuffle: bool) -> tf.data.Dataset:
    return image_dataset_from_directory(
        directory,
        seed=config.seed if shuffle else None,
        image_size=config.target_size,
        batch_size=config.batch_size,
        label_mode="binary",
        shuffle=shuffle
    )

def build_augmentation() -> Sequential:
    return Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        ],
        name="augmentation"
    )

def get_datasets(
        config: Config,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:

    validate_dirs(config)

    train_ds = load_dataset(config.train_dir, config, shuffle=True)
    val_ds = load_dataset(config.valid_dir, config, shuffle=True)
    test_ds = load_dataset(config.test_dir, config, shuffle=False) # keep order stable for evaluation/confusion matrix

    class_names = train_ds.class_names

    logger.info(
        "Loaded %d train / %d valid / %d test images across classes %s",
        count_images(config.train_dir),
        count_images(config.valid_dir),
        count_images(config.test_dir),
        class_names
    )

    rescale = layers.Rescaling(1.0 / 255.0)
    augment = build_augmentation()

    raw_datasets = {
        "train": (train_ds, True),
        "val": (val_ds, False),
        "test": (test_ds, False)
    }

    processed = {}

    for name, (ds, training) in raw_datasets.items():
        if training:
            ds = ds.map(
                lambda x, y: (augment(rescale(x), training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            ds = ds.map(
                lambda x, y: (rescale(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        processed[name] = ds.prefetch(tf.data.AUTOTUNE)

    return processed["train"], processed["val"], processed["test"], class_names