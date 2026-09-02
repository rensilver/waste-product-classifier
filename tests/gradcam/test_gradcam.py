from types import SimpleNamespace

import keras
import numpy as np
import pytest
from PIL import Image

from waste_product_classifier.gradcam.gradcam import (
    find_last_conv_layer,
    get_class_names,
    load_and_preprocess_image,
    make_gradcam_heatmap,
    overlay_heatmap,
)


def _tiny_classifier(input_shape=(16, 16, 3)):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(2, 3, padding="same", name="only_conv")(inputs)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def test_get_class_names_returns_sorted_subdirectory_names(tmp_path):
    (tmp_path / "recyclable").mkdir()
    (tmp_path / "organic").mkdir()
    (tmp_path / "notes.txt").write_text("not a class")

    assert get_class_names(tmp_path) == ["organic", "recyclable"]


def test_load_and_preprocess_image_rescales_to_unit_range_with_batch_dim(tmp_path):
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (32, 32), color=(255, 128, 0)).save(image_path)

    array = load_and_preprocess_image(image_path, target_size=(16, 16))

    assert array.shape == (1, 16, 16, 3)
    assert array.max() <= 1.0
    assert array.min() >= 0.0


def test_find_last_conv_layer_finds_only_conv_layer():
    model = _tiny_classifier()

    assert find_last_conv_layer(model) == "only_conv"


def test_find_last_conv_layer_recurses_into_nested_submodels():
    inner = keras.Sequential([keras.layers.Conv2D(2, 3, padding="same", name="nested_conv")])
    inner.build((None, 8, 8, 3))
    inputs = keras.Input(shape=(8, 8, 3))
    outputs = inner(inputs)
    model = keras.Model(inputs, outputs)

    assert find_last_conv_layer(model) == "nested_conv"


def test_find_last_conv_layer_skips_nested_submodel_without_conv_and_keeps_searching():
    dense_leaf = SimpleNamespace(name="dense_leaf")
    empty_submodel = SimpleNamespace(name="empty_submodel", layers=[dense_leaf])
    conv_layer = keras.layers.Conv2D(2, 3, padding="same", name="real_conv")
    model = SimpleNamespace(layers=[conv_layer, empty_submodel])

    assert find_last_conv_layer(model) == "real_conv"


def test_find_last_conv_layer_raises_when_no_conv_layer_present():
    inputs = keras.Input(shape=(4,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)

    with pytest.raises(ValueError, match="No Conv2D layer found"):
        find_last_conv_layer(model)


def test_make_gradcam_heatmap_returns_normalized_heatmap_and_score():
    model = _tiny_classifier()
    img_array = np.random.rand(1, 16, 16, 3).astype("float32")

    heatmap, score = make_gradcam_heatmap(img_array, model, last_conv_layer_name="only_conv")

    assert heatmap.shape == (16, 16)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    assert isinstance(score, float)


def test_make_gradcam_heatmap_auto_detects_last_conv_layer_when_not_given():
    model = _tiny_classifier()
    img_array = np.random.rand(1, 16, 16, 3).astype("float32")

    heatmap, score = make_gradcam_heatmap(img_array, model)

    assert heatmap.shape == (16, 16)
    assert isinstance(score, float)


def test_overlay_heatmap_matches_original_image_dimensions(tmp_path):
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
    heatmap = np.random.rand(4, 4).astype("float32")

    overlay = overlay_heatmap(image_path, heatmap)

    assert overlay.shape == (10, 20, 3)


def test_overlay_heatmap_does_not_use_deprecated_colormap_api(tmp_path, recwarn):
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
    heatmap = np.random.rand(4, 4).astype("float32")

    overlay_heatmap(image_path, heatmap)

    assert not any("get_cmap" in str(w.message) for w in recwarn.list)


def test_overlay_heatmap_writes_output_file_when_path_given(tmp_path):
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(image_path)
    heatmap = np.random.rand(4, 4).astype("float32")
    output_path = tmp_path / "overlay.jpg"

    overlay_heatmap(image_path, heatmap, output_path=output_path)

    assert output_path.exists()
