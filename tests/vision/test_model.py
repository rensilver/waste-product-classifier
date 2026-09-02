import keras
import pytest

from waste_product_classifier.config import Config
from waste_product_classifier.vision import model as model_module
from waste_product_classifier.vision.model import (
    build_classifier,
    build_model,
    unfreeze_from,
)

INPUT_SHAPE = (8, 8, 3)


@pytest.fixture
def fake_vgg16(monkeypatch):
    """Stands in for keras.applications.VGG16: a tiny real Functional model,
    so tests stay fast and don't depend on network access or imagenet weights."""

    def _fake_vgg16(*, include_top, weights, input_shape):
        inputs = keras.Input(shape=input_shape)
        x = keras.layers.Conv2D(4, 3, padding="same", name="block1_conv1")(inputs)
        outputs = keras.layers.Conv2D(4, 3, padding="same", name="block1_conv2")(x)
        return keras.Model(inputs, outputs, name="fake_vgg16")

    monkeypatch.setattr(model_module.keras.applications, "VGG16", _fake_vgg16)


def test_build_vgg16_base_flattens_output(fake_vgg16):
    base = model_module.build_vgg16_base(INPUT_SHAPE)

    assert base.output_shape == (None, 8 * 8 * 4)


def test_build_vgg16_base_freezes_all_layers(fake_vgg16):
    base = model_module.build_vgg16_base(INPUT_SHAPE)

    assert all(not layer.trainable for layer in base.layers)


def test_build_classifier_ends_with_single_sigmoid_output(fake_vgg16):
    base = model_module.build_vgg16_base(INPUT_SHAPE)

    classifier = build_classifier(base)

    assert classifier.output_shape == (None, 1)
    assert classifier.layers[-1].activation.__name__ == "sigmoid"


def test_unfreeze_from_unfreezes_named_layer_and_everything_after(fake_vgg16):
    base = model_module.build_vgg16_base(INPUT_SHAPE)

    unfreeze_from(base, "block1_conv2")

    assert base.get_layer("block1_conv1").trainable is False
    assert base.get_layer("block1_conv2").trainable is True


def test_build_model_is_compiled_and_ready_to_train(fake_vgg16):
    config = Config(data_dir="unused", artifacts_dir="unused", img_rows=8, img_cols=8)

    model = build_model(config)

    assert model.output_shape == (None, 1)
    assert model.optimizer is not None
    assert model.loss == "binary_crossentropy"
