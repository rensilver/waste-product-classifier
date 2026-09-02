import keras
from keras import Model, Sequential, layers

from waste_product_classifier.config import Config, load_config

def build_vgg16_base(input_shape: tuple[int, int, int]) -> Model:
    """Builds a VGG16 base model without the top classification layers."""
    vgg16_base = keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    output = vgg16_base.output
    output = layers.Flatten()(output)
    base_model = Model(vgg16_base.input, output, name="vgg16_base")

    for layer in base_model.layers:
        layer.trainable = False

    return base_model

def build_classifier(base_model: Model) -> Sequential:
    model = Sequential(name="waste_classifier")
    model.add(base_model)
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

def unfreeze_from(base_model: Model, layer_name: str) -> None:
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == layer_name:
            set_trainable = True
        layer.trainable = set_trainable

def build_model(config: Config) -> Sequential:
    base_model = build_vgg16_base(config.input_shape)
    model = build_classifier(base_model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model