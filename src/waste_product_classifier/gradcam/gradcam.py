from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tensorflow as tf

from keras.utils import load_img, img_to_array
from keras.models import Model
from keras.layers import Conv2D

def get_class_names(directory: Path) -> list[str]:
    return sorted(p.name for p in Path(directory).iterdir() if p.is_dir())

def load_and_preprocess_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0

def find_last_conv_layer(model: Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue
    raise ValueError("No Conv2D layer found in the model.")

def make_gradcam_heatmap(img_array: np.ndarray, model: Model, last_conv_layer_name=None) -> tuple[np.ndarray, float]:
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(predictions[0][0])

def overlay_heatmap(img_path: Path, heatmap: np.ndarray, alpha: float = 0.4, output_path: Path = None) -> np.ndarray:
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    superimposed = jet_heatmap * alpha + img
    superimposed = np.uint8(np.clip(superimposed, 0, 255))

    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))

    return superimposed