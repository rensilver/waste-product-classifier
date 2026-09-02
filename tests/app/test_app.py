from types import SimpleNamespace

import numpy as np
from PIL import Image

from waste_product_classifier.app import app


def test_save_upload_to_tmp_writes_bytes_to_a_file_with_matching_suffix(tmp_path, monkeypatch):
    monkeypatch.setattr(app.tempfile, "gettempdir", lambda: str(tmp_path))
    uploaded_file = SimpleNamespace(name="photo.png", getvalue=lambda: b"image-bytes")

    saved_path = app._save_upload_to_tmp(uploaded_file)

    assert saved_path.suffix == ".png"
    assert saved_path.read_bytes() == b"image-bytes"


def test_classify_and_explain_labels_high_score_as_second_class(tmp_path, monkeypatch):
    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (10, 10)).save(image_path)
    overlay_path = tmp_path / "overlays" / "overlay.jpg"

    monkeypatch.setattr(app, "make_gradcam_heatmap", lambda img_array, model: (np.zeros((4, 4)), 0.9))
    monkeypatch.setattr(app, "overlay_heatmap", lambda img_path, heatmap, output_path: None)

    label, confidence = app._classify_and_explain(
        model=object(), class_names=["organic", "recyclable"],
        img_path=image_path, target_size=(10, 10), overlay_path=overlay_path,
    )

    assert label == "recyclable"
    assert confidence == 0.9
    assert overlay_path.parent.is_dir()
