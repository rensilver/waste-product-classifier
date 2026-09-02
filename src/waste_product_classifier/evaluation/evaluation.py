import numpy as np
from sklearn import metrics

from keras.models import load_model
from waste_product_classifier.config import Config, load_config
from waste_product_classifier.gradcam.gradcam import get_class_names, load_and_preprocess_image
from waste_product_classifier.vision.train import FEATURE_EXTRACT_CHECKPOINT

def load_test_set(config: Config) -> tuple[np.ndarray, list[str], list[str]]:
    class_names = get_class_names(config.test_dir)

    image_paths = []
    true_labels = []
    for label in class_names:
        for path in sorted((config.test_dir / label).glob("*")):
            image_paths.append(path)
            true_labels.append(label)

    images = np.vstack(
        [load_and_preprocess_image(p, config.target_size) for p in image_paths]
    )
    return images, true_labels, class_names

def predict_labels(model, images, class_names) -> list[str]:
    raw_scores = model.predict(images, verbose=0).ravel()
    return [class_names[1] if score >= 0.5 else class_names[0] for score in raw_scores]

def evaluate_models(config: Config) -> dict:
    images, true_labels, class_names = load_test_set(config)

    feature_extract_path = config.artifacts_dir / FEATURE_EXTRACT_CHECKPOINT
    extract_feat_model = load_model(feature_extract_path)
    fine_tune_model = load_model(config.model_path)

    results = {}
    for name, model in [
        ("Feature Extraction", extract_feat_model),
        ("Fine-tuning", fine_tune_model)
    ]:
        predictions = predict_labels(model, images, class_names)
        results[name] = metrics.classification_report(
            true_labels, predictions, output_dict=True
        )

    return results