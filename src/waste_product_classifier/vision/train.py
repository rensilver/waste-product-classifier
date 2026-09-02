import logging

from waste_product_classifier.vision.callbacks import build_callbacks
from waste_product_classifier.config import load_config
from waste_product_classifier.vision.data import get_datasets
from waste_product_classifier.vision.model import build_model
from waste_product_classifier.evaluation.plotting import plot_history

logger = logging.getLogger(__name__)

FEATURE_EXTRACT_CHECKPOINT = "feature_extract_vgg16.keras"
FEATURE_EXTRACT_ACCURACY_CURVE = "feature_extract_accuracy_curve.png"
FEATURE_EXTRACT_LOSS_CURVE = "feature_extract_loss_curve.png"

def train_model():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _, class_names = get_datasets(config)
    logger.info("Feature-extraction training - classes: %s", class_names)

    model = build_model(config)
    checkpoint_path = config.artifacts_dir / FEATURE_EXTRACT_CHECKPOINT
    callbacks = build_callbacks(checkpoint_path)

    history = model.fit(
        train_ds,
        epochs=config.n_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Feature-extraction training complete. Best checkpoint: %s", checkpoint_path)

    accuracy_path = config.artifacts_dir / FEATURE_EXTRACT_ACCURACY_CURVE
    loss_path = config.artifacts_dir / FEATURE_EXTRACT_LOSS_CURVE
    plot_history(
        history, 
        accuracy_path,
        loss_path,
        title_prefix="Feature-Extraction - "
    )
    logger.info(
        "Saved curves: %s, %s", accuracy_path, loss_path
    )

    return history