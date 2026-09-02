import logging

from keras.models import load_model
from keras.optimizers import RMSprop

from waste_product_classifier.vision.callbacks import build_callbacks
from waste_product_classifier.config import load_config
from waste_product_classifier.vision.data import get_datasets
from waste_product_classifier.vision.model import unfreeze_from
from waste_product_classifier.vision.train import FEATURE_EXTRACT_CHECKPOINT
from waste_product_classifier.evaluation.plotting import plot_history

logger = logging.getLogger(__name__)

UNFREEZE_FROM_LAYER = "block5_conv3"
FINE_TUNE_LEARNING_RATE = 1e-4

def fine_tune_model():
    logging.basicConfig(level=logging.INFO)
    config = load_config()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = get_datasets(config)
    logger.info("Fine-tuning - classes: %s", class_names)

    feature_extractor_checkpoint = config.artifacts_dir / FEATURE_EXTRACT_CHECKPOINT
    model = load_model(feature_extractor_checkpoint)
    logger.info("Loaded feature-extraction checkpoint: %s", feature_extractor_checkpoint)

    # model.layers[0] is the VGG16 base (see model.build_classifier)
    base_model = model.layers[0]
    unfreeze_from(base_model, UNFREEZE_FROM_LAYER)
    for layer in base_model.layers:
        logger.debug("%s: trainable=%s", layer.name, layer.trainable)

    model.compile(
        loss="binary_crossentropy",
        optimizer=RMSprop(learning_rate=FINE_TUNE_LEARNING_RATE),
        metrics=["accuracy"]
    )

    callbacks = build_callbacks(config.model_path)

    history = model.fit(
        train_ds,
        epochs=config.n_epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    logger.info("Fine-tuning complete. Best checkpoint: %s", config.model_path)

    plot_history(
        history, 
        config.accuracy_curve_path,
        config.loss_curve_path,
        title_prefix="Fine-Tuned - "
    )
    logger.info(
        "Saved curves: %s, %s", config.accuracy_curve_path, config.loss_curve_path
    )

    return history