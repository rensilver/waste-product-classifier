import logging

import keras
import numpy as np
from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint

logger = logging.getLogger(__name__)

def exp_decay(epoch: int, initial_lrate: float = 1e-4, k: float = 0.1) -> float:
    return initial_lrate * np.exp(-k * epoch)

class LossHistory(Callback):

    def on_train_begin(self, logs=None):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_lr = exp_decay(epoch)
        self.losses.append(logs.get("loss"))
        self.lr.append(current_lr)
        logger.info(
            "Epoch %d - loss: %.4f, scheduled lr: %.6f",
            epoch,
            logs.get("loss", float("nan")),
            current_lr
        )

def build_callbacks(
        checkpoint_path,
        *,
        initial_lrate: float = 1e-4,
        k: float = 0.1,
        patience: int = 4,
        min_delta: float = 0.01,
) -> list[keras.callbacks.Callback]:

    def schedule(epoch: int) -> float:
        return exp_decay(epoch, initial_lrate=initial_lrate, k=k)

    return [
        LossHistory(),
        LearningRateScheduler(schedule),
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            min_delta=min_delta
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min"
        )
    ]    