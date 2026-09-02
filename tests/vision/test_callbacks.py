import numpy as np
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint

from waste_product_classifier.vision.callbacks import build_callbacks, exp_decay, LossHistory


def test_exp_decay_at_epoch_zero_returns_initial_rate():
    assert exp_decay(0, initial_lrate=1e-4, k=0.1) == 1e-4


def test_exp_decay_decreases_with_epoch():
    assert exp_decay(5, initial_lrate=1e-4, k=0.1) < exp_decay(0, initial_lrate=1e-4, k=0.1)


def test_build_callbacks_does_not_raise(tmp_path):
    build_callbacks(tmp_path / "checkpoint.keras")


def test_build_callbacks_configures_early_stopping_to_minimize(tmp_path):
    callbacks = build_callbacks(tmp_path / "checkpoint.keras", patience=2, min_delta=0.5)
    early_stopping = next(cb for cb in callbacks if isinstance(cb, EarlyStopping))
    assert early_stopping.mode == "min"
    assert early_stopping.monitor == "val_loss"
    assert early_stopping.patience == 2


def test_build_callbacks_configures_checkpoint_to_minimize(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.keras"
    callbacks = build_callbacks(checkpoint_path)
    model_checkpoint = next(cb for cb in callbacks if isinstance(cb, ModelCheckpoint))
    assert model_checkpoint.mode == "min"
    assert model_checkpoint.monitor == "val_loss"


def test_build_callbacks_schedule_follows_exp_decay_with_given_rate_and_k(tmp_path):
    callbacks = build_callbacks(tmp_path / "checkpoint.keras", initial_lrate=2e-3, k=0.5)
    scheduler = next(cb for cb in callbacks if isinstance(cb, LearningRateScheduler))

    assert scheduler.schedule(3) == exp_decay(3, initial_lrate=2e-3, k=0.5)


def test_loss_history_records_loss_and_scheduled_lr_per_epoch():
    history = LossHistory()
    history.on_train_begin()

    history.on_epoch_end(0, logs={"loss": 0.5})
    history.on_epoch_end(1, logs={"loss": 0.3})

    assert history.losses == [0.5, 0.3]
    assert history.lr[0] == exp_decay(0)
    assert history.lr[1] == exp_decay(1)
