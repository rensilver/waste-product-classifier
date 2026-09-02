from waste_product_classifier.vision import finetune


class FakeBaseModel:
    def __init__(self):
        self.layers = [_layer("block1_conv1"), _layer("block5_conv3")]


class _layer:
    def __init__(self, name):
        self.name = name
        self.trainable = True


class FakeModel:
    def __init__(self):
        self.layers = [FakeBaseModel()]
        self.compile_calls = []
        self.fit_calls = []

    def compile(self, loss, optimizer, metrics):
        self.compile_calls.append({"loss": loss, "optimizer": optimizer, "metrics": metrics})

    def fit(self, train_ds, epochs, validation_data, callbacks, verbose):
        self.fit_calls.append({"train_ds": train_ds, "epochs": epochs, "callbacks": callbacks})
        return "fake-history"


def test_fine_tune_model_loads_checkpoint_unfreezes_and_trains(config, monkeypatch):
    fake_model = FakeModel()
    unfreeze_calls = []
    plot_calls = []

    monkeypatch.setattr(finetune, "load_config", lambda: config)
    monkeypatch.setattr(finetune, "get_datasets", lambda cfg: ("train_ds", "val_ds", "test_ds", []))
    monkeypatch.setattr(finetune, "load_model", lambda path: fake_model)
    monkeypatch.setattr(finetune, "unfreeze_from", lambda base_model, layer_name: unfreeze_calls.append(
        (base_model, layer_name)
    ))
    monkeypatch.setattr(finetune, "build_callbacks", lambda checkpoint_path: ["callback"])
    monkeypatch.setattr(finetune, "plot_history", lambda *args, **kwargs: plot_calls.append(args))

    history = finetune.fine_tune_model()

    assert history == "fake-history"
    assert unfreeze_calls == [(fake_model.layers[0], finetune.UNFREEZE_FROM_LAYER)]
    assert fake_model.compile_calls[0]["loss"] == "binary_crossentropy"
    assert fake_model.fit_calls[0]["train_ds"] == "train_ds"
    assert fake_model.fit_calls[0]["callbacks"] == ["callback"]
    assert plot_calls
