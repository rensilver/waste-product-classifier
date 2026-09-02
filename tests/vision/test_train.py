from waste_product_classifier.vision import train


class FakeModel:
    def __init__(self):
        self.fit_calls = []

    def fit(self, train_ds, epochs, validation_data, callbacks, verbose):
        self.fit_calls.append(
            {"train_ds": train_ds, "epochs": epochs, "validation_data": validation_data, "callbacks": callbacks}
        )
        return "fake-history"


def test_train_model_wires_datasets_model_and_callbacks_together(config, monkeypatch):
    fake_model = FakeModel()
    plot_calls = []

    monkeypatch.setattr(train, "load_config", lambda: config)
    monkeypatch.setattr(train, "get_datasets", lambda cfg: ("train_ds", "val_ds", "test_ds", ["organic", "recyclable"]))
    monkeypatch.setattr(train, "build_model", lambda cfg: fake_model)
    monkeypatch.setattr(train, "build_callbacks", lambda checkpoint_path: ["callback-a", "callback-b"])
    monkeypatch.setattr(train, "plot_history", lambda history, acc_path, loss_path, title_prefix: plot_calls.append(
        (history, acc_path, loss_path, title_prefix)
    ))

    history = train.train_model()

    assert history == "fake-history"
    assert fake_model.fit_calls[0]["train_ds"] == "train_ds"
    assert fake_model.fit_calls[0]["validation_data"] == "val_ds"
    assert fake_model.fit_calls[0]["epochs"] == config.n_epochs
    assert fake_model.fit_calls[0]["callbacks"] == ["callback-a", "callback-b"]
    assert plot_calls[0][0] == "fake-history"
    assert config.artifacts_dir.is_dir()


def test_train_model_creates_artifacts_dir_if_missing(config, monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(train, "load_config", lambda: config)
    monkeypatch.setattr(train, "get_datasets", lambda cfg: ("train_ds", "val_ds", "test_ds", []))
    monkeypatch.setattr(train, "build_model", lambda cfg: fake_model)
    monkeypatch.setattr(train, "build_callbacks", lambda checkpoint_path: [])
    monkeypatch.setattr(train, "plot_history", lambda *args, **kwargs: None)

    assert not config.artifacts_dir.exists()

    train.train_model()

    assert config.artifacts_dir.is_dir()
