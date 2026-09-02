from types import SimpleNamespace

from waste_product_classifier.evaluation.plotting import plot_history, plot_metric


def _fake_history(**metrics):
    return SimpleNamespace(history=metrics)


def test_plot_metric_writes_an_image_file(tmp_path):
    history = _fake_history(loss=[0.9, 0.5, 0.2], val_loss=[1.0, 0.7, 0.4])
    output_path = tmp_path / "curves" / "loss.png"

    plot_metric(history, "loss", "val_loss", output_path, "Loss")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_history_writes_both_accuracy_and_loss_curves(tmp_path):
    history = _fake_history(
        accuracy=[0.5, 0.7], val_accuracy=[0.4, 0.6],
        loss=[0.9, 0.5], val_loss=[1.0, 0.7],
    )
    accuracy_path = tmp_path / "accuracy.png"
    loss_path = tmp_path / "loss.png"

    plot_history(history, accuracy_path, loss_path)

    assert accuracy_path.exists()
    assert loss_path.exists()
