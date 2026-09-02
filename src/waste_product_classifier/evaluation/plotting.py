from pathlib import Path
import matplotlib.pyplot as plt

def plot_history(history: dict, accuracy_path: Path, 
        loss_path: Path, title_prefix: str = "") -> None:
    plot_metric(history, "accuracy", "val_accuracy", accuracy_path, f"{title_prefix}Accuracy")
    plot_metric(history, "loss", "val_loss", loss_path, f"{title_prefix}Loss")

def plot_metric(history: dict, metric_key: str, val_metric_key: str, 
        output_path: Path, title: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(history.history[metric_key], label="train")
    ax.plot(history.history[val_metric_key], label="validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key.capitalize())
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)