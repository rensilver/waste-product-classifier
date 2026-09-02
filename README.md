# Waste Product Classifier

A binary image classifier that sorts waste into **Organic** or **Recyclable**, built around a VGG16 transfer-learning pipeline, Grad-CAM explainability, and a benchmark against a zero-shot vision-language model — all explorable through a Streamlit app.

## Overview

- **Classification** — a VGG16-based CNN trained in two stages: feature extraction (frozen backbone) followed by fine-tuning (partial unfreeze from `block5_conv3` onward).
- **Explainability** — Grad-CAM heatmaps overlaid on the input image, showing which regions drove a prediction.
- **Benchmarking** — the fine-tuned CNN is compared against a zero-shot vision-language model (via [Ollama](https://ollama.com/), `qwen2.5vl`) on accuracy and latency.
- **Demo app** — a Streamlit UI with three tabs: model evaluation metrics, single-image classification with Grad-CAM, and CNN-vs-VLM benchmark browsing.

## Project Structure

```
src/waste_product_classifier/
├── app/          # Streamlit demo app
├── benchmark/    # CNN vs. zero-shot VLM benchmark
├── evaluation/   # Test-set evaluation and training-curve plotting
├── gradcam/      # Grad-CAM heatmap generation and overlay
├── vision/       # Model architecture, data pipeline, training, fine-tuning
├── config.py     # Central configuration (paths, hyperparameters, env overrides)
└── inference.py  # Shared score → (label, confidence) logic
tests/            # pytest suite mirroring the package layout
data/             # Image splits (train/valid/test), by class
artifacts/        # Generated checkpoints, metrics, and plots
```

## Technologies

| Purpose | Tools |
|---|---|
| Language / tooling | Python 3.11+, [uv](https://docs.astral.sh/uv/) |
| Deep learning | TensorFlow / Keras (`tensorflow-cpu`), VGG16 transfer learning |
| Image processing | OpenCV, Pillow |
| Explainability & plotting | Grad-CAM (custom), Matplotlib |
| Metrics | scikit-learn |
| Data handling | pandas, numpy |
| Zero-shot VLM benchmark | Ollama (`qwen2.5vl`) |
| Web app | Streamlit |
| Testing | pytest, pytest-cov, pytest-mock |

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- (Optional, for the VLM benchmark) [Ollama](https://ollama.com/) running locally with the `qwen2.5vl` model pulled

### Installation

```bash
git clone git@github.com:rensilver/waste-product-classifier.git
cd waste-product-classifier
uv sync
```

### Data layout

Images live under `data/o-vs-r-split/`, one subfolder per class (`O` = organic, `R` = recyclable):

```
data/o-vs-r-split/
├── train/{O,R}/
├── valid/{O,R}/
└── test/{O,R}/
```

`train/` and `test/` splits are included in this repo. **A `valid/` split is required by the training pipeline but is not included** — add your own validation split before running training or fine-tuning.

### Configuration

Defaults live in `config.py` and can be overridden via environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ADC_DATA_DIR` | `data/o-vs-r-split` | Root of the train/valid/test image splits |
| `ADC_ARTIFACTS_DIR` | `artifacts/` | Where checkpoints and plots are written |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server used for the VLM benchmark |
| `OLLAMA_MODEL` | `qwen2.5vl:3b` | Ollama model tag |

## How to Run

### Run the tests

```bash
uv run pytest
```

### Train the classifier

Feature extraction, then fine-tuning (fine-tuning loads the feature-extraction checkpoint, so run it second):

```bash
uv run python -c "from waste_product_classifier.vision.train import train_model; train_model()"
uv run python -c "from waste_product_classifier.vision.finetune import fine_tune_model; fine_tune_model()"
```

Both stages write checkpoints and accuracy/loss curves to `artifacts/`.

### Evaluate on the test set

```bash
uv run python -c "
from waste_product_classifier.config import load_config
from waste_product_classifier.evaluation.evaluation import evaluate_models
print(evaluate_models(load_config()))
"
```

### Benchmark CNN vs. zero-shot VLM

Requires a running Ollama instance with the model pulled:

```bash
ollama pull qwen2.5vl
ollama serve
```

Then:

```bash
uv run python -c "from waste_product_classifier.benchmark.benchmark import run_benchmark; run_benchmark()"
```

### Launch the Streamlit app

```bash
uv run streamlit run src/waste_product_classifier/app/app.py
```

Opens a browser with tabs for model metrics, single-image classification (with Grad-CAM), and CNN-vs-VLM benchmark browsing.

## License

No license file is currently included. Add one (e.g. MIT, Apache-2.0) before treating this as open source.
