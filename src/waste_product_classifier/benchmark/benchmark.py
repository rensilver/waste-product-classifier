from pathlib import Path

import time
import requests
import pandas as pd

from keras.models import load_model
from keras import Model
from waste_product_classifier.config import load_config
from waste_product_classifier.gradcam.gradcam import get_class_names, load_and_preprocess_image
from waste_product_classifier.benchmark.vlm_zero_shot import OLLAMA_HOST, classify_with_vlm

VLM_MODEL = "qwen2.5v1"
RESULTS_FILENAME = "benchmark_results.csv"

# Keep this modest — VLM calls are much slower than a batched CNN forward pass
MAX_IMAGES_PER_CLASS = 100

def check_ollama_available():
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_HOST}. "
            "Is the container running? Try: docker compose up -d"
        ) from exc

def load_test_paths(test_dir: Path, class_names: list[str]) -> list[dict[str, Path | str]]:
    rows = []
    for label in class_names:
        paths = sorted((test_dir / label).glob("*.jpg"))[:MAX_IMAGES_PER_CLASS]
        for p in paths:
            rows.append({"path": p, "true_label": label})
    return rows

def predict_cnn(model: Model, img_path: Path, 
        target_size: tuple[int, int], class_names: list[str]) -> tuple[str, float, float]:
    img_array = load_and_preprocess_image(img_path, target_size)

    start = time.time()
    raw_score = float(model.predict(img_array, verbose=0)[0][0])
    latency = time.time() - start

    label = class_names[1] if raw_score >= 0.5 else class_names[0]
    confidence = raw_score if raw_score >= 0.5 else 1 - raw_score
    return label, confidence, latency

def run_benchmark() -> pd.DataFrame:
    check_ollama_available()
    config = load_config()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(config.model_path)
    class_names = get_class_names(config.test_dir)
    test_set = load_test_paths(config.test_dir, class_names)

    results = []
    for row in test_set:
        cnn_label, cnn_conf, cnn_latency = predict_cnn(
            model, row["path"], config.target_size, class_names
        )
        vlm_result = classify_with_vlm(row["path"], model_name=VLM_MODEL)

        results.append({
            "path": str(row["path"]),
            "true_label": row["true_label"],
            "cnn_label": cnn_label,
            "cnn_confidence": cnn_conf,
            "cnn_latency_s": cnn_latency,
            "cnn_correct": cnn_label == row["true_label"],
            "vlm_label": vlm_result["label"],
            "vlm_confidence": vlm_result["confidence"],
            "vlm_latency_s": vlm_result["latency_s"],
            "vlm_correct": vlm_result["label"] == row["true_label"]
        })

        df = pd.DataFrame(results)
        results_path = config.artifacts_dir / RESULTS_FILENAME
        df.to_csv(results_path, index=False)
        print_summary(df)
        return df

def print_summary(df: pd.DataFrame) -> None:
    summary = pd.DataFrame({
        "accuracy": [df["cnn_correct"].mean(), df["vlm_correct"].mean()],
        "avg_latency_s": [df["cnn_latency_s"].mean(), df["vlm_latency_s"].mean()],
    }, index=["Fine-tuned VGG16", f"Zero-shot VLM ({VLM_MODEL})"])
    print(summary.to_markdown())