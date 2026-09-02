import tempfile
from pathlib import Path
 
import keras
import pandas as pd
import streamlit as st
 
from waste_product_classifier.config import load_config
from waste_product_classifier.evaluation.evaluation import evaluate_models
from waste_product_classifier.gradcam.gradcam import get_class_names, load_and_preprocess_image, make_gradcam_heatmap, overlay_heatmap
from waste_product_classifier.inference import classify_score
from waste_product_classifier.vision.train import FEATURE_EXTRACT_ACCURACY_CURVE, FEATURE_EXTRACT_LOSS_CURVE
from waste_product_classifier.benchmark.vlm_zero_shot import classify_with_vlm
 
st.set_page_config(page_title="Waste Classifier", layout="wide")
 
@st.cache_resource
def get_config():
    return load_config()
 
@st.cache_resource
def load_final_model(_config):
    return keras.models.load_model(_config.model_path)
 
def _save_upload_to_tmp(uploaded_file) -> Path:
    """Streamlit's uploader gives an in-memory file; cv2/keras.utils.load_img
    both expect a filesystem path, so we write it to a temp file once."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)
 
def _classify_and_explain(model, class_names, img_path, target_size, overlay_path):
    """Shared CNN + Grad-CAM logic used by both the Classify and Benchmark tabs."""
    img_array = load_and_preprocess_image(img_path, target_size)
    heatmap, raw_score = make_gradcam_heatmap(img_array, model)
    label, confidence = classify_score(raw_score, class_names)
 
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_heatmap(img_path, heatmap, output_path=overlay_path)
    return label, confidence
 
def render_metrics_tab(config):
    st.header("Model Evaluation Metrics")
    st.caption("Classification report on the held-out test set, for both training stages.")
 
    if st.button("Run evaluation", key="run_eval"):
        with st.spinner("Loading models and scoring the test set..."):
            reports = evaluate_models(config)
 
        for stage_name, report in reports.items():
            st.subheader(stage_name)
            accuracy = report.pop("accuracy", None)
            if accuracy is not None:
                st.metric("Overall accuracy", f"{accuracy:.1%}")
            df = pd.DataFrame(report).transpose()
            st.dataframe(
                df.style.format({
                    "precision": "{:.3f}",
                    "recall": "{:.3f}",
                    "f1-score": "{:.3f}",
                    "support": "{:.0f}",
                })
            )
    else:
        st.info("Click the button to run evaluation on the test set (loads both checkpoints).")
 
    feature_extract_accuracy = config.artifacts_dir / FEATURE_EXTRACT_ACCURACY_CURVE
    feature_extract_loss = config.artifacts_dir / FEATURE_EXTRACT_LOSS_CURVE
 
    if config.accuracy_curve_path.exists() or config.loss_curve_path.exists():
        st.subheader("Fine-Tuned — Training Curves")
        cols = st.columns(2)
        if config.accuracy_curve_path.exists():
            cols[0].image(str(config.accuracy_curve_path), caption="Accuracy")
        if config.loss_curve_path.exists():
            cols[1].image(str(config.loss_curve_path), caption="Loss")
 
    if feature_extract_accuracy.exists() or feature_extract_loss.exists():
        st.subheader("Feature Extraction — Training Curves")
        cols = st.columns(2)
        if feature_extract_accuracy.exists():
            cols[0].image(str(feature_extract_accuracy), caption="Accuracy")
        if feature_extract_loss.exists():
            cols[1].image(str(feature_extract_loss), caption="Loss")
 
def render_classify_tab(config):
    st.header("Classify an Image")
    uploaded_file = st.file_uploader(
        "Upload a waste item image", type=["jpg", "jpeg", "png"], key="classify_upload"
    )
 
    if uploaded_file is None:
        st.info("Upload an image to classify it as recyclable or organic.")
        return
 
    class_names = get_class_names(config.test_dir)
    model = load_final_model(config)
    tmp_path = _save_upload_to_tmp(uploaded_file)
 
    overlay_path = config.artifacts_dir / "gradcam_overlays" / f"upload_{tmp_path.stem}.jpg"
    label, confidence = _classify_and_explain(
        model, class_names, tmp_path, config.target_size, overlay_path
    )
 
    col1, col2 = st.columns(2)
    col1.image(str(tmp_path), caption="Uploaded image")
    col2.image(str(overlay_path), caption="Grad-CAM overlay")
    st.metric("Prediction", label, f"{confidence:.1%} confidence")
 
    tmp_path.unlink(missing_ok=True)
 
def render_benchmark_tab(config):
    st.header("CNN vs. Zero-Shot VLM Benchmark")
 
    results_path = config.artifacts_dir / "benchmark_results.csv"
 
    if results_path.exists():
        df = pd.read_csv(results_path)
 
        st.subheader("Summary")
        summary = pd.DataFrame({
            "accuracy": [df["cnn_correct"].mean(), df["vlm_correct"].mean()],
            "avg_latency_s": [df["cnn_latency_s"].mean(), df["vlm_latency_s"].mean()],
        }, index=["Fine-tuned VGG16", "Zero-shot VLM"])
        st.dataframe(summary.style.format("{:.3f}"))
        st.bar_chart(summary["accuracy"])
 
        st.subheader("Browse individual results")
        selected_path = st.selectbox("Image", df["path"])
        row = df[df["path"] == selected_path].iloc[0]
        st.image(selected_path, width=300)
        st.write(
            f"**True label:** {row['true_label']} | "
            f"**CNN:** {row['cnn_label']} ({row['cnn_confidence']:.1%}) | "
            f"**VLM:** {row['vlm_label']} ({row['vlm_confidence']})"
        )
    else:
        st.warning(
            f"No benchmark results found at `{results_path}`. "
            "Run `python benchmark.py` first (requires the Ollama container — see docker-compose.yml)."
        )
 
    st.divider()
    st.subheader("Live single-image comparison")
    st.caption("Compares the fine-tuned CNN (with Grad-CAM) against the VLM on one image, on demand.")
 
    uploaded_file = st.file_uploader(
        "Upload an image for a live comparison", type=["jpg", "jpeg", "png"], key="benchmark_upload"
    )
    if uploaded_file is None:
        return
 
    class_names = get_class_names(config.test_dir)
    model = load_final_model(config)
    tmp_path = _save_upload_to_tmp(uploaded_file)
 
    overlay_path = config.artifacts_dir / "gradcam_overlays" / f"live_{tmp_path.stem}.jpg"
    cnn_label, cnn_confidence = _classify_and_explain(
        model, class_names, tmp_path, config.target_size, overlay_path
    )
 
    with st.spinner("Asking the VLM..."):
        try:
            vlm_result = classify_with_vlm(tmp_path)
        except Exception as exc:  # e.g. Ollama container not running
            st.error(f"VLM call failed: {exc}")
            vlm_result = None
 
    col1, col2 = st.columns(2)
    with col1:
        st.image(str(overlay_path), caption="CNN + Grad-CAM")
        st.metric("CNN prediction", cnn_label, f"{cnn_confidence:.1%} confidence")
    with col2:
        st.image(str(tmp_path), caption="Original")
        if vlm_result:
            confidence = vlm_result.get("confidence")
            conf_display = f"{confidence:.1%}" if isinstance(confidence, (int, float)) else "N/A"
            st.metric("VLM prediction", vlm_result["label"] or "N/A", conf_display)
            st.caption(vlm_result.get("reason") or "")
 
    tmp_path.unlink(missing_ok=True)
 
def main():
    config = get_config()
    st.title("Waste Classifier — Recyclable vs. Organic")
 
    tab1, tab2, tab3 = st.tabs(["Model Metrics", "Classify an Image", "CNN vs. VLM Benchmark"])
    with tab1:
        render_metrics_tab(config)
    with tab2:
        render_classify_tab(config)
    with tab3:
        render_benchmark_tab(config)