from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

# Allow `streamlit run app.py` to work even when the project has not been
# installed (or an editable install's .pth file is not loaded by a reloader).
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from hri_action.dataset import discover_sequences, load_frames
from hri_action.experiment import run_experiment
from hri_action.model import ActionRecognizer
from hri_action.reporting import write_annotated_video, write_preview_frame


st.set_page_config(page_title="HRI Action Lab", page_icon="◉", layout="wide")
st.title("HRI Action Lab")
st.caption("Classical computer vision · 40-frame sequence · one actor · six KTH actions")

default_model = PROJECT_ROOT / "artifacts/demo/action_model.joblib"


def extract_zip_safely(data: bytes, destination: Path) -> None:
    archive_path = destination.parent / "upload.zip"
    archive_path.write_bytes(data)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe path in ZIP")
        archive.extractall(destination)


analyze_tab, train_tab, about_tab = st.tabs(["Analyze", "Train & evaluate", "How it works"])

with analyze_tab:
    st.subheader("Classify one sequence")
    model_upload = st.file_uploader("Model (.joblib)", type=["joblib"], key="prediction_model")
    sequence_upload = st.file_uploader("40-frame sequence (.zip)", type=["zip"], key="prediction_sequence")
    with st.expander("Expected ZIP format"):
        st.code("sequence/\n  frame_001.png\n  ...\n  frame_040.png\n  annotation.txt  # optional: <class_id> <xc> <yc> <w> <h>")
    if model_upload is None:
        st.info("No model uploaded: the synthetic smoke-test model will be used. Train a real model in the next tab for meaningful KTH predictions.")

    if st.button("Analyze sequence", type="primary", disabled=sequence_upload is None):
        with tempfile.TemporaryDirectory(prefix="hri-action-") as temp:
            temp_path = Path(temp)
            extract_zip_safely(sequence_upload.getvalue(), temp_path / "input")
            records = discover_sequences(temp_path / "input")
            if not records:
                st.error("No folder containing 40 supported image frames was found.")
                st.stop()
            model_path = temp_path / "model.joblib"
            if model_upload:
                model_path.write_bytes(model_upload.getvalue())
            elif default_model.exists():
                model_path = default_model
            else:
                st.error("Upload a trained model, or run `hri-action demo` to create the demo model.")
                st.stop()
            with st.spinner("Segmenting, tracking and analyzing motion…"):
                frames = load_frames(records[0])
                model = ActionRecognizer.load(model_path)
                prediction = model.predict(frames)
                preview = write_preview_frame(frames, prediction, temp_path / "preview.png")
                video = write_annotated_video(frames, prediction, temp_path / "prediction.mp4")
            left, right = st.columns([1.35, 1])
            with left:
                st.image(str(preview), caption="Median frame localization", use_container_width=True)
                st.video(video.read_bytes())
            with right:
                st.metric("Predicted action", prediction.label.title())
                st.metric("Confidence", f"{prediction.confidence:.1%}")
                table = pd.DataFrame({"Action": prediction.probabilities.keys(),
                                      "Probability": prediction.probabilities.values()}).sort_values("Probability", ascending=False)
                st.bar_chart(table.set_index("Action"))
                if model.metadata:
                    st.caption(f"Training set: {model.metadata.get('train_sequences', 'unknown')} sequences")

with train_tab:
    st.subheader("Run the proposal experiment")
    st.write("Upload the complete dataset. The app validates it, creates an action-and-scenario stratified split, extracts features, trains the classifier, and evaluates the held-out sequences.")
    dataset_upload = st.file_uploader("Complete dataset (.zip)", type=["zip"], key="training_dataset")
    strict = st.checkbox("Require the exact proposal design (72 sequences, 12 per action, 3 per action/scenario)", value=True)
    seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42)
    if st.button("Train and evaluate", type="primary", disabled=dataset_upload is None):
        try:
            with tempfile.TemporaryDirectory(prefix="hri-experiment-") as temp:
                root = Path(temp)
                extract_zip_safely(dataset_upload.getvalue(), root / "dataset")
                with st.spinner("Validating data, extracting optical flow, training and evaluating…"):
                    result = run_experiment(
                        root / "dataset", root / "output", seed=int(seed),
                        strict_proposal=strict, qualitative_videos=False,
                    )
                st.session_state["experiment_result"] = {
                    "model": result.model_path.read_bytes(),
                    "metrics": result.metrics,
                    "validation": result.validation.as_dict(),
                    "confusion": (root / "output/evaluation/confusion_matrix.png").read_bytes(),
                    "importance": (root / "output/feature_importance.png").read_bytes(),
                    "manifest": (root / "output/split_manifest.csv").read_bytes(),
                }
        except Exception as error:
            st.error(str(error))

    if "experiment_result" in st.session_state:
        result_data = st.session_state["experiment_result"]
        metrics = result_data["metrics"]
        one, two, three = st.columns(3)
        one.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        two.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
        three.metric("Mean IoU", "N/A" if metrics["mean_iou"] is None else f"{metrics['mean_iou']:.3f}")
        left, right = st.columns(2)
        left.image(result_data["confusion"], caption="Confusion matrix", use_container_width=True)
        right.image(result_data["importance"], caption="Feature importance", use_container_width=True)
        st.download_button("Download trained model", result_data["model"], "action_model.joblib")
        st.download_button("Download split manifest", result_data["manifest"], "split_manifest.csv")

with about_tab:
    st.subheader("Proposal pipeline")
    st.markdown("""
1. **Segmentation and tracking:** temporal foreground energy plus appearance contrast localizes the single actor.
2. **Motion representation:** dense optical flow, centroid speed, silhouette deformation, regional motion and periodicity produce 32 interpretable features.
3. **Classification:** a balanced Random Forest assigns one label to the complete 40-frame window.
4. **Evaluation:** frame-20 IoU/mIoU, accuracy, per-class precision/recall/F1 and a confusion matrix.

The class-ID mapping is `1 boxing`, `2 handclapping`, `3 handwaving`, `4 jogging`, `5 running`, `6 walking`.
""")
