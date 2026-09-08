"""FreshLens: general product recognition and experiment evidence."""
from pathlib import Path
from io import BytesIO
from html import escape
import json
import time
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image, UnidentifiedImageError
from inference import load_artifacts, read_image, predict_product

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="FreshLens | Product Recognition", page_icon="🍊", layout="wide")

with st.sidebar:
    st.title("🍊 FreshLens")
    st.caption("PRODUCT RECOGNITION STUDIO")
    dark = st.toggle("Dark mode", value=False)
    page = st.radio("Navigate", ["Recognize", "Model results & evidence", "About the project"])
    st.divider()
    st.write("**ITI Summer Training**")
    st.caption("Computer Vision · Final Project")
    st.markdown('<div class="sidebar-note">One product.<br>One image.<br>A closer look.</div>', unsafe_allow_html=True)

if dark:
    colors = {"BG": "#101b25", "SURFACE": "#192b38", "TEXT": "#edf5fa", "MUTED": "#b7c9d6", "BORDER": "#3a5262", "ACCENT": "#68d6af"}
else:
    colors = {"BG": "#f5f8fc", "SURFACE": "#ffffff", "TEXT": "#132b40", "MUTED": "#516779", "BORDER": "#d6e2eb", "ACCENT": "#117759"}
css = (ROOT / "style.css").read_text()
for name, value in colors.items():
    css = css.replace("@" + name, value)
st.markdown("<style>" + css + "</style>", unsafe_allow_html=True)

config = json.loads((ROOT / "models/model_config.json").read_text())
class_names = json.loads((ROOT / "models/class_names.json").read_text())
category_mapping = json.loads((ROOT / "models/category_mapping.json").read_text())
test_results = pd.read_csv(ROOT / "reports/test_comparison.csv")
validation_results = pd.read_csv(ROOT / "reports/validation_comparison.csv")
model_names = {"cnn_best": "Simple CNN", "transfer_head_best": "MobileNetV2 · frozen base", "transfer_fine_best": "MobileNetV2 · fine-tuned"}
model_name = model_names[config["selected_model"]]
selected = test_results[(test_results["model"] == config["selected_model"]) & (test_results["domain"] == "natural")].iloc[0]

@st.cache_resource
def get_model():
    return load_artifacts()


def show_table(table):
    st.markdown('<div class="data-table">' + table.to_html(index=False, escape=True, float_format="%.4f") + '</div>', unsafe_allow_html=True)


def show_chart(data, x, y, group, title):
    chart = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X(x, title="Epoch"), y=alt.Y(y, title=title),
        color=alt.Color(group, scale=alt.Scale(range=["#28b78b", "#eea63e"])),
        tooltip=list(data.columns)
    ).properties(height=280, background=colors["SURFACE"])
    chart = chart.configure_axis(labelColor=colors["TEXT"], titleColor=colors["TEXT"], gridColor=colors["BORDER"]).configure_legend(labelColor=colors["TEXT"], titleColor=colors["TEXT"])
    st.altair_chart(chart, use_container_width=True, theme=None)


def show_predictions(results):
    best = results[0]
    st.markdown('<div class="result-card"><div class="eyebrow">PREDICTED PRODUCT</div><h2>' + escape(best["product"]) + '</h2><div class="result-score">' + f'{best["score"] * 100:.1f}%' + '<span> model score</span></div></div>', unsafe_allow_html=True)
    st.subheader("Top 3 predictions")
    for rank, result in enumerate(results, 1):
        left, right = st.columns([4, 1])
        left.write(f"**{rank}. {result['product']}**")
        right.write(f"**{result['score'] * 100:.2f}%**")
        st.progress(float(result["score"]))
    st.caption("A high score does not guarantee a correct prediction. Unknown products can also receive high scores.")

st.markdown('<div class="eyebrow">FRESHLENS / COMPUTER VISION</div>', unsafe_allow_html=True)
if page == "Recognize":
    st.title("Know your produce.")
    st.write("Upload one product to predict its general type, such as Apple, Banana or Tomato.")
    first, second, third = st.columns(3)
    first.metric("General product types", len(class_names))
    second.metric("Natural-photo test accuracy", f"{selected['accuracy'] * 100:.2f}%")
    third.metric("Selected model", model_name)
    st.write("")
    image_column, result_column = st.columns([1.05, 1], gap="large")
    with image_column:
        st.markdown("### 01 / Choose an image")
        uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"],
                                    help="One centered product, ideally against a plain white background. Maximum 10 MB.")
        st.caption("Use one clear product. Explore measured performance in Model results & evidence.")
        image = None
        image_bytes = None
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            if len(image_bytes) > 10 * 1024 * 1024:
                st.error("Please choose an image smaller than 10 MB.")
            else:
                try:
                    image = read_image(BytesIO(image_bytes))
                    st.image(image, caption=uploaded.name, use_container_width=True)
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                    st.error("This image could not be opened. Try a valid JPG or PNG file.")
                    st.caption(str(error))
        else:
            st.markdown('<div class="upload-empty"><div class="empty-icon">🍋</div><h3>Your product goes here</h3><p>JPG, PNG or WEBP · One product per image</p></div>', unsafe_allow_html=True)
        analyze = st.button("Analyze product", type="primary", use_container_width=True, disabled=image is None)
        if analyze:
            try:
                with st.spinner("Loading the model and recognizing your product..."):
                    model, labels, settings = get_model()
                    start = time.perf_counter()
                    results = predict_product(model, labels, image_bytes, settings["image_size"])
                    elapsed = time.perf_counter() - start
                st.session_state["prediction"] = {"source": image_bytes, "results": results, "elapsed": elapsed}
            except Exception as error:
                st.session_state.pop("prediction", None)
                st.error("The model could not run. Check that setup_windows.bat completed successfully.")
                st.code(str(error))
    with result_column:
        st.markdown("### 02 / Recognition result")
        prediction = st.session_state.get("prediction")
        if image is not None and prediction and prediction["source"] == image_bytes:
            show_predictions(prediction["results"])
            st.caption(f"Inference time: {prediction['elapsed']:.2f} seconds (excluding model loading)")
        else:
            st.markdown('<div class="result-empty"><h3>Ready when you are.</h3><p>Your predicted category and three highest model scores will appear here after analysis.</p></div>', unsafe_allow_html=True)
        st.info("Best results: one centered product on a plain background. This classifier does not locate or count multiple products.")

elif page == "Model results & evidence":
    st.title("The evidence, in focus.")
    st.write("Explore all three training runs and their saved evaluation results.")
    a, b, c = st.columns(3)
    a.metric("Selected model", model_name)
    b.metric("Natural test accuracy", f"{selected['accuracy'] * 100:.2f}%")
    c.metric("Natural test images", int(selected["images"]))
    st.info("This run still has limited generalization. Controlled and natural results are shown separately. Natural test data covers 17 of the 84 supported categories.")
    tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "Learning curves", "Prediction evidence", "Data & downloads"])
    with tab1:
        split = st.radio("Evaluation split", ["Test", "Validation"], horizontal=True)
        domain = st.radio("Image domain", ["natural", "controlled"], horizontal=True)
        report = test_results if split == "Test" else validation_results
        table = report[report["domain"] == domain].copy()
        table["model"] = table["model"].replace(model_names)
        show_table(table)
        plot = table.melt(id_vars=["model"], value_vars=["accuracy", "macro_f1"], var_name="Metric", value_name="Score")
        chart = alt.Chart(plot).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=-15)),
            xOffset="Metric:N", y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#28b78b", "#eea63e"])), tooltip=["model", "Metric", alt.Tooltip("Score:Q", format=".2%")]
        ).properties(height=320, background=colors["SURFACE"]).configure_axis(labelColor=colors["TEXT"], titleColor=colors["TEXT"], gridColor=colors["BORDER"]).configure_legend(labelColor=colors["TEXT"], titleColor=colors["TEXT"])
        st.altair_chart(chart, use_container_width=True, theme=None)
        st.caption("Macro metrics average over categories present in each evaluated subset. Image and category counts appear in the table.")
        st.subheader("Selection used validation only")
        st.write("The saved model minimizes the mean of controlled and natural validation loss. Test scores do not change the selection.")
        show_table(pd.read_csv(ROOT / "reports/model_selection.csv"))
    with tab2:
        chosen = st.selectbox("Training run", list(model_names), format_func=model_names.get)
        history = pd.read_csv(ROOT / "reports" / (chosen + "_history.csv"))
        history["Epoch"] = range(1, len(history) + 1)
        left, right = st.columns(2)
        with left:
            st.subheader("Accuracy")
            data = history.melt(id_vars="Epoch", value_vars=["accuracy", "val_accuracy"], var_name="Series", value_name="Accuracy")
            show_chart(data, "Epoch:O", "Accuracy:Q", "Series:N", "Accuracy")
        with right:
            st.subheader("Loss")
            data = history.melt(id_vars="Epoch", value_vars=["loss", "val_loss"], var_name="Series", value_name="Loss")
            show_chart(data, "Epoch:O", "Loss:Q", "Series:N", "Loss")
        if "learning_rate" in history:
            st.subheader("Learning rate")
            data = history[["Epoch", "learning_rate"]].copy()
            data["Series"] = "Learning rate"
            show_chart(data, "Epoch:O", "learning_rate:Q", "Series:N", "Learning rate")
        st.caption("Training includes augmentation and oversampled natural images. Validation loss gives equal weight to both domains; validation accuracy is unweighted. The curves are not measured under identical conditions.")
        with st.expander("View epoch values"):
            show_table(history)
        with st.expander("Original combined learning-curve figure"):
            st.image(str(ROOT / "reports/training_curves.png"))
    with tab3:
        st.subheader("Selected-model prediction evidence")
        st.caption("Per-class reports and confusion matrices were exported for the selected model only.")
        domain = st.selectbox("Evidence domain", ["natural", "controlled"])
        matrix = pd.read_csv(ROOT / "reports" / (domain + "_confusion_matrix.csv"), index_col=0)
        present = matrix.index[matrix.sum(axis=1) > 0]
        product = st.selectbox("Actual product", list(present))
        counts = matrix.loc[product].sort_values(ascending=False)
        counts = counts[counts > 0].rename("Images").reset_index()
        counts.columns = ["Predicted product", "Images"]
        st.write("**Predictions for actual " + product + " images**")
        show_table(counts)
        with st.expander("Full confusion matrix heatmap"):
            heat = matrix.loc[present].copy()
            heat = heat.loc[:, heat.sum(axis=0) > 0]
            heat.index.name = "Actual"
            heat = heat.reset_index().melt(id_vars="Actual", var_name="Predicted", value_name="Images")
            chart = alt.Chart(heat).mark_rect().encode(x="Predicted:N", y="Actual:N", color=alt.Color("Images:Q", scale=alt.Scale(scheme="greens")), tooltip=["Actual", "Predicted", "Images"]).properties(height=max(350, len(present)*18), background=colors["SURFACE"]).configure_axis(labelColor=colors["TEXT"], titleColor=colors["TEXT"]).configure_legend(labelColor=colors["TEXT"], titleColor=colors["TEXT"])
            st.altair_chart(chart, use_container_width=True, theme=None)
        with st.expander("Full per-class report"):
            show_table(pd.read_csv(ROOT / "reports" / (domain + "_classification_report.csv")))
        st.subheader("Saved examples of incorrect predictions")
        st.image(str(ROOT / "reports/prediction_errors.png"))
        st.caption("These are the notebook's saved test examples. Natural-image mistakes are displayed first.")
        predictions = pd.read_csv(ROOT / "reports/test_predictions.csv")
        predictions = predictions[(predictions["domain"] == domain) & (predictions["label"] == product)]
        only_errors = st.checkbox("Show mistakes only", value=True)
        if only_errors:
            predictions = predictions[predictions["label"] != predictions["predicted"]]
        show_table(predictions.head(100))
        st.caption("Showing up to 100 matching rows. Download the full prediction CSV below.")
    with tab4:
        st.subheader("Training data")
        for filename, title in [("training_examples.png", "Natural training examples"), ("class_distribution.png", "Training class distribution")]:
            with st.expander(title):
                st.image(str(ROOT / "reports" / filename))
        st.write("Exact duplicates were removed. Controlled images were grouped by normalized variety. Natural images use capture-year groups: 2018 validation, 2022 test, other known years training. These grouping proxies do not prove independent objects or scenes.")
        with st.expander("Category coverage"):
            coverage = pd.read_csv(ROOT / "reports/class_coverage.csv", header=[0, 1], index_col=0)
            coverage.columns = [" / ".join(column) for column in coverage.columns]
            show_table(coverage.reset_index())
        st.subheader("Download exported evidence")
        report_files = sorted((ROOT / "reports").glob("*"))
        report_path = st.selectbox("Report or figure", report_files, format_func=lambda path: path.name)
        st.download_button("Download selected evidence", report_path.read_bytes(), file_name=report_path.name)
        st.caption("All exported reports and figures are included. Raw test photos and the other two model weights are not part of this application export.")
else:
    st.title("From pixels to products.")
    st.write("FreshLens is a retail product image classifier built for the ITI summer training final project.")
    st.subheader("How it works")
    st.markdown("1. Upload an image of one product.\n2. Correct its orientation and convert it to RGB.\n3. Fit it inside a white 160 × 160 frame and encode it as JPEG at quality 95.\n4. Run the saved model, which includes pixel normalization.\n5. Display the three highest product scores.")
    st.subheader("Training and evaluation")
    st.write("A CNN trained from scratch is compared with MobileNetV2 head training and fine-tuning. Model selection uses equal-domain validation loss. The application loads the selected checkpoint without retraining.")
    st.subheader("Scope")
    st.write("One general category per image. The system does not locate or count products and has no trained unknown category. Natural test results currently show limited reliability on unfamiliar photographs.")
    with st.expander("Supported products"):
        show_table(pd.DataFrame({"Category": class_names}))
    with st.expander("Original labels and general categories"):
        show_table(pd.DataFrame(category_mapping.items(), columns=["Original label", "General category"]))
    st.markdown("[Fruits-360 dataset](https://github.com/fruits-360) · Mihai Oltean · Dataset licenses and source notes are included in the references folder.")

st.divider()
st.caption("FreshLens · 84 general product categories · Local predictions · ITI Summer Training")
