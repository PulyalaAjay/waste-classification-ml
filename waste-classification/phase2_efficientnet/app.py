"""
Phase 2: Streamlit App — 10-Class Waste Classifier with Grad-CAM
Author: Pulyala Ajay Kumar
Run: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
import os
import io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH  = "models/waste_classifier_v2.keras"
LABELS_PATH = "models/class_indices.json"
IMG_SIZE    = 224
CONF_THRESHOLD = 0.60       # flag predictions below this

DISPOSAL_TIPS = {
    "cardboard":  "♻️  Flatten and place in dry paper recycling bin.",
    "e_waste":    "⚠️  Take to a certified e-waste collection centre — never bin it.",
    "food_waste": "🌱  Add to compost bin or wet waste collection.",
    "glass":      "🫙  Rinse and place in glass recycling. Remove lids.",
    "hazardous":  "🔴  Take to a hazardous waste facility. Do NOT mix with general waste.",
    "metal":      "🔩  Rinse cans; place in dry recycling. Scrap metal to a dealer.",
    "paper":      "📄  Keep dry; add to paper recycling or newspaper pickup.",
    "plastic":    "♻️  Check resin code. Rinse and place in plastic recycling.",
    "rubber":     "🔄  Contact tyre/rubber recycler. Do not burn.",
    "textile":    "👕  Donate wearable clothes; take unusable fabric to textile recycler.",
}

# ─── Model Loading ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


# ─── Image Quality Checks ────────────────────────────────────────────────────

def check_brightness(img_array: np.ndarray) -> tuple[bool, str]:
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness < 40:
        return False, "⚠️ Image is too dark. Move to a brighter area."
    if mean_brightness > 220:
        return False, "⚠️ Image is overexposed. Reduce direct light."
    return True, ""


def check_blur(img_array: np.ndarray) -> tuple[bool, str]:
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 80:
        return False, "⚠️ Image is blurry. Hold the camera steady and focus."
    return True, ""


def check_size(img_array: np.ndarray) -> tuple[bool, str]:
    h, w = img_array.shape[:2]
    if h < 100 or w < 100:
        return False, "⚠️ Image is too small. Use a higher-resolution photo."
    return True, ""


def run_quality_checks(img_array: np.ndarray) -> list[str]:
    issues = []
    for check in [check_brightness, check_blur, check_size]:
        ok, msg = check(img_array)
        if not ok:
            issues.append(msg)
    return issues


# ─── Grad-CAM ────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(model, img_tensor, last_conv_layer_name="top_conv"):
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    colormap = cm.get_cmap("jet")
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    return overlay


# ─── Prediction ──────────────────────────────────────────────────────────────

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img_tensor, idx_to_class):
    probs = model.predict(img_tensor, verbose=0)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    results = [(idx_to_class[i], float(probs[i])) for i in top3_idx]
    return results


# ─── UI ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="WasteWise — AI Waste Classifier",
        page_icon="♻️",
        layout="wide",
    )

    st.title("♻️ WasteWise — AI Waste Classifier")
    st.caption("10-category waste classification · EfficientNet-B0 · Built by Pulyala Ajay Kumar")

    # Load model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at `{MODEL_PATH}`. Train the model first using `train_efficientnet.py`.")
        st.stop()

    model, idx_to_class = load_model()

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            "Upload a photo of any waste item and get an instant classification "
            "with disposal guidance. Works on **10 waste categories**."
        )
        st.markdown("**Categories:**")
        for cls in sorted(idx_to_class.values()):
            st.markdown(f"- {cls.replace('_', ' ').title()}")
        st.divider()
        show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
        confidence_warning = st.slider("Low-confidence threshold", 0.3, 0.9, CONF_THRESHOLD, 0.05)

    # Upload
    uploaded = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        img = Image.open(uploaded)
        img_array = np.array(img.convert("RGB"))

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("Your image")
            st.image(img, use_column_width=True)

            # Quality checks
            issues = run_quality_checks(img_array)
            if issues:
                for issue in issues:
                    st.warning(issue)
                if len(issues) >= 2:
                    st.error("Too many image quality issues. Please retake the photo.")
                    st.stop()

        # Preprocess & predict
        img_tensor = preprocess(img)
        with st.spinner("Classifying…"):
            results = predict(model, img_tensor, idx_to_class)

        top_class, top_conf = results[0]

        with col2:
            st.subheader("Classification result")

            if top_conf < confidence_warning:
                st.warning(f"Low confidence ({top_conf:.0%}) — result may not be reliable.")

            st.metric(
                label=top_class.replace("_", " ").title(),
                value=f"{top_conf:.1%} confidence",
            )

            # Bar chart for top-3
            st.markdown("**Top 3 predictions**")
            for cls, prob in results:
                label = cls.replace("_", " ").title()
                bar_color = "green" if cls == top_class else "gray"
                st.progress(prob, text=f"{label}  —  {prob:.1%}")

            # Disposal tip
            st.divider()
            st.markdown("**Disposal guidance**")
            st.info(DISPOSAL_TIPS.get(top_class, "Please consult your local waste management authority."))

        # Grad-CAM
        if show_gradcam:
            try:
                heatmap = make_gradcam_heatmap(model, img_tensor)
                overlay = overlay_gradcam(
                    cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)), heatmap
                )
                st.subheader("Grad-CAM — what the model focused on")
                st.image(overlay, caption="Red/yellow = high attention regions", use_column_width=True)
            except Exception:
                st.info("Grad-CAM not available for this model configuration.")

        # Log low-confidence predictions for active learning
        if top_conf < confidence_warning:
            log_path = "logs/low_confidence_samples.csv"
            os.makedirs("logs", exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"{uploaded.name},{top_class},{top_conf:.4f}\n")


if __name__ == "__main__":
    main()
