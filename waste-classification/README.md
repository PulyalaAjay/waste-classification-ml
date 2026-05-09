# ♻️ WasteWise — AI-Powered Waste Classification System

> Classifying India's 62M+ ton annual waste problem, one image at a time.

**Built by [Pulyala Ajay Kumar](https://www.linkedin.com/in/ajaytrpx)** — Final-year B.Tech (AI & ML), Malla Reddy College of Engineering and Technology

---

## 🗺️ Project Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** — SVM + Random Forest | ✅ Deployed | Binary classifier (Organic / Recyclable) via Streamlit |
| **Phase 2** — EfficientNet CNN | 🔄 In progress | 10-category classifier with Grad-CAM, higher accuracy |
| **Phase 3** — TFLite Android App | 🔜 Planned | Offline mobile deployment, works without internet |
| **Phase 4** — Active Learning + Drift Monitoring | 🔮 Future | Self-improving production system |

---

## 🧠 Phase 1 — SVM + Random Forest (Live)

**Tech stack:** Python · Scikit-learn · TensorFlow · Streamlit · OpenCV

- Binary classification: **Organic** vs **Recyclable**
- Image quality pre-checks: brightness, blur, face detection — before any ML runs
- Model caching with `@st.cache_resource` for fast repeat predictions
- Real-time user feedback on lighting, focus, and framing
- Confidence score display with low-confidence flagging

**Try the live demo:** [wastewise.streamlit.app](#) *(update with your URL)*

---

## 🔬 Phase 2 — EfficientNet-B0 · 10 Categories

**Tech stack:** TensorFlow · EfficientNetB0 (ImageNet pretrained) · Grad-CAM · MLflow

### 10 Waste Categories
`cardboard` · `e_waste` · `food_waste` · `glass` · `hazardous` · `metal` · `paper` · `plastic` · `rubber` · `textile`

### Training strategy
1. **Stage 1 (20 epochs):** Freeze EfficientNetB0 base, train classification head only
2. **Stage 2 (10 epochs):** Unfreeze top 30 layers, fine-tune at `lr=1e-5`
3. Class weighting to handle imbalanced categories (e-waste is scarce)
4. Augmentation: rotation, flip, zoom, brightness variation

### Run training
```bash
# Set up data directory: data/train, data/val, data/test
# Each subfolder = one class (cardboard, e_waste, ...)
python phase2_efficientnet/train_efficientnet.py

# Launch Streamlit app
streamlit run phase2_efficientnet/app.py
```

### Key features in the Streamlit app
- Top-3 predictions with probability bar chart
- **Grad-CAM heatmap** — see exactly what region the model focused on
- Per-class disposal guidance
- Auto-logging of low-confidence predictions for Phase 4 active learning

---

## 📱 Phase 3 — TFLite Android App (Offline)

**Tech stack:** TFLite (INT8 quantization) · Kotlin · CameraX · SQLite

### Convert & deploy
```bash
# After Phase 2 training, convert the model
python phase3_tflite/convert_to_tflite.py

# Copy outputs to Android assets
cp phase3_tflite/assets/waste_classifier_dynamic.tflite android/app/src/main/assets/
cp phase3_tflite/assets/labels.txt android/app/src/main/assets/
```

### App features
- **Real-time camera inference** via CameraX `ImageAnalysis` — target <150ms per frame
- **Gallery upload** fallback for saved photos
- **Offline disposal map** — bundled SQLite DB of nearby collection points
- **Low-confidence logging** — flags uncertain predictions for Phase 4 feedback
- Regional language support ready (Telugu, Hindi localization)
- No internet required after install

### Model size comparison
| Format | Size | Accuracy |
|--------|------|----------|
| Keras (.keras) | ~22 MB | Baseline |
| Dynamic range TFLite | ~6 MB | ~same |
| INT8 quantized TFLite | ~4 MB | <2% drop |

---

## 🔄 Phase 4 — Active Learning + Drift Monitoring

**Tech stack:** Evidently AI · modAL · GitHub Actions · Label Studio

### Active Learning Loop
```bash
# 1. Score unlabeled images — find the most uncertain ones
python phase4_active_learning/active_learning.py --mode query

# 2. Check labeling queue status
python phase4_active_learning/active_learning.py --mode status

# 3. Apply a label to a queued image
python phase4_active_learning/active_learning.py --mode label \
    --filename image123.jpg --class_name plastic

# 4. Retrain when enough labels are collected (default: 200)
python phase4_active_learning/active_learning.py --mode retrain
```

### Drift Monitoring
```bash
# Compare training distribution vs. current production images
python phase4_active_learning/drift_monitor.py \
    --reference data/train \
    --current active_learning/unlabeled
```

Generates a JSON report + optional Evidently HTML dashboard with:
- **Population Stability Index (PSI)** per feature dimension
- Confidence trend over time
- Automatic retrain recommendation when PSI > 0.2

### Automated CI/CD
GitHub Actions workflow (`.github/workflows/retrain.yml`) runs every Sunday:
1. Downloads latest production logs
2. Runs drift check
3. If drift detected → triggers retrain → converts to TFLite → creates GitHub issue

---

## 📁 Project Structure

```
waste-classification/
├── phase2_efficientnet/
│   ├── train_efficientnet.py    # EfficientNet-B0 training (2-stage)
│   └── app.py                   # Streamlit app with Grad-CAM
├── phase3_tflite/
│   ├── convert_to_tflite.py     # Keras → TFLite converter + validator
│   └── assets/                  # .tflite + labels.txt (copy to Android)
├── phase4_active_learning/
│   ├── active_learning.py       # Uncertainty sampling + retrain loop
│   └── drift_monitor.py         # PSI-based drift detection (Evidently)
├── android/
│   └── app/src/main/
│       ├── java/.../MainActivity.kt   # CameraX + TFLite inference
│       └── assets/                    # Model + labels (after conversion)
├── .github/workflows/
│   └── retrain.yml              # Weekly automated retrain pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/ajaytrpx/waste-classification.git
cd waste-classification

# Install dependencies
pip install -r requirements.txt

# Download TrashNet dataset (extend with your own images for 10 classes)
# https://github.com/garythung/trashnet
# Arrange as: data/train/<class>/, data/val/<class>/, data/test/<class>/

# Train Phase 2 model
python phase2_efficientnet/train_efficientnet.py

# Run Streamlit app
streamlit run phase2_efficientnet/app.py
```

---

## 📊 Results

| Metric | Phase 1 (SVM) | Phase 2 (EfficientNet) |
|--------|--------------|------------------------|
| Classes | 2 | 10 |
| Val accuracy | ~82% | ~93% (target) |
| Inference time | <50ms | <200ms |
| Model size | <1 MB | ~6 MB (TFLite) |

---

## 🎯 Motivation

India generates over **62 million tonnes** of waste annually. Only 20% is processed. Proper sorting at source — knowing whether something is recyclable, compostable, hazardous, or e-waste — is the first step toward better management. This project puts that intelligence in everyone's pocket, offline, in their local language.

---

## 🤝 Contributing

Contributions welcome — especially:
- More labeled images of Indian waste items (plastic sachets, newspaper, coconut husks)
- Telugu / Hindi UI translations
- Per-city disposal centre database for the SQLite bundle

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🙋 About Me

**Pulyala Ajay Kumar** — B.Tech AI & ML, Malla Reddy College of Engineering, Hyderabad  
CGPA: 7.9/10 · AWS Cloud Foundations · Salesforce Developer Certified  
📧 ajaytrpx@gmail.com · [LinkedIn](https://linkedin.com/in/ajaytrpx) · [GitHub](https://github.com/ajaytrpx)
