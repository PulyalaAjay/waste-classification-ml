"""
Phase 4: Data Drift Monitoring for Waste Classifier
Author: Pulyala Ajay Kumar

Uses Evidently AI to detect when the distribution of incoming waste images
shifts away from the training distribution — triggering a retrain alert.

Run:
  python drift_monitor.py --reference data/train --current active_learning/unlabeled
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH     = "models/waste_classifier_v2.keras"
IMG_SIZE       = 224
FEATURE_DIM    = 1280           # EfficientNetB0 pooled output dimension
REPORT_DIR     = "drift_reports"
PSI_THRESHOLD  = 0.2            # Population Stability Index — retrain if exceeded
CONF_DROP_THRESHOLD = 0.05      # alert if avg confidence drops by this much

os.makedirs(REPORT_DIR, exist_ok=True)


# ─── Feature extraction (EfficientNet penultimate layer) ─────────────────────

def build_feature_extractor() -> tf.keras.Model:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False
    print(f"Feature extractor ready — output dim: {base.output_shape}")
    return base


def extract_features(model: tf.keras.Model, image_dir: str, max_images: int = 500) -> np.ndarray:
    """Extract pooled features for all images in a directory."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(Path(image_dir).rglob(ext))
    paths = paths[:max_images]

    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"Extracting features from {len(paths)} images in {image_dir} …")
    features = []
    for path in paths:
        img = load_img(str(path), target_size=(IMG_SIZE, IMG_SIZE))
        arr = img_to_array(img) / 255.0
        inp = np.expand_dims(arr, axis=0)
        feat = model.predict(inp, verbose=0)[0]
        features.append(feat)
    return np.array(features)


# ─── Population Stability Index ──────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """
    PSI measures how much a distribution has shifted.
    PSI < 0.1  : no significant change
    PSI 0.1-0.2: moderate change — monitor
    PSI > 0.2  : significant shift — retrain
    """
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, buckets + 1)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current,   bins=breakpoints)

    ref_pct = ref_counts / (ref_counts.sum() + 1e-10)
    cur_pct = cur_counts / (cur_counts.sum() + 1e-10)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-4, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-4, cur_pct)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_feature_psi(ref_features: np.ndarray, cur_features: np.ndarray) -> dict:
    """Compute PSI for each feature dimension; return summary stats."""
    n_dims = min(ref_features.shape[1], 50)    # sample 50 dims for speed
    sampled_dims = np.random.choice(ref_features.shape[1], n_dims, replace=False)

    psis = []
    for dim in sampled_dims:
        psi = compute_psi(ref_features[:, dim], cur_features[:, dim])
        psis.append(psi)

    return {
        "mean_psi"   : float(np.mean(psis)),
        "max_psi"    : float(np.max(psis)),
        "p90_psi"    : float(np.percentile(psis, 90)),
        "n_dims_checked": n_dims,
        "drift_detected": float(np.mean(psis)) > PSI_THRESHOLD,
    }


# ─── Confidence trend monitoring ─────────────────────────────────────────────

def monitor_confidence_trend(log_path: str = "logs/low_confidence_samples.csv") -> dict:
    """
    Read the low-confidence log from the Streamlit / Android app.
    Alert if the rate of low-confidence predictions is rising.
    """
    if not os.path.exists(log_path):
        return {"status": "no_log_found", "low_conf_count": 0}

    df = pd.read_csv(log_path, header=None, names=["filename", "pred_class", "confidence"])
    total = len(df)
    low   = (df["confidence"] < 0.6).sum()
    avg_conf = df["confidence"].mean()

    return {
        "total_predictions" : int(total),
        "low_conf_count"    : int(low),
        "low_conf_rate"     : float(low / total) if total > 0 else 0,
        "avg_confidence"    : float(avg_conf),
        "alert"             : (low / total > 0.3) if total > 0 else False,
    }


# ─── Evidently report (optional — install evidently to enable) ────────────────

def generate_evidently_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame, output_path: str):
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
        report.save_html(output_path)
        print(f"Evidently HTML report saved: {output_path}")
    except ImportError:
        print("Evidently not installed. Run: pip install evidently")
        print("Falling back to manual PSI computation.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drift Monitoring")
    parser.add_argument("--reference", default="data/train",              help="Reference (training) image dir")
    parser.add_argument("--current",   default="active_learning/unlabeled", help="Current production image dir")
    parser.add_argument("--max_images", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 4: Drift Monitoring")
    print("=" * 60)

    extractor = build_feature_extractor()

    print("\n[1/3] Extracting reference features …")
    ref_features = extract_features(extractor, args.reference, args.max_images)

    print("\n[2/3] Extracting current features …")
    cur_features = extract_features(extractor, args.current, args.max_images)

    print("\n[3/3] Computing drift metrics …")
    psi_result = compute_feature_psi(ref_features, cur_features)
    conf_result = monitor_confidence_trend()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp"         : timestamp,
        "reference_dir"     : args.reference,
        "current_dir"       : args.current,
        "reference_samples" : len(ref_features),
        "current_samples"   : len(cur_features),
        "feature_drift"     : psi_result,
        "confidence_trend"  : conf_result,
        "recommendation"    : (
            "RETRAIN — significant distribution shift detected"
            if psi_result["drift_detected"] or conf_result.get("alert", False)
            else "MONITOR — no significant drift detected"
        ),
    }

    report_path = os.path.join(REPORT_DIR, f"drift_report_{timestamp}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 50)
    print("DRIFT REPORT")
    print("=" * 50)
    print(f"Mean PSI         : {psi_result['mean_psi']:.4f}  (threshold={PSI_THRESHOLD})")
    print(f"Max PSI          : {psi_result['max_psi']:.4f}")
    print(f"Drift detected   : {'YES ⚠️' if psi_result['drift_detected'] else 'No ✓'}")
    print(f"Low-conf rate    : {conf_result.get('low_conf_rate', 'N/A')}")
    print(f"Confidence alert : {'YES ⚠️' if conf_result.get('alert') else 'No ✓'}")
    print(f"\n>>> {report['recommendation']}")
    print(f"\nFull report saved: {report_path}")

    # Optionally generate Evidently HTML report
    print("\nGenerating Evidently report (requires `pip install evidently`) …")
    # Convert first 50 features to dataframe columns for Evidently
    n_cols = min(50, ref_features.shape[1])
    col_names = [f"feat_{i}" for i in range(n_cols)]
    ref_df = pd.DataFrame(ref_features[:, :n_cols], columns=col_names)
    cur_df = pd.DataFrame(cur_features[:, :n_cols], columns=col_names)
    evidently_path = os.path.join(REPORT_DIR, f"evidently_{timestamp}.html")
    generate_evidently_report(ref_df, cur_df, evidently_path)


if __name__ == "__main__":
    main()
