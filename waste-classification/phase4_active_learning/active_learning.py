"""
Phase 4: Active Learning Pipeline for Waste Classifier
Author: Pulyala Ajay Kumar

Strategy: Uncertainty sampling — collect predictions where model is least confident,
surface them to a human labeler, retrain on the enriched dataset.

Components:
  1. UncertaintySampler  — scores unlabeled images by entropy / least-confidence
  2. LabelingQueue       — manages samples waiting for annotation
  3. ActiveLearningLoop  — orchestrates query → label → retrain cycle
  4. RetrainTrigger      — decides when to kick off retraining

Run:
  python active_learning.py --mode query   # score and queue low-confidence samples
  python active_learning.py --mode retrain # retrain on newly labeled data
  python active_learning.py --mode status  # show queue stats
"""

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_PATH      = "models/waste_classifier_v2.keras"
LABELS_PATH     = "models/class_indices.json"
DATA_DIR        = "data"
UNLABELED_DIR   = "active_learning/unlabeled"    # new incoming images (no labels)
QUEUE_FILE      = "active_learning/labeling_queue.json"
LABELED_DIR     = "active_learning/labeled"      # human-labeled corrections
LOG_FILE        = "active_learning/al_log.csv"
IMG_SIZE        = 224
BATCH_SIZE      = 32
QUERY_BUDGET    = 50        # how many samples to add to queue per cycle
RETRAIN_TRIGGER = 200       # retrain once this many labeled samples accumulate
ENTROPY_THRESHOLD = 0.5     # flag predictions with entropy above this

os.makedirs(UNLABELED_DIR,  exist_ok=True)
os.makedirs(LABELED_DIR,    exist_ok=True)
os.makedirs("active_learning", exist_ok=True)


# ─── Model & labels ──────────────────────────────────────────────────────────

def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class


# ─── Uncertainty Sampling ────────────────────────────────────────────────────

def prediction_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of probability vector — higher = more uncertain."""
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def least_confidence(probs: np.ndarray) -> float:
    """1 - max prob. Higher = less confident."""
    return float(1.0 - np.max(probs))


def margin_sampling(probs: np.ndarray) -> float:
    """Difference between top-2 probs. Smaller = more uncertain."""
    sorted_probs = np.sort(probs)[::-1]
    return float(1.0 - (sorted_probs[0] - sorted_probs[1]))


def score_uncertainty(probs: np.ndarray, method: str = "entropy") -> float:
    if method == "entropy":
        return prediction_entropy(probs)
    if method == "least_confidence":
        return least_confidence(probs)
    if method == "margin":
        return margin_sampling(probs)
    raise ValueError(f"Unknown method: {method}")


def preprocess_image(path: str) -> np.ndarray:
    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ─── Uncertainty Query ───────────────────────────────────────────────────────

def query_uncertain_samples(
    model,
    idx_to_class: dict,
    budget: int = QUERY_BUDGET,
    method: str = "entropy",
) -> list[dict]:
    """
    Score all images in UNLABELED_DIR, return the `budget` most uncertain ones.
    """
    image_paths = list(Path(UNLABELED_DIR).glob("*.jpg")) + \
                  list(Path(UNLABELED_DIR).glob("*.png")) + \
                  list(Path(UNLABELED_DIR).glob("*.jpeg"))

    if not image_paths:
        print(f"No images found in {UNLABELED_DIR}")
        return []

    print(f"Scoring {len(image_paths)} unlabeled images …")
    scores = []
    for path in image_paths:
        try:
            inp   = preprocess_image(str(path))
            probs = model.predict(inp, verbose=0)[0]
            score = score_uncertainty(probs, method)
            pred_class = idx_to_class[int(np.argmax(probs))]
            pred_conf  = float(np.max(probs))
            scores.append({
                "path":       str(path),
                "filename":   path.name,
                "score":      score,
                "pred_class": pred_class,
                "pred_conf":  pred_conf,
                "method":     method,
                "queued_at":  datetime.now().isoformat(),
                "labeled":    False,
                "true_class": None,
            })
        except Exception as e:
            print(f"  Skipping {path.name}: {e}")

    # Sort by uncertainty descending; take top `budget`
    scores.sort(key=lambda x: x["score"], reverse=True)
    selected = scores[:budget]
    print(f"Selected {len(selected)} samples (highest uncertainty)")
    return selected


# ─── Labeling Queue ──────────────────────────────────────────────────────────

def load_queue() -> list[dict]:
    if not os.path.exists(QUEUE_FILE):
        return []
    with open(QUEUE_FILE) as f:
        return json.load(f)


def save_queue(queue: list[dict]):
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)


def add_to_queue(new_samples: list[dict]):
    queue = load_queue()
    existing = {s["filename"] for s in queue}
    added = 0
    for s in new_samples:
        if s["filename"] not in existing:
            queue.append(s)
            added += 1
    save_queue(queue)
    print(f"Added {added} samples to labeling queue ({len(queue)} total)")


def apply_label(filename: str, true_class: str):
    """Mark a sample as labeled and move to labeled data directory."""
    queue = load_queue()
    for s in queue:
        if s["filename"] == filename and not s["labeled"]:
            s["labeled"]    = True
            s["true_class"] = true_class
            s["labeled_at"] = datetime.now().isoformat()

            # Copy image to labeled dir, under the true_class subfolder
            dst = Path(LABELED_DIR) / true_class
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(s["path"], dst / filename)
            print(f"  Labeled: {filename} → {true_class}")
            break
    save_queue(queue)


# ─── Retrain Trigger ─────────────────────────────────────────────────────────

def count_labeled_samples() -> int:
    return sum(
        len(list(Path(LABELED_DIR, cls).glob("*")))
        for cls in os.listdir(LABELED_DIR)
        if os.path.isdir(Path(LABELED_DIR, cls))
    )


def should_retrain(threshold: int = RETRAIN_TRIGGER) -> bool:
    n = count_labeled_samples()
    print(f"Labeled samples: {n} / {threshold} needed to trigger retrain")
    return n >= threshold


# ─── Retrain ─────────────────────────────────────────────────────────────────

def retrain_with_labeled(model, idx_to_class: dict):
    """
    Merge newly labeled samples with original training data and fine-tune.
    """
    from sklearn.utils.class_weight import compute_class_weight

    print("\n[Retrain] Merging labeled corrections into training data …")

    # Combine original train dir + labeled corrections
    combined_dir = "active_learning/combined_train"
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)
    shutil.copytree(os.path.join(DATA_DIR, "train"), combined_dir)

    for cls_dir in Path(LABELED_DIR).iterdir():
        if cls_dir.is_dir():
            dst = Path(combined_dir) / cls_dir.name
            dst.mkdir(exist_ok=True)
            for img in cls_dir.glob("*"):
                shutil.copy(img, dst / img.name)

    aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.15,
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = aug.flow_from_directory(
        combined_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(weights))

    # Fine-tune top layers only
    for layer in model.layers[:-10]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"models/waste_classifier_al_{timestamp}.keras"

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_accuracy", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        ],
    )

    model.save(checkpoint_path)
    print(f"\nRetrained model saved: {checkpoint_path}")

    # Log retrain event
    with open(LOG_FILE, "a") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, "retrain", count_labeled_samples(), checkpoint_path])

    return model


# ─── Status ──────────────────────────────────────────────────────────────────

def print_status():
    queue = load_queue()
    labeled   = [s for s in queue if s["labeled"]]
    unlabeled = [s for s in queue if not s["labeled"]]
    print(f"\n{'='*50}")
    print("Active Learning Status")
    print(f"{'='*50}")
    print(f"Queue total    : {len(queue)}")
    print(f"  - Labeled    : {len(labeled)}")
    print(f"  - Pending    : {len(unlabeled)}")
    print(f"Labeled on disk: {count_labeled_samples()}")
    print(f"Retrain trigger: {RETRAIN_TRIGGER} samples")
    if unlabeled:
        print(f"\nTop 5 pending (highest uncertainty):")
        for s in sorted(unlabeled, key=lambda x: x["score"], reverse=True)[:5]:
            print(f"  {s['filename']:40s}  score={s['score']:.4f}  pred={s['pred_class']} ({s['pred_conf']:.2f})")
    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Active Learning Pipeline")
    parser.add_argument(
        "--mode",
        choices=["query", "retrain", "status", "label"],
        default="status",
        help="Pipeline step to run",
    )
    parser.add_argument("--filename", help="Image filename (for --mode label)")
    parser.add_argument("--class_name", help="True class label (for --mode label)")
    parser.add_argument(
        "--method",
        choices=["entropy", "least_confidence", "margin"],
        default="entropy",
        help="Uncertainty sampling method",
    )
    args = parser.parse_args()

    if args.mode == "status":
        print_status()
        return

    if args.mode == "label":
        if not args.filename or not args.class_name:
            print("--filename and --class_name are required for --mode label")
            return
        apply_label(args.filename, args.class_name)
        return

    model, idx_to_class = load_model_and_labels()

    if args.mode == "query":
        samples = query_uncertain_samples(model, idx_to_class, budget=QUERY_BUDGET, method=args.method)
        add_to_queue(samples)
        print_status()

    elif args.mode == "retrain":
        if not should_retrain():
            print("Not enough labeled samples yet. Keep annotating!")
        else:
            retrain_with_labeled(model, idx_to_class)


if __name__ == "__main__":
    main()
