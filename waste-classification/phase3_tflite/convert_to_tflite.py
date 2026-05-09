"""
Phase 3: Convert Keras model → TFLite with INT8 quantization
Author: Pulyala Ajay Kumar

Steps:
  1. Load trained .keras model
  2. Convert with dynamic-range quantization (default) for speed
  3. Optionally run full INT8 quantization with a representative dataset
  4. Validate: run TFLite interpreter on test set, confirm <2% accuracy drop
  5. Output: waste_classifier.tflite + labels.txt  (copy both to Android assets/)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

KERAS_MODEL   = "models/waste_classifier_v2.keras"
LABELS_PATH   = "models/class_indices.json"
DATA_DIR      = "data"
OUTPUT_DIR    = "phase3_tflite/assets"
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_CALIB_BATCHES = 50        # batches used for INT8 calibration

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Helper: representative dataset for full INT8 quantization ───────────────

def representative_dataset():
    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=True,
    )
    for i, (img_batch, _) in enumerate(gen):
        if i >= NUM_CALIB_BATCHES * BATCH_SIZE:
            break
        yield [img_batch.astype(np.float32)]


# ─── Conversion ──────────────────────────────────────────────────────────────

def convert_dynamic_range(model) -> str:
    """Fast option — float16 weights, float32 activations. ~50% size reduction."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    path = os.path.join(OUTPUT_DIR, "waste_classifier_dynamic.tflite")
    Path(path).write_bytes(tflite_model)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Dynamic-range TFLite saved: {path}  ({size_mb:.2f} MB)")
    return path


def convert_int8(model) -> str:
    """Full INT8 — requires representative dataset. Best for edge devices."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    path = os.path.join(OUTPUT_DIR, "waste_classifier_int8.tflite")
    Path(path).write_bytes(tflite_model)
    size_mb = os.path.getsize(path) / 1e6
    print(f"INT8 TFLite saved: {path}  ({size_mb:.2f} MB)")
    return path


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_tflite(tflite_path: str, is_int8: bool = False):
    """Run TFLite interpreter on the test set, report accuracy."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        shuffle=False,
    )

    correct = 0
    total = 0
    for img_batch, label_batch in test_gen:
        if total >= len(test_gen.filenames):
            break
        if is_int8:
            inp = (img_batch * 255).astype(np.uint8)
        else:
            inp = img_batch.astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        pred = np.argmax(output[0])
        true = np.argmax(label_batch[0])
        correct += int(pred == true)
        total += 1

    acc = correct / total
    print(f"TFLite validation accuracy ({tflite_path}): {acc:.4f}  ({correct}/{total})")
    return acc


# ─── Labels file for Android ─────────────────────────────────────────────────

def write_labels():
    with open(LABELS_PATH) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    lines = [idx_to_class[i] for i in range(len(idx_to_class))]
    labels_out = os.path.join(OUTPUT_DIR, "labels.txt")
    with open(labels_out, "w") as f:
        f.write("\n".join(lines))
    print(f"Labels written: {labels_out}")
    return labels_out


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3: TFLite Conversion Pipeline")
    print("=" * 60)

    print(f"\nLoading model from {KERAS_MODEL} …")
    model = tf.keras.models.load_model(KERAS_MODEL)
    model.summary()

    # Write labels
    write_labels()

    # Convert — dynamic range (always)
    print("\n[1/2] Dynamic-range quantization …")
    dyn_path = convert_dynamic_range(model)
    dyn_acc  = validate_tflite(dyn_path, is_int8=False)

    # Convert — INT8 (requires calibration data; skip if data not present)
    if os.path.exists(os.path.join(DATA_DIR, "train")):
        print("\n[2/2] INT8 full quantization …")
        int8_path = convert_int8(model)
        int8_acc  = validate_tflite(int8_path, is_int8=True)
        print(f"\nDynamic accuracy : {dyn_acc:.4f}")
        print(f"INT8 accuracy    : {int8_acc:.4f}")
        print(f"Drop             : {(dyn_acc - int8_acc):.4f}  (target < 0.02)")
    else:
        print("\n[2/2] Skipped INT8 — training data not found for calibration.")

    print(f"\nCopy these files to your Android project's assets/ folder:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  {OUTPUT_DIR}/{f}")


if __name__ == "__main__":
    main()
