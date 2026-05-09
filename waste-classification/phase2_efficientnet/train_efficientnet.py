"""
Phase 2: EfficientNet-B0 Fine-tuning for 10-Category Waste Classification
Author: Pulyala Ajay Kumar
Dataset: TrashNet (extended) — https://github.com/garythung/trashnet
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import json
import datetime

# ─── Config ──────────────────────────────────────────────────────────────────

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_FROZEN  = 20
EPOCHS_FINETUNE = 10
LEARNING_RATE   = 1e-3
FINETUNE_LR     = 1e-5
UNFREEZE_LAYERS = 30          # top N layers to unfreeze during fine-tuning
DATA_DIR    = "data"          # expects data/train, data/val, data/test
MODEL_DIR   = "models"
LOG_DIR     = "logs"

CLASSES = [
    "cardboard", "e_waste", "food_waste", "glass",
    "hazardous", "metal",   "paper",     "plastic",
    "rubber",    "textile"
]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Data Pipeline ───────────────────────────────────────────────────────────

def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
    )
    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True,
    )
    val_gen = val_aug.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )
    test_gen = val_aug.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


def compute_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return dict(enumerate(weights))


# ─── Model ───────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 10, trainable_base: bool = False) -> Model:
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = trainable_base

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)


def unfreeze_top_layers(model: Model, n: int = UNFREEZE_LAYERS):
    """Unfreeze the top N layers of the EfficientNet base."""
    base = model.layers[1]          # EfficientNetB0 is layers[1]
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    print(f"Unfrozen top {n} layers of EfficientNetB0")


# ─── Training ────────────────────────────────────────────────────────────────

def get_callbacks(tag: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return [
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"best_{tag}.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=os.path.join(LOG_DIR, f"{tag}_{timestamp}")),
    ]


def train():
    print("=" * 60)
    print("Waste Classification — EfficientNet-B0 Training")
    print("=" * 60)

    train_gen, val_gen, test_gen = build_generators()
    class_weights = compute_weights(train_gen)
    print(f"\nClass weights: {class_weights}\n")

    # Save class index map
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # ── Stage 1: Train head only (frozen base) ────────────────────────────
    print("\n[Stage 1] Training classification head — base frozen")
    model = build_model(num_classes=len(CLASSES), trainable_base=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )
    model.summary()

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FROZEN,
        class_weight=class_weights,
        callbacks=get_callbacks("stage1"),
    )

    # ── Stage 2: Fine-tune top layers ─────────────────────────────────────
    print(f"\n[Stage 2] Fine-tuning top {UNFREEZE_LAYERS} layers")
    unfreeze_top_layers(model, UNFREEZE_LAYERS)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        class_weight=class_weights,
        callbacks=get_callbacks("stage2_finetune"),
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    print("\n[Evaluation] Test set performance")
    test_loss, test_acc, test_top3 = model.evaluate(test_gen, verbose=1)
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test top-3 acc: {test_top3:.4f}")

    y_true = test_gen.classes
    y_pred = np.argmax(model.predict(test_gen, verbose=1), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.title("Confusion Matrix — EfficientNet-B0")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("Confusion matrix saved.")

    # Save final model
    model.save(os.path.join(MODEL_DIR, "waste_classifier_v2.keras"))
    print(f"\nModel saved to {MODEL_DIR}/waste_classifier_v2.keras")
    return model


if __name__ == "__main__":
    train()
