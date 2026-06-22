# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 13:02:54 2026

@author: CSE-AIML
"""

# -*- coding: utf-8 -*-
"""
Hybrid RGB + LBP Ablation Study (A1–A5)
5-Fold Cross Validation
EarlyStopping + 50 Epochs
Saves:
- best_model.h5
- model.h5
- confusion_matrix.png
- roc_curve.png
- metrics.json
- Final_Summary.csv
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from skimage.feature import local_binary_pattern

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Multiply,
    Reshape, Conv2D, Input, Lambda,
    Concatenate, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# =====================================================
# CONFIGURATION
# =====================================================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
NUM_CLASSES = 9

TRAIN_DIR = r"D:\Mrityunjoy\Disease(after data imbalancing)\train"
VAL_DIR   = r"D:\Mrityunjoy\Disease(after data imbalancing)\val"

RESULT_DIR = r"D:\Mrityunjoy\Ablation_Final_result(50 epochs)"
os.makedirs(RESULT_DIR, exist_ok=True)


# =====================================================
# LBP FUNCTION
# =====================================================
def lbp_rgb(image):
    lbp_channels = []
    for c in range(3):
        lbp = local_binary_pattern(image[:, :, c], 8, 1, method="uniform")
        lbp = lbp / (lbp.max() + 1e-8)
        lbp_channels.append(lbp)
    return np.stack(lbp_channels, axis=-1)


# =====================================================
# LOAD DATA
# =====================================================
records = []

def collect_images(root_dir):
    for cls in sorted(os.listdir(root_dir)):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in os.listdir(cls_dir):
            records.append({
                "filename": os.path.join(cls_dir, f),
                "label": cls
            })

collect_images(TRAIN_DIR)
collect_images(VAL_DIR)

df = pd.DataFrame(records)
df["label"] = df["label"].astype("category")

y = df["label"].cat.codes.values
X = df.index.values


# =====================================================
# 5-FOLD
# =====================================================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X, y))


# =====================================================
# DATA GENERATOR
# =====================================================
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

class HybridDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, base_gen):
        self.base_gen = base_gen

    def __len__(self):
        return len(self.base_gen)

    def __getitem__(self, idx):
        x_rgb, y = self.base_gen[idx]

        lbp_batch = []
        for img in x_rgb:
            img_uint8 = ((img + 1) * 127.5).astype(np.uint8)
            lbp_batch.append(lbp_rgb(img_uint8))

        lbp_batch = np.array(lbp_batch, dtype=np.float32)
        return (x_rgb, lbp_batch), y


# =====================================================
# SE BLOCK
# =====================================================
def se_block(inputs, ratio=8):
    filters = inputs.shape[-1]
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(filters // ratio, activation="relu")(x)
    x = Dense(filters, activation="sigmoid")(x)
    x = Reshape((1, 1, filters))(x)
    return Multiply()([inputs, x])


# =====================================================
# MODEL BUILDERS
# =====================================================
def build_A1():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=inp)
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_A2():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=inp)
    x = se_block(base.output)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_A3():
    rgb = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    lbp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    base = MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=rgb)
    cnn_feat = GlobalAveragePooling2D()(base.output)
    cnn_feat = Dense(128, activation="relu")(cnn_feat)

    y = Conv2D(32, 3, activation="relu", padding="same")(lbp)
    y = GlobalAveragePooling2D()(y)
    lbp_feat = Dense(128, activation="relu")(y)

    fused = Concatenate()([cnn_feat, lbp_feat])
    out = Dense(NUM_CLASSES, activation="softmax")(fused)

    model = Model([rgb, lbp], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_A4():
    rgb = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    lbp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    base = MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=rgb)
    cnn_feat = GlobalAveragePooling2D()(base.output)
    cnn_feat = Dense(128, activation="relu")(cnn_feat)

    y = Conv2D(32, 3, activation="relu", padding="same")(lbp)
    y = GlobalAveragePooling2D()(y)
    lbp_feat = Dense(128, activation="relu")(y)

    alpha = Dense(1, activation="sigmoid")(cnn_feat)
    cnn_w = Multiply()([cnn_feat, alpha])
    lbp_w = Multiply()([lbp_feat, Lambda(lambda a: 1 - a)(alpha)])
    fused = Add()([cnn_w, lbp_w])

    out = Dense(NUM_CLASSES, activation="softmax")(fused)

    model = Model([rgb, lbp], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_A5():
    rgb = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    lbp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    base = MobileNetV3Small(include_top=False, weights="imagenet", input_tensor=rgb)
    x = se_block(base.output)
    cnn_feat = GlobalAveragePooling2D()(x)
    cnn_feat = Dense(128, activation="relu")(cnn_feat)

    y = Conv2D(32, 3, activation="relu", padding="same")(lbp)
    y = GlobalAveragePooling2D()(y)
    lbp_feat = Dense(128, activation="relu")(y)

    fused = Concatenate()([cnn_feat, lbp_feat])
    out = Dense(NUM_CLASSES, activation="softmax")(fused)

    model = Model([rgb, lbp], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


ABLATIONS = {
    "A1": build_A1,
    "A2": build_A2,
    "A3": build_A3,
    "A4": build_A4,
    "A5": build_A5
}

INPUT_MODE = {
    "A1": "rgb",
    "A2": "rgb",
    "A3": "dual",
    "A4": "dual",
    "A5": "dual"
}


# =====================================================
# TRAINING LOOP
# =====================================================
all_results = {}

for ab in ABLATIONS:

    fold_metrics = []

    for fold_no, (tr_idx, va_idx) in enumerate(folds, 1):

        save_dir = os.path.join(RESULT_DIR, ab, f"Fold{fold_no}")
        os.makedirs(save_dir, exist_ok=True)

        train_df = df.iloc[tr_idx]
        val_df   = df.iloc[va_idx]

        train_base = datagen.flow_from_dataframe(
            train_df,
            x_col="filename",
            y_col="label",
            target_size=(IMG_SIZE, IMG_SIZE),
            class_mode="categorical",
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_base = datagen.flow_from_dataframe(
            val_df,
            x_col="filename",
            y_col="label",
            target_size=(IMG_SIZE, IMG_SIZE),
            class_mode="categorical",
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        if INPUT_MODE[ab] == "rgb":
            train_gen = train_base
            val_gen   = val_base
        else:
            train_gen = HybridDataGenerator(train_base)
            val_gen   = HybridDataGenerator(val_base)

        model = ABLATIONS[ab]()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8,
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(os.path.join(save_dir, "best_model.h5"),
                            monitor="val_loss", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                              patience=4, verbose=1)
        ]

        model.fit(train_gen,
                  validation_data=val_gen,
                  epochs=EPOCHS,
                  callbacks=callbacks,
                  verbose=1)

        model.save(os.path.join(save_dir, "model.h5"))

        y_prob = model.predict(val_gen)
        y_pred = np.argmax(y_prob, axis=1)
        y_true = val_base.classes

        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.close()

        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        aucs = []

        plt.figure()
        for i in range(NUM_CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")

        plt.plot([0,1],[0,1],'k--')
        macro_auc = np.mean(aucs)
        plt.title(f"ROC Curve (Macro AUC={macro_auc:.2f})")
        plt.legend(fontsize=6)
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.close()

        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump({
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "macro_f1": float(f1),
                "macro_auc": float(macro_auc)
            }, f, indent=4)

        fold_metrics.append({
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "macro_f1": f1,
            "macro_auc": macro_auc
        })

    all_results[ab] = fold_metrics


# =====================================================
# FINAL SUMMARY CSV
# =====================================================
summary = []

for ab, res in all_results.items():
    summary.append({
        "Ablation": ab,
        "Accuracy Mean": np.mean([m["accuracy"] for m in res]),
        "Accuracy Std": np.std([m["accuracy"] for m in res]),
        "Balanced Accuracy Mean": np.mean([m["balanced_accuracy"] for m in res]),
        "Macro F1 Mean": np.mean([m["macro_f1"] for m in res]),
        "Macro AUC Mean": np.mean([m["macro_auc"] for m in res])
    })

pd.DataFrame(summary).to_csv(
    os.path.join(RESULT_DIR, "Final_Summary.csv"),
    index=False
)

print("Training Completed Successfully 🚀")