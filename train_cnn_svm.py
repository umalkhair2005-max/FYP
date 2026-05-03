"""
Train DenseNet121 + head → CNN, then extract features → SVM.

Generalization (Google / other datasets / different scanners):
  • Train on as DIVERSE chest X-rays as you can (multiple sources if possible).
  • Strong augmentation below simulates brightness/contrast/position changes — not a
    guarantee for every web image, but reduces "only my folder works" overfitting.
  • SVM features MUST match labels: extraction uses shuffle=False (fixed bug vs shuffled train).

Optional env (better generalization, slower on CPU):
  set FINE_TUNE_LAYERS=40   — unfreeze last N base layers after phase 1 (default 0 = off)
  FINE_TUNE_EPOCHS=8
"""
import os
import pickle

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# PATHS
# -------------------------
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

# -------------------------
# HYPERPARAMETERS
# -------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Stronger augmentation → better robustness to new sources / lighting / framing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=8,
    zoom_range=0.18,
    horizontal_flip=True,
    brightness_range=(0.75, 1.25),
    channel_shift_range=40,
    fill_mode="nearest",
)

# Validation / test: no random aug (honest metrics)
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# For SVM feature extraction: NO augmentation + fixed order (labels align with rows)
feat_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    seed=42,
)
val_data = val_test_datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)
test_data = val_test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

train_data_feat = feat_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

# -------------------------
# CNN MODEL
# -------------------------
base_model = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu", name="feature_layer")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

cnn_model = Model(inputs=base_model.input, outputs=output)

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(MODEL_PATH, "cnn_best_model.h5"),
        monitor="val_loss",
        save_best_only=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.4,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# -------------------------
# PHASE 1 — train CNN (frozen base)
# -------------------------
history = cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks,
)

# -------------------------
# PHASE 2 — optional fine-tune last N base layers (helps adapt beyond ImageNet)
# -------------------------
fine_tune_layers = int(os.environ.get("FINE_TUNE_LAYERS", "0"))
fine_tune_epochs = int(os.environ.get("FINE_TUNE_EPOCHS", "8"))

if fine_tune_layers > 0:
    for layer in base_model.layers[-fine_tune_layers:]:
        layer.trainable = True

    # Keep batch norm stable: very small LR on unfrozen blocks
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-6),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    ft_callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ModelCheckpoint(
            os.path.join(MODEL_PATH, "cnn_best_model.h5"),
            monitor="val_loss",
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-8,
            verbose=1,
        ),
    ]
    _start = len(history.history["loss"])
    _end = _start + fine_tune_epochs
    history_ft = cnn_model.fit(
        train_data,
        validation_data=val_data,
        epochs=_end,
        callbacks=ft_callbacks,
        initial_epoch=_start,
    )
    # Merge histories for plotting
    for k in history.history:
        if k in history_ft.history:
            history.history[k].extend(history_ft.history[k])

# -------------------------
# FEATURE EXTRACTOR + SVM (aligned rows ↔ labels)
# -------------------------
feature_extractor = Model(
    inputs=cnn_model.input, outputs=cnn_model.get_layer("feature_layer").output
)

print("\nExtracting training features for SVM (no shuffle — labels aligned)...")
X_train = feature_extractor.predict(train_data_feat, verbose=1)
y_train = train_data_feat.classes

print("Extracting test features...")
X_test = feature_extractor.predict(test_data, verbose=1)
y_test = test_data.classes

# -------------------------
# TRAIN SVM
# -------------------------
svm_model = SVC(kernel="rbf", probability=True, class_weight="balanced")
svm_model.fit(X_train, y_train)

# -------------------------
# EVALUATE
# -------------------------
y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

report = classification_report(y_test, y_pred)
print(report)

np.savetxt(os.path.join(MODEL_PATH, "confusion_matrix.txt"), cm, fmt="%d")

with open(os.path.join(MODEL_PATH, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
    f.write("\n\nconfusion_matrix (test set, rows=actual, cols=predicted):\n")
    f.write(np.array2string(cm))
    f.write(
        "\n\nNote: Good test accuracy on YOUR val/test folders does not guarantee "
        "perfect accuracy on random Google images — add diverse training data + "
        "keep augmentation strong.\n"
    )

# -------------------------
# SAVE MODELS
# -------------------------
feature_extractor.save(os.path.join(MODEL_PATH, "cnn_feature_extractor.h5"))
joblib.dump(svm_model, os.path.join(MODEL_PATH, "svm_model.pkl"))
with open(os.path.join(MODEL_PATH, "training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# -------------------------
# PLOTS
# -------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.legend()
plt.title("CNN Accuracy")
plt.savefig(os.path.join(MODEL_PATH, "cnn_accuracy.png"))

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("CNN Loss")
plt.savefig(os.path.join(MODEL_PATH, "cnn_loss.png"))

plt.show()

print("\nDone. Copy models/*.h5 and svm_model.pkl into your project models/ folder if needed.")
