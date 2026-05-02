import tensorflow as tf
import numpy as np
import os
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# PATHS
# -------------------------
MODEL_PATH = "models"
TEST_PATH = "dataset/test"
IMG_SIZE = (224,224)

# -------------------------
# LOAD MODELS
# -------------------------
feature_extractor = tf.keras.models.load_model(
    os.path.join(MODEL_PATH, "cnn_feature_extractor.h5")
)

svm_model = joblib.load(
    os.path.join(MODEL_PATH, "svm_model.pkl")
)

# -------------------------
# LOAD TRAINING HISTORY (FOR GRAPHS)
# -------------------------
with open(os.path.join(MODEL_PATH, "training_history.pkl"), "rb") as f:
    history = pickle.load(f)

# -------------------------
# SAVE ACCURACY & LOSS GRAPHS
# -------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("CNN Accuracy")
plt.savefig(os.path.join(MODEL_PATH, "cnn_accuracy.png"))

plt.subplot(1,2,2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title("CNN Loss")
plt.savefig(os.path.join(MODEL_PATH, "cnn_loss.png"))

plt.show()

# -------------------------
# TEST DATA GENERATOR (NO SHUFFLE)
# -------------------------
datagen = ImageDataGenerator(rescale=1./255)

test_gen = datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)
print(test_gen.class_indices)

# -------------------------
# FEATURE EXTRACTION
# -------------------------
X_test = feature_extractor.predict(test_gen)
y_test = test_gen.classes
print("X_test shape:", X_test.shape)

# -------------------------
# PREDICTIONS
# -------------------------
y_pred = svm_model.predict(X_test)

# -------------------------
# CONFUSION MATRIX
# -------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid CNN + SVM Confusion Matrix")
plt.savefig(os.path.join(MODEL_PATH, "confusion_matrix.png"))
plt.show()

# -------------------------
# CLASSIFICATION REPORT
# -------------------------
report = classification_report(y_test, y_pred)

with open(os.path.join(MODEL_PATH, "classification_report.txt"), "w") as f:
    f.write(report)

print(report)
print("✅ Evaluation complete. All graphs & reports saved.")