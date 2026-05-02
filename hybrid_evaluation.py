import tensorflow as tf
import numpy as np
import os
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------
# PATHS
# -------------------------
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH  = os.path.join(DATASET_PATH, "test")
MODEL_PATH = "models"

os.makedirs(MODEL_PATH, exist_ok=True)

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# -------------------------
# LOAD TRAINED CNN
# -------------------------
cnn_model = tf.keras.models.load_model(
    os.path.join(MODEL_PATH, "cnn_best_model.h5")
)

# -------------------------
# FEATURE EXTRACTOR
# -------------------------
feature_extractor = Model(
    inputs=cnn_model.input,
    outputs=cnn_model.get_layer("feature_layer").output
)

# -------------------------
# DATA GENERATORS (NO SHUFFLE)
# -------------------------
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_gen = datagen.flow_from_directory(
    TEST_PATH,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# -------------------------
# FEATURE EXTRACTION
# -------------------------
X_train = feature_extractor.predict(train_gen)
y_train = train_gen.classes

X_test = feature_extractor.predict(test_gen)
y_test = test_gen.classes

# -------------------------
# TRAIN SVM
# -------------------------
svm_model = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced'
)

svm_model.fit(X_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
y_pred = svm_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(cm)
print(report)

# -------------------------
# SAVE CONFUSION MATRIX
# -------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Hybrid CNN + SVM Confusion Matrix")
plt.savefig(os.path.join(MODEL_PATH, "confusion_matrix.png"))
plt.show()

# -------------------------
# SAVE CLASSIFICATION REPORT
# -------------------------
with open(os.path.join(MODEL_PATH, "classification_report.txt"), "w") as f:
    f.write(report)

# -------------------------
# SAVE MODELS
# -------------------------
feature_extractor.save(os.path.join(MODEL_PATH, "cnn_feature_extractor.h5"))
joblib.dump(svm_model, os.path.join(MODEL_PATH, "svm_model.pkl"))

print("✅ Hybrid model evaluation completed & files saved.")