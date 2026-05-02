import tensorflow as tf
import numpy as np
import os
import joblib
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------------------------
# PATHS
# -------------------------
DATASET_PATH = "dataset"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH   = os.path.join(DATASET_PATH, "val")
TEST_PATH  = os.path.join(DATASET_PATH, "test")
MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

# -------------------------
# IMAGE AUGMENTATION
# -------------------------
IMG_SIZE = (224,224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(TRAIN_PATH, target_size=IMG_SIZE,
                                               batch_size=BATCH_SIZE, class_mode='binary', shuffle=True)
val_data = val_test_datagen.flow_from_directory(VAL_PATH, target_size=IMG_SIZE,
                                                batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
test_data = val_test_datagen.flow_from_directory(TEST_PATH, target_size=IMG_SIZE,
                                                 batch_size=1, class_mode='binary', shuffle=False)

# -------------------------
# CNN MODEL
# -------------------------
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', name="feature_layer")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs=base_model.input, outputs=output)

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# -------------------------
# CALLBACKS
# -------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_PATH,"cnn_best_model.h5"), save_best_only=True)
]

# -------------------------
# TRAIN CNN
# -------------------------
history = cnn_model.fit(train_data,
                        validation_data=val_data,
                        epochs=25,
                        callbacks=callbacks)

# -------------------------
# FEATURE EXTRACTION (SVM READY)
# -------------------------
feature_extractor = Model(inputs=cnn_model.input,
                          outputs=cnn_model.get_layer("feature_layer").output)

X_train = feature_extractor.predict(train_data)
y_train = train_data.classes

X_test = feature_extractor.predict(test_data)
y_test = test_data.classes

# -------------------------
# TRAIN SVM
# -------------------------
svm_model = SVC(kernel='rbf', probability=True, class_weight='balanced')
svm_model.fit(X_train, y_train)

# -------------------------
# EVALUATE
# -------------------------
y_pred = svm_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

report = classification_report(y_test, y_pred)
print(report)

# Save classification report
with open(os.path.join(MODEL_PATH, "classification_report.txt"), "w") as f:
    f.write(report)

# -------------------------
# SAVE MODELS
# -------------------------
feature_extractor.save(os.path.join(MODEL_PATH,"cnn_feature_extractor.h5"))
joblib.dump(svm_model, os.path.join(MODEL_PATH,"svm_model.pkl"))
with open(os.path.join(MODEL_PATH,"training_history.pkl"), "wb") as f:
    pickle.dump(history.history, f)

# -------------------------
# PLOTS (SAVE TO FILES)
# -------------------------
plt.figure(figsize=(12,4))

# Accuracy graph
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("CNN Accuracy")
plt.savefig(os.path.join(MODEL_PATH, "cnn_accuracy.png"))  # save file

# Loss graph
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("CNN Loss")
plt.savefig(os.path.join(MODEL_PATH, "cnn_loss.png"))  # save file

plt.show()