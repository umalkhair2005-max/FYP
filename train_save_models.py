import joblib
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

# -------------------------
# 1️⃣ Load Models
# -------------------------
cnn = load_model("models/cnn_feature_extractor.h5")
svm = joblib.load("models/svm_model.pkl")

# -------------------------
# 2️⃣ Load Training History (if saved)
# -------------------------
with open("models/training_history.pkl", "rb") as f:
    history = pickle.load(f)

# -------------------------
# 3️⃣ Plot & Save Accuracy / Loss Graphs
# -------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history['accuracy'], label='Train Acc')
plt.plot(history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("CNN Accuracy")
plt.savefig("models/cnn_accuracy.png")  # save file

plt.subplot(1,2,2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title("CNN Loss")
plt.savefig("models/cnn_loss.png")  # save file

plt.show()

# -------------------------
# 4️⃣ Load Test Data Features (SVM Prediction)
# -------------------------
# Agar features pehle save kiye hue hain to load karo
# Ya phir dobara extract karo test_data se
# Example: (agar test_data flow_from_directory me hai)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
TEST_PATH = "dataset/test"

test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(TEST_PATH,
                                             target_size=(224,224),
                                             batch_size=1,
                                             class_mode='binary',
                                             shuffle=False)

X_test = cnn.predict(test_data)
y_test = test_data.classes

# -------------------------
# 5️⃣ SVM Prediction + Classification Report
# -------------------------
from sklearn.metrics import classification_report, confusion_matrix

y_pred = svm.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# Save classification report to file
with open("models/classification_report.txt", "w") as f:
    f.write(report)

# Save confusion matrix if needed
cm = confusion_matrix(y_test, y_pred)
np.savetxt("models/confusion_matrix.txt", cm, fmt='%d')