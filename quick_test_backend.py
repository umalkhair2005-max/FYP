import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

# LOAD MODELS
cnn = tf.keras.models.load_model("models/cnn_feature_extractor.h5")
svm = joblib.load("models/svm_model.pkl")

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

# TEST IMAGE (ek known image use karo)
img_path = "dataset/test/NORMAL/IM-0001-0001.jpeg"  # ya pneumonia wali

img = preprocess(img_path)
features = cnn.predict(img)
prob = svm.predict_proba(features)[0]

print("Normal Probability   :", round(prob[0]*100,2))
print("Pneumonia Probability:", round(prob[1]*100,2))