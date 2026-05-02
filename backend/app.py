import os
from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import joblib
import numpy as np
from PIL import Image
import cv2
import sqlite3
import webbrowser

# -------------------------
# FLASK APP
# -------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend/template"),
    static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend/static")
)

app.secret_key = "supersecretkey"

# -------------------------
# UPLOAD FOLDER
# -------------------------
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------
# DATABASE INITIALIZATION
# -------------------------
conn = sqlite3.connect("users.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT,
    password TEXT
)
""")

conn.commit()

conn.close()

# -------------------------
# LOAD MODELS
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "../models")

# FEATURE EXTRACTOR
cnn = tf.keras.models.load_model(
    os.path.join(MODEL_PATH, "cnn_feature_extractor.h5"),
    compile=False
)

# FULL CNN MODEL FOR GRAD-CAM
full_cnn = tf.keras.models.load_model(
    os.path.join(MODEL_PATH, "cnn_best_model.h5"),
    compile=False
)

# SVM MODEL
svm = joblib.load(
    os.path.join(MODEL_PATH, "svm_model.pkl")
)

# -------------------------
# IMAGE PREPROCESSING
# -------------------------
def preprocess(img):

    img = img.resize((224, 224))

    img = np.array(img) / 255.0

    img = np.expand_dims(img, axis=0)

    return img

# -------------------------
# GRAD-CAM
# -------------------------
LAST_CONV_LAYER = "conv5_block16_2_conv"

def get_gradcam(model, img_array):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(LAST_CONV_LAYER).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# -------------------------
# SAVE GRAD-CAM
# -------------------------
def save_gradcam(img_path, heatmap, output_path, alpha=0.40):

    img = cv2.imread(img_path)

    img = cv2.resize(img, (224,224))

    heatmap = cv2.resize(heatmap, (224,224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = cv2.addWeighted(
        heatmap,
        alpha,
        img,
        1 - alpha,
        0
    )

    cv2.imwrite(output_path, superimposed_img)

    return output_path

# -------------------------
# ROUTES
# -------------------------

# ===== HOMEPAGE =====
@app.route("/")
def home():

    if "user" in session:
        return redirect(url_for("project"))

    return redirect(url_for("login"))

# ===== SIGNUP =====
@app.route("/signup", methods=["GET","POST"])
def signup():

    if request.method == "POST":

        username = request.form["username"]

        email = request.form["email"]

        password = request.form["password"]

        conn = sqlite3.connect("users.db")

        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (username,email,password) VALUES (?,?,?)",
            (username,email,password)
        )

        conn.commit()

        conn.close()

        session["user"] = username

        return redirect(url_for("project"))

    return render_template("signup.html")

# ===== LOGIN =====
@app.route("/login", methods=["GET","POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        conn = sqlite3.connect("users.db")

        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username,password)
        )

        user = cursor.fetchone()

        conn.close()

        if user:

            session["user"] = username

            return redirect(url_for("project"))

        else:

            return render_template(
                "login.html",
                error="Invalid Username or Password."
            )

    return render_template("login.html")

# ===== LOGOUT =====
@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))
# ===== PROJECT PAGE =====
@app.route("/project", methods=["GET","POST"])
def project():

    if "user" not in session:
        return redirect(url_for("login"))

    result = None
    confidence = None
    img_name = None
    gradcam_static_path = None
    original_static_path = None
    description = None
    suggestions = []

    if request.method == "POST":

        img_file = request.files["image"]

        if img_file:

            img_name = img_file.filename

            img_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                img_name
            )

            img_file.save(img_path)

            # IMAGE
            img = Image.open(img_path).convert("RGB")

            # PREPROCESS
            img_array = preprocess(img)

            # FEATURE EXTRACTION
            features = cnn.predict(img_array)

            # SVM PREDICTION
            prediction = svm.predict(features)[0]

            # PROBABILITIES
            prob = svm.predict_proba(features)[0]

            # CONFIDENCE
            confidence = round(max(prob) * 100, 2)

            # CLASS LABEL
            if prediction == 1:

                result = "PNEUMONIA"

                description = """
                Pneumonia detected. This condition may affect
                the lungs and breathing system. Please consult
                a healthcare professional for proper medical evaluation.
                """

                suggestions = [
                    "Consult a chest specialist immediately.",
                    "Take proper rest and stay hydrated.",
                    "Avoid smoking and cold environments.",
                    "Monitor fever and breathing difficulty.",
                    "Use prescribed medicines only.",
                    "Seek emergency care if symptoms worsen."
                ]

            else:

                result = "NORMAL"

                description = """
                No pneumonia detected. The chest X-ray appears normal.
                Maintain a healthy lifestyle to keep lungs healthy.
                """

                suggestions = [
                    "Maintain a healthy diet.",
                    "Exercise regularly.",
                    "Avoid smoking and pollution.",
                    "Drink enough water.",
                    "Get regular medical checkups.",
                    "Follow proper hygiene practices."
                ]

            print("Prediction:", prediction)

            print("Probabilities:", prob)

            # -------------------------
            # GRAD-CAM
            # -------------------------
            gradcam_filename = f"gradcam_{img_name}"

            gradcam_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                gradcam_filename
            )

            heatmap = get_gradcam(
                full_cnn,
                img_array
            )

            save_gradcam(
                img_path,
                heatmap,
                gradcam_path
            )

            gradcam_static_path = f"uploads/{gradcam_filename}"

            original_static_path = f"uploads/{img_name}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        description=description,
        suggestions=suggestions,
        img_name=img_name,
        gradcam_path=gradcam_static_path,
        original_image=original_static_path
    )

# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":

    webbrowser.open("http://127.0.0.1:5000")

    app.run(debug=True)