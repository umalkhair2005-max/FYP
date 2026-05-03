"""
Recompute confusion matrix + classification_report from saved CNN feature extractor + SVM.
No training — loads weights from disk and evaluates on dataset/test (or given folder).
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def recompute_eval_metrics(
    models_dir: str,
    test_dir: str,
    *,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
) -> dict:
    """
    Load cnn_feature_extractor.h5 + svm_model.pkl, run on test_dir, write:
      - confusion_matrix.txt, classification_report.txt, confusion_matrix.png

    Returns a dict with ok, confusion_matrix (list of rows), test_samples, message.
    """
    fe_path = os.path.join(models_dir, "cnn_feature_extractor.h5")
    svm_path = os.path.join(models_dir, "svm_model.pkl")
    if not os.path.isfile(fe_path):
        raise FileNotFoundError(f"Missing CNN extractor: {fe_path}")
    if not os.path.isfile(svm_path):
        raise FileNotFoundError(f"Missing SVM model: {svm_path}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    tf.keras.backend.clear_session()
    feature_extractor = tf.keras.models.load_model(fe_path, compile=False)
    svm_model = joblib.load(svm_path)

    datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    X_test = feature_extractor.predict(test_gen, verbose=0)
    y_test = test_gen.classes
    y_pred = svm_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    np.savetxt(os.path.join(models_dir, "confusion_matrix.txt"), cm, fmt="%d")

    report_path = os.path.join(models_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n\nconfusion_matrix (test set, rows=actual, cols=predicted):\n")
        f.write(np.array2string(cm))
        f.write(
            "\n\nNote: Good test accuracy on YOUR val/test folders does not guarantee "
            "perfect accuracy on random Google images — add diverse training data + "
            "keep augmentation strong.\n"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Hybrid CNN + SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "confusion_matrix.png"), dpi=120)
    plt.close()

    return {
        "ok": True,
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test)),
        "message": "Saved confusion_matrix.txt, classification_report.txt, and confusion_matrix.png from saved models (no training).",
    }
