"""
Evaluate saved CNN + SVM on dataset/test — no training required.

Writes the same files as training would for analytics:
  models/confusion_matrix.txt, classification_report.txt, confusion_matrix.png

Optional: replots CNN accuracy/loss from training_history.pkl (local dev).
"""
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from recompute_eval_metrics import recompute_eval_metrics  # noqa: E402

MODEL_PATH = ROOT / "models"
TEST_PATH = ROOT / "dataset" / "test"


def plot_training_history() -> None:
    hp = MODEL_PATH / "training_history.pkl"
    if not hp.is_file():
        print("(skip) No training_history.pkl for accuracy/loss plots.")
        return
    with open(hp, "rb") as f:
        history = pickle.load(f)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.legend()
    plt.title("CNN Accuracy")
    plt.savefig(MODEL_PATH / "cnn_accuracy.png")
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("CNN Loss")
    plt.savefig(MODEL_PATH / "cnn_loss.png")
    plt.close()
    print("Saved cnn_accuracy.png and cnn_loss.png from training_history.pkl")


if __name__ == "__main__":
    if not TEST_PATH.is_dir():
        print(f"ERROR: Test folder not found: {TEST_PATH}")
        sys.exit(1)
    plot_training_history()
    out = recompute_eval_metrics(str(MODEL_PATH), str(TEST_PATH))
    print(out["message"])
    print("confusion_matrix:", out["confusion_matrix"])
    print(f"test_samples: {out['test_samples']}")
    print("Done.")
