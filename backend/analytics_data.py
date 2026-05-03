"""
Load training metrics from disk for the AI Analytics dashboard.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Any, Optional


def load_training_analytics(models_dir: str) -> dict[str, Any]:
    """
    Reads training_history.pkl + classification_report.txt from models folder.
    Returns chart-ready lists and parsed metrics (safe if files missing).
    """
    out: dict[str, Any] = {
        "has_history": False,
        "has_report": False,
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "epochs": [],
        "classification_report_text": "",
        "accuracy_pct": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "weighted_precision": None,
        "weighted_recall": None,
        "weighted_f1": None,
        "confusion_matrix": None,
        "class_rows": [],
    }

    hp = os.path.join(models_dir, "training_history.pkl")
    if os.path.isfile(hp):
        try:
            with open(hp, "rb") as f:
                hist = pickle.load(f)
            out["has_history"] = True
            out["train_acc"] = [float(x) for x in hist.get("accuracy", [])]
            out["val_acc"] = [float(x) for x in hist.get("val_accuracy", [])]
            out["train_loss"] = [float(x) for x in hist.get("loss", [])]
            out["val_loss"] = [float(x) for x in hist.get("val_loss", [])]
            n = max(
                len(out["train_acc"]),
                len(out["val_acc"]),
                len(out["train_loss"]),
                len(out["val_loss"]),
            )
            out["epochs"] = list(range(1, n + 1))
        except (OSError, pickle.PickleError, TypeError, ValueError):
            pass

    cr_path = os.path.join(models_dir, "classification_report.txt")
    if os.path.isfile(cr_path):
        try:
            with open(cr_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            out["has_report"] = True
            out["classification_report_text"] = text

            m_acc = re.search(
                r"accuracy\s+([\d.]+)\s+(\d+)\s*$", text, re.I | re.MULTILINE
            )
            if not m_acc:
                m_acc = re.search(r"accuracy\s+([\d.]+)", text, re.I)
            if m_acc:
                acc_v = float(m_acc.group(1))
                out["accuracy_pct"] = round(acc_v * 100, 2) if acc_v <= 1.0 else round(
                    acc_v, 2
                )

            for label, pref in (
                ("macro avg", "macro"),
                ("weighted avg", "weighted"),
            ):
                block = re.search(
                    rf"{label}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
                    text,
                    re.I | re.MULTILINE,
                )
                if block:
                    out[f"{pref}_precision"] = float(block.group(1))
                    out[f"{pref}_recall"] = float(block.group(2))
                    out[f"{pref}_f1"] = float(block.group(3))

            flat = re.sub(r"\s+", " ", text)
            cm = re.search(
                r"\[\[?\s*(\d+)\s+(\d+)\s*\]\s*\[\s*(\d+)\s+(\d+)\s*\]", flat
            )
            if cm:
                out["confusion_matrix"] = [
                    [int(cm.group(1)), int(cm.group(2))],
                    [int(cm.group(3)), int(cm.group(4))],
                ]

            rows = []
            for cls in ("0", "1"):
                line = re.search(
                    rf"^\s*{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
                    text,
                    re.MULTILINE,
                )
                if line:
                    label_name = "NORMAL" if cls == "0" else "PNEUMONIA"
                    rows.append(
                        {
                            "class": label_name,
                            "precision": float(line.group(1)),
                            "recall": float(line.group(2)),
                            "f1": float(line.group(3)),
                            "support": int(line.group(4)),
                        }
                    )
            out["class_rows"] = rows
        except OSError:
            pass

    return out


def format_pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.2f}%" if val <= 1.0 else f"{val:.2f}%"
