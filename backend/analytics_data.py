"""
Load training metrics from disk for the AI Analytics dashboard.
"""
from __future__ import annotations

import os
import pickle
import re
from typing import Any, Optional


def _load_confusion_matrix_txt(models_dir: str) -> Optional[list[list[int]]]:
    """2×2 counts from models/confusion_matrix.txt (saved by train_cnn_svm)."""
    path = os.path.join(models_dir, "confusion_matrix.txt")
    if not os.path.isfile(path):
        return None
    try:
        rows: list[list[int]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    rows.append([int(parts[0]), int(parts[1])])
        if len(rows) == 2 and all(len(r) == 2 for r in rows):
            return rows
    except (OSError, ValueError):
        pass
    return None


def _infer_cm_from_class_rows(class_rows: list[dict[str, Any]]) -> Optional[list[list[int]]]:
    """
    Rebuild 2×2 counts from per-class recall + support when CM file is missing.
    Layout: [[TN, FP], [FN, TP]] for Normal=0, Pneumonia=1.
    """
    by_class: dict[str, dict[str, Any]] = {}
    for r in class_rows:
        by_class[str(r.get("class", ""))] = r
    n = by_class.get("NORMAL")
    p = by_class.get("PNEUMONIA")
    if not n or not p:
        return None
    try:
        n0 = int(n["support"])
        n1 = int(p["support"])
        r0 = float(n["recall"])
        r1 = float(p["recall"])
    except (KeyError, TypeError, ValueError):
        return None
    if n0 <= 0 or n1 <= 0:
        return None
    tn = int(round(r0 * n0))
    fp = n0 - tn
    tp = int(round(r1 * n1))
    fn = n1 - tp
    return [[tn, fp], [fn, tp]]


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
        "confusion_matrix_inferred": False,
        "class_rows": [],
    }

    cm_file = _load_confusion_matrix_txt(models_dir)
    if cm_file:
        out["confusion_matrix"] = cm_file

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

            if out["confusion_matrix"] is None:
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

    if out["confusion_matrix"] is None and out["class_rows"]:
        inferred = _infer_cm_from_class_rows(out["class_rows"])
        if inferred:
            out["confusion_matrix"] = inferred
            out["confusion_matrix_inferred"] = True

    return out


def format_pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.2f}%" if val <= 1.0 else f"{val:.2f}%"
