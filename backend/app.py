import os
import re
import json
import sys
import uuid
import socket
import threading
import time
import webbrowser
from functools import wraps
from datetime import date, datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


def _load_env_file(path: str) -> None:
    """Load KEY=value pairs into os.environ (backend .env overrides project root). Works without python-dotenv."""
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and val:
                    os.environ[key] = val
    except OSError:
        pass


# Always load .env from disk first (so chat works even if python-dotenv is missing).
_load_env_file(os.path.join(ROOT_DIR, ".env"))
_load_env_file(os.path.join(BASE_DIR, ".env"))

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)
    load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)
    _found = find_dotenv(usecwd=True)
    if _found and os.path.normpath(_found) not in {
        os.path.normpath(os.path.join(ROOT_DIR, ".env")),
        os.path.normpath(os.path.join(BASE_DIR, ".env")),
    }:
        load_dotenv(_found, override=True)
except ImportError:
    pass

import cv2
import joblib
import numpy as np
import tensorflow as tf
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    send_file,
    abort,
    Response,
)
from PIL import Image
from werkzeug.utils import secure_filename

import database as db
from analytics_data import load_training_analytics
from recompute_eval_metrics import recompute_eval_metrics
from chatbot import followup_suggestions, get_chat_reply

# -------------------------
# Paths
# -------------------------
MODEL_PATH = os.path.join(ROOT_DIR, "models")

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "frontend", "template"),
    static_folder=os.path.join(ROOT_DIR, "frontend", "static"),
)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey-change-in-production")

UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

db.init_db()

# -------------------------
# Models (optional if files missing)
# -------------------------
cnn = full_cnn = svm = None
MODEL_ERR = None
try:
    cnn = tf.keras.models.load_model(
        os.path.join(MODEL_PATH, "cnn_feature_extractor.h5"), compile=False
    )
    full_cnn = tf.keras.models.load_model(
        os.path.join(MODEL_PATH, "cnn_best_model.h5"), compile=False
    )
    svm = joblib.load(os.path.join(MODEL_PATH, "svm_model.pkl"))
except Exception as e:
    MODEL_ERR = str(e)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)

    return decorated


def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


LAST_CONV_LAYER = "conv5_block16_2_conv"


def get_gradcam(model, img_array):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(LAST_CONV_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = tf.reduce_mean(predictions)
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def save_gradcam(img_path, heatmap, output_path, alpha=0.40):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(output_path, superimposed_img)
    return output_path


def unique_upload_name(original_filename):
    ext = os.path.splitext(secure_filename(original_filename or ""))[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        ext = ".png"
    return f"{uuid.uuid4().hex}{ext}"


def static_path_from_fs(abs_path):
    rel = os.path.relpath(abs_path, app.static_folder)
    return rel.replace(os.sep, "/")


def abs_from_static_rel(rel):
    if not rel:
        return ""
    return os.path.normpath(os.path.join(app.static_folder, rel.replace("/", os.sep)))


def safe_report_filename(name):
    s = re.sub(r"[^\w\s\-]", "", name or "Patient", flags=re.UNICODE).strip() or "Patient"
    s = re.sub(r"[\s\-]+", "_", s)[:48]
    return f"{s}_Report.pdf"


def weekly_chart_data(for_user: str | None = None):
    raw = db.weekly_detection_counts(for_user)
    by_date = {r["date"]: r["count"] for r in raw}
    end = date.today()
    labels, values = [], []
    for i in range(6, -1, -1):
        d = (end - timedelta(days=i)).isoformat()
        labels.append(d[5:])
        values.append(by_date.get(d, 0))
    return labels, values


# -------------------------
# Auth
# -------------------------
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    err = None
    if request.method == "POST":
        username = request.form.get("username", "")
        email = request.form.get("email", "")
        password = request.form.get("password", "")
        if not username or not email or not password:
            err = "All fields are required."
        else:
            ok, msg = db.create_user(username, email, password)
            if ok:
                session["user"] = username.strip()
                return redirect(url_for("dashboard"))
            err = msg
    return render_template("signup.html", error=err)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if db.verify_login(username, password):
            session["user"] = username
            return redirect(url_for("dashboard"))
        return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# -------------------------
# App pages
# -------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    user = session.get("user") or ""
    stats = db.dashboard_stats(created_by=user)
    ml = load_training_analytics(MODEL_PATH)
    stats["ai_accuracy_pct"] = ml.get("accuracy_pct")
    line_labels, line_values = weekly_chart_data(session.get("user"))
    return render_template(
        "dashboard.html",
        nav="dashboard",
        stats=stats,
        ml_online=_llm_configured(),
        line_labels=line_labels,
        line_values=line_values,
    )


@app.route("/detection", methods=["GET", "POST"])
@login_required
def detection():
    ctx = {
        "nav": "detection",
        "result": None,
        "confidence": None,
        "description": None,
        "suggestions": [],
        "img_name": None,
        "gradcam_path": None,
        "original_image": None,
        "form_vals": {},
        "saved_patient_id": None,
        "report_generated_at": None,
        "model_error": MODEL_ERR if not cnn or not svm else None,
    }

    if request.method == "POST":
        ctx["form_vals"] = {
            "patient_name": request.form.get("patient_name", ""),
            "age": request.form.get("age", ""),
            "gender": request.form.get("gender", ""),
            "phone": request.form.get("phone", ""),
            "address": request.form.get("address", ""),
        }
        if ctx["model_error"]:
            return render_template("detection.html", **ctx)

        img_file = request.files.get("image")
        if not img_file or not img_file.filename:
            ctx["model_error"] = "Please upload an X-ray image."
            return render_template("detection.html", **ctx)

        fname = unique_upload_name(img_file.filename)
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        img_file.save(img_path)

        img = Image.open(img_path).convert("RGB")
        img_array = preprocess(img)

        features = cnn.predict(img_array, verbose=0)
        prediction = svm.predict(features)[0]
        prob = svm.predict_proba(features)[0]
        confidence = round(float(max(prob)) * 100, 2)

        if prediction == 1:
            result = "PNEUMONIA"
            description = (
                "Pneumonia detected. This condition may affect the lungs and breathing system. "
                "Please consult a healthcare professional for proper medical evaluation."
            )
            suggestions = [
                "Consult a chest specialist immediately.",
                "Take proper rest and stay hydrated.",
                "Avoid smoking and cold environments.",
                "Monitor fever and breathing difficulty.",
                "Use prescribed medicines only.",
                "Seek emergency care if symptoms worsen.",
            ]
        else:
            result = "NORMAL"
            description = (
                "No pneumonia detected. The chest X-ray appears normal. "
                "Maintain a healthy lifestyle to keep lungs healthy."
            )
            suggestions = [
                "Maintain a healthy diet.",
                "Exercise regularly.",
                "Avoid smoking and pollution.",
                "Drink enough water.",
                "Get regular medical checkups.",
                "Follow proper hygiene practices.",
            ]

        gradcam_filename = f"gradcam_{fname}"
        gradcam_fs = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)
        heatmap = get_gradcam(full_cnn, img_array)
        save_gradcam(img_path, heatmap, gradcam_fs)

        rel_original = static_path_from_fs(img_path)
        rel_grad = static_path_from_fs(gradcam_fs)

        age_int = None
        try:
            age_int = int(ctx["form_vals"]["age"]) if ctx["form_vals"]["age"] else None
        except ValueError:
            age_int = None

        pid = db.insert_patient(
            patient_name=ctx["form_vals"]["patient_name"].strip(),
            age=age_int,
            gender=ctx["form_vals"]["gender"],
            phone=ctx["form_vals"]["phone"],
            address=ctx["form_vals"]["address"],
            image_path=rel_original,
            gradcam_path=rel_grad,
            result=result,
            confidence=confidence,
            description=description.strip(),
            suggestions=suggestions,
            created_by_user=session["user"],
        )

        ctx.update(
            {
                "result": result,
                "confidence": confidence,
                "description": description,
                "suggestions": suggestions,
                "img_name": fname,
                "gradcam_path": rel_grad.replace("\\", "/"),
                "original_image": rel_original.replace("\\", "/"),
                "saved_patient_id": pid,
                "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        )

    return render_template("detection.html", **ctx)


def _enrich_report_rows(rows: list) -> list:
    """Parse suggestions_json for reports UI."""
    out = []
    for row in rows:
        r = dict(row)
        try:
            r["suggestions_list"] = json.loads(r.get("suggestions_json") or "[]")
        except json.JSONDecodeError:
            r["suggestions_list"] = []
        out.append(r)
    return out


@app.route("/patients")
@login_required
def patients_list():
    q = request.args.get("q", "").strip()
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    per_page = 12
    rows, total = db.list_patients(
        q=q,
        result_filter="",
        page=page,
        per_page=per_page,
        created_by=session.get("user"),
        sort_order="desc",
    )
    total_pages = max(1, (total + per_page - 1) // per_page)
    return render_template(
        "patient_records.html",
        nav="patients",
        patients=rows,
        q=q,
        page=page,
        total_pages=total_pages,
    )


@app.route("/reports")
@login_required
def reports_page():
    q = request.args.get("q", "").strip()
    result_filter = request.args.get("result", "").strip()
    sort_order = request.args.get("sort", "desc").strip().lower()
    if sort_order not in ("asc", "desc"):
        sort_order = "desc"
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    per_page = 6
    rows, total = db.list_patients(
        q=q,
        result_filter=result_filter,
        page=page,
        per_page=per_page,
        created_by=session.get("user"),
        sort_order=sort_order,
    )
    rows = _enrich_report_rows(rows)
    total_pages = max(1, (total + per_page - 1) // per_page)
    return render_template(
        "reports.html",
        nav="reports",
        reports=rows,
        q=q,
        result_filter=result_filter,
        sort_order=sort_order,
        page=page,
        total_pages=total_pages,
    )


@app.route("/analytics")
@login_required
def analytics_page():
    data = load_training_analytics(MODEL_PATH)
    return render_template(
        "analytics.html",
        nav="analytics",
        analytics=data,
        model_error=MODEL_ERR,
    )


@app.route("/api/analytics/recompute-from-models", methods=["POST"])
@login_required
def api_recompute_eval_metrics():
    """Load saved CNN+SVM, evaluate on dataset/test, refresh confusion matrix files (no training)."""
    if MODEL_ERR:
        return jsonify({"ok": False, "error": f"Models unavailable: {MODEL_ERR}"}), 400
    test_path = os.path.join(ROOT_DIR, "dataset", "test")
    if not os.path.isdir(test_path):
        return jsonify(
            {
                "ok": False,
                "error": "dataset/test not found. Add a test split (Normal/Pneumonia folders).",
            }
        ), 400
    try:
        result = recompute_eval_metrics(MODEL_PATH, test_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/export/patients.csv")
@login_required
def export_patients_csv():
    import csv
    from io import StringIO

    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "Patient ID",
            "Name",
            "Age",
            "Gender",
            "Phone",
            "Result",
            "Confidence_pct",
            "Prediction_date",
            "Created_by",
        ]
    )
    user = session.get("user") or ""
    for r in db.list_patients_for_export(created_by=user):
        w.writerow(
            [
                r.get("id"),
                r.get("patient_name"),
                r.get("age"),
                r.get("gender"),
                r.get("phone"),
                r.get("result"),
                r.get("confidence"),
                r.get("prediction_date"),
                r.get("created_by_user"),
            ]
        )
    out = buf.getvalue()
    return Response(
        out,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": "attachment; filename=pneumonia_patient_export.csv"
        },
    )


@app.route("/patients/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    row = db.get_patient(patient_id)
    if not row:
        abort(404)
    p = dict(row)
    try:
        sug = json.loads(p.get("suggestions_json") or "[]")
    except json.JSONDecodeError:
        sug = []
    src = (request.args.get("src") or "").strip()
    nav_key = "reports" if src == "reports" else "patients"
    delete_dest = "reports" if src == "reports" else "patients"
    return render_template(
        "patient_detail.html",
        nav=nav_key,
        patient=p,
        suggestions=sug,
        delete_dest=delete_dest,
    )


@app.route("/patients/<int:patient_id>/delete", methods=["POST"])
@login_required
def delete_patient(patient_id):
    row = db.get_patient(patient_id)
    if row:
        r = dict(row)
        for key in ("image_path", "gradcam_path"):
            rel = r.get(key)
            if rel:
                ap = abs_from_static_rel(rel)
                if ap and os.path.isfile(ap):
                    try:
                        os.remove(ap)
                    except OSError:
                        pass
        db.delete_patient(patient_id)
    dest = (request.form.get("dest") or "patients").strip()
    if dest == "reports":
        return redirect(url_for("reports_page"))
    return redirect(url_for("patients_list"))


@app.route("/patients/<int:patient_id>/pdf")
@login_required
def download_patient_pdf(patient_id):
    try:
        from pdf_generator import build_patient_report_pdf
    except ImportError:
        return Response(
            "PDF requires ReportLab. Install with: pip install reportlab\n"
            "(Use the same Python/venv you use to run app.py.)",
            status=503,
            mimetype="text/plain",
        )
    row = db.get_patient(patient_id)
    if not row:
        abort(404)
    p = dict(row)
    try:
        sug = json.loads(p.get("suggestions_json") or "[]")
    except json.JSONDecodeError:
        sug = []
    rid = p.get("id")
    pdf_buf = build_patient_report_pdf(
        patient_name=p.get("patient_name") or "",
        age=p.get("age"),
        gender=p.get("gender") or "",
        phone=p.get("phone") or "",
        address=p.get("address") or "",
        result=p.get("result") or "",
        confidence=float(p.get("confidence") or 0),
        description=(p.get("description") or "").strip(),
        suggestions=sug,
        original_image_abs_path=abs_from_static_rel(p.get("image_path") or ""),
        gradcam_image_abs_path=abs_from_static_rel(p.get("gradcam_path") or ""),
        report_code=f"PR-{rid}" if rid is not None else None,
    )
    fname = safe_report_filename(p.get("patient_name"))
    return send_file(
        pdf_buf,
        as_attachment=True,
        download_name=fname,
        mimetype="application/pdf",
    )


def _patient_context_block(user: str, patient_id: int | None) -> str:
    if not patient_id:
        return ""
    row = db.get_patient(patient_id)
    if not row:
        return ""
    r = dict(row)
    if (r.get("created_by_user") or "") != user:
        return ""
    name = r.get("patient_name") or "—"
    age = r.get("age")
    gender = r.get("gender") or "—"
    result = r.get("result") or "—"
    conf = r.get("confidence")
    conf_s = f"{float(conf):.1f}%" if conf is not None else "—"
    date_s = r.get("prediction_date") or "—"
    desc = (r.get("description") or "").strip()
    if len(desc) > 800:
        desc = desc[:800] + "…"
    lines = [
        f"Patient label: {name}",
        f"Age / gender: {age} / {gender}",
        f"Demo classifier output: {result} (confidence {conf_s})",
        f"Prediction timestamp: {date_s}",
    ]
    if desc:
        lines.append(f"App description / notes: {desc}")
    return "\n".join(lines)


def _llm_configured() -> bool:
    if (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip():
        return True
    if (os.environ.get("GROQ_API_KEY") or "").strip():
        return True
    if (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        return True
    return False


ASSISTANT_ONLINE_ONLY_MSG = (
    "The AI assistant is online-only. Add GOOGLE_API_KEY (Gemini, recommended), GROQ_API_KEY, or "
    "OPENROUTER_API_KEY to your .env file, keep this computer connected to the internet, install "
    "python-dotenv if needed, then restart Flask. Open Settings in the app for details."
)

ASSISTANT_AI_UNREACHABLE_MSG = (
    "Could not get a reply from the cloud AI. Check that this PC has internet, your API key is valid, "
    "and no firewall or VPN is blocking the request. Then try again."
)


@app.route("/assistant")
@login_required
def assistant_page():
    return render_template(
        "assistant.html", nav="chat", llm_online=_llm_configured()
    )


@app.route("/api/assistant/patients")
@login_required
def api_assistant_patients():
    if not _llm_configured():
        return jsonify({"patients": [], "error": "online_assistant_disabled"}), 403
    user = session.get("user") or ""
    rows = db.list_patients_for_user(user, limit=80)
    out = [
        {
            "id": r["id"],
            "patient_name": r.get("patient_name"),
            "result": r.get("result"),
            "confidence": r.get("confidence"),
            "prediction_date": r.get("prediction_date"),
        }
        for r in rows
    ]
    return jsonify({"patients": out})


def _ai_key_mask() -> str:
    k = (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    ).strip()
    if len(k) <= 10:
        return ""
    return k[:7] + "…" + k[-4:]


@app.route("/settings")
@login_required
def settings_page():
    user = session.get("user") or ""
    row = db.find_user_by_username(user)
    email = (row["email"] if row else "") or "—"
    ai_status = {
        "gemini": bool(
            (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
        ),
        "groq": bool((os.environ.get("GROQ_API_KEY") or "").strip()),
        "openrouter": bool((os.environ.get("OPENROUTER_API_KEY") or "").strip()),
        "any": _llm_configured(),
        "gemini_mask": _ai_key_mask(),
    }
    return render_template(
        "settings.html",
        nav="settings",
        account_username=user,
        account_email=email,
        ai_status=ai_status,
        env_file_path=os.path.join(ROOT_DIR, ".env"),
        db_file_path=os.path.join(ROOT_DIR, "database", "app.db"),
    )


@app.route("/api/settings/test-ai", methods=["POST"])
@login_required
def api_settings_test_ai():
    if not _llm_configured():
        return jsonify(
            {"ok": False, "message": "No cloud API key loaded. Check .env and restart Flask."}
        ), 400
    reply = get_chat_reply("Reply with exactly: OK", [])
    if reply and reply.strip():
        return jsonify(
            {
                "ok": True,
                "message": "Cloud AI is reachable.",
                "preview": reply.strip()[:200],
            }
        )
    return jsonify(
        {"ok": False, "message": "No response from Gemini / Groq / OpenRouter. Check internet and key."}
    ), 502


@app.route("/api/chat/history", methods=["GET"])
@login_required
def api_chat_history():
    user = session.get("user") or ""
    messages = db.assistant_chat_list(user)
    return jsonify({"messages": messages})


@app.route("/api/chat/history", methods=["DELETE"])
@login_required
def api_chat_history_clear():
    user = session.get("user") or ""
    db.assistant_chat_clear(user)
    return jsonify({"ok": True})


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    if not _llm_configured():
        return jsonify(
            {
                "reply": ASSISTANT_ONLINE_ONLY_MSG,
                "suggestions": [],
                "code": "no_api_key",
            }
        ), 403

    user = session.get("user") or ""
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify(
            {
                "reply": "Type a question to get started.",
                "suggestions": followup_suggestions(False),
            }
        ), 400

    pid = data.get("patient_id")
    patient_id: int | None = None
    if pid is not None and str(pid).strip() != "":
        try:
            patient_id = int(pid)
        except (TypeError, ValueError):
            patient_id = None
    extra = _patient_context_block(user, patient_id)
    history = db.assistant_chat_list(user)
    reply = get_chat_reply(msg, history, extra_context=extra or None)
    if not reply:
        return jsonify(
            {
                "reply": ASSISTANT_AI_UNREACHABLE_MSG,
                "suggestions": [],
                "code": "ai_unreachable",
            }
        ), 502

    db.assistant_chat_append(user, "user", msg)
    db.assistant_chat_append(user, "assistant", reply)
    return jsonify(
        {
            "reply": reply,
            "suggestions": followup_suggestions(bool(extra)),
        }
    )


# Legacy route
@app.route("/project")
@login_required
def project_legacy():
    return redirect(url_for("detection"))


def _print_llm_startup_status():
    """Helps debug why GOOGLE_API_KEY is not picked up."""
    root_p = os.path.join(ROOT_DIR, ".env")
    back_p = os.path.join(BASE_DIR, ".env")
    print("\n======== Pneumonia AI Lab — AI assistant ========")
    print(f"  [.] .env at project root: {'YES ' if os.path.isfile(root_p) else 'NO  '} {root_p}")
    print(f"  [.] .env in backend folder: {'YES ' if os.path.isfile(back_p) else 'NO  '} {back_p}")
    if _llm_configured():
        k = (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or "").strip()
        if k:
            print(f"  [OK] Gemini-style key is set (length {len(k)} chars). Assistant works when internet is on.")
        else:
            print("  [OK] GROQ or OpenRouter key is set.")
    else:
        print("  [!!] No cloud key in environment. Edit .env and set GOOGLE_API_KEY=your_key")
        print("       from Google AI Studio, SAVE, then start Flask again. Internet = required on this PC.")
    print("==================================================\n")


def _open_local_browser(port: int, timeout_sec: float = 120.0) -> None:
    """Open default browser once 127.0.0.1:port accepts connections (works after slow TF import)."""
    url = f"http://127.0.0.1:{port}/"

    def _try_open() -> None:
        if not webbrowser.open(url):
            if sys.platform == "win32":
                import subprocess

                subprocess.run(
                    ["cmd", "/c", "start", "", url],
                    shell=False,
                    check=False,
                )

    def task() -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            print(f"[browser] Timed out waiting for port {port}; open manually: {url}")
            return
        try:
            _try_open()
            print(f"[browser] Opened: {url}")
        except Exception as ex:
            print(f"[browser] Could not open browser ({ex}). Open manually: {url}")

    threading.Thread(target=task, daemon=True).start()


if __name__ == "__main__":
    _print_llm_startup_status()
    port = int(os.environ.get("PORT", "5000"))
    use_reloader = os.environ.get("USE_RELOADER", "1").lower() not in ("0", "false", "no")
    no_browser = os.environ.get("NO_BROWSER", "").lower() in ("1", "true", "yes")
    if not no_browser:
        # Reloader: only the child process runs the server (avoid duplicate tabs).
        if not use_reloader or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            _open_local_browser(port)
    app.run(debug=True, host="0.0.0.0", port=port, use_reloader=use_reloader)
