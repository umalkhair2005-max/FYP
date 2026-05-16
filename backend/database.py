"""
SQLite access: users + patients. Single file under database/app.db
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "database")
DB_PATH = os.path.join(DB_DIR, "app.db")

# Legacy single-file DB in project root (migrate rows once if present)
LEGACY_USERS_DB = os.path.join(ROOT, "users.db")


def _ensure_dir():
    os.makedirs(DB_DIR, exist_ok=True)


@contextmanager
def get_db():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create tables and optionally import users from legacy users.db."""
    _ensure_dir()
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                phone TEXT,
                address TEXT,
                image_path TEXT,
                gradcam_path TEXT,
                result TEXT,
                confidence REAL,
                description TEXT,
                suggestions_json TEXT,
                prediction_date TEXT,
                created_by_user TEXT
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_patients_date ON patients(prediction_date DESC)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_patients_result ON patients(result)"
        )

        c.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_chat (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_assistant_chat_user ON assistant_chat(username, id)"
        )

        # One-time migration from users.db in project root
        if os.path.isfile(LEGACY_USERS_DB):
            cur = c.execute("SELECT COUNT(*) AS n FROM users")
            if cur.fetchone()["n"] == 0:
                try:
                    leg = sqlite3.connect(LEGACY_USERS_DB)
                    leg.row_factory = sqlite3.Row
                    rows = leg.execute("SELECT username, email, password FROM users").fetchall()
                    leg.close()
                    for r in rows:
                        try:
                            c.execute(
                                "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                                (r["username"], r["email"], r["password"]),
                            )
                        except sqlite3.IntegrityError:
                            pass
                except sqlite3.Error:
                    pass


def find_user_by_username(username: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()


def find_user_by_email(email: str) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()


def create_user(username: str, email: str, password: str) -> tuple[bool, Optional[str]]:
    """Returns (ok, error_message)."""
    u = username.strip()
    e = email.strip().lower()
    if find_user_by_username(u):
        return False, "Username already registered."
    if find_user_by_email(e):
        return False, "Email already registered."
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?,?,?)",
                (u, e, password),
            )
            return True, None
        except sqlite3.IntegrityError:
            return False, "Could not complete registration."


def verify_login(username: str, password: str) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        return row is not None


def insert_patient(
    patient_name: str,
    age: Optional[int],
    gender: str,
    phone: str,
    address: str,
    image_path: str,
    gradcam_path: str,
    result: str,
    confidence: float,
    description: str,
    suggestions: list,
    created_by_user: str,
) -> int:
    suggestions_json = json.dumps(suggestions)
    prediction_date = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO patients (
                patient_name, age, gender, phone, address,
                image_path, gradcam_path, result, confidence,
                description, suggestions_json, prediction_date, created_by_user
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                patient_name,
                age,
                gender,
                phone,
                address,
                image_path,
                gradcam_path,
                result,
                confidence,
                description,
                suggestions_json,
                prediction_date,
                created_by_user,
            ),
        )
        return int(cur.lastrowid)


def get_patient(pid: int) -> Optional[sqlite3.Row]:
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM patients WHERE id = ?", (pid,)
        ).fetchone()


def list_patients_for_user(username: str, limit: int = 100) -> list[dict]:
    """Recent records created by this user (for assistant context picker)."""
    limit = max(1, min(limit, 200))
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, patient_name, age, gender, result, confidence,
                   prediction_date, description, created_by_user
            FROM patients
            WHERE created_by_user = ?
            ORDER BY prediction_date DESC
            LIMIT ?
            """,
            (username.strip(), limit),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_patient(pid: int) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM patients WHERE id = ?", (pid,))
        return cur.rowcount > 0


def list_patients(
    q: str = "",
    result_filter: str = "",
    page: int = 1,
    per_page: int = 10,
    created_by: Optional[str] = None,
    sort_order: str = "desc",
) -> tuple[list, int]:
    """Returns (rows as dicts, total_count). sort_order: 'desc' (latest first) or 'asc'."""
    page = max(1, page)
    offset = (page - 1) * per_page
    where = []
    params: list[Any] = []
    if q:
        where.append("patient_name LIKE ?")
        params.append(f"%{q}%")
    if result_filter in ("NORMAL", "PNEUMONIA"):
        where.append("result = ?")
        params.append(result_filter)
    if created_by is not None and str(created_by).strip():
        where.append("created_by_user = ?")
        params.append(created_by.strip())
    wh = (" WHERE " + " AND ".join(where)) if where else ""
    order_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

    with get_db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM patients{wh}", params
        ).fetchone()["c"]
        rows = conn.execute(
            f"""
            SELECT * FROM patients
            {wh}
            ORDER BY prediction_date {order_dir}
            LIMIT ? OFFSET ?
            """,
            [*params, per_page, offset],
        ).fetchall()
    return [dict(r) for r in rows], total


def dashboard_stats(created_by: Optional[str] = None) -> dict:
    """Aggregate counts; if created_by set, scope to that user's records."""
    today_iso = date.today().isoformat()
    with get_db() as conn:
        if created_by is not None and str(created_by).strip():
            u = created_by.strip()
            base = "SELECT COUNT(*) AS c FROM patients WHERE created_by_user = ?"
            total_patients = conn.execute(base, (u,)).fetchone()["c"]
            pneumonia = conn.execute(
                base + " AND result = 'PNEUMONIA'", (u,)
            ).fetchone()["c"]
            normal = conn.execute(base + " AND result = 'NORMAL'", (u,)).fetchone()[
                "c"
            ]
            scans_today = conn.execute(
                base + " AND substr(prediction_date, 1, 10) = ?",
                (u, today_iso),
            ).fetchone()["c"]
        else:
            total_patients = conn.execute(
                "SELECT COUNT(*) AS c FROM patients"
            ).fetchone()["c"]
            pneumonia = conn.execute(
                "SELECT COUNT(*) AS c FROM patients WHERE result = 'PNEUMONIA'"
            ).fetchone()["c"]
            normal = conn.execute(
                "SELECT COUNT(*) AS c FROM patients WHERE result = 'NORMAL'"
            ).fetchone()["c"]
            scans_today = conn.execute(
                """
                SELECT COUNT(*) AS c FROM patients
                WHERE substr(prediction_date, 1, 10) = ?
                """,
                (today_iso,),
            ).fetchone()["c"]
    return {
        "total_patients": total_patients,
        "pneumonia_cases": pneumonia,
        "normal_cases": normal,
        "total_reports": total_patients,
        "scans_today": scans_today,
        "total_scans": total_patients,
    }


def list_patients_for_export(created_by: Optional[str] = None) -> list[dict]:
    """Records for CSV export; scope to user when created_by is set."""
    with get_db() as conn:
        if created_by is not None and str(created_by).strip():
            rows = conn.execute(
                """
                SELECT id, patient_name, age, gender, phone, address, result, confidence,
                       prediction_date, created_by_user
                FROM patients
                WHERE created_by_user = ?
                ORDER BY prediction_date DESC
                """,
                (created_by.strip(),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, patient_name, age, gender, phone, address, result, confidence,
                       prediction_date, created_by_user
                FROM patients
                ORDER BY prediction_date DESC
                """
            ).fetchall()
    return [dict(r) for r in rows]


def assistant_chat_list(username: str, limit: int = 100) -> list[dict]:
    """Chronological messages for the assistant UI (role: user | assistant)."""
    limit = max(1, min(limit, 200))
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM assistant_chat
            WHERE username = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (username.strip(), limit),
        ).fetchall()
    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def assistant_chat_append(username: str, role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        return
    text = (content or "").strip()
    if not text:
        return
    text = text[:12000]
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO assistant_chat (username, role, content, created_at)
            VALUES (?,?,?,?)
            """,
            (username.strip(), role, text, ts),
        )


def assistant_chat_clear(username: str) -> None:
    with get_db() as conn:
        conn.execute("DELETE FROM assistant_chat WHERE username = ?", (username.strip(),))


def weekly_detection_counts(username: Optional[str] = None) -> list:
    """Counts per calendar day (YYYY-MM-DD prefix of ISO timestamp)."""
    with get_db() as conn:
        if username is not None and str(username).strip():
            rows = conn.execute(
                """
                SELECT substr(prediction_date, 1, 10) AS d, COUNT(*) AS c
                FROM patients
                WHERE created_by_user = ?
                GROUP BY substr(prediction_date, 1, 10)
                ORDER BY d DESC
                LIMIT 14
                """,
                (username.strip(),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT substr(prediction_date, 1, 10) AS d, COUNT(*) AS c
                FROM patients
                GROUP BY substr(prediction_date, 1, 10)
                ORDER BY d DESC
                LIMIT 14
                """
            ).fetchall()
    return [{"date": r["d"], "count": r["c"]} for r in rows]
