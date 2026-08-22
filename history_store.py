"""Local SQLite storage for per-user analysis history.

Each analysis report is saved against the phone number the user entered on the
( currently local/demo ) login screen. Data never leaves the user's own machine:
the database and chart images live under the per-user application data directory.

Storage layout:
    <data_dir>/quantagent_history.db
    <data_dir>/charts/<record_id>_pattern.png
    <data_dir>/charts/<record_id>_trend.png

Charts are decoded from the base64 payloads returned by the analysis and stored
as PNG files; the database only keeps their paths, so it stays small and fast.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_FILENAME = "quantagent_history.db"
CHARTS_SUBDIR = "charts"

# A single process-wide lock guards SQLite writes. Flask/waitress serves requests
# on threads, so we serialise writes and open short-lived connections per call.
_lock = threading.Lock()


def get_data_dir() -> Path:
    """Return a writable per-user data directory, creating it if needed.

    The desktop (Tauri) wrapper passes its per-app data directory via the
    ``QUANTAGENT_DATA_DIR`` environment variable; honour that when present so
    history lives inside the app's own data folder. Otherwise fall back to the
    platform-standard per-user location.
    """
    override = os.environ.get("QUANTAGENT_DATA_DIR")
    if override:
        candidate = Path(override)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            pass  # fall through to platform defaults

    if sys.platform == "win32":
        root = os.environ.get("APPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Roaming"
        candidate = base / "QuantAgent"
    elif sys.platform == "darwin":
        candidate = Path.home() / "Library" / "Application Support" / "QuantAgent"
    else:
        root = os.environ.get("XDG_DATA_HOME")
        base = Path(root) if root else Path.home() / ".local" / "share"
        candidate = base / "quantagent"

    # Fall back to a local directory if the per-user location is not writable.
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        test_file = candidate / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except OSError:
        candidate = Path("data")
        candidate.mkdir(parents=True, exist_ok=True)

    return candidate


def _db_path() -> Path:
    return get_data_dir() / DB_FILENAME


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the history table and indexes if they do not already exist.

    Called on every connection so that, regardless of which data directory the
    DB resolves to (the per-user APPDATA dir or the local `data/` fallback),
    the schema is always present. ``CREATE TABLE IF NOT EXISTS`` is cheap on an
    existing database.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            asset TEXT,
            asset_name TEXT,
            timeframe TEXT,
            start_date TEXT,
            end_date TEXT,
            data_source TEXT,
            decision_direction TEXT,
            entry_price TEXT,
            stop_loss TEXT,
            take_profit TEXT,
            report_json TEXT NOT NULL,
            pattern_chart_path TEXT,
            trend_chart_path TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_history_phone_time "
        "ON analysis_history(phone, created_at DESC)"
    )
    conn.commit()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    # Ensure schema on every connection (handles the local data/ fallback, too).
    _ensure_schema(conn)
    return conn


def init_db() -> None:
    """Create the history table and indexes if they do not already exist."""
    with _lock, _connect() as conn:
        _ensure_schema(conn)


def _charts_dir() -> Path:
    path = get_data_dir() / CHARTS_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _decode_and_save_chart(record_id: int, kind: str, b64_data: str) -> Optional[str]:
    """Decode a base64 PNG and write it to disk. Return the stored path or None."""
    if not b64_data:
        return None
    try:
        # Some payloads may include a "data:image/png;base64," prefix.
        if "," in b64_data and b64_data.lstrip().lower().startswith("data:"):
            b64_data = b64_data.split(",", 1)[1]
        png_bytes = base64.b64decode(b64_data)
        file_path = _charts_dir() / f"{record_id}_{kind}.png"
        file_path.write_bytes(png_bytes)
        return str(file_path)
    except Exception as error:  # noqa: BLE001 - chart storage must never break saving
        print(f"[history] could not save {kind} chart for record {record_id}: {error}")
        return None


def _load_chart_as_b64(path_str: Optional[str]) -> str:
    if not path_str:
        return ""
    try:
        data = Path(path_str).read_bytes()
        return base64.b64encode(data).decode("ascii")
    except OSError:
        return ""


def save_record(
    phone: str,
    payload: Dict[str, Any],
    start_date: str = "",
    end_date: str = "",
    data_source: str = "",
) -> Optional[int]:
    """Persist one successful analysis for the given phone.

    `payload` is the full response dict returned by the analysis pipeline.
    Returns the new record id, or None if the save failed.
    """
    if not phone:
        return None

    full = dict(payload or {})
    pattern_b64 = full.pop("pattern_chart", "") or ""
    trend_b64 = full.pop("trend_chart", "") or ""

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _lock, _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analysis_history (
                phone, asset, asset_name, timeframe, start_date, end_date,
                data_source, decision_direction, entry_price, stop_loss,
                take_profit, report_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                phone,
                full.get("asset", full.get("asset_name", "")),
                full.get("asset_name", ""),
                full.get("timeframe", ""),
                start_date,
                end_date,
                data_source,
                full.get("decision_direction", ""),
                _stringify(full.get("entry_price")),
                _stringify(full.get("stop_loss")),
                _stringify(full.get("take_profit")),
                json.dumps(full, ensure_ascii=False),
                created_at,
            ),
        )
        record_id = cursor.lastrowid

        pattern_path = _decode_and_save_chart(record_id, "pattern", pattern_b64)
        trend_path = _decode_and_save_chart(record_id, "trend", trend_b64)
        if pattern_path or trend_path:
            conn.execute(
                "UPDATE analysis_history SET pattern_chart_path = ?, trend_chart_path = ? WHERE id = ?",
                (pattern_path, trend_path, record_id),
            )
        conn.commit()
        return record_id


def list_records(
    phone: str, asset: Optional[str] = None, limit: int = 200
) -> List[Dict[str, Any]]:
    """Return summary records for a phone, newest first, optionally filtered by asset."""
    if not phone:
        return []

    sql = (
        "SELECT id, asset, asset_name, timeframe, start_date, end_date, "
        "data_source, decision_direction, entry_price, stop_loss, take_profit, created_at "
        "FROM analysis_history WHERE phone = ?"
    )
    params: List[Any] = [phone]
    if asset:
        sql += " AND (asset LIKE ? OR asset_name LIKE ?)"
        like = f"%{asset}%"
        params.extend([like, like])
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


def list_asset_buttons(phone: str) -> List[Dict[str, Any]]:
    """Return distinct assets this phone has analysed, for a filter dropdown."""
    if not phone:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT asset, asset_name, MAX(created_at) AS last_seen, COUNT(*) AS times
            FROM analysis_history
            WHERE phone = ?
            GROUP BY asset
            ORDER BY last_seen DESC
            """,
            (phone,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_record(phone: str, record_id: int) -> Optional[Dict[str, Any]]:
    """Return one full record (reconstructed report + charts), scoped to phone."""
    if not phone:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_history WHERE id = ? AND phone = ?",
            (record_id, phone),
        ).fetchone()

    if row is None:
        return None

    try:
        report = json.loads(row["report_json"])
    except (TypeError, json.JSONDecodeError):
        report = {}

    report["pattern_chart"] = _load_chart_as_b64(row["pattern_chart_path"])
    report["trend_chart"] = _load_chart_as_b64(row["trend_chart_path"])

    return {
        "id": row["id"],
        "phone": row["phone"],
        "asset": row["asset"],
        "asset_name": row["asset_name"],
        "timeframe": row["timeframe"],
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "data_source": row["data_source"],
        "decision_direction": row["decision_direction"],
        "entry_price": row["entry_price"],
        "stop_loss": row["stop_loss"],
        "take_profit": row["take_profit"],
        "created_at": row["created_at"],
        "report": report,
    }


def delete_record(phone: str, record_id: int) -> bool:
    """Delete one record and its chart files. Returns True if a row was removed."""
    if not phone:
        return False
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT pattern_chart_path, trend_chart_path FROM analysis_history WHERE id = ? AND phone = ?",
            (record_id, phone),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "DELETE FROM analysis_history WHERE id = ? AND phone = ?",
            (record_id, phone),
        )
        conn.commit()

    for path_str in (row["pattern_chart_path"], row["trend_chart_path"]):
        if path_str:
            try:
                Path(path_str).unlink(missing_ok=True)
            except OSError:
                pass
    return True


def clear_records(phone: str) -> int:
    """Delete all records for a phone. Returns the number removed."""
    if not phone:
        return 0
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT pattern_chart_path, trend_chart_path FROM analysis_history WHERE phone = ?",
            (phone,),
        ).fetchall()
        removed = conn.execute(
            "DELETE FROM analysis_history WHERE phone = ?", (phone,)
        ).rowcount
        conn.commit()

    for row in rows:
        for path_str in (row["pattern_chart_path"], row["trend_chart_path"]):
            if path_str:
                try:
                    Path(path_str).unlink(missing_ok=True)
                except OSError:
                    pass
    return removed


def export_records(phone: str) -> List[Dict[str, Any]]:
    """Return all records for a phone as plain dicts (without chart binaries),
    suitable for JSON export / backup."""
    if not phone:
        return []
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_history WHERE phone = ? ORDER BY created_at DESC, id DESC",
            (phone,),
        ).fetchall()

    result: List[Dict[str, Any]] = []
    for row in rows:
        try:
            report = json.loads(row["report_json"])
        except (TypeError, json.JSONDecodeError):
            report = {}
        item = dict(row)
        item.pop("report_json", None)
        item["report"] = report
        result.append(item)
    return result


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
