# eval_storage.py
import sqlite3
import json
import time

DB_PATH = "./eval_history.db"

def init_eval_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            params TEXT,
            avg_relevancy REAL,
            avg_faithfulness REAL,
            avg_completeness REAL,
            avg_total REAL,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_eval_run(params: dict, summary: dict, details: list) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """INSERT INTO eval_runs
           (params, avg_relevancy, avg_faithfulness, avg_completeness, avg_total, details)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            json.dumps(params, ensure_ascii=False),
            summary["avg_relevancy"],
            summary["avg_faithfulness"],
            summary["avg_completeness"],
            summary["avg_total"],
            json.dumps(details, ensure_ascii=False),
        )
    )
    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return run_id

def list_eval_runs() -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT id, created_at, params, avg_relevancy, avg_faithfulness,
                  avg_completeness, avg_total
           FROM eval_runs ORDER BY created_at DESC"""
    ).fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "created_at": r[1],
            "params": json.loads(r[2]),
            "avg_relevancy": r[3],
            "avg_faithfulness": r[4],
            "avg_completeness": r[5],
            "avg_total": r[6],
        }
        for r in rows
    ]

def get_eval_run(run_id: int) -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "params": json.loads(row[2]),
        "avg_relevancy": row[3],
        "avg_faithfulness": row[4],
        "avg_completeness": row[5],
        "avg_total": row[6],
        "details": json.loads(row[7]),
    }

init_eval_db()