"""SQLite database layer for the Knowledge Assistant."""
import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            content TEXT NOT NULL,
            priority INTEGER NOT NULL,
            hours_spent REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def insert_note(title, category, content, priority, hours_spent):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notes (title, category, content, priority, hours_spent, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (title, category, content, priority, hours_spent, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def insert_many(rows):
    """rows: list of tuples (title, category, content, priority, hours_spent, created_at)"""
    conn = get_connection()
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO notes (title, category, content, priority, hours_spent, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def fetch_all_df():
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM notes ORDER BY id DESC", conn)
    except Exception:
        df = pd.DataFrame(columns=["id", "title", "category", "content", "priority", "hours_spent", "created_at"])
    conn.close()
    return df


def delete_note(note_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    conn.commit()
    conn.close()


def clear_all():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM notes")
    conn.commit()
    conn.close()


def row_count():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM notes")
    n = cur.fetchone()[0]
    conn.close()
    return n
