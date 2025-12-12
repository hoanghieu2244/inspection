import sqlite3
from typing import List, Tuple, Optional

def init_db(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def upsert_document(db_path: str, doc_id: str, text: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("INSERT INTO documents(id, text) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET text=excluded.text", (doc_id, text))
    conn.commit()
    conn.close()

def get_all_documents(db_path: str) -> List[Tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM documents ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_document(db_path: str, doc_id: str) -> Optional[Tuple[str, str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM documents WHERE id = ?", (doc_id,))
    row = cur.fetchone()
    conn.close()
    return row
