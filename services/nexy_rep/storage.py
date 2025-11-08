# storage.py
import sqlite3
from datetime import datetime
from typing import Optional

def init_db(db_path: str) -> None:
    """
    Initialize the SQLite database if not exists.
    
    Args:
        db_path (str): Path to the database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS captured_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            image_path TEXT NOT NULL,
            extracted_text TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def ensure_conversation_table(db_path: str) -> None:
    """
    Ensure the conversations table exists.

    Schema:
      - id: PK
      - session_id: text
      - timestamp: datetime
      - role: text ('user'|'assistant')
      - message: text
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)")
    conn.commit()
    conn.close()


def store_chat_message(db_path: str, session_id: str, role: str, message: str, timestamp: Optional[datetime] = None) -> None:
    """
    Store a chat message (user or assistant) for a session.
    """
    if timestamp is None:
        timestamp = datetime.now()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO conversations (session_id, timestamp, role, message)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, timestamp, role, message)
    )
    conn.commit()
    conn.close()


def get_chat_history(db_path: str, session_id: str, limit: int | None = None):
    """
    Retrieve chat history for a session ordered by timestamp ascending.

    Returns list of tuples: (timestamp, role, message)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    q = "SELECT timestamp, role, message FROM conversations WHERE session_id = ? ORDER BY timestamp ASC"
    if limit:
        q = q + " LIMIT %d" % int(limit)
    cursor.execute(q, (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def store_data(
    db_path: str,
    timestamp: datetime,
    image_path: str,
    extracted_text: str
) -> None:
    """
    Store the data in the SQLite database.
    
    Args:
        db_path (str): Path to the database.
        timestamp (datetime): Timestamp of capture.
        image_path (str): Path to stored image.
        extracted_text (str): Extracted text.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO captured_data (timestamp, image_path, extracted_text)
        VALUES (?, ?, ?)
    """, (timestamp, image_path, extracted_text))
    conn.commit()
    conn.close()