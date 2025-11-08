# storage.py
import sqlite3
from datetime import datetime

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