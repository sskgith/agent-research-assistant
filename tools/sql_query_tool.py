"""
SQL query tool, backed by a synthetic 'papers' dataset loaded from CSV.

Data is entirely synthetic (fabricated titles/authors/citation counts)
for portfolio-safety reasons — no real paper metadata or scraped data
is used. Loaded fresh into an in-memory SQLite database on each call,
same pattern as before, just sourced from a real file instead of
hardcoded Python tuples, so data and code are properly separated.
"""

import csv
import sqlite3
from pathlib import Path

_CSV_PATH = Path(__file__).parent.parent / "data" / "papers.csv"


def _get_seeded_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY,
            title TEXT,
            authors TEXT,
            year INTEGER,
            topic TEXT,
            citation_count INTEGER
        )
        """
    )

    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            (row["title"], row["authors"], int(row["year"]), row["topic"], int(row["citation_count"]))
            for row in reader
        ]

    conn.executemany(
        "INSERT INTO papers (title, authors, year, topic, citation_count) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def sql_query(query: str) -> str:
    """
    Run a read-only SQL query against the synthetic papers database.

    Args:
        query: A SQL SELECT statement.

    Returns:
        The query results as a plain-text string, or an error message.
    """
    if not query.strip().lower().startswith("select"):
        return "[ERROR] Only SELECT statements are permitted."
    try:
        conn = _get_seeded_connection()
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return str(rows)
    except Exception as exc:
        return f"[ERROR] Query failed: {exc}"