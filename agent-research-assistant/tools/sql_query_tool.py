"""
SQL query tool, backed by a small local SQLite database.

The schema/seed data here is a placeholder — swap in real sample data once
the graph loop is confirmed working. Kept as a real (not hardcoded-string)
SQLite call so the tool-selection tests exercise actual query execution.
"""

import sqlite3


def _get_seeded_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"
    )
    conn.executemany(
        "INSERT INTO products (name, price) VALUES (?, ?)",
        [
            ("Widget A", 9.99),
            ("Widget B", 14.50),
            ("Widget C", 22.00),
        ],
    )
    conn.commit()
    return conn


def sql_query(query: str) -> str:
    """
    Run a read-only SQL query against the seeded sample database.

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
