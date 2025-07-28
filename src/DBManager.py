import os
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

class DBManager:
    """Handle SQLite database introspection and query execution."""

    def __init__(self, db_path: str):
        """Open a connection to the SQLite database."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def extract_column_table_pairs(self) -> Dict[str, List[str]]:
        """Return a mapping of table names to their column names."""
        result: Dict[str, List[str]] = {}
        for table in self._get_tables():
            columns = self._get_columns(table)
            result[table] = [col["name"] for col in columns]
        return result

    def extract_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """Return metadata (columns and foreign keys) for a table."""
        return {
            "columns": self._get_columns(table_name),
            "foreign_keys": self._get_foreign_keys(table_name),
        }

    def _get_tables(self) -> List[str]:
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        return [row[0] for row in self.cursor.fetchall()]

    def _get_columns(self, table: str) -> List[Dict[str, Any]]:
        self.cursor.execute(f"PRAGMA table_info({table});")
        return [
            {
                "name": col[1],
                "type": col[2],
                "not_null": bool(col[3]),
                "default_value": col[4],
                "primary_key": bool(col[5]),
            }
            for col in self.cursor.fetchall()
        ]

    def _get_foreign_keys(self, table: str) -> List[Dict[str, str]]:
        self.cursor.execute(f"PRAGMA foreign_key_list({table});")
        return [
            {"from_column": fk[3], "to_table": fk[2], "to_column": fk[4]}
            for fk in self.cursor.fetchall()
        ]

    def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Any:
        """Execute a SQL query and return results or affected row count."""
        if params is None:
            params = ()
        self.cursor.execute(query, params)
        if self.cursor.description:  # SELECT-like query
            columns = [col[0] for col in self.cursor.description]
            rows = self.cursor.fetchall()
            return {"columns": columns, "rows": rows}
        else:
            self.conn.commit()
            return {"rowcount": self.cursor.rowcount}

    def close(self) -> None:
        self.conn.close()

if __name__ == "__main__":
    db_path = "databases/spider/test_database/bakery_1/bakery_1.sqlite"
    assert os.path.exists(db_path), f"Database file {db_path} does not exist."
    db = DBManager(db_path)
    print(f"✅ Connected to {db_path!r}")
    tables = db.extract_column_table_pairs()
    print("Tables and their columns:")
    for table, columns in tables.items():
        print(f"  {table}: {', '.join(columns)}")
    db.close()
    print("Connection closed.")