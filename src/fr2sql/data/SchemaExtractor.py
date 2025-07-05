import sqlite3
from typing import Dict, List, Any

class SchemaExtractor:
    """Introspect SQLite databases to extract table schemas and relations."""

    def __init__(self, db_path: str):
        """Open the database and prepare metadata extraction."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def extract(self) -> Dict[str, Any]:
        """Return a structured representation of the schema."""
        schema = {}
        tables = self._get_tables()
        for table in tables:
            schema[table] = {
                "columns": self._get_columns(table),
                "foreign_keys": self._get_foreign_keys(table)
            }
        return schema

    def _get_tables(self) -> List[str]:
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        return [row[0] for row in self.cursor.fetchall()]

    def _get_columns(self, table: str) -> List[Dict[str, Any]]:
        self.cursor.execute(f"PRAGMA table_info({table});")
        return [
            {
                "name": col[1],
                "type": col[2],
                "not_null": bool(col[3]),
                "default_value": col[4],
                "primary_key": bool(col[5])
            }
            for col in self.cursor.fetchall()
        ]

    def _get_foreign_keys(self, table: str) -> List[Dict[str, str]]:
        self.cursor.execute(f"PRAGMA foreign_key_list({table});")
        return [
            {
                "from_column": fk[3],
                "to_table": fk[2],
                "to_column": fk[4]
            }
            for fk in self.cursor.fetchall()
        ]

    def close(self):
        self.conn.close()
        
if __name__ == "__main__":
    extractor = SchemaExtractor("data/sqlite/employee_db.sqlite")
    schema = extractor.extract()
    
    import json
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    
    extractor.close()
