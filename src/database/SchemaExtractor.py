import sqlite3
from typing import List, Dict, Any

class SchemaExtractor:
    """Introspect SQLite databases to extract column-table pairs and table metadata."""

    def __init__(self, db_path: str):
        """Open the database and prepare metadata extraction."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def extract_column_table_pairs(self) -> Dict[str, List[str]]:
        result = {}
        tables = self._get_tables()
        for table in tables:
            columns = self._get_columns(table)
            result[table] = [col['name'] for col in columns]
        return result

    def extract_table_metadata(self, table_name: str) -> Dict[str, Any]:
        """
        Return metadata (columns + foreign keys) for a specific table.
        """
        return {
            "columns": self._get_columns(table_name),
            "foreign_keys": self._get_foreign_keys(table_name)
        }

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

    # Affiche les paires colonne-table
    print("(colonne-table)")
    for item in extractor.extract_column_table_pairs().items():
        print(item)

    # Demande à l'utilisateur une table à analyser
    table_name = input("\nEntrez le nom d'une table pour voir ses métadonnées : ").strip()

    try:
        metadata = extractor.extract_table_metadata(table_name)
        import json
        print(f"\n Métadonnées de la table '{table_name}' ")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Erreur : impossible d'extraire les métadonnées de la table '{table_name}'.")
        print(f"Détail : {e}")

    extractor.close()
