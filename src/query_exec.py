import sqlite3
import os
import sys


def execute_query(query: str, db_path: str = "data/sqlite/employee_db.sqlite"):
    """
    Exécute une requête SQL sur la base SQLite spécifiée.
    Affiche le résultat pour les SELECT ou le nombre de lignes affectées pour les autres commandes.
    """
    if not os.path.exists(db_path):
        print(f"Erreur : fichier de base de données introuvable : {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        if cursor.description:
            # SELECT-like query
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            # Affichage des en-têtes
            print(" | ".join(columns))
            print("-" * (len(" | ".join(columns))))
            # Affichage des lignes
            for row in rows:
                print(" | ".join(str(item) for item in row))
        else:
            # Non-SELECT (INSERT/UPDATE/DELETE)
            conn.commit()
            print(f"Commandes exécutées, {cursor.rowcount} ligne(s) affectée(s)")
    except sqlite3.Error as e:
        print(f"Erreur SQLite : {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    sql_query = """
        SELECT
        d.id            AS department_id,
        d.name          AS department_name,
        ROUND(AVG(e.salary), 2) AS average_salary
        FROM departments d
        LEFT JOIN employees e
        ON e.department_id = d.id
        GROUP BY
        d.id,
        d.name
        ;
    """

    execute_query(sql_query)
