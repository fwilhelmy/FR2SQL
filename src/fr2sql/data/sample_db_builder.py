import sqlite3
import os

def build_sample_db(db_path="employee_db.sqlite"):
    if not os.path.exists("data/sqlite"):
        os.makedirs("data/sqlite")

    conn = sqlite3.connect(os.path.join("data/sqlite", db_path))
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS employees;")
    cursor.execute("DROP TABLE IF EXISTS departments;")

    cursor.execute("""
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        salary REAL,
        department_id INTEGER,
        FOREIGN KEY(department_id) REFERENCES departments(id)
    );
    """)

    # Ajout de données d'exemple
    cursor.execute("INSERT INTO departments (id, name) VALUES (1, 'Informatique'), (2, 'Ressources Humaines');")

    cursor.execute("""
    INSERT INTO employees (id, name, salary, department_id) VALUES
    (1, 'Alice', 70000, 1),
    (2, 'Bob', 65000, 1),
    (3, 'Claire', 62000, 2);
    """)

    conn.commit()
    conn.close()
    print("✔ Base de données créée avec succès : data/sqlite/employee_db.sqlite")

if __name__ == "__main__":
    build_sample_db()
