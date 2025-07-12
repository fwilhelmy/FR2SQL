import sqlite3
import os


def build_sample_db(db_path="employee_db.sqlite"):
    # Ensure directory exists
    if not os.path.exists("database/sqlite"):
        os.makedirs("database/sqlite")

    conn = sqlite3.connect(os.path.join("database/sqlite", db_path))
    cursor = conn.cursor()

    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS employee_projects;")
    cursor.execute("DROP TABLE IF EXISTS projects;")
    cursor.execute("DROP TABLE IF EXISTS employee_roles;")
    cursor.execute("DROP TABLE IF EXISTS roles;")
    cursor.execute("DROP TABLE IF EXISTS employees;")
    cursor.execute("DROP TABLE IF EXISTS departments;")
    cursor.execute("DROP TABLE IF EXISTS addresses;")

    # Main tables
    cursor.execute("""
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        location TEXT    -- New: Office location for department
    );
    """)

    cursor.execute("""
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,      -- New: Work email address
        hire_date TEXT,                  -- New: ISO date of hiring (YYYY-MM-DD)
        salary REAL,
        is_active INTEGER DEFAULT 1,     -- New: Employment status (1=active, 0=inactive)
        department_id INTEGER,
        FOREIGN KEY(department_id) REFERENCES departments(id)
    );
    """)

    # Address table for employee contact info
    cursor.execute("""
    CREATE TABLE addresses (
        id INTEGER PRIMARY KEY,
        employee_id INTEGER NOT NULL,    -- FK to employees
        street TEXT,
        city TEXT,
        state TEXT,
        postal_code TEXT,
        country TEXT,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    );
    """)

    # Roles and assignment
    cursor.execute("""
    CREATE TABLE roles (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,            -- e.g., 'Manager', 'Developer'
        description TEXT                -- Role description or responsibilities
    );
    """)

    cursor.execute("""
    CREATE TABLE employee_roles (
        employee_id INTEGER,
        role_id INTEGER,
        assigned_date TEXT,             -- When the role was assigned
        PRIMARY KEY(employee_id, role_id),
        FOREIGN KEY(employee_id) REFERENCES employees(id),
        FOREIGN KEY(role_id) REFERENCES roles(id)
    );
    """)

    # Projects and assignments
    cursor.execute("""
    CREATE TABLE projects (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        start_date TEXT,
        end_date TEXT,
        budget REAL                    -- New: Project budget
    );
    """)

    cursor.execute("""
    CREATE TABLE employee_projects (
        employee_id INTEGER,
        project_id INTEGER,
        role TEXT,                     -- Role in project (e.g., 'Lead', 'Contributor')
        hours_allocated REAL,          -- Estimated hours
        PRIMARY KEY(employee_id, project_id),
        FOREIGN KEY(employee_id) REFERENCES employees(id),
        FOREIGN KEY(project_id) REFERENCES projects(id)
    );
    """)

    # Insert sample data
    cursor.execute("INSERT INTO departments (id, name, location) VALUES (1, 'Informatique', 'Montreal'), (2, 'Ressources Humaines', 'Quebec City');")

    cursor.execute("""
    INSERT INTO employees (id, name, email, hire_date, salary, is_active, department_id) VALUES
    (1, 'Alice', 'alice@example.com', '2022-05-15', 70000, 1, 1),
    (2, 'Bob', 'bob@example.com', '2021-11-01', 65000, 1, 1),
    (3, 'Claire', 'claire@example.com', '2023-01-22', 62000, 1, 2);
    """)

    cursor.execute("""
    INSERT INTO addresses (employee_id, street, city, state, postal_code, country) VALUES
    (1, '123 Main St', 'Montreal', 'QC', 'H1A1A1', 'Canada'),
    (2, '456 Elm St', 'Montreal', 'QC', 'H2B2B2', 'Canada'),
    (3, '789 Oak St', 'Quebec City', 'QC', 'G1C3C3', 'Canada');
    """)

    cursor.execute("""
    INSERT INTO roles (id, title, description) VALUES
    (1, 'Manager', 'Oversees team operations'),
    (2, 'Developer', 'Writes and maintains code');
    """)

    cursor.execute("""
    INSERT INTO employee_roles (employee_id, role_id, assigned_date) VALUES
    (1, 2, '2022-05-15'),
    (2, 2, '2021-11-01'),
    (3, 1, '2023-01-22');
    """)

    cursor.execute("""
    INSERT INTO projects (id, name, start_date, end_date, budget) VALUES
    (1, 'Intranet Redesign', '2024-01-10', '2024-06-30', 50000),
    (2, 'HR Onboarding App', '2024-03-01', NULL, 30000);
    """)

    cursor.execute("""
    INSERT INTO employee_projects (employee_id, project_id, role, hours_allocated) VALUES
    (1, 1, 'Lead Developer', 800),
    (2, 1, 'Developer', 600),
    (3, 2, 'Project Manager', 400);
    """)

    conn.commit()
    conn.close()
    print("✔ Base de données créée avec succès : database/sqlite/employee_db.sqlite")


if __name__ == "__main__":
    build_sample_db()
