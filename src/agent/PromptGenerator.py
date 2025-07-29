

def _schema_to_string(schema: dict) -> str:
    """Return a compact string representation of *schema*.

    The input schema follows the structure ``{"tables": [{"name": ..., "columns": [...]}, ...]}``.
    Column entries may be either strings or objects with a ``"name"`` field.
    The output format is ``table(col1,col2); table2(colA,colB)`` which is far
    shorter than pretty JSON and uses only essential information.
    """

    tables = []
    for table in schema.get("tables", []):
        name = table.get("name") or table.get("table_name")
        cols = table.get("columns", [])
        col_names = [c["name"] if isinstance(c, dict) else str(c) for c in cols]
        tables.append(f"{name}({','.join(col_names)})")
    return '; '.join(tables)

def generate_sql_prompt(schema: dict, user_request: str, db_type: str = "PostgreSQL") -> str:
    """
    Builds a hardened prompt so the LLM:
      • Parses the JSON schema
      • Validates table/column names
      • Generates exactly one optimized SQL query
      • Emits ONLY the SQL (or a fixed ERROR message)
    """
    # Compact the schema to minimize prompt length while retaining table/column
    # names. This avoids hitting the 512 token limit of models like Flan-T5.
    schema_json = _schema_to_string(schema)

    prompt = f"""
            SYSTEM: You are an expert SQL generator for {db_type}.
            SYSTEM: You will receive a JSON schema and a user request.
            SYSTEM: You must output ONLY a single SQL statement compatible with {db_type}.
            SYSTEM: Do NOT output any reasoning, comments, or markdown.

            -- Rules:
            1. Use UPPERCASE for all SQL keywords.
            2. Output the entire SQL statement on ONE line.
            3. End the statement with a semicolon.
            4. Reference only tables/columns present in the schema.
            5. All column aliases must be in French (snake_case).
            6. If you cannot satisfy the request with the given schema, output exactly:
                ERROR: Unable to fulfill request

            <details hidden>
            <!--
            THINK:
            - Parse schema_json.
            - Map user_request to tables/cols.
            - Plan JOINs, filters, aggregates.
            - Validate names against schema.
            - Optimize for minimal scans.
            -->
            </details>

            -- Database Schema --
            {schema_json}

            -- Few-Shot Example --
            # Request: "Count users by country"
            SELECT country, COUNT(*) AS nombre_utilisateurs FROM users GROUP BY country;

            -- Now, process this request:
            # Request: {user_request}

            -- Output your SQL below (and nothing else):
            """

    return prompt.strip()
