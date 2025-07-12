import json

def generate_sql_prompt(schema: dict, user_request: str, db_type: str = "PostgreSQL") -> str:
    """
    Builds a hardened prompt so the LLM:
      • Parses the JSON schema
      • Validates table/column names
      • Generates exactly one optimized SQL query
      • Emits ONLY the SQL (or a fixed ERROR message)
    """
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)

    prompt = f"""
        SYSTEM: You are an expert SQL generator for {db_type}.
        SYSTEM: You will receive a JSON schema and a user request.  
        SYSTEM: You must output ONLY a single SQL statement compatible with {db_type}.  
        SYSTEM: Do NOT output any reasoning, comments, or markdown.

        -- Rules:
        1. Use UPPERCASE for all SQL keywords.
        2. Indent with 2 spaces.
        3. End the statement with a semicolon.
        4. Reference only tables/columns present in the schema.
        5. If you cannot satisfy the request with the given schema, output exactly:
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
        SELECT
        country,
        COUNT(*) AS user_count
        FROM
        users
        GROUP BY
        country;

        -- Now, process this request:
        # Request: {user_request}
        
        -- Output your SQL below (and nothing else):
        """
    return prompt.strip()
