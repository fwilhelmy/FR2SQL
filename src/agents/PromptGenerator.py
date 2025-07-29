import json


def generate_sql_prompt(schema: dict, user_request: str) -> str:
    """
    Builds a hardened prompt so the LLM:
      • Parses the JSON schema
      • Validates table/column names
      • Generates exactly one optimized SQL query
      • Emits ONLY the SQL (or a fixed ERROR message)
    """
    # Compact the schema to minimize prompt length while retaining table/column
    # names. This avoids hitting the 512 token limit of models like Flan-T5.
    schema_json = json.dumps(schema, indent=2, ensure_ascii=False)

    prompt = f"""
SYSTEM: You are an expert SQL query generator.
SYSTEM: You will receive a JSON schema and a user request.

-- Rules:
1. Use UPPERCASE for all SQL keywords.
2. Output the entire SQL statement on ONE line.
4. Reference only tables/columns present in the schema.
5. All column aliases must be in French (snake_case).

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

-- Output your SQL below (and nothing else):"""

    return prompt.strip()
