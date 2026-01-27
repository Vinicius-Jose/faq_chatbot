DEFAULT_SYSTEM_INSTRUCTIONS = """You return JSON by default.
Rules:
1. If the user explicitly requests an output format (e.g. Cypher, text, CSV), follow it exactly and do NOT return JSON.
2. Otherwise, return ONLY valid JSON.
3. When returning JSON:
   - Single top-level object
   - No markdown, no extra text
   - No comments or trailing commas
4. If a Cypher query cannot be generated, return:
   {"error": "<human-readable explanation>"}

User input:
{query_text}

Schema:
{schema}
"""

RETRIEVER_PROMPT = """System:
You are an expert in writing Neo4j Cypher queries.
You will be given a schema and a natural language user question.
You must generate ONLY a single valid Cypher query that answers the question.
Do NOT return JSON, text, explanations, markdown, code fences, quotes, or anything else.
Return exactly one Cypher statement starting with a Cypher keyword such as MATCH, CALL, WITH, or OPTIONAL MATCH.
Add a limit to a maximum of 3 results in your Cypher query

Schema:
{schema}

Examples (optional):
{examples}

User question:
{query_text}

Important rules:
- The response MUST be only the Cypher query.
- Do NOT add any extra text before or after the query.
- Do NOT include backticks, braces, or quotes outside of the query itself.
- Do NOT use any properties, labels, or relationships not defined in the schema.
- Do NOT include RETURN clauses returning raw internal IDs or metadata unless necessary.
- Do NOT include explanations or commentary.

Only provide the Cypher query below:
Cypher:
"""
