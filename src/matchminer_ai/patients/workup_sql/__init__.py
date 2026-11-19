"""SQL-agent workup checks: an LLM writes one cohort-level SQL query per workup requirement.

A research review aid beside the compiled-query method
(:mod:`matchminer_ai.patients.workup_queries`). Structured extraction records
become relational tables (:func:`build_extraction_database`, written to DuckDB
by :func:`write_extraction_database`). For each guideline workup requirement, an
LLM agent explores an exploration database with read-only SQL in an isolated
worker. It sees codes, spellings and distributions, submits a final query, sees
that query's results over the whole exploration population, and accepts or
revises it (:func:`compile_workup_sql`). Code counts what the store holds first
(:func:`store_profile`, in the agent's prompt), and the agent's search action
(:func:`search_values`) lists the codes and text values containing a few words.
The agent's only power is read-only SQL: no files, network, shell or other tools. Code then runs each accepted query
once over every cohort it is applied to (:func:`check_workup_with_sql`) and
reports met, not met (not documented), unknown or not applicable, with the
cited items' evidence quotes.

Exploration results are patient-derived extracted values (never patient ids,
note text or evidence quotes) and reach the configured LLM endpoint. Use only
an endpoint authorized for the records. Results need human review against the
source notes. Requires the optional ``duckdb`` dependency
(``pip install matchminer-ai[sql]``).
"""

from .agent import (
    ACTIONS,
    AGENT_VERSION,
    RESULT_COLUMNS,
    SQL_REASONS,
    WorkupSQLLimits,
    agent_contract,
    agent_limits,
    check_result,
    compile_workup_sql,
    render_system_prompt,
    resolve_agent_llm,
    response_schema,
    reusable_sql_entry,
    run_final_query,
    workup_sql_cache_key,
)
from .database import (
    DATABASE_VERSION,
    ExtractionDatabase,
    build_extraction_database,
    database_info,
    read_database_info,
    record_extensions,
    render_database_guide,
    write_extraction_database,
)
from .evaluate import METHOD, NOTICE, check_workup_with_sql
from .profile import PROFILE_VERSION, render_profile, search_values, store_profile
from .sandbox import SandboxLimits, SQLSandbox, SQLSandboxError

__all__ = [
    "ACTIONS",
    "AGENT_VERSION",
    "DATABASE_VERSION",
    "ExtractionDatabase",
    "METHOD",
    "NOTICE",
    "PROFILE_VERSION",
    "RESULT_COLUMNS",
    "SQLSandbox",
    "SQLSandboxError",
    "SQL_REASONS",
    "SandboxLimits",
    "WorkupSQLLimits",
    "agent_contract",
    "agent_limits",
    "build_extraction_database",
    "check_result",
    "check_workup_with_sql",
    "compile_workup_sql",
    "database_info",
    "read_database_info",
    "record_extensions",
    "render_database_guide",
    "render_profile",
    "render_system_prompt",
    "resolve_agent_llm",
    "response_schema",
    "reusable_sql_entry",
    "run_final_query",
    "search_values",
    "store_profile",
    "workup_sql_cache_key",
    "write_extraction_database",
]
