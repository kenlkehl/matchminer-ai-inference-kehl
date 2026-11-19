"""The SQL workup agent: one cohort-level SQL query per guideline workup requirement.

For each distinct requirement an LLM explores an extraction database
(:mod:`.database`) with read-only SQL in the isolated worker (:mod:`.sandbox`):
schema, codes, spellings and distributions. It submits a final query, sees how
that query's results are distributed over the whole exploration population, and
accepts or revises it, within a bounded number of actions. Code first counts
what the store holds (:mod:`.profile`, in the system prompt), and the agent's
``search`` action lists the codes and text values containing a few words. The
agent's only power is read-only SQL on the extracted items (its own queries, and
the code-written ones behind the profile and search): no files, network, shell
or other tools. The accepted query is cached (one per requirement, population
context and exploration database) and later run once over each cohort
(:func:`~matchminer_ai.patients.workup_sql.check_workup_with_sql`).

Exploration results are patient-derived extracted values (no patient ids, note
text or evidence quotes) and reach the configured LLM endpoint: use only an
endpoint authorized for the records the database was built from. The cache keeps
the SQL, the summary and aggregate counts, never query results. A failed run is
saved too, with its steps and code-made diagnostics, so that it can be diagnosed;
it is never reused.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, MutableMapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

from matchminer_ai._storage import digest
from matchminer_ai.cancellation import check_cancelled, submit_cancellable
from matchminer_ai.config import MMAIConfig, load_default_preset
from matchminer_ai.llm.backends import build_llm_runtime_config
from matchminer_ai.llm.prompts import load_prompt_text
from matchminer_ai.llm.remote_auth import DATABRICKS_ANTHROPIC_PROVIDER, remote_provider_name
from matchminer_ai.llm.structured import (
    CredentialsError,
    EndpointError,
    StructuredConfig,
    resolve_structured_config,
)
from matchminer_ai.patients.extraction.schema import SCHEMA_VERSION
from matchminer_ai.patients.workup_queries.compile import requirement_text
from matchminer_ai.patients.workup_queries.evaluate import APPLICABILITY, STATUSES

from .database import DATABASE_VERSION, ExtractionDatabase, read_database_info, render_database_guide
from .profile import (
    PROFILE_VERSION,
    SearchFailed,
    render_profile,
    render_search,
    search_values,
    store_profile,
)
from .sandbox import SandboxLimits, SQLSandbox, SQLSandboxError

logger = logging.getLogger(__name__)

SECTION = "workup_sql_agent"
PROMPT_FILE = "patient.workup_sql.system.txt"
AGENT_VERSION = "mmai-workup-sql-agent-v4"
RESULT_COLUMNS = ("patient_key", "status", "applicability", "reason", "evidence_item_keys",
                  "detail")
#: The reasons a query may give, by status; code owns the others (no_extraction,
#: incomplete_extraction, not_expressible, query_unavailable).
SQL_REASONS = {
    "met": ("documented",),
    "not_met": ("not_documented", "planned_only"),
    "unknown": ("uncertain_dates", "fields_not_stated", "anchor_unavailable"),
    "not_applicable": ("documented_not_applicable",),
    # Only for a requirement that advises against the step (recommends_against).
    "done_not_recommended": ("documented_not_recommended",),
}
ACTIONS = ("search", "query", "submit", "accept", "not_expressible")
#: Actions that explore (and count as checking queries after a valid submission).
_EXPLORING = ("search", "query")
_FIELDS = ("action", "purpose", "terms", "sql", "summary", "approximations",
           "not_expressible_reason",
           "recommends_against", "memory")


@dataclass(frozen=True)
class WorkupSQLLimits:
    """Bounds of one requirement's agent run (preset ``workup_sql_agent.limits``)."""

    max_actions: int = 12
    max_submits: int = 3
    #: The action by which the agent is asked to submit a first candidate.
    submit_by: int = 6
    #: Checking queries allowed after a valid submission before the agent must accept
    #: it or submit a revision (code removes the query action).
    max_review_queries: int = 3
    max_rows: int = 30
    max_cell_chars: int = 160
    max_result_chars: int = 6000
    max_memory_chars: int = 4000
    max_recent_steps: int = 8
    #: Words per search, and matching text values a search shows.
    max_search_terms: int = 8
    max_search_rows: int = 50
    max_sql_chars: int = 16000
    max_detail_chars: int = 300
    max_evidence_items: int = 8
    sample_rows_per_status: int = 3
    max_concurrent_items: int = 8
    query_timeout_seconds: float = 20.0
    final_timeout_seconds: float = 120.0
    worker_memory_mb: int = 4096
    duckdb_memory_mb: int = 2048

    def __post_init__(self):
        if self.max_actions < 2 or not 1 <= self.max_submits <= self.max_actions:
            raise ValueError("max_actions must be >= 2 and max_submits from 1 to max_actions.")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"workup_sql_agent.limits.{item.name} must be positive.")

    @property
    def first_submit(self) -> int:
        return min(self.submit_by, self.max_actions)

    @classmethod
    def from_mapping(cls, values) -> WorkupSQLLimits:
        values = dict(values or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown workup_sql_agent.limits: {', '.join(sorted(unknown))}")
        return cls(**values)

    def sandbox(self) -> SandboxLimits:
        return SandboxLimits(memory_mb=self.worker_memory_mb,
                             duckdb_memory_mb=self.duckdb_memory_mb,
                             query_timeout_seconds=self.query_timeout_seconds,
                             # Room for the code-written profile and search statements;
                             # the agent's own SQL is held to max_sql_chars.
                             max_sql_chars=self.max_sql_chars + 24000)


def _section(config: MMAIConfig) -> dict:
    values = dict(getattr(config, SECTION, None) or {})
    if not values:
        raise ValueError(f"The configuration has no {SECTION} section")
    return values


def agent_limits(config: MMAIConfig | None = None) -> WorkupSQLLimits:
    return WorkupSQLLimits.from_mapping(_section(config or load_default_preset()).get("limits"))


#: Connection keys the section's ``remote`` may set for the agent alone, overriding
#: ``config.remote`` (the other sections, such as extraction, keep that connection).
_CONNECTION_KEYS = ("server_urls", "provider", "google_project_id", "databricks_profile")


def resolve_agent_llm(config: MMAIConfig) -> StructuredConfig:
    """The agent's endpoint: the ``workup_sql_agent`` section on ``config.remote``.

    Any OpenAI-compatible endpoint works (a local vLLM server, Vertex, others), and
    so does Claude through a Databricks AI Gateway (provider ``databricks_anthropic``).
    The section's ``remote`` may name its own ``server_urls``, ``provider``,
    ``google_project_id`` and ``databricks_profile``. The agent cannot drive an
    in-process engine.
    """
    values = _section(config)
    llm = {k: copy.deepcopy(values[k]) for k in
           ("local", "remote", "reasoning_parser", "sampling_profile", "reasoning_effort")
           if k in values}
    runtime = build_llm_runtime_config(SECTION, llm, config=config)
    if runtime["backend_mode"] != "remote":
        raise ValueError("The SQL workup agent needs an OpenAI-compatible endpoint: enable "
                         "remote and set remote.server_urls (a local vLLM server works).")
    connection = values.get("remote") or {}
    if "provider" in connection:
        # A provider of the agent's own: drop the shared connection's credentials scope.
        runtime.pop("google_project_id", None)
        runtime.pop("databricks_profile", None)
    for key in _CONNECTION_KEYS:
        if key in connection:
            runtime[key] = copy.deepcopy(connection[key])
    runtime["max_retries"] = int(values.get("max_attempts", 3))
    runtime["response_format"] = "json_schema" if values.get("structured_outputs", True) \
        else "none"
    runtime["tokenizer_mode"] = values.get("tokenizer_mode", "bytes")
    # A turn's own read timeout: the shared remote.request_timeout (hours, for long
    # summaries) would let one stalled call hold an item that long.
    runtime["request_timeout"] = float(values.get("request_timeout", 600))
    # Enough request slots for every item explored at once.
    runtime["max_concurrent_requests"] = max(
        int(runtime.get("max_concurrent_requests") or 1),
        int(values.get("max_concurrent_requests") or agent_limits(config).max_concurrent_items),
    )
    # A busy endpoint (429/503) is waited out, within max_attempts, instead of
    # costing the item an attempt immediately.
    if not runtime.get("capacity_retry_initial_seconds"):
        runtime["capacity_retry_initial_seconds"] = 10.0
        runtime["capacity_retry_max_seconds"] = max(
            60.0, float(runtime.get("capacity_retry_max_seconds") or 60.0))
    context = values.get("context_window") or runtime.get("max_model_len")
    if context:
        runtime["context_window"] = int(context)
    resolved, _ = resolve_structured_config(runtime, cache_dir=None)
    return resolved


def response_schema(limits: WorkupSQLLimits, actions: Sequence[str] = ACTIONS) -> dict:
    text = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(actions)},
            "purpose": {**text, "description": "What this step is for, one short sentence."},
            "terms": {"type": "array", "items": text, "maxItems": limits.max_search_terms,
                      "description": "search: words or word stems to look for (case does not "
                      "matter); otherwise []."},
            "sql": {**text, "description": "query/submit: one DuckDB SELECT of at most "
                    f"{limits.max_sql_chars} characters; otherwise ''."},
            "summary": {**text, "description": "submit: one plain sentence saying what the "
                        "final query checks; otherwise ''."},
            "approximations": {"type": "array", "items": text, "maxItems": 8,
                               "description": "submit: parts of the requirement left out or "
                               "approximated, one short sentence each; otherwise []."},
            "not_expressible_reason": {**text, "description": "not_expressible: why; "
                                       "otherwise ''."},
            "recommends_against": {"type": "boolean", "description": "submit: true when the "
                                   "requirement advises against the step (not recommended, "
                                   "not routinely indicated); otherwise false."},
            "memory": {**text, "description": "Notes to yourself for later turns: findings, "
                       f"codes, spellings, plan (at most {limits.max_memory_chars} characters)."},
        },
        "required": list(_FIELDS),
        "additionalProperties": False,
    }


#: Stands for the store profile in the prompt the contract hashes: the profile is a
#: function of the database (its fingerprint is in the cache key) and of PROFILE_VERSION.
_PROFILE_PLACEHOLDER = "(the exploration database's profile, computed by code)"


def render_system_prompt(limits: WorkupSQLLimits, extensions=(), profile: str | None = None) -> str:
    return (load_prompt_text(PROMPT_FILE)
            .replace("{store_profile}", profile or _PROFILE_PLACEHOLDER)
            .replace("{max_search_terms}", str(limits.max_search_terms))
            .replace("{max_rows}", str(limits.max_rows))
            .replace("{submit_by}", str(limits.first_submit))
            .replace("{max_review_queries}", str(limits.max_review_queries))
            .replace("{max_detail_chars}", str(limits.max_detail_chars))
            .replace("{database_guide}", render_database_guide(extensions)))


def agent_llm_identity(config: MMAIConfig | None = None) -> dict:
    """The provider, model and reasoning effort that write the queries, from the configuration.

    Part of the cache key, so each model's (and effort's) queries are cached apart.
    Read without contacting the endpoint (a model left to discovery counts as "").
    """
    config = config or load_default_preset()
    section = _section(config)
    connection = section.get("remote") or {}
    provider = connection.get("provider") or (getattr(config, "remote", None) or {}).get(
        "provider")
    effort = (connection.get("request_params") or {}).get("reasoning_effort") or section.get(
        "reasoning_effort")
    return {"provider": remote_provider_name({"provider": provider}),
            "model": str(connection.get("model_name") or ""), "effort": str(effort or "")}


def agent_contract(config: MMAIConfig | None = None, extensions=()) -> dict:
    """What an accepted query depends on besides its requirement, population and database."""
    limits = agent_limits(config)
    return {
        "agent_version": AGENT_VERSION,
        "llm": agent_llm_identity(config),
        "database_version": DATABASE_VERSION,
        "profile_version": PROFILE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "extensions": sorted(extensions),
        "prompt_sha256": digest(render_system_prompt(limits, extensions)),
        "response_schema_sha256": digest(response_schema(limits)),
        "result_columns": list(RESULT_COLUMNS),
    }


def workup_sql_cache_key(requirement: dict, population_context: str | None = None, *,
                         database_fingerprint: str, contract: dict) -> str:
    """The cache key of one requirement's accepted query."""
    return digest({
        "requirement": requirement_text(requirement),
        "population": (population_context or "").strip(),
        "database": database_fingerprint,
        **contract,
    })


# ---------------------------------------------------------------------------
# Checking a query's output
# ---------------------------------------------------------------------------


def _strip(sql: str) -> str:
    return re.sub(r"[\s;]+$", "", sql.strip())


@dataclass
class ResultCheck:
    problems: list[str]
    rows: dict[int, dict]
    distribution: dict
    #: Short code-made names of the problems, safe to save (no values or error text).
    kinds: list[str] = field(default_factory=list)


def check_result(reply: dict, patient_count: int, limits: WorkupSQLLimits, *,
                 recommends_against: bool = False) -> ResultCheck:
    """Check a final query's full output against the contract (one row per patient).

    For a requirement that advises against its step (``recommends_against``), a
    documented step is ``done_not_recommended``, never ``met``; otherwise
    ``done_not_recommended`` is not allowed.
    """
    problems: list[str] = []
    if reply["error"]:
        return ResultCheck([f"The query failed: {reply['error']}: {reply['error_detail']}"],
                           {}, {}, [f"query error {_error_name(reply['error'])}"])
    columns = [str(c).lower() for c in reply["columns"]]
    if sorted(columns) != sorted(RESULT_COLUMNS) or len(set(columns)) != len(columns):
        return ResultCheck([f"Return exactly the columns {', '.join(RESULT_COLUMNS)} "
                            f"(got {', '.join(columns) or 'none'})."], {}, {}, ["wrong columns"])
    if reply["more_rows"] or reply["row_count"] > patient_count:
        return ResultCheck([f"The query returned {reply['row_count']} rows for "
                            f"{patient_count} patients: return one row per patient."], {}, {},
                           ["more rows than patients"])
    index = {name: columns.index(name) for name in RESULT_COLUMNS}
    rows: dict[int, dict] = {}
    bad: dict[str, list] = {}
    tags: dict[str, str] = {}

    def flag(kind, key, tag=None):
        bad.setdefault(kind, []).append(key)
        tags.setdefault(kind, tag or kind)

    for raw in reply["rows"]:
        row = {name: raw[i] for name, i in index.items()}
        key = row["patient_key"]
        if type(key) is not int:
            flag("patient_key is not an integer", key, "patient_key not an integer")
            continue
        if key in rows:
            flag("patient_key repeats", key)
            continue
        status, reason = row["status"], row["reason"]
        if status not in STATUSES:
            flag(f"status must be one of {', '.join(STATUSES)}", key, "invalid status")
        elif reason not in SQL_REASONS[status]:
            flag(f"reason does not fit its status (allowed pairs: {_pairs()})", key,
                 "reason does not fit status")
        elif recommends_against and status == "met":
            flag("this requirement advises against the step (recommends_against true): "
                 "report a documented step as done_not_recommended / "
                 "documented_not_recommended, never met", key, "met for a step advised against")
        elif not recommends_against and status == "done_not_recommended":
            flag("done_not_recommended is only for a requirement that advises against the "
                 "step: set recommends_against true, or report met", key,
                 "done_not_recommended without recommends_against")
        if row["applicability"] not in APPLICABILITY:
            flag(f"applicability must be one of {', '.join(APPLICABILITY)}", key,
                 "invalid applicability")
        if status == "not_applicable" and row["applicability"] != "not_applicable":
            flag("status not_applicable needs applicability not_applicable", key,
                 "not_applicable status with other applicability")
        keys = row["evidence_item_keys"]
        if keys is None:
            keys = []
        if not isinstance(keys, list) or any(type(k) is not int for k in keys):
            flag("evidence_item_keys must be a list of item_key integers ([] when none)", key,
                 "evidence_item_keys not a list of integers")
            keys = []
        detail = row["detail"]
        if detail is not None and not isinstance(detail, str):
            flag("detail must be text", key, "detail not text")
            detail = None
        rows[key] = {"status": status, "reason": reason, "applicability": row["applicability"],
                     "evidence_item_keys": list(dict.fromkeys(keys))[:limits.max_evidence_items],
                     "detail": (detail or "").strip()[:limits.max_detail_chars]}
    missing = patient_count - len([k for k in rows if 1 <= k <= patient_count])
    kinds = []
    if missing:
        problems.append(f"{missing} patients have no row: start FROM patients so that every "
                        "patient gets exactly one row.")
        kinds.append("patients without a row")
    for kind, keys in bad.items():
        shown = ", ".join(str(k) for k in keys[:3])
        problems.append(f"{kind}: {len(keys)} rows (for example patient_key {shown}).")
        kinds.append(tags[kind])
    statuses = Counter(f"{r['status']}/{r['reason']}" for r in rows.values())
    applicability = Counter(r["applicability"] for r in rows.values())
    with_evidence = Counter(r["status"] for r in rows.values() if r["evidence_item_keys"])
    distribution = {"patients": patient_count, "status_reason": dict(statuses.most_common()),
                    "applicability": dict(applicability.most_common()),
                    "with_evidence": dict(with_evidence)}
    return ResultCheck(problems, rows, distribution, kinds)


def _error_name(error) -> str:
    """A DuckDB error type (BinderException), never its message."""
    name = str(error or "")
    return name if re.fullmatch(r"[A-Za-z_]{1,60}", name) else "error"


def _pairs() -> str:
    return "; ".join(f"{s}/{r}" for s, reasons in SQL_REASONS.items() for r in reasons)


def ownership_sql(sql: str) -> str:
    """A trusted wrapper counting evidence keys that are not the row's patient's items."""
    return ("WITH result AS (\n" + _strip(sql) + "\n), cited AS (SELECT patient_key, "
            "unnest(evidence_item_keys) AS k FROM result) SELECT count(*) AS foreign_keys "
            "FROM cited LEFT JOIN items i ON i.item_key = cited.k "
            "WHERE i.item_key IS NULL OR i.patient_key <> cited.patient_key")


def run_final_query(box: SQLSandbox, sql: str, patient_count: int,
                    limits: WorkupSQLLimits, *, recommends_against: bool = False) -> ResultCheck:
    """Run a final query once over the whole database and check its output."""
    sql = _strip(sql)
    reply = box.query(sql, max_rows=patient_count + 1, count_limit=patient_count + 2,
                      max_cell_chars=None, timeout=limits.final_timeout_seconds)
    check = check_result(reply, patient_count, limits, recommends_against=recommends_against)
    if not check.problems:
        owned = box.query(ownership_sql(sql), max_rows=1, timeout=limits.final_timeout_seconds)
        foreign = owned["rows"][0][0] if not owned["error"] and owned["rows"] else None
        if foreign is None:
            check.problems.append(f"evidence_item_keys could not be checked: {owned['error']}: "
                                  f"{owned['error_detail']}")
            check.kinds.append("evidence check failed")
        elif foreign:
            check.problems.append(f"{foreign} evidence_item_keys are not item_key values of "
                                  "that row's patient.")
            check.kinds.append("evidence keys of other patients")
    return check


# ---------------------------------------------------------------------------
# One requirement's agent run
# ---------------------------------------------------------------------------


def _render(reply: dict, limits: WorkupSQLLimits) -> str:
    if reply["error"]:
        return f"Error ({reply['error']}): {reply['error_detail']}"
    head = "columns: " + ", ".join(f"{c} {t}" for c, t in zip(reply["columns"], reply["types"]))
    total = f"{reply['row_count']}{'+' if reply['more_rows'] else ''}"
    lines = [head, f"rows: {total} (showing {len(reply['rows'])})"]
    lines += [json.dumps(r, ensure_ascii=False, default=str) for r in reply["rows"]]
    text = "\n".join(lines)
    if len(text) > limits.max_result_chars:
        text = text[:limits.max_result_chars] + "\n[output cut]"
    return text


def _rows(reply: dict) -> str:
    count = reply["row_count"]
    return f"{count}{'+' if reply['more_rows'] else ''} row{'' if count == 1 else 's'}"


def _render_check(check: ResultCheck, limits: WorkupSQLLimits) -> str:
    if check.problems:
        return "Not valid:\n" + "\n".join(f"- {p}" for p in check.problems)
    d = check.distribution
    lines = [f"Valid. Results over all {d['patients']} exploration patients:",
             "status/reason: " + ", ".join(f"{k} {n}" for k, n in d["status_reason"].items()),
             "applicability: " + ", ".join(f"{k} {n}" for k, n in d["applicability"].items()),
             "rows citing evidence, by status: " + (", ".join(
                 f"{k} {n}" for k, n in d["with_evidence"].items()) or "none"),
             "examples (patient_key: status, reason, applicability, evidence_item_keys, detail):"]
    shown = Counter()
    for key in sorted(check.rows):
        row = check.rows[key]
        if shown[row["status"]] >= limits.sample_rows_per_status:
            continue
        shown[row["status"]] += 1
        lines.append(f"{key}: {row['status']}, {row['reason']}, {row['applicability']}, "
                     f"{row['evidence_item_keys']}, {row['detail']!r}")
    lines.append("Accept it unless these results show a specific defect you can name (a count "
                 "implausible for patients like the guideline population, or a status its own "
                 "detail contradicts); then fix that defect and submit again. You may first run "
                 f"up to {limits.max_review_queries} checking queries (for example looking up a "
                 "few of these patients' items). Do not explore for further edge cases.")
    return "\n".join(lines)


class _Run:
    """State of one requirement's agent loop."""

    def __init__(self, requirement, population, box, patient_count, llm, limits, system,
                 profile=None, extensions=()):
        self.requirement = requirement_text(requirement)
        self.population = (population or "").strip()
        self.box, self.patient_count, self.limits = box, patient_count, limits
        self.system, self.llm = system, llm
        self.profile, self.extensions = profile, tuple(extensions)
        self.memory, self.steps = "", []
        self.pending = None
        #: The last valid submission, kept when a later revision is not valid.
        self.last_valid = None
        self.submits = 0
        #: Checking queries since the pending (valid) submission.
        self.reviews = 0

    def reviews_left(self) -> int:
        return max(0, self.limits.max_review_queries - self.reviews) if self.pending else 0

    def actions(self, left: int) -> list[str]:
        can_submit = self.submits < self.limits.max_submits
        if left == 1:
            return ["accept"] if self.pending else (
                ["submit", "not_expressible"] if can_submit else ["not_expressible"])
        # After a valid submission only a few checking queries are allowed: then the
        # agent accepts it or submits a revision.
        can_query = not self.pending or self.reviews_left() > 0
        exploring = [a for a in _EXPLORING if a != "search" or self.profile is not None]
        actions = (exploring if can_query else []) + (["submit"] if can_submit else [])
        return actions + (["accept"] if self.pending else []) + ["not_expressible"]

    def payload(self, turn, left, actions, recent):
        if self.pending and self.reviews_left():
            guidance = ("Your last submission is valid. Accept it now unless its distribution "
                        "or examples show a specific defect you can name; then fix that defect "
                        "and submit a revision. You may run up to "
                        f"{self.reviews_left()} more checking queries first. Do not explore for "
                        "further edge cases.")
        elif self.pending:
            guidance = ("Your last submission is valid and no checking queries are left: accept "
                        "it, or submit a revision that fixes a specific defect you found.")
        elif left <= 3:
            guidance = "Few actions are left: submit your best final query now."
        elif turn == 0:
            guidance = ("Start with search: the requirement's words, word stems, abbreviations "
                        "and synonyms. The store profile already gives table and code counts."
                        if self.profile is not None else
                        "Start by exploring the tables that can document this requirement: "
                        "codes, text spellings and counts.")
        elif turn + 1 >= self.limits.first_submit and not self.submits:
            guidance = ("Submit a first candidate now; its results will show what to revise.")
        elif self.submits:
            guidance = ("Your last submission was not valid: fix every problem listed and "
                        "submit again.")
        else:
            guidance = ("Explore what decides the requirement, then submit a first candidate by "
                        f"action {self.limits.first_submit}.")
        older = self.steps[:len(self.steps) - len(recent)]
        return {
            "requirement": self.requirement,
            "guideline_population": self.population or "(not given)",
            "exploration_patients": self.patient_count,
            "turn": turn + 1,
            "actions_left": left,
            "submissions_left": self.limits.max_submits - self.submits,
            "checking_queries_left": self.reviews_left() if self.pending else None,
            "allowed_actions": actions,
            "pending_submission": ({"sql": self.pending["sql"], "summary": self.pending["summary"],
                                    "recommends_against": self.pending["recommends_against"]}
                                   if self.pending else None),
            "memory": self.memory,
            "earlier_steps": [f"{s['step']}. {s['action']}: {s['purpose']} -> {s['outcome']}"
                              for s in older],
            "recent_steps": [{k: s[k] for k in ("step", "action", "purpose", "terms", "sql",
                                                "result") if k != "terms" or s["terms"]}
                             for s in recent],
            "next_step": guidance,
        }

    def validator(self, actions):
        limits = self.limits

        def check(value):
            if not isinstance(value, dict) or set(value) != set(_FIELDS):
                raise ValueError("Return exactly the fields of the response schema.")
            if value["action"] not in actions:
                raise ValueError("action must be one of the allowed actions.")
            for name in _FIELDS:
                if name not in {"approximations", "recommends_against", "terms"} and not \
                        isinstance(value[name], str):
                    raise ValueError(f"{name} must be a string.")
            if type(value["recommends_against"]) is not bool:
                raise ValueError("recommends_against must be true or false.")
            if not isinstance(value["approximations"], list) or len(value["approximations"]) > 8 \
                    or any(not isinstance(a, str) for a in value["approximations"]):
                raise ValueError("approximations must be a list of at most 8 strings.")
            if not isinstance(value["terms"], list) or any(
                    not isinstance(t, str) or len(t) > 60 for t in value["terms"]) \
                    or len(value["terms"]) > limits.max_search_terms:
                raise ValueError(f"terms must be a list of at most {limits.max_search_terms} "
                                 "words of at most 60 characters.")
            if value["action"] == "search" and not any(len(t.strip()) >= 2
                                                       for t in value["terms"]):
                raise ValueError("search needs terms: at least one word of 2 or more "
                                 "characters.")
            if len(value["memory"]) > limits.max_memory_chars:
                raise ValueError("memory is longer than allowed.")
            if value["action"] in {"query", "submit"}:
                if not value["sql"].strip():
                    raise ValueError("query and submit need sql.")
                if len(value["sql"]) > limits.max_sql_chars:
                    raise ValueError(f"sql is longer than {limits.max_sql_chars} characters: "
                                     "shorten it (shared CTEs instead of repeated conditions).")
            if value["action"] == "submit" and not value["summary"].strip():
                raise ValueError("submit needs a one-sentence summary.")
            if value["action"] == "not_expressible" and not value["not_expressible_reason"].strip():
                raise ValueError("not_expressible needs a reason.")
            return value

        return check

    def step(self, action, purpose, sql, result, outcome, terms=()):
        self.steps.append({"step": len(self.steps) + 1, "action": action, "purpose": purpose,
                           "terms": list(terms), "sql": sql, "result": result,
                           "outcome": outcome})


def _compile_one(requirement, population, box, patient_count, llm, limits, system,
                 on_turn=None, profile=None, extensions=()) -> dict:
    from matchminer_ai.patients.note_search_qa import _MeasuredClient

    run = _Run(requirement, population, box, patient_count, llm, limits, system, profile,
               extensions)
    client = _MeasuredClient(llm)
    started = time.monotonic()
    outcome = {"status": "failed", "expressible": True, "not_expressible_reason": "",
               "sql": None, "summary": "", "approximations": [], "accepted_by": None,
               "recommends_against": False, "distribution": None,
               "termination": "budget_exhausted", "error": None}
    try:
        for turn in range(limits.max_actions):
            check_cancelled()
            if on_turn is not None:
                on_turn(turn + 1)
            left = limits.max_actions - turn
            actions = run.actions(left)
            schema = response_schema(limits, actions)
            schema_text = "Response JSON schema:\n" + json.dumps(schema)
            # Claude caches the instructions as their own block; the schema, whose actions
            # change from turn to turn, follows in a second one. Chat endpoints get one.
            head = ([{"role": "system", "content": system},
                     {"role": "system", "content": schema_text}]
                    if llm.provider == DATABRICKS_ANTHROPIC_PROVIDER
                    else [{"role": "system", "content": system + "\n\n" + schema_text}])
            recent = run.steps[-limits.max_recent_steps:]
            while True:
                messages = [*head,
                            {"role": "user", "content": json.dumps(
                                run.payload(turn, left, actions, recent), ensure_ascii=False)}]
                if client.fits(messages) or not recent:
                    break
                recent = recent[1:]
            if not client.fits(messages):
                outcome["termination"] = "context_limit"
                _fall_back(run, outcome, "context_accepted")
                break
            value = client.complete("workup-sql-agent", messages, schema, run.validator(actions))
            run.memory = value["memory"]
            action, purpose = value["action"], value["purpose"].strip()
            if action in _EXPLORING and run.pending:
                run.reviews += 1
            if action == "search":
                try:
                    found = search_values(box, value["terms"], profile, extensions=extensions,
                                          max_rows=limits.max_search_rows,
                                          max_cell_chars=limits.max_cell_chars)
                except SearchFailed as exc:
                    run.step("search", purpose, "", f"The search failed ({exc}): try fewer or "
                             "longer words, or a query.", f"error {exc}", terms=value["terms"])
                else:
                    run.step("search", purpose, "", render_search(found, limits.max_result_chars),
                             f"{len(found['codes'])} codes, "
                             f"{len(found['values']) + found['more_values']} values",
                             terms=found["words"])
            elif action == "query":
                reply = box.query(value["sql"], max_rows=limits.max_rows,
                                  max_cell_chars=limits.max_cell_chars)
                run.step("query", purpose, value["sql"], _render(reply, limits),
                         f"error {reply['error']}" if reply["error"] else _rows(reply))
            elif action == "submit":
                run.submits += 1
                run.reviews = 0
                check = run_final_query(box, value["sql"], patient_count, limits,
                                        recommends_against=value["recommends_against"])
                candidate = {"sql": _strip(value["sql"]), "summary": value["summary"].strip(),
                             "approximations": [a.strip() for a in value["approximations"]
                                                if a.strip()],
                             "recommends_against": value["recommends_against"],
                             "distribution": check.distribution}
                run.pending = None if check.problems else candidate
                if run.pending:
                    run.last_valid = candidate
                run.step("submit", purpose or "final query", value["sql"],
                         _render_check(check, limits),
                         "invalid: " + "; ".join(check.kinds or ["contract"])
                         if check.problems else "valid")
                if run.pending and left == 1:
                    outcome.update(_accepted(run.pending, "budget"))
                    outcome["termination"] = "budget_accepted"
                    break
            elif action == "accept":
                outcome.update(_accepted(run.pending, "agent"))
                outcome["termination"] = "accepted"
                break
            else:
                outcome.update(status="compiled", expressible=False,
                               not_expressible_reason=value["not_expressible_reason"].strip(),
                               termination="not_expressible", summary=purpose)
                break
        else:
            _fall_back(run, outcome, "budget_accepted")
    except CredentialsError as exc:
        # A fixed login hint (no provider or patient text).
        outcome.update(status="failed", termination="endpoint_failure", error=str(exc))
    except EndpointError:
        outcome.update(status="failed", termination="endpoint_failure")
    except SQLSandboxError:
        outcome.update(status="failed", termination="sandbox_failure",
                       error="The isolated SQL worker failed.")
    diagnostics = _diagnostics(client)
    if outcome["termination"] == "endpoint_failure" and outcome["error"] is None:
        outcome["error"] = _endpoint_error(diagnostics)
    if outcome["status"] == "failed" and outcome["error"] is None:
        submits = [s["outcome"] for s in run.steps if s["action"] == "submit"]
        outcome["error"] = ("No valid final query was accepted within the agent's budget"
                            f" ({outcome['termination']}; {len(run.steps)} actions, "
                            f"{len(submits)} submissions"
                            + (f", the last {submits[-1]}" if submits else "") + ").")
    if outcome["accepted_by"] == "budget":
        outcome["approximations"] = [*outcome["approximations"], (
            "The query was used at the end of the agent's action budget, without its "
            "final review of the results.")]
    outcome.update(
        steps=[{k: s[k] for k in ("step", "action", "purpose", "terms", "sql", "outcome")}
               for s in run.steps],
        action_count=len(run.steps) + (outcome["termination"] in {
            "accepted", "not_expressible"}),
        submit_count=run.submits,
        seconds=round(time.monotonic() - started, 2),
        usage={"requests": client.requests, "prompt_tokens": client.prompt_tokens,
               "completion_tokens": client.completion_tokens},
        diagnostics=diagnostics,
        model=llm.model,
    )
    return outcome


def _diagnostics(client) -> dict:
    """What went wrong with the agent's requests, from code-made values only.

    Validation messages are the harness's fixed texts; transport failures keep only
    their HTTP status. No provider text, prompts or results.
    """
    statuses = Counter(str(m.get("http_status", "no status")) for m in client.request_metrics
                       if m.get("transport_error"))
    return {
        "invalid_answers": dict(Counter(client.validation_errors).most_common()),
        "finish_reasons": dict(Counter(str(r) for r in client.finish_reasons).most_common()),
        "transport_errors": dict(statuses.most_common()),
    }


def _endpoint_error(diagnostics) -> str:
    if diagnostics["transport_errors"]:
        shown = ", ".join(f"HTTP {k} x{n}" if k.isdigit() else f"{k} x{n}"
                          for k, n in diagnostics["transport_errors"].items())
        return f"The agent's LLM endpoint failed ({shown})."
    if diagnostics["invalid_answers"]:
        last = next(iter(diagnostics["invalid_answers"]))
        return f"The agent's answers stayed invalid (most often: {last})"
    if diagnostics["finish_reasons"].get("length"):
        return "The agent's answers were cut off at the output token limit."
    return "The agent's LLM endpoint failed or its answers stayed invalid."


_PROFILES: dict[tuple, dict] = {}
_PROFILES_LOCK = threading.Lock()


def _profile(box, fingerprint, extensions) -> dict:
    """The exploration database's profile, computed once per process and database."""
    key = (fingerprint, tuple(sorted(extensions)), PROFILE_VERSION)
    with _PROFILES_LOCK:
        if key in _PROFILES:
            return _PROFILES[key]
    profile = store_profile(box, extensions)
    with _PROFILES_LOCK:
        while len(_PROFILES) >= 4:
            _PROFILES.pop(next(iter(_PROFILES)))
        _PROFILES[key] = profile
    return profile


def _fall_back(run, outcome, termination) -> None:
    """At the end of the budget, use the pending or else the last valid submission."""
    candidate = run.pending or run.last_valid
    if candidate is None:
        return
    outcome.update(_accepted(candidate, "budget"))
    outcome["termination"] = termination
    if run.pending is None:
        outcome["approximations"].append(
            "A later revision was not valid; the last valid submission was used.")


def _accepted(candidate, by) -> dict:
    return {"status": "compiled", "expressible": True, "sql": candidate["sql"],
            "summary": candidate["summary"], "approximations": list(candidate["approximations"]),
            "recommends_against": candidate["recommends_against"], "accepted_by": by,
            "distribution": candidate["distribution"]}


# ---------------------------------------------------------------------------
# Many requirements
# ---------------------------------------------------------------------------


def reusable_sql_entry(entry, contract) -> bool:
    """Whether a cache entry is an accepted query of ``contract`` (saved failures are not)."""
    return (isinstance(entry, dict) and entry.get("status") == "compiled"
            and entry.get("contract") == contract
            and (entry.get("expressible") is False or isinstance(entry.get("sql"), str)))


def _log(entry) -> None:
    """One line per explored requirement: guideline text and counts, never data."""
    usage = entry.get("usage") or {}
    args = (entry["requirement"].get("name") or "requirement", entry["termination"],
            entry.get("action_count", 0), entry.get("seconds", 0), usage.get("requests", 0),
            usage.get("prompt_tokens", 0))
    if entry["status"] == "failed":
        logger.warning("SQL agent failed for %r: %s after %d actions, %.0f s, %d requests, "
                       "%d prompt tokens: %s", *args, entry.get("error"))
    else:
        logger.info("SQL agent %r: %s after %d actions, %.0f s, %d requests, %d prompt tokens",
                    *args)


def compile_workup_sql(
    requirements: Sequence[dict],
    *,
    database: str | Path | ExtractionDatabase,
    config: MMAIConfig | None = None,
    population_context: str | None = None,
    cache: MutableMapping[str, dict] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    return_metadata: bool = False,
    llm: StructuredConfig | None = None,
    activity_callback: Callable[[list[dict]], None] | None = None,
):
    """One accepted cohort-level SQL query per workup requirement.

    ``database``: an exploration database, either a DuckDB file from
    :func:`write_extraction_database` or an :class:`ExtractionDatabase`. The
    agent explores it and checks each submission on all of its patients.
    ``cache``: any mapping from cache keys to entries. Accepted queries are reused
    while their contract matches. A failed run is saved under its key too, with its
    steps and ``diagnostics``, so that it can be diagnosed; it is never reused, and
    the next call explores the requirement again (and overwrites it). ``llm`` overrides the endpoint
    resolved from ``config.workup_sql_agent``. ``progress_callback(done, total)``
    reports finished requirements; ``activity_callback(items)`` reports, whenever
    it changes, the requirements being explored: ``{"name", "action",
    "max_actions"}`` each, ``action`` being the turn the agent is on (both are
    called from the calling thread).

    Returns one entry per requirement, in order (duplicates share an entry). Each
    entry has ``cache_key``, ``requirement``, ``status`` (compiled or failed),
    ``expressible``, ``sql``, ``summary``, ``approximations``, ``steps`` (SQL
    and outcomes, no results), ``distribution`` (aggregate counts over the
    exploration population), ``termination``, ``error``, ``usage``,
    ``diagnostics`` (code-made counts of invalid answers, finish reasons and
    transport failures), ``model``, ``contract`` and ``cached``. With ``return_metadata``, also returns metadata.
    """
    config = config or load_default_preset()
    limits = agent_limits(config)
    if isinstance(database, ExtractionDatabase):
        fingerprint, extensions = database.fingerprint, database.extensions
        patient_count = len(database.patient_ids)
        items = len(database.views)
    else:
        info = read_database_info(database)
        if info.get("database_version") != DATABASE_VERSION:
            raise ValueError("The extraction database is from another version; rebuild it.")
        fingerprint = info["fingerprint"]
        extensions = json.loads(info.get("extensions") or "[]")
        patient_count, items = int(info["patients"]), int(info["items"])
    if patient_count < 1:
        raise ValueError("The exploration database has no patients.")
    contract = agent_contract(config, extensions)
    texts = [requirement_text(r) for r in requirements]
    keys = [workup_sql_cache_key(r, population_context, database_fingerprint=fingerprint,
                                 contract=contract) for r in requirements]
    distinct = list(dict.fromkeys(keys))
    progress = progress_callback or (lambda done, total: None)
    results: dict[str, dict] = {}
    for key in distinct:
        saved = cache.get(key) if cache is not None else None
        if reusable_sql_entry(saved, contract):
            results[key] = {**copy.deepcopy(saved), "cached": True}
    todo = [key for key in distinct if key not in results]
    progress(len(distinct) - len(todo), len(distinct))
    model = None
    if todo:
        llm = llm or resolve_agent_llm(config)
        model = llm.model
        population_sha = digest((population_context or "").strip())
        first = {key: texts[keys.index(key)] for key in todo}
        box = (SQLSandbox(database=database, limits=limits.sandbox())
               if isinstance(database, ExtractionDatabase)
               else SQLSandbox(database_path=database, limits=limits.sandbox()))
        active: dict[str, int] = {}
        lock = threading.Lock()
        shown: list[dict] | None = None

        def on_turn(key):
            def record(action):
                with lock:
                    active[key] = action
            return record

        def report():
            nonlocal shown
            if activity_callback is None:
                return
            with lock:
                items = [{"name": first[key]["name"], "action": action,
                          "max_actions": limits.max_actions}
                         for key, action in sorted(active.items(),
                                                   key=lambda kv: (-kv[1], kv[0]))]
            if items != shown:
                shown = items
                activity_callback(copy.deepcopy(items))

        with box, ThreadPoolExecutor(max_workers=min(limits.max_concurrent_items,
                                                     len(todo))) as pool:
            try:
                profile = _profile(box, fingerprint, extensions)
            except SQLSandboxError:
                # The agent still works without it: no profile and no search.
                logger.warning("The exploration database's profile could not be computed.")
                profile = None
            system = render_system_prompt(
                limits, extensions,
                render_profile(profile) if profile else "(not available for this database)")
            futures = {submit_cancellable(pool, _compile_one, first[key], population_context,
                                          box, patient_count, llm, limits, system,
                                          on_turn(key), profile, extensions): key
                       for key in todo}
            pending = set(futures)
            try:
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    check_cancelled()
                    for future in done:
                        key = futures[future]
                        with lock:
                            active.pop(key, None)
                        outcome = future.result()
                        entry = {
                            "cache_key": key,
                            "requirement": first[key],
                            "population_context_sha256": population_sha,
                            **outcome,
                            "exploration": {"fingerprint": fingerprint,
                                            "patients": patient_count, "items": items},
                            "contract": contract,
                            "compiled_at": datetime.now(timezone.utc).isoformat(
                                timespec="seconds"),
                        }
                        if cache is not None:
                            cache[key] = copy.deepcopy(entry)
                        _log(entry)
                        results[key] = {**entry, "cached": False}
                        progress(len(results), len(distinct))
                    report()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
    entries = [{**copy.deepcopy(results[key]), "requirement": text}
               for key, text in zip(keys, texts, strict=True)]
    if not return_metadata:
        return entries
    fresh = [results[k] for k in todo]
    metadata = {
        "contract": contract,
        "model": model,
        "compiled_by": sorted({e.get("model") for e in results.values() if e.get("model")}),
        "requirements": len(requirements),
        "distinct": len(distinct),
        "cached": len(distinct) - len(todo),
        "compiled": sum(e["status"] == "compiled" for e in results.values()),
        "failed": sum(e["status"] == "failed" for e in results.values()),
        "requests": sum(e["usage"]["requests"] for e in fresh),
        "prompt_tokens": sum(e["usage"]["prompt_tokens"] for e in fresh),
        "completion_tokens": sum(e["usage"]["completion_tokens"] for e in fresh),
        "exploration": {"fingerprint": fingerprint, "patients": patient_count, "items": items},
    }
    return entries, metadata


__all__ = [
    "ACTIONS",
    "AGENT_VERSION",
    "RESULT_COLUMNS",
    "SQL_REASONS",
    "WorkupSQLLimits",
    "agent_contract",
    "agent_llm_identity",
    "agent_limits",
    "check_result",
    "compile_workup_sql",
    "render_system_prompt",
    "resolve_agent_llm",
    "response_schema",
    "reusable_sql_entry",
    "run_final_query",
    "workup_sql_cache_key",
]
