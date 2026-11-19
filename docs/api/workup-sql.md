# SQL-Agent Workup Checks

!!! warning "Review aids, not determinations"
    Results say whether structured items extracted from the notes document a
    guideline workup step. An LLM agent wrote each query and an LLM extracted the
    items: check every cited quote against its note. *Not documented* means no
    extracted item matched, not that the step was not done. This does not establish
    overall guideline concordance.

This is a third way to check guideline workup, beside the note-reading reviews and
the compiled-query method ([Extraction-Based Workup Checks](workup-queries.md)). Like
the compiled-query method, it reads the records of
[Structured Clinical Extraction](extraction.md). The difference is that an LLM agent
writes each requirement's check as SQL **after looking at the data**: which codes
are used, how text fields are spelled, how often things occur and how dates look.

1. **Database.** `build_extraction_database` turns extraction records into
   relational tables: one table per item type, `patients`, `documents`, an `items`
   view, and `vocabulary`. `write_extraction_database` writes them as a DuckDB
   file.
2. **One query per requirement.** For each distinct workup item (name, conditions,
   category), `compile_workup_sql` runs a bounded agent over an *exploration
   database*, normally a whole extraction store.
   - Each turn the agent returns one JSON action: `query`, `submit`, `accept` or
     `not_expressible`.
   - A submission must return one row per patient. Code runs it over every
     exploration patient, checks its output contract and shows the agent the
     distribution of results with example rows. The agent then accepts it or
     revises it.
   - The accepted query is cached.
3. **Run once per cohort.** `check_workup_with_sql` builds the cohort's database
   from its own records and runs each accepted query **once** over it. It reports
   in the shape of `check_workup_with_extractions`: status, reason, applicability,
   `bottom_line`, evidence quotes and limitations. The report's
   `metadata.method` is `extraction_sql`.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.patients import (
    build_extraction_database, check_workup_with_sql, compile_workup_sql,
    write_extraction_database,
)

config = load_default_preset()
config.remote.update(enabled=True, server_urls=["http://localhost:8000/v1"])
config.workup_sql_agent["remote"]["model_name"] = "served/model"

store = build_extraction_database([document_records, patient_records])
write_extraction_database("store.duckdb", store)            # the exploration database

compiled = compile_workup_sql(
    space["diagnostic_workup"], database="store.duckdb", config=config,
    population_context=space["cohort_key_criteria"], cache=my_cache,
)
report = check_workup_with_sql([cohort_documents, cohort_patients], compiled,
                               patient_ids=cohort_ids)
```

For a population, `matchminer_ai.population.check_population_workup_with_sql` takes a
cohort (with its space) and the accepted queries. It links records to the cohort's
own notes by note id, as `check_population_workup_with_extractions` does. The report
has `method: extraction_sql` and one `queries` entry per item: the SQL, the summary,
how the agent ended, and how the query ran on this cohort.

## What the agent can do

The agent's only tool is read-only SQL, run in an isolated worker process
(`SQLSandbox`).

**What the worker accepts:**
- Exactly one statement of type SELECT: `SELECT`, `WITH`, `FROM`, `DESCRIBE`,
  `SUMMARIZE` or `SHOW`. No DDL, `COPY`, `ATTACH`, `SET`, `INSTALL` or
  multi-statement strings.
- Statements run with DuckDB's file and network access disabled, extension loading
  and installing off, and the configuration locked.

**How the worker is isolated:**
- It opens the database before any model SQL runs: a file read-only, or in-memory
  tables sent over a pipe and never written to disk.
- It then installs a seccomp allowlist with no `open`, `socket` or process creation.
- An address-space limit, DuckDB's memory limit and a per-statement interrupt bound
  each statement. The worker is killed if it stops answering.
- Results are capped in rows and characters.

**What the agent cannot do:** it has no files, shell, network, Python or browsing.
Model-written SQL never runs in the calling process.

## The database

**Columns** come from the item type declarations (`table_columns`), never from the
data. A query written against one database therefore runs on any other database of
the same `DATABASE_VERSION` (`mmai-extraction-sql-v2`).

**Every item table** starts with these shared columns:
- `item_key`, `patient_key`, `item_type`, `level`, `note_id`, `note_date`,
  `document_type`.
- `occurrence`: performed or planned.
- The primary date as `date_value`, `date_precision`, `date_source`, `date_bound`,
  `date_lo` and `date_hi`.
- A code-made `label`.

**Then each field** of the type:
- List fields become `VARCHAR[]`.
- Unstated text becomes NULL.
- Each date field `x` also gets `x_lo` and `x_hi`.

**Date intervals** follow the compiled-query evaluator's rules:
- A month or year spans its whole period.
- An undated mentioned item was on or before its note (`date_lo` NULL).
- An undated planned item is on or after its note.

**`patients`** carries `reference_date`, which by default is the latest note date.
It also carries the extraction's completeness (`extraction_incomplete`,
`notes_without_records`) and `extracted_item_types`: the item types every one of the
patient's records was extracted for. Records of an older extraction schema lack some
types; v1 records have no lab tests or counseling. `extraction_coverage` counts the
covered patients per item type.

**Privacy:** the tables hold no patient ids (patients are numbered) and no note
text. Evidence quotes stay in Python and are attached to results by `item_key`.

## Results and overrides

**The final query** returns these columns:
- `patient_key`
- `status` and `reason`, one of the pairs `met/documented`,
  `not_met/not_documented`, `not_met/planned_only`, `unknown/uncertain_dates`,
  `unknown/fields_not_stated`, `unknown/anchor_unavailable`,
  `not_applicable/documented_not_applicable` or, only for a requirement that
  advises against its step, `done_not_recommended/documented_not_recommended`
- `applicability`
- `evidence_item_keys`
- `detail`

**Code checks** that the query returns one row per patient, that the values are
valid, and that every cited key is one of that patient's items.

**Steps the guideline advises against** ("FDG-PET/CT not routinely indicated"). The
agent submits these with `recommends_against: true`:
- A patient who had the step anyway is `done_not_recommended`, never `met`.
- The aggregate reports their count and no completion fraction (`recommends_against`
  on the row).

**Code then applies the compiled-query method's overrides:**
- no extraction records gives unknown / `no_extraction`;
- an incomplete extraction turns not_met into unknown / `incomplete_extraction`;
- so does a query that reads an item table the patient's records were never extracted
  for (for example `lab_tests` on v1 records): an empty table there says nothing;
- a requirement the agent declared not expressible gives unknown / `not_expressible`;
- a missing or failed query gives unknown / `query_unavailable`.

`aggregate_extraction_workup` counts the results, as for compiled queries.

## Rules the agent follows

The system prompt (`patient.workup_sql.system.txt`) holds the full rules. The main
ones:
- **Timing:** any time up to the reference date counts unless the requirement is
  defined by timing relative to an event. Phase labels such as "initial evaluation"
  set no window.
- **Undated anchors:** when the anchor event has no exact date, it is dated by the
  first note that mentions it, and a step dated before that note counts as before
  the event.
- **Labs:** a lab requirement is met only when every lab it names is documented.
  Tumor markers are biomarkers.
- **History and physical** is always done and is ignored inside combined
  requirements.
- **Population:** neither the population nor its site is re-checked. Items about an
  unrelated cancer are excluded through `cancer_ids`, never through lists of site
  words.

## Privacy

**What reaches the endpoint:** exploration results are patient-derived extracted
values. They are never patient ids, note text or evidence quotes. They reach the
configured LLM endpoint. Use only an endpoint authorized for the records the
exploration database was built from, the same rule as for note-reading reviews.

**What reaches no LLM:** evaluating a cohort makes no model calls.

**What the cache keeps:** the SQL, the summary, approximations, the exploration
steps' SQL and outcomes, and aggregate counts (`distribution`). It never keeps
query results.

## Settings, budget and caching

**Section and endpoint:** the `workup_sql_agent` preset section configures the agent.
- It needs an OpenAI-compatible endpoint (`remote.enabled`): a local vLLM server,
  Vertex or another provider. It cannot drive an in-process engine.
- The context window comes from `context_window`, else `local.engine.max_model_len`,
  else the endpoint's `/v1/models`.

**`limits`:**
- `max_actions` (12) and `max_submits` (3) bound the run.
- The agent is asked to submit a first candidate by `submit_by` (6), so that it
  reviews results rather than spending the budget exploring.
- After a valid submission it may run at most `max_review_queries` (3) checking
  queries. Code then removes the query action: the agent accepts the submission,
  or submits a revision that fixes a defect it found, which allows 3 more.
- `max_rows`, `max_cell_chars` and `max_result_chars` bound what it sees of each
  result.
- `max_recent_steps` sets how many recent steps are replayed with their results.
  Earlier ones are listed by outcome, and the agent's own `memory` carries the rest.
- Up to `max_concurrent_items` (8) requirements are explored at once, sharing one
  worker; the agent's request cap is raised to match. `activity_callback` reports
  which requirements are being explored and the turn each is on.
- A stalled request is retried after `request_timeout` (600 s) instead of the
  shared `remote.request_timeout`.

**Ending a run:**
- A valid submission on the last turn is accepted, with a limitation saying so.
- When the budget (or the context window) runs out after a valid submission whose
  revision was not valid, the last valid submission is used, with limitations
  saying so.
- If no valid submission is made, the entry fails and its results are unknown.

**Caching:** distinct requirements are explored once per call. Pass `cache` (any
mapping) to reuse accepted queries. The key (`workup_sql_cache_key`) covers:
- the requirement text and the population context;
- the exploration database's fingerprint;
- the agent, database and extraction-schema versions;
- the prompt and response-schema hashes.

**Failed runs** are saved under their key too, but never reused: the next call
explores the requirement again and overwrites the entry. A saved failure keeps
what is needed to diagnose it:
- its steps (SQL and outcomes; an invalid submission names its problems, such
  as `invalid: patients without a row`, never values or error text);
- `error`, which says how the run ended;
- `diagnostics`: counts of the harness's own messages for invalid answers, of
  finish reasons (`length` means cut off at the token limit), and of transport
  failures by HTTP status. Never provider text, prompts or results.

Each explored requirement is also logged in one line (requirement name, how the
run ended, actions, seconds, requests and prompt tokens): INFO for an accepted
query, WARNING for a failure.

::: matchminer_ai.patients.workup_sql
    options:
      members:
        - build_extraction_database
        - write_extraction_database
        - compile_workup_sql
        - check_workup_with_sql
        - SQLSandbox
        - WorkupSQLLimits
        - workup_sql_cache_key
        - render_database_guide
