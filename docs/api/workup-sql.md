# SQL-Agent Workup Checks

!!! warning "Review aids, not determinations"
    Results say whether structured items extracted from the notes document a
    guideline workup step. An LLM agent wrote each query and an LLM extracted the
    items: check every cited quote against its note. *Not documented* means no
    extracted item matched, not that the step was not done. This does not establish
    overall guideline concordance.

This method reads the records of [Structured Clinical Extraction](extraction.md), like the compiled-query method
([Extraction-Based Workup Checks](workup-queries.md)), and gives results in the same shape. The difference is that
an LLM agent writes each requirement's check as SQL **after looking at the data**: which codes are used, how text
fields are spelled, how often things occur and how dates look.

It needs the optional DuckDB dependency (`pip install 'matchminer-ai[sql]'`) and Linux with libseccomp, which the
isolated SQL worker uses.

1. **Database.** `build_extraction_database` turns extraction records into
   relational tables (below), and `write_extraction_database` writes them as a DuckDB file.
2. **One query per requirement.** For each distinct workup item (name, conditions,
   category), `compile_workup_sql` runs a bounded agent over an *exploration
   database*, normally a whole extraction store.
    - Before it starts, code counts what the database holds (the **store
      profile**, below) and puts it in the agent's system prompt.
    - Each turn the agent returns one JSON action: `search`, `query`, `submit`,
      `accept` or `not_expressible`.
    - A submission must return one row per patient. Code runs it over every
      exploration patient, checks its output contract and shows the agent the
      distribution of results with example rows. The agent then accepts it or
      revises it.
    - The accepted query is cached.
3. **Run once per cohort.** `check_workup_with_sql` builds the cohort's database
   from its own records and runs each accepted query **once** over it, with no model calls. It reports
   in the shape of `check_workup_with_extractions`: status, reason, applicability,
   `bottom_line`, evidence quotes and limitations, with `metadata.method` `extraction_sql`.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.patients import (
    build_extraction_database, check_workup_with_sql, compile_workup_sql,
    write_extraction_database,
)
from matchminer_ai.trials import load_guideline_catalog

config = load_default_preset()
config.remote.update(enabled=True, server_urls=["http://localhost:8000/v1"])
config.workup_sql_agent["remote"]["model_name"] = "served/model"   # the agent's served model

store = build_extraction_database([document_records, patient_records])
write_extraction_database("store.duckdb", store, private=True)     # the exploration database

space = load_guideline_catalog("catalogs/nsclc").iloc[0]          # one guideline space
sql_cache = {}                                                     # any mapping; reuse it across calls
compiled = compile_workup_sql(
    space["diagnostic_workup"], database="store.duckdb", config=config,
    population_context=space["cohort_key_criteria"], cache=sql_cache,
)
report = check_workup_with_sql([cohort_documents, cohort_patients], compiled,
                               patient_ids=cohort_ids)
```

`compile_workup_sql` also accepts an in-memory `ExtractionDatabase` as `database`. `check_workup_with_sql` takes
`reference_dates` and `coverage` as `check_workup_with_extractions` does, and returns `patients`, `aggregate`,
`notice` and `queries` (how each query ran on this cohort: `ran` with its distribution, `failed`, or `not_run`).

For a cohort, `matchminer_ai.population.check_population_workup_with_sql` takes the cohort (with its space) and
the accepted queries. It links records to the cohort's own notes by note id, as
`check_population_workup_with_extractions` does. The report has `method: extraction_sql` and one `queries` entry
per item: the SQL, the summary, how the agent ended, and how the query ran on this cohort. See
[Population workup review](../user-guide/population-workup.md#extraction-based-checks).

## What the agent can do

The agent's only tool is read-only SQL, run in an isolated worker process
(`SQLSandbox`).

**What the worker accepts:**

- Exactly one statement of type SELECT: `SELECT`, `WITH`, `FROM`, `DESCRIBE`,
  `SUMMARIZE` or `SHOW`. No DDL, `COPY`, `ATTACH`, `SET`, `INSTALL` or
  multi-statement strings.
- Statements run with DuckDB's file and network access disabled, extension loading
  and installing off, spilling to disk off, and the configuration locked.

**How the worker is isolated:**

- It opens the database before any model SQL runs: a file read-only, or in-memory
  tables sent over a pipe and never written to disk.
- It then installs a seccomp allowlist with no `open`, `socket` or process creation, and may not write files or
  core dumps.
- An address-space limit, DuckDB's memory limit and a per-statement interrupt bound
  each statement. The worker is killed if it stops answering.
- Results are capped in rows and characters.

**What the agent cannot do:** it has no files, shell, network, Python or browsing.
Model-written SQL never runs in the calling process.

## Store profile and search

Two code-made aids (`matchminer_ai.patients.workup_sql.profile`) save the agent from rediscovering the same facts
for every requirement. Both run code-written SELECT statements in the same worker, over the same tables.

**Store profile** (`store_profile`, `render_profile`). Computed once per process and exploration database, it
gives:

- patients, notes with document-level records, and patients with an incomplete extraction;
- per item table: rows at each level, patients, planned items, items with an exact date stated in a note, and
  undated items;
- patients per code of every code column, and per document type (codes with no items are left out);
- patients per cancer site group, and how many of them have stage IV, M1 or
  distant extent documented. Code matches `primary_site` and `histology` text
  against fixed patterns (`SITE_GROUPS`), so the groups are approximate. The
  agent is told to use them only to judge plausibility, never in a final query.

**`search`** (`search_values`). The agent gives up to `max_search_terms` (8) words or word stems; a word of three
or fewer characters (CEA, CT, EGD) matches whole words only. Code returns:

- every code whose code or meaning contains one, with its patients;
- every distinct value of every text column that contains one, rare values
  included, with item and patient counts, most patients first. Values that
  differ only in case are merged. At most `max_search_rows` (50) are shown, and
  per-column totals show what was cut.

A search after a valid submission counts as a checking query.

The profile is a function of the database and of `PROFILE_VERSION`, which is in
the agent contract, so the cache key (which holds the database fingerprint)
covers it. Like exploration results, both reach the agent's endpoint: aggregate
counts and distinct extracted values, never patient ids, note text or quotes.

## The database

**Tables:**

- one per item type (`imaging_studies`, `procedures`, `diagnoses`, `biomarkers`, `treatments`,
  `disease_assessments`, `performance_status`, `lab_tests`, `counseling`, `events`, and any extension's), one row
  per item at either level;
- `items`, a view of the columns every item table shares;
- `patients` and `documents` (one row per note with a document-level record);
- `vocabulary`: every code of every code column, with its meaning and `code_group`;
- `extraction_coverage`: for each item type, how many patients' records were all extracted for it;
- `database_info` (in a written file): the database version, fingerprint, extensions and counts.

**Columns** come from the item type declarations (`table_columns`), never from the
data. A query written against one database therefore runs on any other database of
the same `DATABASE_VERSION` (`mmai-extraction-sql-v2`). `render_database_guide` writes them, with every column's
codes, for the agent's prompt.

**Every item table** starts with these shared columns:

- `item_key`, `patient_key`, `item_type`, `level`, `note_id`, `note_date`,
  `document_type` (the note columns are NULL for patient-level items), and `record_item_id` (a patient-level
  item's `D1`, `I2`, ...).
- `occurrence`: performed or planned.
- The primary date as `date_value`, `date_precision`, `date_source`, `date_bound`,
  `date_lo` and `date_hi`.
- A code-made `label`.

**Then each field** of the type:

- List fields become `VARCHAR[]`; unstated text becomes NULL.
- Each date field `x` becomes `x`, `x_precision`, `x_source`, `x_lo` and `x_hi`.
- Imaging studies, procedures, lab tests and counseling add `relation`, `source_of_document` and `status`; item
  types that link to cancers add `cancer_ids`; procedures add `procedure_group`; treatments add
  `agents_ncit_codes`.

**Date intervals** follow the compiled-query evaluator's rules:

- A month or year spans its whole period.
- An undated mentioned item was on or before its note (`date_lo` NULL).
- An undated planned item is on or after its note (`date_hi` NULL).

**`patients`** carries `reference_date`, which by default is the latest note date.
It also carries the extraction's completeness (`extraction_incomplete`, `notes_without_records`, failed and
partial document records) and `extracted_item_types`: the item types every one of the patient's records was
extracted for. Records of an older extraction schema lack some types; `mmai-clinical-extraction-v1` records have no
lab tests or counseling.

**Privacy:** the tables hold no patient ids (patients are numbered) and no note
text. Evidence quotes stay in Python and are attached to results by `item_key`.
A written database still holds patient-derived values: keep it where the extraction records may be kept
(`private=True` writes it 0600).

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
- a missing query, or one that fails or breaks its contract on the cohort, gives unknown / `query_unavailable`.

`aggregate_extraction_workup` counts the results, as for compiled queries.

## Rules the agent follows

The system prompt (`patient.workup_sql.system.txt`) holds the full rules. The main
ones:

- **Not expressible:** physical examination, history taking, organ function tests such as PFTs or ECG, and
  multidisciplinary or tumor board evaluation. So is a requirement whose item type the store's records were not
  extracted for (`extraction_coverage`).
- **Timing:** any time up to the reference date counts unless the requirement is
  defined by timing relative to an event. Phase labels such as "initial evaluation"
  set no window.
- **Undated anchors:** when the anchor event has no exact date, it is dated by the
  first note that mentions it, and a step dated before that note counts as before
  the event.
- **Labs:** a lab requirement is met only when every lab it names is documented.
  Tumor markers are biomarkers.
- **Unstated details:** a detail notes often leave out (contrast, a tracer, laterality) gives unknown /
  `fields_not_stated`, never not_met.
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

## Settings and endpoint

The `workup_sql_agent` preset section configures the agent; [Configuration](../reference/configuration.md) lists
every key.

- **Endpoint:** it needs a remote endpoint (`remote.enabled`): an OpenAI-compatible server such as a local vLLM
  server, Vertex or another provider, or Claude through a Databricks AI Gateway (below). It cannot drive an
  in-process engine.
- **Its own connection:** the section's `remote` may set its own `server_urls`, `provider`, `google_project_id`
  and `databricks_profile`, overriding `config.remote` for the agent alone. The other sections (such as
  extraction) keep the shared connection.
- **Model:** Nemotron 3.5 Super by default, with `reasoning_effort: high` and 32,000 completion tokens per turn.
- **Requests:** each turn is one structured request; an invalid action is retried up to `max_attempts` (3). A
  stalled request is retried after `request_timeout` (600 s) instead of the shared `remote.request_timeout`, and a
  busy endpoint (429 or 503) is waited out.
- **Context window:** `context_window`, else `local.engine.max_model_len`, else the endpoint's `/v1/models`.
  Prompt budgets count UTF-8 bytes (`tokenizer_mode: bytes`) unless `tokenizer_mode: endpoint` counts with
  `/tokenize`.

### Claude through a Databricks AI Gateway

Provider `databricks_anthropic` sends the agent's requests to Claude on a Databricks AI Gateway's Anthropic
Messages route.

- **Install:** `pip install 'matchminer-ai[databricks]'` (the `anthropic` and `databricks-sdk` packages).
- **Log in once:** `databricks auth login --host <workspace URL> --profile unity-gateway`. The Databricks SDK then
  gets and refreshes access tokens through the Databricks CLI, which must be on the process's `PATH`.
- **Connection settings:** in the section's `remote`, set:
    - `provider: databricks_anthropic`;
    - `server_urls: ["https://<workspace host>/ai-gateway/anthropic"]`;
    - `databricks_profile: unity-gateway`, or none to use a Databricks notebook's or job's identity, or a service
      principal from `ARM_*` variables;
    - `model_name`: the gateway endpoint's full three-part name (`<catalog>.<schema>.<endpoint>`, the last part
      ending in the Claude model id, such as `...-claude-sonnet-5-5`);
    - `request_params`: `max_tokens` and `reasoning_effort`, which becomes Claude's effort (`low`, `medium`,
      `high`, `xhigh` or `max`);
    - `extra_body: {}` (the preset's vLLM fields are not accepted).

- **Section settings:** `sampling_profile: none`, `context_window` (the model's context window, such as 1,000,000
  tokens) and `tokenizer_mode: bytes`, because the gateway has no `/models` or tokenizer route.
- **What is sent:** requests go through the official Anthropic SDK to the Messages
  route, with structured outputs (`output_config.format`).
    - **Schema limits:** the SDK moves limits Claude does not enforce, such as
      `maxItems`, into descriptions. The harness still validates them in code.
    - **Prompt caching:** the instructions are a cached system block. The turn's response
      schema follows in a second block, because its allowed actions change from turn
      to turn.
    - **Refusal fallbacks:** Opus and Sonnet 5.5 requests opt into server-side
      fallbacks (`fallbacks: "default"`), so a declined turn is answered by another
      Claude model.
- **Credentials:** the token goes only to an `https` gateway URL on a Databricks workspace host, which must be the
  profile's workspace. A missing or expired login fails the requirement's run at once with a login hint. Other
  failures keep only their HTTP status.
- **Scope:** only structured requests support this provider; other stages reject it.

The [guideline fetcher](../apps/guideline-fetcher.md) offers this provider for its SQL-agent task, with the
gateway taken from the site config's `databricks` keys ([Site configuration](../reference/site-config.md)).

## Budget, ending and caching

**`limits`:**

- `max_actions` (12) and `max_submits` (3) bound the run.
- The agent is asked to submit a first candidate by `submit_by` (6), so that it
  reviews results rather than spending the budget exploring.
- After a valid submission it may run at most `max_review_queries` (3) checking
  queries or searches. Code then removes the query action: the agent accepts the submission,
  or submits a revision that fixes a defect it found, which allows 3 more.
- `max_rows`, `max_cell_chars` and `max_result_chars` bound what it sees of each
  result; `max_search_terms` and `max_search_rows` bound a search.
- `max_recent_steps` sets how many recent steps are replayed with their results.
  Earlier ones are listed by outcome, and the agent's own `memory` carries the rest.
- Up to `max_concurrent_items` (8) requirements are explored at once, sharing one
  worker; the agent's request cap is raised to match. `activity_callback` reports
  which requirements are being explored and the turn each is on.
- `query_timeout_seconds` (20) and `final_timeout_seconds` (120) bound exploration and final queries;
  `worker_memory_mb` and `duckdb_memory_mb` bound the worker.

**Ending a run:**

- The agent accepts a valid submission, or declares the requirement not expressible.
- A valid submission on the last turn is accepted, with a limitation saying so.
- When the budget or the context window runs out, the pending valid submission is used or, when a later revision
  was not valid, the last valid one, with limitations saying so.
- If no valid submission is made, the entry fails and its results are unknown.

**Entries:** one per requirement, in order (duplicates share one): `cache_key`, `requirement`, `status`
(`compiled` or `failed`), `expressible`, `sql`, `summary`, `approximations`, `recommends_against`, `accepted_by`,
`termination`, `steps` (SQL and outcomes, no results), `distribution`, `error`, `usage`, `diagnostics`, `model`,
`contract` and `cached`. `return_metadata=True` adds counts of distinct, cached, compiled and failed requirements,
requests and tokens.

**Caching:** distinct requirements are explored once per call. Pass `cache` (any
mapping) to reuse accepted queries. The key (`workup_sql_cache_key`) covers:

- the requirement text and the population context;
- the exploration database's fingerprint;
- the agent, database, store-profile and extraction-schema versions and the extensions;
- the prompt and response-schema hashes;
- the provider, model and reasoning effort that write the queries
  (`agent_llm_identity`), so each model's and effort's queries are cached apart.

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
        - store_profile
        - search_values
