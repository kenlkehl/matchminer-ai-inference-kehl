# Guideline fetcher

`matchminer_ai.apps.guideline_fetcher` is a local web workspace that turns a patient summary into
source-preserving guideline considerations, with patient evaluation and population management views. It is a
FastAPI backend with a plain HTML/CSS/JavaScript frontend (no build step, CDN, analytics or browser storage).
Install it with the `apps` extra (add `databricks` for Claude in the SQL agent):

```bash
uv pip install -e ".[apps]"
```

## Two modes

The mode is chosen only by how the app is started, never by the environment:

- **Default** (`mmai-guideline-fetcher`): the non-PHI app, for synthetic data or deployments authorized for your
  records. Population management samples the configured synthetic Parquet files, and model metadata and guideline
  embeddings are cached under the data root's `no_phi/guideline_fetcher_cache`. It never opens a population DuckDB
  database and never reads or writes a `phi` folder. It binds `0.0.0.0:8034` by default (`--host`, `--port`).
- **PHI mode** (`mmai-guideline-fetcher --phi`): the deployment for real patient records. It loads populations from
  a population DuckDB database or a preprocessed population cache, listens only on loopback (`127.0.0.1:8044`; a
  non-loopback `--host` is refused), accepts only loopback Host headers, requires an access token on every page and
  API route except sign-in, and writes only its own `phi/guideline_fetcher_phi_cache`, never a `no_phi` folder. The
  token is `GUIDELINE_ACCESS_TOKEN`, else a private file (`GUIDELINE_ACCESS_TOKEN_FILE`, default
  `~/.config/guideline-fetcher-phi/access_token`) created on first use. Reach the app from another machine with
  `ssh -L`.

`python -m matchminer_ai.apps.guideline_fetcher` is the same command. Notes and summaries reach the LLM endpoints
chosen in the model picker, so in either mode use only endpoints authorized for the data.

## Workflow

1. Paste a summary, or import or paste notes and create a patient summary.
2. Review the summary and choose a guideline catalog (or all completed catalogs).
3. **Find guideline spaces** embeds the summary with TrialSpace, retrieves candidate populations and ranks them
   with the decision model's match questions.
4. Review workup and treatment menus, their conditions, key criteria, source excerpts and scores, and export the
   Markdown report.
5. Review a space's workup in the raw notes (full-note question answering or agentic note search), with structured
   extraction records, or with the SQL agent over extraction records (`extraction_sql`).

The model picker sets one LLM connection per task: patient summarization, note compression, guideline concordance
checks, the SQL agent (which falls back to the concordance connection) and the two backup reviews. The default backend
is the Vertex profile `gemini4_vertex` (`MMAI_LLM_BACKEND`); a local OpenAI-compatible server (`MMAI_ENDPOINT`,
default `http://localhost:8000/v1`) is also available. Claude through a Databricks AI Gateway (`claude_databricks`) is
offered for the SQL agent only.

## Data root and site config

Default paths sit under the data root (`--data-root`, `MMAI_DATA_ROOT` or the site config's `paths.data_root`),
which has no default. Institution-specific values come from the site config (`--site-config` or
`MMAI_SITE_CONFIG`); see [Site configuration](../reference/site-config.md).

| Default (under the data root) | Purpose |
| --- | --- |
| `no_phi/space_paradigms_nccn` | Finalized guideline catalogs (`GUIDELINE_CATALOG_ROOT`) |
| `no_phi/guideline_fetcher_cache`, `phi/guideline_fetcher_phi_cache` | Each mode's cache (`GUIDELINE_CACHE`, which must stay on its mode's side) |
| `no_phi/patient_summaries.parquet`, `no_phi/all_synthetic_notes.parquet` | Synthetic population (`GUIDELINE_POPULATION_SUMMARIES`, `GUIDELINE_POPULATION_NOTES`; site `guideline_fetcher.population_summaries`, `.population_notes`) |
| `no_phi/guideline_fetcher_extractions/synthetic` | Non-PHI extraction records (`GUIDELINE_EXTRACTION_RECORDS`) |
| `phi/guideline_fetcher_population.duckdb` | PHI mode's population database (`GUIDELINE_POPULATION_DUCKDB`; site `guideline_fetcher.population_duckdb`); its preprocessed cache and extraction records sit beside it |

Other settings are environment variables: `GUIDELINE_ALLOWED_HOSTS`, `GUIDELINE_DUCKDB_ROOTS` (folders PHI mode may
browse; site `guideline_fetcher.duckdb_roots`), `GUIDELINE_POPULATION_CACHE`, `MMAI_LLM_MODEL`, `MMAI_TOKENIZER`,
`MMAI_DEVICE`, `GUIDELINE_DATABRICKS_HOST`, `GUIDELINE_DATABRICKS_PROFILE`, `GUIDELINE_DATABRICKS_SONNET_ENDPOINT`
(else the site config's `databricks.*`) and `OPENAI_API_KEY`.

A population DuckDB database has a `patient_summary` table (`mrn`, `patient_summary`, `patient_boilerplate`,
`is_current`) and a `patient_notes` table (`id`, `mrn`, `rpt_text`, `narrative_text`, `event_date`,
`note_source`).

## Offline tools

- `python -m matchminer_ai.apps.guideline_fetcher.preprocess_population` builds a PHI population cache (summaries
  with precomputed TrialSpace embeddings, created 0700 with 0600 files) from a population database.
- `python -m matchminer_ai.apps.guideline_fetcher.preprocess_extractions` builds structured extraction records
  (`--phi` for a PHI population cache). Every note goes to the LLM endpoint given, so use one authorized for the notes.
- `examples/guideline_fetcher/workup_sql_spot_check.py` spot-checks SQL-agent workup answers;
  `benchmark_workup_methods.py` and `benchmark_biliary_workup.py` compare workup methods on synthetic notes.

The app is a research prioritization aid; its considerations need clinician review against the current guideline.
