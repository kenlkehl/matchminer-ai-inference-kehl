# Dashboard

`matchminer_ai.apps.dashboard` is a per-disease-center trials dashboard over a site's extracted trial cohorts, with
patient matching over a synthetic training run's patients. It is a FastAPI server with a React frontend (source
in `frontend/dashboard/`, built into the package's `static/` folder). Research use: patients are synthetic, and
scores are screening signals for review, not eligibility determinations.

## What it shows

- **Trials overview per disease center:** an accordion of trial groups (rules in a grouping file), one row per
  cohort with its PI, key inclusion and exclusion criteria, the trial's drugs and mechanisms (from a GoodOption
  catalog), contacts and **Find patients**. Trial details show the summary, drugs, contacts, sites and both
  versions' cohorts (protocol and ClinicalTrials.gov).
- **Find patients:** the synthetic patients nearest a cohort's key criteria in TrialSpace, scored by the decision
  model as the page opens; the per-criterion and GoodOption LLM checks run on demand.
- **AI search:** trials for a patient summary, with the same scores.
- **Status:** the data sources, the TrialSpace round, the decision model and the background jobs.

## Run

```bash
pip install "matchminer-ai[apps]"
cd frontend/dashboard && pnpm install && pnpm build   # once, and after frontend changes
mmai-dashboard --site-config site.yaml --device cuda:0
```

`python -m matchminer_ai.apps.dashboard` is the same command; `mmai-dashboard build` builds the store once and
exits. The server binds `127.0.0.1:8781` only (it shows confidential protocol content); forward the port with
`ssh -L` to reach it from another machine. It accepts only loopback Host headers (`--allowed-hosts` adds names)
and refuses cross-origin writes. `--require-token` also requires an access token (`MMAI_DASH_ACCESS_TOKEN`, else
a private `~/.config/mmai-dashboard/access_token` created on first use), which keeps out other accounts on the
same machine; the server prints a one-click sign-in link to its terminal. `pnpm dev` serves the frontend on :5173
and proxies `/api` to a running server.

## Inputs

Paths come from the command line, `MMAI_DASH_*` variables, the site config's `dashboard` section, or folders under
the data root (`--data-root`, `MMAI_DATA_ROOT` or `paths.data_root`):

| Setting | Default under the data root | Contents |
| --- | --- | --- |
| `--trial-data` (`dashboard.trial_data`) | `phi/real_time/trial_data` | The trial extraction run (`inputs.csv`, `protocol/` and `ctgov/` with per-trial results and `cohorts.parquet`); new results appear within a poll |
| `--oa-db` (`dashboard.oa_db`) | `phi/real_time/oa_db_*.duckdb` | The DuckDB copy holding the trial table (the newest file of a glob) |
| `--store-dir` (`dashboard.store_dir`) | `phi/real_time/dashboard` | The store, vectors and caches (owner-only) |
| `--good-options-catalog` (`dashboard.good_options_catalog`) | `phi/real_time/good_options_catalog/catalog` | Trials' drugs and mechanism classes; missing: no drugs |
| `--grouping` (`dashboard.grouping`) | the bundled example rules | Trial-group rules per center |
| `--synthetic-run` (`dashboard.synthetic_run`) | `no_phi/synthetic_run` | Patients, TrialSpace rounds and the decision model |

The trial table's name and columns, the site names shown, the home institution's center prefix, branding and
protocol links are site-config keys (see [Site configuration](../reference/site-config.md)). The LLM checks use a Vertex
profile (default Gemini 4 Hydrogen, in the site's `vertex.project`) or an OpenAI-compatible endpoint (default the
site's `endpoints.teacher.url`); they send the patient summary and the cohort's criteria to that endpoint, so use
one authorized for the data.
