# Site configuration

Values that belong to one installation (endpoints, cloud projects, local paths, institution-specific schema maps) live
in a site config, a YAML file outside the repository named by `--site-config` or `MMAI_SITE_CONFIG`
(`matchminer_ai.site`). Nothing in the repository depends on a particular site; without a site config, site-only
features are off or their options must be given explicitly. Code reads it with dotted keys
(`load_site_config().get("endpoints.teacher.url")`).

The **data root** holds `phi/` (real clinical data) and `no_phi/` (synthetic and public data). It comes from
`--data-root`, `MMAI_DATA_ROOT` or `paths.data_root`, has no default, and may not be inside the source checkout.

## Training and evaluation keys

| Key | Used by | Meaning |
| --- | --- | --- |
| `paths.data_root` | every training and evaluation stage | The data root, when neither `--data-root` nor `MMAI_DATA_ROOT` is set |
| `paths.models_root` | `training.from_summaries` | Where trained models go (default `<data root>/../models`) |
| `paths.hf_cache` | `evaluation.retro_fidelity` | Hugging Face cache for the teacher tokenizer |
| `paths.embeddinggemma2_cache` | `evaluation.retro_fidelity` | A cache holding the pinned EmbeddingGemma 2 snapshot (default: the Hugging Face cache) |
| `paths.retro_source_root` | `evaluation.retro_cohorts` | The institution's structured EHR extract (`structured_data/`, `derived_data/`) |
| `paths.phi_roots` | `evaluation.retro_cohorts.build_note_level_dataset` | Extra directories (besides `<data root>/phi`) where note text may be written |
| `endpoints.teacher.url` | `evaluation.retro_fidelity`, `evaluation.retro_cohorts.summarize_retro_cohorts` | The on-prem teacher server (OpenAI-compatible `/v1`), approved for PHI |
| `endpoints.secondary_teacher.url` | `evaluation.retro_fidelity` | An optional second teacher (the FP8 build), reached through `teacher_proxy` |
| `vertex.project` | `evaluation.dual_judge.judge_worker` | The Vertex AI project of the Claude judge |

## App keys

The [dashboard](../apps/dashboard.md) and the [guideline fetcher](../apps/guideline-fetcher.md) also read
`paths.data_root`, `endpoints.teacher.url`, `vertex.project` (the Vertex Agent Platform profiles) and the Databricks
gateway of the SQL agent, plus their own sections. Without a site config the Vertex profiles have no project and the
apps' paths fall back to folders under the data root. Placeholder values:

```yaml
paths:
  data_root: /srv/mmai/data            # holds phi/ and no_phi/

vertex:
  project: my-gcp-project              # Google Cloud project of the Vertex (Agent Platform) profiles

databricks:
  host: https://workspace.example.cloud.databricks.com   # gateway serving Claude for the SQL agent
  profile: unity-gateway               # ~/.databrickscfg profile
  sonnet_endpoint: ai_gateway_model_services_prod.team_project.team_project-claude-sonnet-5-5

endpoints:
  teacher:
    url: http://gpu-server.example:8000/v1   # the on-prem OpenAI-compatible server offered first

dashboard:
  synthetic_run: /srv/mmai/data/no_phi/synthetic_run
  trial_data: /srv/mmai/data/phi/real_time/trial_data
  oa_db: /srv/mmai/data/phi/real_time/trials_*.duckdb
  store_dir: /srv/mmai/data/phi/real_time/dashboard
  good_options_catalog: /srv/mmai/data/phi/real_time/good_options_catalog/catalog
  grouping: /srv/mmai/site/grouping.yaml
  embedding_python: /opt/envs/encoder/bin/python   # default: the server's own Python
  embedding_caches: [/srv/models/hf]               # default: the standard Hugging Face cache
  trial_table:
    name: trials                       # the DuckDB copy's trial table
    columns:                           # dashboard field: column (unlisted fields use their own name)
      management_group: managing_group
      site_pi: local_pi
  site_names:                          # site names in the table's site_list and the short names shown
    Example Cancer Center: ECC Main
  center_prefix: '^ECC/\s*'           # stripped from center names in the sidebar
  higher_threshold_groups: [ECC Phase I]
  branding:
    institution: Example Cancer Center
    confidentiality: The information shown is confidential.
    pi_label: ECC PI
    institutions:                      # sidebar groups of the disease centers, by name prefix
      - {label: ECC, prefixes: [ECC/]}
    trial_links:                       # links from a trial's protocol number
      - {label: Portal, url: 'https://portal.example.org/protocols/{protocol_no}'}

guideline_fetcher:
  population_duckdb: /srv/mmai/data/phi/population.duckdb
  population_summaries: /srv/mmai/data/no_phi/patient_summaries.parquet
  population_notes: /srv/mmai/data/no_phi/all_synthetic_notes.parquet
  duckdb_roots: [/srv/mmai/data]       # folders PHI mode may browse for DuckDB files
```

## Training and evaluation example

```yaml
paths:
  data_root: /srv/mmai/data
  models_root: /srv/mmai/models
  hf_cache: /srv/models/hf
  retro_source_root: /srv/ehr/extract_2026
endpoints:
  teacher:
    url: http://teacher.internal.example:8000/v1
  secondary_teacher:
    url: http://teacher2.internal.example:8002/v1
vertex:
  project: my-vertex-project
```
