# Site configuration

Values that belong to one installation (endpoint hosts, cloud projects, database layouts, site names, intranet
links, local paths) are never committed to this repository. They live in a YAML site config named by
`--site-config` or `MMAI_SITE_CONFIG`, read with `matchminer_ai.site.load_site_config()`. Data paths sit under the
data root (`--data-root`, `MMAI_DATA_ROOT` or `paths.data_root`), which has no default; a data root inside the
source checkout is refused.

Keys the apps read (placeholder values):

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

Without a site config, site-only features are unavailable (for example, the Vertex profiles have no project) and
paths fall back to folders under the data root.
