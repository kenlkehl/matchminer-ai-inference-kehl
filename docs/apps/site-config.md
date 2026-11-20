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

guideline_fetcher:
  population_duckdb: /srv/mmai/data/phi/population.duckdb
  population_summaries: /srv/mmai/data/no_phi/patient_summaries.parquet
  population_notes: /srv/mmai/data/no_phi/all_synthetic_notes.parquet
  duckdb_roots: [/srv/mmai/data]       # folders PHI mode may browse for DuckDB files
```

Without a site config, site-only features are unavailable (for example, the Vertex profiles have no project) and
paths fall back to folders under the data root.
