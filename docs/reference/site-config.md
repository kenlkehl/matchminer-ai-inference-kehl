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

## Example

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
