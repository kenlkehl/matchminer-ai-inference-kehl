# Config loading

Three functions return an `MMAIConfig`:

- `load_preset("default")` loads the preset packaged with `matchminer-ai` (`presets/default.yaml`, the only one).
- `load_default_preset()` is the same as `load_preset("default")`.
- `load_config(path)` loads your own YAML file with the same layout.

Every function that takes `config=` uses the default preset when it is omitted. Change settings in Python on the
returned object, or copy the preset into your own file. The sections and their keys are described in the
[configuration reference](../reference/configuration.md).

## Sections

`MMAIConfig` holds the most used sections as dictionary attributes. The others are read from `config.raw`, the parsed
YAML. The attributes are separate copies of their sections, and the package reads them, so edit a section through
its attribute when it has one.

| Access | Sections |
| --- | --- |
| Attributes | `remote`, `trial`, `patient`, `embedding`, `protocol_cohort_extraction`, `trial_space_criteria_extraction`, `synthetic_protocol_generation`, `llm_match_quality`, `llm_exclusion_criteria`, `llm_good_option`, `good_option_catalog`, `guideline`, `document_extraction`, `patient_extraction`, `workup_query_compile`, `workup_sql_agent` |
| `config.raw` | `decision_model`, `match_quality`, `exclusion_criteria`, `criteria_check`, `llm_criteria_check`, `good_option_prompt`, `good_option_check`, `good_option_checker` |

```python
from matchminer_ai import load_preset

config = load_preset("default")
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8000/v1"]
config.embedding["model_path"] = "/path/to/trialspace"
config.raw["decision_model"]["model_name"] = "/path/to/decision_model"   # a fine-tuned checkpoint
```

`debug_mode` and `model_metadata_cache_dir` are attributes too.

::: matchminer_ai.config
    options:
      members:
        - load_config
        - load_default_preset
        - load_preset
        - MMAIConfig
