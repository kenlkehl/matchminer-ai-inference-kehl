# Good Option Catalog and Scoring

Good Option scoring asks whether a trial's experimental drugs have evidence behind them for one patient. The score
ranks patient-trial pairs for human review; it is not an eligibility determination, a response probability, or a
treatment recommendation. It has two separate, versioned stages: a patient-free catalog of drug and drug-class
evidence, built once from a list of trials, and patient scoring against that catalog.

```python
from matchminer_ai.trials import build_good_option_catalog
from matchminer_ai.matching import check_good_options

# Patient-free research stage; run once before patient scoring.
catalog = await build_good_option_catalog(nct_ids, "drug_catalog", config=config)

# candidate_pairs: patient_id, trial_id, cancer_history_summary
scores = check_good_options(candidate_pairs, catalog="drug_catalog", config=config)
```

`check_good_options` is the supported entry point for scoring; see [On-demand check](#on-demand-check).
[How the Good Options pipeline works](../user-guide/good-options-pipeline.md) walks through every stage in order.

**Privacy boundary.** Catalog construction receives NCT IDs and drug names only, and its searches never receive
patient text, so it may run against a shared endpoint. Scoring sends the patient's cancer-history summary to the
configured `llm_good_option` endpoint; use an endpoint authorized for the sensitivity of that input.

Catalog code is in `trials/drug_catalog.py` (build, validation, loading), `trials/drug_research.py` (evidence
sources) and `trials/drug_evidence.py` (data models); scoring is in `matching/good_options.py`. The
`matchminer_ai.good_options` namespace re-exports the same names.

## Prompt resources

All templates are bundled under `src/matchminer_ai/prompts/`:

| Files | Purpose |
| --- | --- |
| `trial_drug_screen.system.txt`, `.user.txt`, `.retry.txt` | Identify anticancer agents and classify their roles in each trial. |
| `trial_drug_class.system.txt`, `.user.txt`, `.retry.txt` | Assign each drug to pharmacologic classes. |
| `trial_drug_synthesis.system.txt`, `.delta.txt`, `.delta_retry.txt` | Synthesize passage-grounded drug and class evidence one chunk at a time, returning only new or revised facts. |
| `llm_good_option.system.txt`, `.user.txt`, `.rubric.txt`, `.retry.txt` | Apply the four-criterion patient-drug rubric and correct invalid responses. |
| `good_option_checker_template.txt` | Patient-drug input for the deprecated GoodOptionChecker classifier. |

The catalog-build templates are listed in `CATALOG_PROMPTS`; `catalog_prompts_sha256()` fingerprints their text for
checkpoint reuse and the catalog manifest.

## Build the catalog

`build_good_option_catalog(nct_ids, output_path, ...)` is a coroutine. It fetches each trial's current
ClinicalTrials.gov record, screens its interventions, researches every unique drug and drug class, synthesizes the
evidence, and writes and validates a Parquet bundle at `output_path`, returning the loaded `GoodOptionCatalog`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `nct_ids` | required | NCT IDs, normalized and deduplicated in order. |
| `output_path` | required | Catalog directory. An existing one raises `FileExistsError` unless `overwrite=True`. |
| `config` | default preset | Uses `llm_good_option` and `good_option_catalog`. |
| `settings` | `ResearchSettings()` | Source caps, facets, axes, retries and timeouts for research. |
| `sources` | `default_sources()` | Evidence adapters implementing `DrugEvidenceSource`. |
| `web_provider` | `DDGSWebProvider()` | General-web search provider (`GeneralWebProvider`). |
| `role_resolver`, `synthesizer`, `class_resolver` | LLM stages | Callables that replace intervention screening, evidence synthesis, or class assignment. |
| `checkpoint_path` | `<output>_checkpoints` | Checkpoint directory; must not contain or sit inside the output. |
| `reset_checkpoint` | `False` | Discard a recognized checkpoint directory before starting. |
| `overwrite` | `False` | Replace an existing catalog. |
| `progress_callback` | `None` | Called as `(stage, completed, total, label)`; stages are `ctgov`, `screening`, `research`, `classify`, `class_research`, `class_synthesis`, `synthesis`. |
| `subject_concurrency` | 1 | Drugs, then classes, researched at once. |
| `web_search_concurrency` | 12 | General-web searches in flight across all subjects. |
| `host_requests_per_second` | `None` | Per-host request rates; override `good_option_catalog.host_requests_per_second`. |
| `ncbi_api_key_file` | `None` | File holding an NCBI API key for PubMed; overrides `good_option_catalog.ncbi_api_key_file` and `NCBI_API_KEY`. |
| `llm_batch_size` | `None` | Trials screened, and drugs classified, at once; overrides `screening_checkpoint_batch_size` and `class_checkpoint_batch_size`. |

The last five parameters only schedule work. They are left out of the checkpoint fingerprint, so a build can resume
with different values.

### Intervention screening

The configured LLM reads every `DRUG`, `BIOLOGICAL`, and `COMBINATION_PRODUCT` entry in its trial and arm context
and keeps only concrete named agents with direct anticancer treatment intent. Antitumor medicines, biologics,
cell and gene therapies, therapeutic radiopharmaceuticals, and named anticancer comparators are kept. Supportive and
procedural medicines (including toxicity protection and GVHD prophylaxis unless the trial uses them as anticancer
therapy), anesthetics, hemostatic agents, diagnostic tracers, devices, prevention-only agents, dosing or cohort
labels, and unnamed standard-of-care placeholders are excluded, failing closed.

- **Decisions:** `research_disposition` is one of `INTERVENTION_SCREENING_DISPOSITIONS` (`include`, `exclude`,
  `uncertain`); an excluded entry gets one of `INTERVENTION_EXCLUSION_CATEGORIES` (such as
  `supportive_or_procedural`, `diagnostic_or_imaging`, `unspecified_standard_of_care`).
- **Roles:** each kept agent is `investigational`, `uncertain`, `control`, `background` (given the same way to every
  participant), or `supportive`. An agent given only in a comparator arm is control. Only investigational and
  uncertain drugs are scored.
- **Names:** each agent is one drug named by the entry's own text (ctgov name, aliases, description, or its arms'
  descriptions), or a component of a regimen named there. "Pemetrexed + Cisplatin /Carboplatin" is three agents,
  and a standard acronym such as FOLFIRI is expanded into fluorouracil, leucovorin, and irinotecan, with the acronym
  in `regimen_names`. Agents are named by generic name (INN/USAN) or development code, never by a treatment
  category; an abbreviation, misspelling, or brand name that a generic name replaces ("LV", "5-Flurouracil") is
  listed in `regimen_names` the same way.
- **Grounding:** a name of three or more letters and digits may be any part of the entry's text with spacing and
  punctuation removed; a two-character name ("SG", "LV") must be whole words of it, and a single character a whole
  field. Validation rejects agent names absent from the text unless a listed regimen or spelling covers them, and
  rejects a regimen name listed as an agent. A listed regimen name the text does not contain is dropped with a
  logged count and grounds no agent. A combination-spelled or ungrounded name gets one more attempt.
- **Structured output:** with a remote backend the answer is decoded under `INTERVENTION_SCREENING_JSON_SCHEMA`
  (`good_option_catalog.structured_outputs`).

`build_intervention_screening_messages` renders the screening prompt for one trial
(`build_role_resolution_messages` is an alias). Every decision is kept in `trial_intervention_screening.parquet`:
disposition, exclusion category, confidence, role, rationale, the supported agent names
(`active_entity_names_json`), and the regimen names (`regimen_names_json`). Excluded entries never enter
`trial_drug_index.parquet`, research, or synthesis.

### Drug identity

Kept agents are normalized with the bundled NCI Thesaurus snapshot (`good_option_catalog.ncit_resource`,
`Thesaurus_26.07d.FLAT.zip`) and deduplicated across the run. A name, its parenthetical-free form, or a
parenthetical alone that is an exact NCIt name or synonym wins. Otherwise `NCIThesaurusDrugIndex.resolve_drug_name`
accepts a concept only when one rule finds exactly one, in order: the same `drug_name_key` (case, spacing,
separators, salt abbreviations, isotope notation, liposome word order); the same key without salts (the parent
molecule when salts tie); route, dosage-form and biosimilar words moved or dropped; one misspelled word of eight or
more letters corrected. A matched drug's `drug_id` is `NCIT:<code>`; an unmatched drug's is `NAME:` plus a hash of
its `drug_name_key`.

### Research

Each unique drug is researched once on five facets (`FACETS`: `mechanism_targets`, `efficacy_by_tumor`,
`biomarker_prevalence`, `biomarker_directed_efficacy`, `safety`) from seven sources (`default_sources()`:
ClinicalTrials.gov, PubMed, Europe PMC, CIViC, DailyMed, cancer.gov, and full-text general web). General-web queries
include the phrase `cancer treatment` to reduce name collisions. Technical failures are retried with `Retry-After`
or exponential backoff, and successful empty results stay distinct from exhausted technical failures. A drug whose
every source failed on some facet is `research_status="blocked"` and is never synthesized.

Retrieval runs on three axes, recorded per passage in `query_scope`:

- **Drug** (`drug`): the agent's own name and aliases, all five facets, three adaptive rounds.
- **Indication** (`drug_indication`, `class_indication`): efficacy by tumor, biomarker prevalence, and
  biomarker-directed efficacy with a disease as a required clause. The diseases are the `conditionsModule.conditions`
  of the trials the drug (or a class's member drugs) appears in, taken round-robin across trials, so the axis stays
  patient-free and cacheable. `max_indication_passages_per_drug` bounds a drug's indication passages, interleaved
  across diseases, sources, and facets.
- **Class** (`class`): the pharmacologic class's name and aliases, once per class rather than once per drug, so
  agents that share a class share a corpus.

Only PubMed, Europe PMC, cancer.gov, and general web run on the indication and class axes; the ctgov, label, and
curation adapters search by exact agent name. Their settings live on `ResearchSettings`: `indication_terms_per_drug`
and `class_indication_terms` (25 each), `indication_facets`, `indication_sources`, `class_facets`,
`class_sources`, `class_research_rounds` (1), and the `max_*` record caps. Research settings are part of the
checkpoint fingerprint.

Research requests to ClinicalTrials.gov, DailyMed, and the NCBI E-utilities are paced per host across every subject;
see [`good_option_catalog`](../reference/configuration.md#good_option_catalog) for the default rates.

### Drug classes

A `classify` stage between research and synthesis assigns classes from the bundled NCIt hierarchy. A class is an
NCIt concept above a drug concept, in the Pharmacologic Substance branch, other than the fixed
`NCIT_UMBRELLA_CLASSES` (Pharmacologic Substance, Antineoplastic Agent, Enzyme Inhibitor, and the like); the
chemical-structure branch is not used.

- **NCIt-matched drugs** get every class above their concept, in code. A local search of NCIt class names and
  synonyms against the drug's name, definition, and mechanism passages finds up to `class_candidate_limit` (25)
  candidates, and the LLM may add, by code, those the passages show the drug belongs to. An added class brings its
  NCIt ancestors; codes outside the candidates are rejected; hierarchy classes are never removed. A drug with no
  candidates gets no LLM call.
- **Drugs NCIt leaves unclassified** get their classes from the LLM: candidates, or a new class
  (`class_source="llm"`) only where none fits. A new class's `class_id` is a hash of its normalized name, so
  "PD-L1 inhibitor" and "PD-L1 Inhibitors" share one corpus, and a new name that spells an NCIt class is stored as
  that class.
- **Limits:** `class_max_per_drug` (3) caps the classes the LLM adds or names per drug. Only the drug's record and
  its candidates' names and short definitions enter the prompt, decoded under `DRUG_CLASS_JSON_SCHEMA` with a remote
  backend.

An NCIt class's `class_id` is `NCIT:<code>`, its name NCIt's display or preferred name, and its aliases NCIt
synonyms. Assignments are stored in `drug_classes.parquet` with `ncit_code`, `class_source` (`ncit` or `llm`), and
`assigned_by` (`ncit_hierarchy`, `llm_added_ncit`, `llm_added_ncit_ancestor`, or `llm_new`); only LLM-chosen rows
carry a `basis` (`target`, `mechanism`, or `modality`), `target`, `confidence`, and support IDs.

### Synthesis and stored summaries

The configured LLM synthesizes each class's and each drug's passages into structured facts in six categories
(`mechanism_and_targets`, `efficacy_by_tumor`, `biomarker_prevalence`, `biomarker_directed_efficacy`, `safety`,
`limitations`), reading the corpus in chunks of `synthesis_chunk_max_tokens` (drugs) or `class_evidence_max_tokens`
(classes), 60,000 each. Each chunk returns only new or revised facts, which code merges into the facts so far.

- **Citations:** every fact cites passage support IDs, validated in code. A matched NCIt definition is stored as its
  own citable passage.
- **Scope:** every fact carries `scope` `agent` or `class`, assigned in code from the corpus it was synthesized
  from, so the rubric's preference for agent-specific evidence has something to bind to.
- **Evidence levels:** supported evidence is kept at every maturity level (ontology, preclinical, first-in-human,
  phase 1, ctgov, and mature clinical evidence) and labelled, without requiring approval, randomization,
  publication, or mature outcomes.
- **Retries:** token-limited, blank, or invalid responses are retried with the validation error, up to
  `synthesis_max_attempts` (3). The default synthesis completion budget is 100,000 tokens (`synthesis_llm`); if
  every ordinary attempt is token-limited, the final attempt disables thinking.

Each drug row in `drug_summaries.parquet` stores the facts (`structured_facts_json`) and two plain-text renders:
`good_option_summary`, without safety, and `help_me_choose_summary`, the same render with safety, sized by
`good_option_summary_max_tokens` and `help_me_choose_summary_max_tokens`. Each class row in
`class_summaries.parquet` stores its facts and `class_option_summary`. URLs, queries, source labels, ctgov metadata,
and failure notices stay in the internal tables and never appear in a patient-bearing prompt.

### Bundle layout

| File | Content |
| --- | --- |
| `manifest.json` | File hashes, `compatibility_id` (a fingerprint of those hashes), trial IDs, `prompts_sha256`, NCIt version, research settings, teacher config, checkpoint fingerprint, counts. |
| `trial_ctgov.parquet` | One row per trial: title, status, phases, brief summary, `ctgov_status`, and the full study JSON. |
| `trial_intervention_screening.parquet` | Every screening decision. |
| `trial_drug_index.parquet` | Trial-drug pairs with role, `role_confidence`, and `scoreable`. |
| `drug_summaries.parquet` | One row per drug: identity, `research_status`, `synthesis_status`, facts, renders, `evidence_count`, technical failures. |
| `drug_evidence/` | Drug passages: source, URL, license, query, `query_scope`, facet, content hash. |
| `drug_research_attempts.parquet` | Every search attempt and its outcome. |
| `drug_classes.parquet` | Drug-class assignments. |
| `class_evidence/` | Class passages. |
| `class_summaries.parquet` | One row per class: facts, `class_option_summary`, `synthesis_status`. |

Catalogs whose ClinicalTrials.gov table is named `trial_registry.parquet`, with `registry_*` columns, are still read.

### Checkpoints and validation

Catalog publication and its intermediate JSON checkpoints are atomic. Completed ctgov fetches, intervention screens,
drug research, class assignments and research, and syntheses are reused when the same build is restarted;
technically blocked fetches are retried. The checkpoint manifest fingerprints the NCT list, research settings and
sources, the NCIt resource and version, the text of the catalog prompts, any custom resolver callables, and the
`llm_good_option` and `good_option_catalog` sections (minus the pacing keys). An incompatible checkpoint fails
explicitly instead of mixing evidence generations; pass `checkpoint_path` to use another directory or
`reset_checkpoint=True` to discard it.

`validate_good_option_catalog(path)` verifies file hashes and the `compatibility_id`, screening decisions,
trial-drug references, terminal research and synthesis states, query scopes, fact scopes, and that every fact's
support IDs exist for its drug or class. It returns the manifest. `load_good_option_catalog(path, validate=True,
cache=False)` validates by default; with `cache=True` it keeps the catalog in memory until any bundle file's size or
modification time changes.

### Command line

`matchminer-ai-build-good-option-catalog` builds and validates a complete catalog, drug classes included, against
one or more OpenAI-compatible endpoints:

```bash
matchminer-ai-build-good-option-catalog \
  --nct-ids-file trial_ids.txt \
  --output good_option_catalog \
  --server-url http://localhost:8000/v1
```

The served model is read from the endpoint's `/v1/models` (every server must report the same one), and its
registered sampling profile (see [LLM Server Helper](llm.md#model-sampling-profiles)) is applied to
`llm_good_option` and every `good_option_catalog.*_llm` stage override. Rerun the same command to resume. The command
prints the request rates it uses and whether an NCBI key is set, never the key.

| Option | Default | Meaning |
| --- | --- | --- |
| `--nct-ids-file` | | Text file (one ID per line, `#` comments) or CSV/Parquet with an `nct_id` or `trial_id` column. |
| `--nct-id` | | One NCT ID; repeatable. |
| `--output` | required | Catalog directory. |
| `--checkpoint-dir` | `<output>_checkpoints` | Resumable checkpoint directory. |
| `--server-url` | required | OpenAI-compatible endpoint; repeat for several identical servers. |
| `--model` | discovered | Expected served model. |
| `--config` | default preset | Config YAML. |
| `--max-concurrent-requests` | 64 | Requests in flight per server (`remote.max_concurrent_requests`). |
| `--llm-request-timeout` | 86,400 s | Seconds per LLM request, including time queued on the server. |
| `--llm-batch-size` | preset (64) | Trials screened, and drugs classified, at once. |
| `--research-concurrency` | 6 | Concurrent requests within one subject (`ResearchSettings.max_concurrency`). |
| `--subject-concurrency` | 8 | Drugs, then classes, researched at once. |
| `--web-search-concurrency` | 12 | General-web searches in flight. |
| `--host-requests-per-second HOST=RATE` | | Per-host request rate; repeatable; `0` leaves a host unpaced. |
| `--ncbi-api-key-file` | | File holding an NCBI API key. |
| `--research-request-timeout` | 30 s | Per research request (`ResearchSettings.request_timeout`). |
| `--reset-checkpoints` | off | Discard a recognized checkpoint directory. |
| `--overwrite` | off | Replace an existing catalog. |

`--llm-batch-size`, `--subject-concurrency`, `--web-search-concurrency`, `--host-requests-per-second`, and
`--ncbi-api-key-file` are not part of the checkpoint fingerprint.

## Score patient-trial pairs

`check_good_options`, `score_good_options_with_llm`, and `evaluate_good_options` take a `candidate_pairs` DataFrame
with `patient_id`, `trial_id`, and `cancer_history_summary` (the training names `pseudo_mrn`, `nct_id`, and
`patient_summary` are also accepted). Pairs are deduplicated on `patient_id` and `trial_id`. The scorers other than
`check_good_options` require a loaded `GoodOptionCatalog`.

The prompt contains the patient's cancer-history summary first, then evidence for the trial's scoreable drugs
(investigational or uncertain). Control, background, and supportive drugs are indexed but never scored. After the
per-drug sections the prompt carries one labelled `DRUG CLASS EVIDENCE` block per distinct class across those
drugs, naming which drugs each block covers; two drugs sharing a class produce one block.

The rubric awards each scoreable drug up to four binary points (`RUBRIC_CRITERIA`):

1. `disease_type_benefit`: evidence of human benefit in the patient's disease type, from the drug or its class (0
   when the patient has already received or is receiving the drug);
2. `common_biomarker_in_disease`: the targeted biomarker is present in 20% or more of that disease;
3. `patient_biomarker_targeted`: the patient's tumor is documented to have the targeted biomarker; and
4. `biomarker_targeted_benefit`: human evidence of benefit from targeting that biomarker.

The code-derived score is `total_points / (4 * scoreable_drug_count)`. `parse_good_option_response` validates an
answer: it must assess exactly the supplied drugs (names compared without case), with each point the integer 0 or 1
and a non-empty rationale. Trials with missing catalog data, exhausted research, or missing synthesis are returned
unscored; there is no live-search fallback.

| Column | Content |
| --- | --- |
| `patient_id`, `trial_id` | The pair (`trial_id` normalized). |
| `good_option_score` | Points over maximum points; NaN when unscored. |
| `good_option_points`, `good_option_max_points` | Points earned and `4 * good_option_drug_count`. |
| `good_option_drug_count` | Scoreable drugs in the trial. |
| `good_option_status` | `ok`, `parse_failed`, `missing_catalog_trial`, `trial_ctgov_blocked`, `no_scoreable_drug`, `missing_drug_summary`, or `drug_research_blocked`. |
| `good_option_method` | `llm` (`classifier` for the deprecated checker). |
| `good_option_patient_disease_type` | The model's statement of the patient's disease. |
| `good_option_drug_assessments` | Per drug: `drug_name`, `targeted_biomarkers`, and each criterion's `point` and `rationale`. |
| `good_option_uncertainties` | The model's key uncertainties. |

With `debug_mode`, each row also carries `good_option_answer_text`, `good_option_reasoning_text`,
`good_option_finish_reason`, `good_option_parse_error`, `good_option_parse_attempts` (requests sent for the pair),
`good_option_reasoning_off_fallback_used`, and `good_option_evidence_packing`. With `return_metadata=True` the
metadata records the config snapshot, the catalog's `compatibility_id`, an evidence-packing summary (granularity
counts and truncated subjects), the rubric criteria, and the retry settings.

`score_good_options_with_llm` retries only invalid answers: `max_parse_attempts` (default 1) attempts in total, each
retry carrying the previous answer and the validator's exact error, then one attempt with thinking disabled when
`reasoning_off_fallback=True`. An answer stopped by its token limit counts as invalid. `evaluate_good_options`
dispatches to it with `method="llm"` (the default).

### Evidence packing

The catalog's stored renders are fixed-size. The scoring prompt does not use them; `pack_good_option_evidence`
re-renders every drug and class from its stored structured facts, sized to how much evidence the trial has in total:

1. `good_option_evidence_budget` takes the scoring model's context window (`good_option_prompt.context_tokens`, or
   `llm_good_option.local.engine.max_model_len` when unset), subtracts the active backend's `llm_good_option`
   `max_tokens` completion allowance, `safety_tokens`, room for a parse retry, and the fixed prompt text, and
   converts the rest at `chars_per_token` (3.5, conservative for the default Nemotron 3.5 Super and for Gemma 4).
   With the default preset that leaves about 530,000 characters, or about 700,000 under `check_good_options`'
   50,000-token output cap.
2. Drugs and classes share it by weighted max-min fair share, drugs at twice a class's weight. A subject needing less
   than its share releases the rest. `max_drug_section_tokens` and `max_class_section_tokens` (20,000 each) bound any
   one subject.
3. A subject that fits its share is rendered in full. One that does not steps down to `condensed` (attribution capped
   at 160 characters, statements at 300) or `brief` (100 and 160) if that fits, so every fact stays visible; the
   space saved is offered to the others. `EVIDENCE_GRANULARITIES` defines the levels.
4. Only when even `brief` overflows are facts dropped, by the render's ranking (strongest evidence first, tumor types
   interleaved) with an omission count; the survivors keep `condensed` detail so their numbers survive.

Subjects without stored facts, such as hand-built summaries, fall back to their stored render, truncated if needed.
A trial whose budget leaves under 1,000 characters per subject raises `ValueError`. `build_good_option_messages`
returns the packed prompt for one patient and one trial's drug and class evidence.

### On-demand check

`check_good_options(candidate_pairs, *, catalog, config=None, return_metadata=False)` wraps
`score_good_options_with_llm` with the settings of the `good_option_check` preset section, applied to a copy of the
config. The caller's config, and the `llm_good_option` section that fingerprints catalog builds, are left unchanged.

- **Catalog:** a loaded `GoodOptionCatalog` or a bundle path. A path is loaded with
  `load_good_option_catalog(path, cache=True)`, which validates once and reuses the catalog until any bundle file
  changes.
- **Output cap:** the `llm_good_option` completion is capped at `max_output_tokens` (50,000), leaving more of the
  context for evidence.
- **Context:** unless `good_option_prompt.context_tokens` is set, a remote OpenAI-compatible backend sizes the
  evidence to the smallest `max_model_len` its servers report at `/v1/models` (`discover_context_tokens`,
  `discovery_timeout_seconds`). If the lookup fails, packing uses the configured context and returns the warning.
- **Retries:** invalid answers get `max_parse_attempts` (3) attempts in total, then one with thinking disabled when
  `reasoning_off_fallback` is true.

Output columns match `score_good_options_with_llm`. With `return_metadata=True`, `metadata["good_option_check"]`
also reports `max_output_tokens`, `context_tokens`, `context_source` (`configured`, `endpoint`, or `preset`), and
`context_warning`.

### Deprecated: GoodOptionChecker

The trained four-logit GoodOptionChecker classifier is deprecated. It saw one patient summary and one drug summary
per example, never the class evidence the LLM prompt carries, and no checker artifact is published.
`score_good_options`, `evaluate_good_options(method="classifier")`, and `build_good_option_checker_text` still run
but emit `DeprecationWarning`; the `good_option_checker` preset section remains only so existing configs load.

::: matchminer_ai.trials.drug_catalog
    options:
      members:
        - build_good_option_catalog
        - validate_good_option_catalog
        - load_good_option_catalog
        - build_intervention_screening_messages

::: matchminer_ai.matching.good_options
    options:
      members:
        - check_good_options
        - score_good_options_with_llm
        - evaluate_good_options
        - build_good_option_messages
        - pack_good_option_evidence
        - good_option_evidence_budget
        - parse_good_option_response
        - score_good_options
        - build_good_option_checker_text
