# Good Option Catalog and Scoring

Good Option scoring is a patient-trial ranking signal, not an eligibility
determination, response probability, or treatment recommendation. Research and
patient scoring are separate, versioned stages.

Use the existing trial and matching stage namespaces:

```python
from matchminer_ai.trials import build_good_option_catalog, load_good_option_catalog
from matchminer_ai.matching import check_good_options

# Public-only research stage; run once before patient scoring.
catalog = await build_good_option_catalog(nct_ids, "drug_catalog", config=config)

# candidate_pairs: patient_id, trial_id, cancer_history_summary
scores = check_good_options(candidate_pairs, catalog="drug_catalog", config=config)
```

`check_good_options` is the supported on-demand entry point; see
[On-demand check](#on-demand-check). `score_good_options_with_llm` is the
lower-level scorer it wraps.

Patient summaries reach the configured LLM endpoint in LLM mode. Use an
endpoint authorized for the sensitivity of the input. Catalog construction
accepts NCT IDs only and never receives patient context.

The old `matchminer_ai.good_options` namespace and its submodules remain
compatibility aliases. Trial research is implemented in `trials/drug_catalog.py`,
`trials/drug_research.py`, and `trials/drug_evidence.py`; patient scoring is in
`matching/good_options.py`.

## Prompt resources

All templates below are bundled under `src/matchminer_ai/prompts/`:

| Files | Purpose |
| --- | --- |
| `trial_drug_screen.system.txt`, `.user.txt`, `.retry.txt` | Identify anticancer agents and classify their roles in each trial. |
| `trial_drug_synthesis.system.txt`, `.user.txt`, `.retry.txt` | Synthesize passage-grounded drug evidence. |
| `llm_good_option.system.txt`, `.user.txt`, `.rubric.txt`, `.retry.txt` | Apply the four-criterion patient-drug rubric and correct invalid responses. |
| `good_option_checker_template.txt` | Deprecated. Patient-drug input for the retired GoodOptionChecker classifier. |

This relocation preserves the rendered prompts, rubric versions, catalog
compatibility IDs, and checkpoint source fingerprints. Existing compatible
catalogs, label shards, and classifier inputs do not require regeneration.
Future prompt changes still require the usual prompt/artifact-version review.

## Build the patient-free catalog

`build_good_option_catalog` accepts a list of NCT IDs. It fetches each current
ClinicalTrials.gov record, then uses the configured LLM to screen every `DRUG`
and `BIOLOGICAL` entry in its public trial and arm context. Only concrete named
agents with direct anticancer treatment intent proceed to normalization and
research. Antitumor medicines, biologics, cell/gene therapies, therapeutic
radiopharmaceuticals, and genuine named anticancer comparators are retained;
supportive/procedural medicines, anesthetics, hemostatic agents, diagnostic
tracers, prevention-only agents, dosing/cohort labels, and unnamed standard-of-
care placeholders are excluded fail-closed. The screen also resolves retained
agents' investigational, uncertain, control, background, or supportive roles.

Every decision is retained in `trial_intervention_screening.parquet`, including
the disposition, exclusion category, confidence, rationale, and supported
active-entity names. Excluded entries never enter `trial_drug_index.parquet`,
the unique-drug set, web queries, evidence retrieval, or synthesis. Retained
identities are normalized with the bundled NCIt snapshot and deduplicated across
the complete run.

Each unique drug is researched once across mechanism and targets, human efficacy
by tumor type and histology, biomarker prevalence, biomarker-directed human
efficacy, and safety. The retriever uses bounded authoritative-source adapters
plus full-text general-web documents. General-web queries include the explicit
phrase `cancer treatment` to reduce name-collision results. Technical failures
are retried with
`Retry-After` or exponential backoff; successful empty results remain distinct
from exhausted technical failures. Searches accept drug identity only and can
never receive patient text.

Retrieval runs on three axes, and every passage records which one found it in
`query_scope`.

The **drug** axis is the one above: the agent's own name and aliases.

The **indication** axis reruns the efficacy facets with a disease as a required
clause. The diseases come from the `conditionsModule.conditions` of the trials
the drug appears in, which the build already fetches and stores, so the axis
stays patient-free and cacheable. Without it the drug evidence never names the
patient's cancer for most patient-drug pairs, and the first rubric criterion
fires far less often when the disease is absent than when it is present. The
axis is bounded as a whole by `max_indication_passages_per_drug`, interleaved
across diseases and sources, so it cannot swamp the agent's own corpus.

The **class** axis retrieves literature for the pharmacologic classes a drug
belongs to, once per class rather than once per drug, so agents that share a
class share a corpus. A `classify` stage between `research` and `synthesis` reads
each drug's mechanism passages and names its classes; NCIt is offered only as a
hint, because it has nothing usable for the first-in-human agents that most need
class evidence. Class identity is a hash of the normalized class name, so
"PD-L1 inhibitor", "PD-L1 Inhibitors", and "anti-PD-L1 inhibitor" resolve to one
corpus. Assignments are stored in `drug_classes.parquet` and the corpus in
`class_evidence/`.

The configured LLM synthesizes the evidence ledger into structured facts with
validated passage support IDs. A matched bundled NCIt definition is materialized
as its own citable ledger passage rather than supplied as uncitable side context.
Synthesis retains supported evidence across maturity levels—including ontology,
preclinical, first-in-human, phase 1, registry, and mature clinical evidence—and
labels its level without requiring approval, randomization, publication, or
mature outcomes. Token-limited or blank final responses are retried, and the
default synthesis completion budget is 100,000 tokens; if every ordinary
attempt is token-limited, the final attempt disables thinking. The bundle stores two
clean projections: a
GoodOption summary without safety and a Help Me Choose summary with safety.
URLs, queries, source labels, registry metadata, and failure notices remain in
the internal Parquet ledger and do not appear in either patient-bearing prompt.

Each class corpus is synthesized in its own call on its own budget, and stored
in `class_summaries.parquet`. Keeping it separate is deliberate: the drugs that
most need class evidence already fill the agent budget, so pooling the two would
let a flood of same-class passages evict the agent's own data. Every fact carries
a `scope` of `agent` or `class`, assigned in code from the corpus synthesis ran
over rather than asked of the model, and both projections print it, so the
rubric's instruction to prefer agent-specific evidence has something to bind to.

Catalog publication and its intermediate JSON checkpoints are atomic. By
default, checkpoints are stored beside the requested catalog in
`<catalog>_checkpoints`. Completed registry fetches, intervention screens, drug
research, and drug syntheses are reused when the same build is restarted.
Technically blocked fetches are retried. The checkpoint manifest fingerprints
the NCT list, source/research settings, ontology version, prompt/schema versions,
and teacher configuration; incompatible checkpoints fail explicitly instead of
silently mixing evidence generations. Pass `checkpoint_path` to select another
directory or `reset_checkpoint=True` to discard a recognized checkpoint bundle.

`validate_good_option_catalog` verifies hashes, component versions, NCIt
compatibility, intervention-screen decisions, trial-drug references, terminal
research states, and structured-fact evidence support.
`load_good_option_catalog` validates by default.

### Command line

`matchminer-ai-build-good-option-catalog` builds and validates a complete
catalog, drug classes included, against an OpenAI-compatible endpoint:

```bash
matchminer-ai-build-good-option-catalog \
  --nct-ids-file trial_ids.txt \
  --output data/no_phi/good_option_catalog \
  --server-url http://gpu-host:8001/v1
```

The served model is read from the endpoint's `/v1/models`, and its registered
sampling profile (see [LLM Server Helper](llm.md#model-sampling-profiles)) is
applied to `llm_good_option` and every `good_option_catalog.*_llm` stage
override. Checkpoints default to `<output>_checkpoints`; rerun the same command
to resume.

## Score a patient-trial candidate

`evaluate_good_options` and `score_good_options_with_llm` require a loaded
catalog. The prompt contains the patient cancer-history summary first, followed
only by evidence for scoreable investigational or unresolved drugs in the trial.
Control, background, and supportive drugs are indexed but are never scored.

Agent and class evidence are merged only here, at prompt time. After the per-drug
sections the prompt carries one labelled `DRUG CLASS EVIDENCE` block per distinct
class across the trial's drugs, naming which drugs each block covers; two drugs
sharing a class produce one block, not two.

### Evidence packing

The catalog's stored projections are fixed-size renders made at build time. The
scoring prompt does not use them; `pack_good_option_evidence` re-renders every
drug and class from its stored structured facts, sized to how much evidence the
trial has in total:

1. `good_option_evidence_budget` takes the teacher's context window
   (`good_option_prompt.context_tokens`, or
   `llm_good_option.local.engine.max_model_len` when unset), subtracts the
   active backend's `llm_good_option` `max_tokens` completion allowance,
   `safety_tokens`, room for a parse retry, and the fixed prompt text, and
   converts the rest at `chars_per_token` (3.5, conservative for Gemma 4, which
   averages ~4.4 on catalog text). With the default preset that leaves roughly
   500,000 characters.
2. Drugs and classes share it by weighted max-min fair share, drugs at twice a
   class's weight. A subject needing less than its share releases the rest.
   `max_drug_section_tokens` and `max_class_section_tokens` (20,000 each) bound
   any one subject.
3. A subject that fits its share is rendered in full. One that does not steps
   down to `condensed` (attribution capped at 160 characters, statements at
   300) or `brief` (100 and 160) if that fits, so every fact stays visible; the
   space saved is offered to the others. On v12 class syntheses the two levels
   render at a median 88% and 58% of full size.
4. Only when even `brief` overflows are facts dropped, by the projection's
   ranking (strongest evidence first, tumor types interleaved) with an omission
   count; the survivors keep `condensed` detail so their numbers survive.

Subjects without stored facts, such as hand-built summaries, fall back to their
stored projection, truncated if needed. Scoring metadata records
`evidence_packing_version` and a summary of granularities and truncations, and
debug mode adds per-row `good_option_evidence_packing`. The prompt template and
`GOOD_OPTION_PROMPT_VERSION` are unchanged, so catalogs remain compatible.

The unchanged rubric awards each scoreable drug up to four binary points:

1. evidence of benefit in the patient's disease type;
2. the targeted biomarker is commonly expressed in that disease type;
3. the patient's tumor is documented to have the targeted biomarker; and
4. human evidence of benefit from targeting that biomarker.

The code-derived LLM score is `total_points / (4 * scoreable_drug_count)`.
Trials with missing catalog data, exhausted research, or missing synthesis are
returned as explicitly unscored; there is no live-search or legacy-snippet
fallback.

### On-demand check

`check_good_options` runs the LLM rubric for patient-trial pairs with
production defaults from the `good_option_check` preset section:

- `catalog` may be a loaded `GoodOptionCatalog` or a bundle path. A path is
  loaded with `load_good_option_catalog(path, cache=True)`, which validates once
  and reuses the catalog until any bundle file's size or modification time
  changes.
- The `llm_good_option` completion is capped at `max_output_tokens` (50,000)
  on a copy of the config, leaving more of the context for evidence.
- Unless `good_option_prompt.context_tokens` is set, the evidence budget follows
  the smallest `max_model_len` that the configured OpenAI-compatible servers
  report at `/v1/models`. If the lookup fails, packing uses the configured
  context and the warning is returned.
- Invalid answers get `max_parse_attempts` (3) attempts in total, then one
  attempt with thinking disabled when `reasoning_off_fallback` is true.

Output columns match `score_good_options_with_llm`. With
`return_metadata=True`, `metadata["good_option_check"]` reports
`max_output_tokens`, `context_tokens`, `context_source` (`configured`,
`endpoint`, or `preset`), and `context_warning`. The caller's config is never
mutated, and the `llm_good_option` section that fingerprints catalog builds is
left unchanged.

### Deprecated: GoodOptionChecker

The trained four-logit GoodOptionChecker classifier is deprecated. It saw one
patient summary and one drug summary per example, never the class evidence the
teacher prompt carries, and no checker artifact is published. `score_good_options`,
`evaluate_good_options(method="classifier")`, and
`build_good_option_checker_text` still run but emit `DeprecationWarning`; the
`good_option_checker` preset section remains only so existing configs load.

::: matchminer_ai.trials
    options:
      members:
        - build_good_option_catalog
        - validate_good_option_catalog
        - load_good_option_catalog
        - build_intervention_screening_messages
        - build_role_resolution_messages
        - build_synthesis_messages

::: matchminer_ai.matching
    options:
      members:
        - check_good_options
        - build_good_option_messages
        - pack_good_option_evidence
        - good_option_evidence_budget
        - score_good_options_with_llm
        - evaluate_good_options
        - score_good_options
        - build_good_option_checker_text
