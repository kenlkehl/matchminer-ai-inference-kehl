# MatchMiner-AI

MatchMiner-AI is an open-source pipeline for matching patients with cancer to clinical trials. It reads trial
documents and longitudinal patient notes and writes comparable summaries of each. From those summaries it retrieves
candidate patient-trial pairs and scores whether each pair deserves a closer look. It works in both directions:
trials for a patient, and patients for a trial. The method is described in
[Altreuter et al., *MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching*](https://doi.org/10.48550/arXiv.2412.17228).

This repository holds the whole pipeline in one Python package, `matchminer_ai`:
- the runtime stages, prompts and default configuration;
- the training code that produces the models;
- the evaluations;
- two web applications built on the package.

Training renders its examples with the same prompts, text formats and schemas the runtime uses, so a model is
trained on exactly what it sees in use.

> [!IMPORTANT]
> MatchMiner-AI is a research tool for prioritizing human review. It is not an eligibility engine, a medical device
> or a source of treatment recommendations. Its scores rank and screen candidates. Every suggested match needs
> review by a clinician and the trial team against the complete, current protocol.

> [!WARNING]
> The package is pre-v1 and under active development. APIs, configuration and outputs may change.

## How matching works

Trials are matched at the level of **cohorts**. A cohort is one target population of a trial with its own
eligibility criteria, so one trial can have several. Each cohort's criteria are split in two:
- **Key criteria** define the population: cancer type, histology, extent, prior treatment, biomarkers.
- **Boilerplate criteria** are the general requirements most trials share: performance status, organ function,
  brain metastases, other cancers.

Patient summaries are split the same way:
- a **cancer history** (the key section);
- an **other relevant history** section, which holds the evidence for the boilerplate criteria.

| Step | Function | What it does |
| --- | --- | --- |
| 1. Trials | `trials.process_trials` | Screens each trial for cancer treatment, names its cohorts and extracts each cohort's complete eligibility criteria. It merges cohorts with equivalent criteria and tags every criteria line as inclusion or exclusion and key or boilerplate. It reads ClinicalTrials.gov listings or full protocols (`source="protocol"`). Each step is also available on its own (`extract_cohorts`, `merge_equivalent_cohorts`, `remove_boilerplate_criteria`). |
| 2. Patients | `patients.summarize_patients` | Summarizes each patient's notes serially. The notes are sorted by date, deduplicated and split into overlapping chunks, and each chunk updates the running summary. The result is a cancer history plus the other relevant history. PDFs are read locally, with OCR when needed. |
| 3. Embed | `embedding.embed_for_matching` | Encodes patients' cancer histories and cohorts' key criteria into one vector space with **TrialSpace**. |
| 4. Retrieve | `matching.generate_candidate_matches` | Ranks candidates by cosine similarity: the top-k cohorts for each patient, or the top-k patients for each cohort. |
| 5. Score | `matching.score_match_quality` | The **decision model** answers five yes/no questions about the pair: reasonableness, cancer type, disease burden, treatment history and biomarkers. Its ranking score is `match_quality_score`, and a pair passes when the model calls it potentially eligible. |
| 6. Boilerplate | `matching.exclusion_criteria_check` | Asks whether the patient clearly fails any boilerplate criterion. |
| 7. Line by line | `matching.check_eligibility_criteria` | Asks, for every tagged criteria line, whether the patient clearly fails it. |

Steps 5-7 also have LLM versions that return a reason with each answer: `score_match_quality_with_llm`,
`exclusion_criteria_check_with_llm` and `check_eligibility_criteria_with_llm`. The match and per-criterion versions
use the training teacher's prompts.

## Models

- **LLM.** Every LLM step of the default preset uses `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`. It runs
  either in-process with vLLM or on any OpenAI-compatible server. Other models (Gemma 4, Qwen 3.5, Reflection Beam)
  are selected per section with `model_name`, and their sampling settings come with them; see
  [LLM runtime](docs/reference/llm-runtime.md).
- **TrialSpace.** The retriever is `google/embeddinggemma-2` fine-tuned by the training pipeline. The default preset leaves
  `embedding.model_path` unset. Set it to your own TrialSpace model (or `paths.trialspace_model` in a site config):
  embedding fails rather than fall back to a different model. Never compare vectors made by different TrialSpace
  models; re-embed both sides instead.
- **Decision model.** [`LiquidAI/d1-2B`](https://huggingface.co/LiquidAI/d1-2B) answers yes/no questions about a
  pair in a single forward pass and runs in-process with Transformers. The default is the pinned base model, which
  is zero-shot, so treat its scores as provisional. Point `decision_model.model_name` at a checkpoint fine-tuned by
  the training pipeline to use one.

Published models and synthetic datasets are on [Hugging Face](https://huggingface.co/ksg-dfci). The latest
release, [collection 0526](https://huggingface.co/collections/ksg-dfci/matchminer-ai-0526), predates the cohort and
decision-model pipeline in this repository.

## Installation

Python 3.12 or later on Linux. A GPU is needed for the in-process models; see the
[requirements](docs/getting-started/requirements.md) and [GPU recommendations](docs/getting-started/gpu-recommendations.md).
Install from a checkout with [`uv`](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/kenlkehl/matchminer-ai-kehl.git
cd matchminer-ai-kehl
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

Optional parts are extras:

| Extra | Adds |
| --- | --- |
| `training` | The training pipeline (`mmai-train`) |
| `evaluation` | The evaluations (install together with `training`) |
| `apps` | The dashboard and the guideline fetcher |
| `sql` | SQL-agent workup checks over extraction records |
| `databricks` | Claude through a Databricks AI Gateway |
| `guideline-conversion` | PDF guideline conversion |
| `dev`, `docs` | Tests, linters and the documentation site |

For example: `uv pip install -e ".[training,evaluation,apps]"`.

## Quickstart

Input is two tables.
- **Trials:** one row per trial with `trial_id`, `trial_title`, `brief_summary` and `eligibility_criteria`. These
  are the ClinicalTrials.gov fields described in the [requirements](docs/getting-started/requirements.md#data).
- **Notes:** one row per clinical note with `patient_id`, `note_text` and `note_date`.

This example uses an OpenAI-compatible server for the LLM steps:

```python
import pandas as pd

from matchminer_ai import load_preset
from matchminer_ai.embedding import embed_for_matching
from matchminer_ai.matching import (
    check_eligibility_criteria,
    exclusion_criteria_check,
    generate_candidate_matches,
    score_match_quality,
)
from matchminer_ai.patients import summarize_patients
from matchminer_ai.trials import process_trials

config = load_preset("default")
config.remote["enabled"] = True                           # LLM steps go to a running server
config.remote["server_urls"] = ["http://localhost:8000/v1"]
config.embedding["model_path"] = "/path/to/trialspace"    # your TrialSpace checkpoint

trials = pd.read_csv("trials.csv")
notes = pd.read_csv("notes.csv")

cohorts = process_trials(trials, config=config)           # one row per cohort
patients = summarize_patients(notes, config=config)       # one row per patient

# The 20 nearest cohorts for each patient; swap the two tables for the nearest patients per cohort.
candidates = generate_candidate_matches(
    embed_for_matching(patients, entity_type="patient", config=config),
    embed_for_matching(cohorts, entity_type="trial", config=config),
    k=20,
)
pairs = candidates.merge(patients, on="patient_id").merge(cohorts, on="cohort_trial_id")

quality = score_match_quality(pairs, config=config)       # the pairs that pass, with their scores
pairs = pairs.merge(quality, on=["patient_id", "cohort_trial_id"])
boilerplate = exclusion_criteria_check(pairs, config=config)
failing_lines = check_eligibility_criteria(pairs, config=config, filter_failing=True)
```

The [example notebook](examples/run_examples.ipynb) runs the same workflow on the sample data in `examples/data/`,
in both directions, and ends with a review table of the scored pairs.
[How matching works](docs/user-guide/patient-centric-matching.md) explains the concepts behind each step.
[Choosing an inference setup](docs/user-guide/local-vs-remote.md) compares in-process and server-based LLMs.

## Configuration

- **The preset.** Behavior comes from a YAML preset
  ([`presets/default.yaml`](src/matchminer_ai/presets/default.yaml)), loaded with `load_preset` or `load_config`. It
  has one section per step: models, endpoints, prompts' token budgets, chunking and cutoffs. Change it in code, as
  above, or in your own YAML file. See the [configuration reference](docs/reference/configuration.md).
- **Site config.** Values that belong to one installation are kept outside the repository: endpoints, cloud
  projects, local paths and institution-specific table maps. They come from a site config (`--site-config` or
  `MMAI_SITE_CONFIG`).
- **Data root.** Training, the evaluations and the apps read and write under a data root (`--data-root` or
  `MMAI_DATA_ROOT`). It holds `phi/` and `no_phi/` and may not be inside the checkout. See
  [site configuration](docs/reference/site-config.md).

## Beyond matching

The package also has components for related review work. Each produces research aids that need human review.

| Component | Entry points | Guide |
| --- | --- | --- |
| GoodOption drug evidence | A patient-free catalog of a trial list's drugs, classed with the NCI Thesaurus, with evidence syntheses: `trials.build_good_option_catalog` (`matchminer-ai-build-good-option-catalog`). Patient-specific option scoring: `matching.check_good_options`. | [Good Options](docs/user-guide/good-options-pipeline.md) |
| Guideline catalogs | Populations, workup and treatment menus with citations from a local NCCN library (`trials.summarize_guidelines`) or NCI PDQ summaries (`trials.summarize_pdq`). Retrieval for a patient: `matching.retrieve_guideline_considerations`. | [Guideline extraction](docs/user-guide/guideline-extraction.md), [PDQ](docs/user-guide/pdq-catalogs.md) |
| Structured clinical extraction | Evidence-linked, schema-validated records of diagnoses, biomarkers, treatments, procedures and imaging, with exact quotes located in the notes: `patients.extract_document_records`, `patients.extract_patient_records`. | [Extraction](docs/api/extraction.md) |
| Workup checks | Whether a guideline's workup was done, from the extraction records (compiled queries, or an LLM SQL agent in `patients.workup_sql`) or from the notes (`patients.review_patient_workup`). | [Extraction-based](docs/api/workup-queries.md), [SQL agent](docs/api/workup-sql.md), [Patients](docs/api/patients.md#workup-documentation-review) |
| Population workup | Patients who fit a guideline space, retrieved and scored, then reviewed against its workup: `matching.retrieve_guideline_patients`, `population.review_population_workup`. | [Population workup](docs/user-guide/population-workup.md) |
| Note-search question answering | An LLM answers questions about a patient by searching the notes in an isolated Python REPL: `patients.answer_patient_questions`. | [Note search](docs/user-guide/note-search-qa.md) |
| Supporting tools | Local PDF text extraction with OCR (`ocr_pdf`); token attribution for a decision-model answer (`matching.interpret_match_quality`); choosing how far down a trial's ranked patient list to score (`matching.find_trial_centric_cutoff`). | [Documents](docs/api/documents.md), [Match quality](docs/api/match-quality.md), [Candidate matching](docs/api/candidate-matching.md) |

## Training

`matchminer_ai.training` produces the models from synthetic and public data only. It runs these stages:
1. Parse a ClinicalTrials.gov export, screen it for cancer treatment trials and sample from them.
2. Write a synthetic protocol for each sampled trial and extract cohorts from both versions.
3. Generate synthetic patients and their notes for those cohorts, and summarize them with the runtime summarizer.
4. Over three rounds, mine candidate pairs with TrialSpace and label them with an LLM teacher. TrialSpace is
   fine-tuned after each of the first two rounds.
5. Fine-tune the decision model on the teacher's labels for all seven question types.
6. Optionally, distill the teacher's outputs into OncoReasoning, a small reasoning model.

Every stage is a module (`python -m matchminer_ai.training.<stage>`) that writes a path-free receipt and answers
`--check`, so a stage can run on another machine and be adopted. `mmai-train` runs a range of stages
(`--start-at`, `--stop-after`). See [Training](docs/training/index.md).

## Evaluation

`matchminer_ai.evaluation` evaluates the pipeline on real patients. It runs only where that data is authorized:
- `retro_cohorts`: retrospective cohorts from an EHR extract;
- `retro_fidelity`: distillation fidelity against the teacher;
- `dual_judge`: two frontier-LLM judges with human adjudication;
- `cohort_lineitems`: cohorts for a table of ClinicalTrials.gov-style listings.

See [Evaluations](docs/evaluation/index.md).

## Apps

- **Dashboard** (`mmai-dashboard`): a per-disease-center view of a site's trial cohorts. It finds synthetic
  patients for a cohort and trials for a patient summary. FastAPI with a React frontend in `frontend/dashboard/`, built with
  `pnpm build`. See [Dashboard](docs/apps/dashboard.md).
- **Guideline fetcher** (`mmai-guideline-fetcher`): turns a patient summary into cited guideline considerations,
  with patient-evaluation and population views. A PHI mode listens on loopback only and requires an access token.
  See [Guideline fetcher](docs/apps/guideline-fetcher.md).

## Repository layout

```text
src/matchminer_ai/
  trials/ patients/ embedding/ matching/   the runtime stages
  llm/ labeling/ prompts/ presets/         LLM clients, the teacher labeler, prompt templates, default configuration
  schema.py                                training-to-runtime column names
  training/ evaluation/ apps/              training, evaluations, web apps
frontend/dashboard/                        the dashboard's frontend source
docs/                                      documentation (mkdocs)
examples/                                  example notebook and scripts
tests/
```

## Development

```shell
uv pip install -e ".[dev]"
pre-commit install          # formatting, linting, type checks and secret scanning
pytest
```

Build the documentation site with `uv pip install -e ".[docs]"` and `mkdocs serve`. See
[contributing](CONTRIBUTING.md).

## Data and privacy

- Use synthetic or de-identified records for tests, examples and screenshots.
- Patient text goes only to the LLM endpoints you configure, so send clinical data only to endpoints approved for
  it.
- GoodOption catalog builds receive NCT IDs and drug names only, never patient context.
- Trial status and criteria change over time, so keep source identifiers and dates with derived trial data.

## License

The code is licensed under [CC BY-NC-ND 4.0](LICENSE). The models and datasets the pipeline downloads have their own
licenses; check each before use. For example, LiquidAI d1-2B is under the LFM Open License v1.0.

The GoodOption catalog bundles **NCI Thesaurus (NCIt) 26.07d**, produced by the National Cancer Institute
Enterprise Vocabulary Services group, Center for Biomedical Informatics and Information Technology, National Cancer
Institute, Maryland, USA. It is licensed under the
[Creative Commons Attribution 4.0 International License](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.pdf)
(source: [NCI EVS](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/)). It is used to normalize drug names, assign drug
classes and cite definitions.
- **Unmodified:** the bundled snapshot is unchanged; MatchMiner-AI adds local search, hierarchy traversal and output
  formatting around it.
- **What reaches LLM prompts:** only matched drugs' records and a short list of candidate classes per drug.
- **Trademark:** NCI Thesaurus is a trademark of the National Cancer Institute, and its use does not imply NCI
  endorsement.

## Citation

If you use MatchMiner-AI, please cite:

> Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA, Sammons S, Lindsay J, Mallaber E,
> Klein HR, Gungor G, Galvin M, Deletto M, Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O,
> Ortiz T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI: An Open-Source Solution for
> Cancer Clinical Trial Matching. *arXiv*. 2026. doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)
