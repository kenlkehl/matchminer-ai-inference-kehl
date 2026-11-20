# NCI PDQ catalogs

The PDQ source adapter builds guideline catalogs from NCI PDQ health-professional summaries instead of a local NCCN
library. It uses the same pipeline and produces the same records: spaces with nine population fields and
cohort-style key criteria, workup and treatment menus, exact source evidence and uncertainties. Configuration,
extraction stages, checks and retries, output files, audit, resume, loading and retrieval work as described in
[Guideline extraction](guideline-extraction.md). This page covers what differs for PDQ.

[NCI describes PDQ](https://www.cancer.gov/publications/pdq/information-summaries) summaries as reviewed,
referenced evidence summaries, explicitly not clinical practice guidelines. A PDQ catalog is a distinct,
source-defined reference for care review. It keeps that identity and separates expected care, acceptable
alternatives and descriptions of research. Like any catalog, it is research extraction that needs human review, not
treatment advice or an eligibility determination.

## Snapshot sources

Choose complete PDQ **health-professional** summary URLs. A disease can use one document or an ordered list of
related documents.

```python
from matchminer_ai.trials import list_guidelines, prepare_pdq_library

sources = "/path/to/pdq/sources-2026-10"
prepare_pdq_library(
    {
        "nsclc": "https://www.cancer.gov/types/lung/hp/non-small-cell-lung-treatment-pdq",
        "breast": "https://www.cancer.gov/types/breast/hp/breast-treatment-pdq",
    },
    output_dir=sources,
    progress_callback=print,
)
print(list_guidelines(sources))
```

| Parameter | Meaning |
| --- | --- |
| `summaries` | Disease names (lowercase slugs such as `nsclc` or `small-cell-lung`) mapped to a URL or an ordered list of URLs |
| `output_dir` | The snapshot directory, outside the code repository: empty, or an earlier snapshot of the same selection |
| `request_timeout` | Seconds per download (default 60) |
| `progress_callback` | Receives progress strings |

It returns the same inventory table as `list_guidelines`.

- **What is fetched:** only the listed documents, over HTTPS from `www.cancer.gov`, following validated redirects, at
  most 32 MiB each. It does not follow article links or citations, download images, or search with patient data.
- **What is kept:** the original HTML bytes, retrieval times, document URLs, headings and anchors, all article text,
  nested lists, table cells with their span attributes, references and reuse notices. Images are not interpreted;
  that limitation is recorded with the source.
- **Sections:** each document is split into sections (the citation unit, in place of NCCN's PDF pages) under
  `markdown/<disease>/sections/`. "Latest Updates", "Changes to This Summary" and "About This PDQ" sections are kept
  but used only as context, never mined for populations. References are context and are never turned into
  recommendations.
- **Verification:** every section is checked by reparsing the original HTML and comparing its text, provenance and
  hashes. An unsupported page layout fails rather than falling back to a shortened excerpt. Sections are never cut
  to a character budget; a required section larger than the LLM's context fails explicitly.

Repeating the call with the same mapping verifies and reuses the snapshot without network calls, and an interrupted
download resumes from the diseases already captured. A different selection needs a **new directory**, as does an
update of the articles. The edition (`version`, `snapshot-<hash>`) comes from the HTML hashes. The page's visible
updated date and NCI's content metadata date are kept separately when they differ; the retrieval time is not
presented as a publication date.

## Extract a catalog

```python
from matchminer_ai import load_default_preset
from matchminer_ai.trials import summarize_pdq

config = load_default_preset()
config.remote.update(
    enabled=True,
    server_urls=["http://localhost:8000/v1"],
    max_concurrent_requests=8,
    request_timeout=7200,
)

spaces, metadata, qc = summarize_pdq(
    sources,
    disease="nsclc",
    output_dir="/path/to/pdq/catalogs-2026-10/nsclc",
    config=config,
    return_metadata=True,
    return_qc=True,
    progress_callback=print,
)
```

`summarize_pdq` takes the same parameters as `summarize_guidelines`, including `allow_endpoint_change`, and rejects a
library that is not a PDQ snapshot. `summarize_guidelines` also accepts a PDQ snapshot and selects the PDQ profile
from its manifest. Both use `config.guideline` and `config.remote` with the defaults in
[Configure the endpoint](guideline-extraction.md#configure-the-endpoint); the endpoint must be given explicitly.

**Prompts.** The PDQ system message (`pdq.system.txt`, `pdq.trial_space_reference.txt`) states the library's purpose
and that organizing PDQ evidence does not make it a guideline. It includes the package's trial-space prompt
(`trial.user.primer.txt`, `trial.user.question.txt`) verbatim as the reference for what a population is, followed by
PDQ adaptations: JSON output, care populations defined by the source only, explicit treatment history and fixed
definitions in later stages. The trial prompt's plain-text output, boilerplate section, assumed sex criteria and
washout rules do not apply. These files are part of a PDQ run's checkpoint identity, not of NCCN runs.

**Content rules.** The prompts separate explicit care options from trial enrollment criteria, historical
comparators, protocol-mandated tests and exploratory research.

- **Prior treatment:** the fields keep the source's prior drugs or classes, disease setting and response, and the
  biomarkers that define a population. Names and fields use explicit history rather than treatment-line shorthand,
  and an undefined line heading does not license invented regimen counts or progression requirements.
- **Category:** an item's `category` keeps PDQ's evidence-grade wording where given; it is not converted to an NCCN
  recommendation category.
- **Conditions:** applicability, timing, alternative methods and investigational status stay explicit conditions or
  uncertainties.

These are prompt rules whose effect needs evaluation on real outputs; a passing audit does not establish clinical
correctness or completeness.

**Records.** The output files and columns are those of
[an NCCN catalog](guideline-extraction.md#catalog-records), with these differences:

| Field | PDQ value |
| --- | --- |
| `space_trial_id`, `paradigm_id` | `pdq-<disease>-<hash>` |
| `trial_id` | `pdq:<disease>:<version>` |
| `evidence` items | `page_id` (the section), `line_ids`, `source_lines`, `quote`, `source_url`, `section_number`, `section_path`, `printed_label`; no PDF page |
| `source` | Also `source_kind: nci_pdq`, `is_clinical_practice_guideline: false`, the documents and the PDQ notice |

`report.md` links each cited section to its URL and its saved copy. The patient retrieval report links evidence to
the section URLs and names each source document with its attribution and update date.

## Use a PDQ catalog

The catalog APIs take PDQ catalogs without conversion:

```python
from matchminer_ai.matching import retrieve_guideline_considerations
from matchminer_ai.patients import review_patient_workup
from matchminer_ai.trials import audit_guideline_catalog, get_guideline_considerations

# patient_summaries: DataFrame with patient_id and cancer_history_summary
# raw_patient_notes: one patient's notes (text, or a DataFrame with note_text and optional note_date)
catalog = "/path/to/pdq/catalogs-2026-10/nsclc"
audit_guideline_catalog(sources, disease="nsclc", output_dir=catalog)    # offline
record = get_guideline_considerations(catalog, space_trial_id=spaces.iloc[0]["space_trial_id"])

ranked = retrieve_guideline_considerations(patient_summaries, catalog, top_n=1, candidate_k=5, config=config)
review = review_patient_workup(
    raw_patient_notes,
    ranked.iloc[0]["diagnostic_workup"],
    population_context=ranked.iloc[0]["cohort_key_criteria"],
    config=config,
)
```

Retrieval runs TrialSpace and the decision model locally (set `config.embedding["model_path"]` to your TrialSpace
checkpoint; see [Retrieve considerations for patients](guideline-extraction.md#retrieve-considerations-for-patients)).
`review_patient_workup` sends raw notes to the configured patient LLM, so the endpoint must be authorized for patient
data. It checks whether the stored PDQ workup items are documented and applicable, keeping their conditions and
evidence. It does not imply that every alternative test was required and does not establish NCCN concordance.

## Command-line runners

All runners keep sources, catalogs and collection metadata outside the code repository, take an explicit
`--endpoint`, and discover the served model unless `--model` gives its exact ID.

**One snapshot, several diseases.** `examples/extract_pdq.py` takes the URL mapping as a JSON file, prepares the
snapshot, then extracts each disease to `<output-dir>/<disease>/` with progress in `<output-dir>/collection.json`:

```bash
python examples/extract_pdq.py \
  --summaries /path/to/pdq/summaries.json \
  --source-dir /path/to/pdq/sources-2026-10 \
  --output-dir /path/to/pdq/catalogs-2026-10 \
  --endpoint http://localhost:8000/v1 --concurrency 8
```

| Option | Meaning |
| --- | --- |
| `--summaries`, `--source-dir` | Required: the URL mapping and the snapshot directory |
| `--download-only` | Prepare the snapshot only; `--output-dir` and `--endpoint` are then not needed |
| `--output-dir`, `--endpoint` | Required for extraction |
| `--model` | Exact served model ID; omit to discover it |
| `--concurrency` | Requests in flight (default 8), shared by all diseases |
| `--disease-workers` | Optional cap on diseases running at once; default up to `--concurrency` |
| `--request-timeout` | Seconds per request (default 7200) |
| `--allow-endpoint-change` | Resume unfinished catalogs on another server of the same model |

`examples/extract_guideline_collection.py` also accepts a prepared PDQ snapshot as `--source`.

**The full treatment collection.** `examples/plan_pdq_treatment_collection.py` snapshots NCI's adult and pediatric
treatment indexes and writes a job plan; `examples/extract_pdq_collection.py` runs it:

```bash
python examples/plan_pdq_treatment_collection.py \
  --inventory-dir /path/to/pdq/treatment-index \
  --source-dir /path/to/pdq/treatment-sources \
  --catalog-dir /path/to/pdq/treatment-catalogs

python examples/extract_pdq_collection.py \
  --plan /path/to/pdq/treatment-index/jobs.json \
  --output-dir /path/to/pdq/treatment-collection \
  --endpoint http://localhost:8000/v1 --concurrency 32
```

- **Planner:** writes the index snapshots, `inventory.json` and `jobs.json` to `--inventory-dir`, with one job per
  summary whose source and catalog go under `--source-dir/<disease>` and `--catalog-dir/<disease>`. A summary listed
  under several labels becomes one job; adult and pediatric summaries stay separate. Overview indexes, financial
  toxicity, survivorship and late effects, genomics and the general transplant overview are recorded as excluded
  topics, and an unrecognized link fails for review rather than being dropped. A changed plan needs a new
  `--inventory-dir`.
- **Reusing an `extract_pdq.py` run:** give the planner `--existing-summaries`, `--existing-source` and
  `--existing-output` together. Those diseases keep their names, snapshot and catalog directories, and accepted
  checkpoints.
- **Runner options:** `--plan`, `--output-dir` (collection metadata) and `--endpoint` are required. `--model`,
  `--concurrency` (default 32), `--source-workers` (concurrent downloads, default 2), `--disease-workers`,
  `--request-timeout` (default 7200), `--allow-endpoint-change` and `--parent-collection` (see below) are optional.

The Python API is `summarize_pdq_collection(jobs, output_dir=..., config=...)`, with `source_workers`,
`disease_workers`, `allow_endpoint_change`, `progress_callback` (receives dicts) and `parent_collection`.

- **Jobs:** each maps a disease slug to `source_directory`, `output_directory` and optional `urls`. A job with `urls`
  downloads its snapshot first; one without starts at once.
- **Scheduling:** downloads run beside generation, so a slow source or a serial consolidation stage does not hold up
  other catalogs. A failed source or catalog is recorded in `collection.json` while the others continue, and a rerun
  reuses verified sources and compatible results.
- **Result:** the returned state's `completed_catalogs` can be passed straight to `load_guideline_catalog`.
- **Changes:** a changed plan needs a new collection directory, but compatible disease directories can stay where
  they are.

## Interrupt, resume and recover

The PDQ runners share the NCCN runner's scheduling: each active disease can use the endpoint's full request cap, so
finishing catalogs free their slots at once, and `--disease-workers` is only an optional limit. `collection.json`
records the endpoint, concurrency, effective disease workers, the runner's PID, stage messages, failures and each
disease's completion and audit status. A directory lock prevents two runners from writing the same output.

- **Stop:** send SIGTERM or Ctrl-C, or create an empty `STOP` file in the output directory. Local dispatch stops and
  the batch is marked interrupted. A request already accepted by the server may briefly continue there; only fully
  saved, validated responses are reused, and an interrupted stream is retried.
- **Resume:** wait for the process to exit, remove `STOP` and repeat the command. Concurrency and `--disease-workers`
  may change without affecting checkpoints. Separate processes have independent request caps, so keep one runner per
  endpoint where possible.
- **Another server of the same model:** repeat the command with the new `--endpoint` and `--allow-endpoint-change`
  (`allow_endpoint_change=True` in the API). The rules are those of
  [endpoint changes](guideline-extraction.md#resume-and-change-endpoints); no fallback server is chosen
  automatically.
- **Another model, context size or generation settings:** use new output directories. Keep completed catalogs and
  the previous run's files, and never relabel old responses as coming from the new model.

**Recover a failed subset.** Write a plan with only the failed diseases, copying each job's source and output paths
and URLs from the original `collection.json` (`identity.jobs`). Run it with a separate metadata directory and point
`--parent-collection` (`parent_collection=` in the API) at the original report:

```bash
python examples/extract_pdq_collection.py \
  --plan /path/to/pdq/failed-jobs.json \
  --output-dir /path/to/pdq/recovery \
  --parent-collection /path/to/pdq/treatment-collection/collection.json \
  --endpoint http://localhost:8000/v1 --concurrency 32
```

Stop the original runner first. The recovery locks both directories, refuses jobs whose paths differ from the parent
plan or that would rerun completed catalogs, and updates the original report as each disease progresses. Diagnose
the recorded errors before retrying a failed run.

## Source reuse

[PDQ evidence grades](https://www.cancer.gov/publications/pdq/levels-evidence/treatment) describe evidence, not NCCN
recommendation categories. [NCI's reuse policy](https://www.cancer.gov/policies/copyright-reuse) permits reuse of
much of its text with attribution, with page-specific exceptions for third-party material; tables and reproduced
material can carry such notices. Do not present a derived catalog as an unchanged official PDQ product or as
endorsed by NCI. Source snapshots and generated catalogs are not bundled with this package; review the relevant
notices before redistributing them.
