# NCI PDQ catalogs

The PDQ source adapter creates the same TrialSpace population, diagnostic/workup,
and treatment-menu records as the guideline extraction API. It reuses extraction
ownership, source-backed consolidation, population coverage, exact excerpt repair,
checkpoint resume, export and final audit. It does not use the contextualization
adapter's ranked, partial PDQ excerpts as a complete catalog source.

## Snapshot sources

Choose complete NCI PDQ **health-professional** summary URLs. A disease can use one
document or an ordered list of related documents. The package downloads only these
documents; it does not follow article citations, fetch images or search using
patient data.

```python
from matchminer_ai.trials import prepare_pdq_library, list_guidelines

sources = "/local/data/pdq/sources-2026-10-07"
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

The external directory retains original HTML bytes, timestamps, document URLs,
headings and anchors, all article text, nested lists, table cells with span
attributes, references and reuse notices. Images are not interpreted; visual-source
limitations are recorded. Update histories and editorial sections remain saved
but are excluded from population-extraction ownership. References remain available
as context and must not be turned into clinical recommendations.

Each derived section is verified by reparsing the original HTML, comparing its
text and provenance, and checking hashes. Unsupported page layouts fail rather
than silently falling back to a shortened text excerpt. Sections are never cut
to an arbitrary character budget. Very large required sections can exceed the
configured LLM context; that remains an explicit failure, not a truncated success.

Repeating the call with the same URL mapping verifies and reuses the snapshot
without network calls. An interrupted download can reuse completed disease
snapshots. Use a **new source directory** to fetch updated articles. The visible
updated date and NCI content metadata date are retained separately when they
differ. The version identifier derives from the original HTML hashes; retrieval
time is not presented as a clinical publication date.

## Extract populations and considerations

```python
from matchminer_ai import load_default_preset
from matchminer_ai.trials import summarize_pdq

config = load_default_preset()
config.remote.update(
    enabled=True,
    server_urls=["http://your-selected-server:8001/v1"],
    max_concurrent_requests=8,
    request_timeout=7200,
)
config.guideline["remote"]["model_name"] = None  # discover the served model

spaces, metadata, qc = summarize_pdq(
    sources,
    disease="nsclc",
    output_dir="/local/data/pdq/catalogs-2026-10-07/nsclc",
    config=config,
    return_metadata=True,
    return_qc=True,
    progress_callback=print,
)
```

`summarize_guidelines` also accepts the same prepared library and selects the PDQ
profile from its verified manifest. `summarize_pdq` is an explicit wrapper that
rejects non-PDQ libraries. Both use `config.guideline` and `config.remote`, full
discovered context, the normal 100,000-token output reserve, reasoning enabled,
and automatic vendor sampling. Qwen 3.8 uses xhigh effort by default. An endpoint
must be supplied explicitly; neither function starts or substitutes a server.
Separate processes must have their combined concurrency budget managed by the
caller when sharing a server with other catalog jobs.

All clinical extraction is performed by the configured endpoint. PDQ prompting
distinguishes explicit care options from trial enrollment criteria, historical
comparators, protocol-mandated tests and exploratory research. It preserves
treatment lines in prior-treatment fields and population-defining biomarkers.
The `category` field retains PDQ evidence-grade wording when supplied; it is not
converted to an NCCN recommendation category. Applicability, timing, alternative
methods and investigational status remain explicit conditions or uncertainties.
These prompt rules require evaluation on real outputs; a passing mechanical audit
does not establish clinical correctness or completeness.

Exports remain `paradigms.jsonl`, `trial_spaces.csv`, `report.md`, `sources.json`,
`run_config.json` and `validation.json`, with source and model checkpoints.
Identifiers use a `pdq-`/`pdq:` namespace. Each evidence item has exact text and line
addresses plus `source_url`, `section_number` and `section_path`; it has no invented
PDF page number. Catalog and patient-retrieval reports link to the source section.
Prompt identities for existing NCCN runs exclude PDQ-only prompt files.

```python
from matchminer_ai.trials import audit_guideline_catalog, get_guideline_considerations
from matchminer_ai.matching import retrieve_guideline_considerations
from matchminer_ai.patients import review_patient_workup

catalog = "/local/data/pdq/catalogs-2026-10-07/nsclc"
audit_guideline_catalog(sources, disease="nsclc", output_dir=catalog)  # offline
exact = get_guideline_considerations(catalog, space_trial_id=spaces.iloc[0].space_trial_id)
ranked = retrieve_guideline_considerations(
    patient_summaries, catalog, candidate_k=5, top_n=1, config=config,
)
review = review_patient_workup(
    raw_patient_notes,
    ranked.iloc[0].diagnostic_workup,
    population_context=ranked.iloc[0].clinical_space_summary,
    config=config,
)
```

The retrieval API uses local TrialSpace and TrialChecker. Workup review sends raw
notes to the configured patient LLM. It checks documentation and applicability
of the stored PDQ considerations, retaining their conditions and source evidence.
It does not imply that all alternative tests were required or establish NCCN
concordance. Patient data must only reach an authorized endpoint.

For a source-checkout CLI, put the URL mapping in an external JSON file:

```bash
PYTHONPATH=src python examples/extract_pdq.py \
  --summaries /local/data/pdq/summaries.json \
  --source-dir /local/data/pdq/sources-2026-10-07 \
  --output-dir /local/data/pdq/catalogs-2026-10-07 \
  --endpoint http://your-selected-server:8001/v1 --concurrency 8
```

Use `--download-only` without endpoint/output arguments to prepare sources without
LLM calls. The existing `extract_guideline_collection.py` runner can process the
prepared library with multiple disease workers sharing its endpoint cap.

## Expand a treatment collection without waiting for catalog tails

`examples/plan_pdq_treatment_collection.py` snapshots NCI's adult and pediatric
treatment indexes and writes an external `inventory.json` and `jobs.json`.
Repeated index aliases become one job; distinct adult/child sources remain
separate. Overview indexes, financial toxicity, survivorship, genomics and the
general transplant overview are explicitly recorded as excluded supporting
topics. Unrecognized links fail for review instead of silently disappearing.

```bash
PYTHONPATH=src python examples/plan_pdq_treatment_collection.py \
  --inventory-dir /local/data/pdq/treatment-index-snapshot \
  --source-dir /local/data/pdq/treatment-sources \
  --catalog-dir /local/data/pdq/treatment-catalogs

PYTHONPATH=src python examples/extract_pdq_collection.py \
  --plan /local/data/pdq/treatment-index-snapshot/jobs.json \
  --output-dir /local/data/pdq/treatment-catalogs \
  --endpoint http://your-selected-server:8000/v1 --concurrency 32
```

To expand an existing `extract_pdq.py` run, also give the planner
`--existing-summaries`, `--existing-source` and `--existing-output`. It preserves
those disease names and source/output paths, including all accepted checkpoints.
Stop the old worker before starting the expanded collection: the shared cap is
process-wide, not shared between independent processes. Changing the plan requires
a new collection metadata directory, but compatible disease output directories
can stay unchanged. Changing an endpoint requires `--allow-endpoint-change` and
the normal per-catalog compatibility checks.

The public API is `summarize_pdq_collection(jobs, output_dir=..., config=...)`.
Each job supplies `source_directory`, `output_directory`, and optionally `urls`.
Existing-source jobs resume immediately; newly downloaded sources enter generation
as soon as ready. Downloads default to two concurrent workers independently of
the shared generation cap. A slow source or a serial consolidation stage does not
hold up other catalogs. Source failures remain visible in `collection.json` while
healthy jobs continue; rerunning reuses verified sources and compatible results.
The returned state's `completed_catalogs` list can be passed directly to
`load_guideline_catalog`, including catalogs in their original directories.

## Interrupt, resume, or move to another endpoint

The PDQ and NCCN collection runners share the same capacity-allocation policy.
By default, up to `--concurrency` available diseases run in parallel. Each disease
can use the full shared endpoint request cap, so finishing catalogs immediately
free slots for remaining work, including the last catalog. `--disease-workers`
is an optional explicit limit on disease preparation, rather than a default
two-disease restriction. Dependencies, source packing and the remaining number
of independent jobs can keep actual traffic below the cap.

The runner writes `collection.json` with the effective disease-worker count,
allocation policy,
its PID, endpoint, stage messages, failures and per-disease completion/audit counts.
Each disease retains the pipeline's independently saved request/response checkpoints.
A directory lock prevents two batch owners from writing to the same output.

Send SIGTERM or Ctrl-C to the runner, or create an empty `STOP` file in its output
directory. These stop local dispatch and mark the batch interrupted. A request
already accepted by the remote server may briefly continue there. Only fully
saved, validated responses are reusable; an interrupted stream is retried.
Remove `STOP` and repeat the original command to resume. Do not start a replacement
worker until the original process has exited.

Concurrency and the explicit disease-worker override may change on resume without
changing clinical checkpoint identities. Omit `--disease-workers` to restore
automatic allocation. Run the collection in one process where possible: separate
processes have independent request caps and need an explicitly managed combined
budget. Automatic allocation does not raise `--concurrency` or choose a server.

To move unfinished work to another instance of the **same model**, repeat the
command with the new explicit `--endpoint` and `--allow-endpoint-change`. The
new server must advertise the same model ID and full context limit. Sampling,
reasoning, source and clinical prompt settings must still match. No fallback
server is selected automatically. Cached token counts are scoped to the actual
destination so the new server's tokenizer performs new packing checks.

For the public API, pass `allow_endpoint_change=True` to `summarize_pdq` or
`summarize_guidelines`. Both PDQ and NCCN catalogs support this opt-in migration.
Original checkpoint identities and raw responses stay intact; `run_config.json`
retains the original `base_url` and current `transport_base_url`, and
`endpoint_history.json` records destinations. Each new request attempt records
its actual transport, which the final audit checks against that history.
The existing general collection runner retains its stricter endpoint ownership;
use the public API for migrating an existing NCCN disease run.

Switching model IDs, context sizes, or generation settings requires a separate
output directory. Keep completed audited catalogs and the previous run's files;
do not relabel old responses as having come from the new model. The endpoint's
advertised ID is a compatibility check, not a verification of model weights.
Resume/retry failed runs only after diagnosing their recorded errors.

## Recover a failed subset

Create a new job plan containing only failed diseases, preserving each job's exact
source/output paths and URLs from the original collection's `identity.jobs`.
Use a separate directory for recovery metadata and point `--parent-collection` at
the original report:

```bash
PYTHONPATH=src python examples/extract_pdq_collection.py \
  --plan /local/data/pdq/failed-jobs.json \
  --output-dir /local/data/pdq/recovery \
  --parent-collection /local/data/pdq/treatment-catalogs/collection.json \
  --endpoint http://your-selected-server:8000/v1 --concurrency 32
```

The recovery worker locks both collection directories, updates the original
report as each disease progresses, and leaves completed catalogs outside its plan
untouched. Stop the original worker first. The Python API accepts the same option
as `parent_collection=...`. Interrupt/resume and explicit endpoint migration work
the same way as for a full collection.

Recovery is part of the shared PDQ/NCCN pipeline. Exhausted extraction tasks can
split into smaller, uniquely owned page groups; final population-coverage failures
can narrow affected consolidation batches. Both keep durable split records and
reuse compatible accepted checkpoints. When optional cross-batch deduplication
repeatedly omits or rewrites populations, the pipeline retains all validated
definitions, records that semantic duplicates may remain, and still requires the
full population-coverage check to pass.

The same retention path handles a repetitive deduplication response. If selection
passes its local review but fails the final original-population review, the
pipeline first restores the complete validated union and checks coverage again.
The rejected selection and review stay in the audit record; source batches are
revisited only if the complete union still fails. Exhausted extraction responses
with ambiguous Boolean grouping receive focused feedback and retry headroom while
retaining all owned source pages and the full output reserve.
If an exhausted extraction confuses table rows or display wrapping with literal
source line addresses, a bounded retry repeats the owned section and restricts
the address schema to actual primary pages and nonblank line numbers. The source
is not renumbered, clinical content is generated by the endpoint, and all original
field, page-accounting, ownership and citation checks still apply. A branch on a
different owner's page does not qualify for this retry.

Targeted retries can constrain immutable population fields and coverage names to
their exact supplied values. Citation repair chooses literal supporting source
text, treating the rejected draft as fallible. These retries do not relax exact
quotation, population coverage, source ownership, or final audit requirements.
Clinical output remains model-generated and retains its raw-response provenance.

When exact-definition deduplication combines records, their original clinical names
are retained in `population_names`, together with the union of source evidence and
batch provenance. Detail generation receives those names and associated candidate
menus. Coverage can recognize an exact original population through that audited name
history; a matching name with different clinical restrictions still requires review.

Output-limited coverage reviews split into smaller input groups while retaining the
complete proposed catalog. Exhausted extraction prompts can repack whole optional
supporting pages to reserve 8,192 input tokens for retry feedback; every owned primary
page and the configured output-token reserve remain intact. Boolean diagnostics
identify the complete offending parenthesis group, including errors late in a field.

If a singleton consolidation exhausts its coverage retries, the pipeline can send its
original extraction owner back to the endpoint for source review. Both the original
extraction and the conflicting review are presented as fallible. The model re-extracts
the entire owned packet with the original validation rules. This is bounded to one
review per packet and two packets per catalog. `extraction_source_reviews.json` keeps
the original extraction and trigger; changed candidates cause downstream artifacts
to move to `superseded_candidate_artifacts/` before rebuilding. Raw provider responses
remain intact, and final audit verifies both original and replacement provenance.
An unresolved disagreement still fails; no clinical field is patched by Python.

## Source meaning and reuse

[NCI describes PDQ](https://www.cancer.gov/publications/pdq/information-summaries)
as reviewed, referenced evidence summaries, explicitly not clinical practice
guidelines. A PDQ-derived catalog can provide a distinct, source-defined benchmark
for care review; it must retain that identity and distinguish expected care,
acceptable alternatives and research descriptions.

[PDQ evidence grades](https://www.cancer.gov/publications/pdq/levels-evidence/treatment)
describe evidence rather than NCCN recommendation categories.
[NCI's reuse policy](https://www.cancer.gov/policies/copyright-reuse) permits reuse
of much of its text with attribution, subject to page-specific exceptions for
third-party material. Tables and reproduced material may carry such notices.
Do not treat a derived catalog as an unchanged official PDQ product or an NCI
endorsement. Source snapshots and generated catalogs are deliberately not bundled
with this package. Review the relevant notices before redistributing them.
