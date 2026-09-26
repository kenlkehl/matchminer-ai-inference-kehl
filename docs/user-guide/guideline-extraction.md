# Guideline extraction

`matchminer_ai.trials.summarize_guidelines` turns one locally supplied NCCN
guideline into canonical disease-state populations in the same nine-field text
format returned by `summarize_trials`. Each population also has diagnostic/workup
considerations, treatment/management options, source quotations, and uncertainties.
All clinical content is extracted by the configured LLM. Python manages source
ownership, record identity, validation, persistence, and export.

This is an optional source-grounded extraction workflow, separate from trial
matching. A guideline population is not a clinical trial, a treatment recommendation,
or an eligibility determination. Structural checks do not establish clinical
correctness or completeness. Review the original PDF when graphical relationships
or clinical applicability are uncertain.

## Supply a local source library

No NCCN source files or generated catalogs are distributed with this package.
Supply your own permitted local copies. The reader accepts the page-preserving
`nccn-pdf-to-markdown` manifest format: a collection `markdown/manifest.json`,
per-disease manifests and Markdown page files, plus the original PDFs alongside
`markdown/`. Each page has a fenced text block and a recorded hash. PDF hashes,
page coverage, and Markdown hashes must agree before any model call. Incomplete
copies fail explicitly and can be retried after copying finishes.

For a new collection, the repository includes `scripts/convert_guidelines.py`.
Install `matchminer-ai[guideline-conversion]` and Poppler's `pdftotext`, then run
from a source checkout:

```bash
python scripts/convert_guidelines.py \
  --input-dir /path/to/local/guideline-pdfs \
  --output-dir /path/to/local/guideline-pdfs/markdown
```

The converter requires explicit external paths and makes no network or model calls.
It preserves text, page addresses, and links; it does not perform OCR or recover
every arrow, table relationship, superscript, or graphical decision branch. Retain
the PDF for human review. Conversion provenance is in `scripts/README.md`.

## Extract one disease

```python
from matchminer_ai import load_default_preset
from matchminer_ai.trials import list_guidelines, summarize_guidelines

source = "/path/to/local/guideline-pdfs/markdown"
print(list_guidelines(source))  # Use the disease folder name from this table.

config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://camus:8002/v1"]
config.remote["request_timeout"] = 7200
config.remote["max_concurrent_requests"] = 32

spaces, metadata, qc = summarize_guidelines(
    source,
    disease="breast",
    output_dir="/path/to/private-derived-data/breast-run-1",
    config=config,
    return_metadata=True,
    return_qc=True,
    progress_callback=print,
)
```

The function accepts one disease per call and returns the same flag combinations
as `summarize_trials`: a DataFrame, optionally followed by metadata and/or a QC
DataFrame. The standard columns are `space_trial_id`, `trial_id`,
`clinical_space_number`, `clinical_space_summary`, and `general_exclusion_criteria`.
Here `trial_id` has a `guideline:` namespace, and `general_exclusion_criteria` is
`NA` because no trial protocol was summarized. These IDs must not be sent to
services expecting NCT identifiers.

Additional columns include the nine-field `space` object, `diagnostic_workup`,
`treatment_options`, `evidence`, `source`, `uncertainties`, and omitted context.
Each menu item has its conditions, source-stated category, and evidence. The
standard text/ID columns can be passed to the package's TrialSpace embedding API;
this function does not load an embedding model or perform matching.

## Endpoint and budget configuration

The collection runner can give unstarted catalogs to another same-model server:

```bash
python examples/extract_guideline_collection.py \
  --source /path/to/local/guideline-pdfs/markdown \
  --output-dir /path/to/private-derived-data/collection \
  --endpoint http://original-server:8001/v1 \
  --additional-endpoint http://additional-server:8060/v1 \
  --new-disease-endpoint http://additional-server:8060/v1 \
  --model served-model-id --concurrency 32 --disease-workers 8
```

Retain the original invocation's source, output, primary endpoint, model and
completed-catalog arguments when resuming. Additional endpoints are preflighted
for the same model ID and advertised context window. Each disease with an existing
`run_config.json` keeps its recorded endpoint and checkpoints. With
`--new-disease-endpoint`, every unstarted catalog goes to that server; otherwise
unstarted work is balanced between configured endpoints. Separate disease-worker
pools allow the additional server to start immediately. The concurrency and
disease-worker caps apply **per endpoint**. `collection.json` records the endpoint
assigned to each disease and the preflight model information. Stop the original
collection process before resuming; do not run two collection writers together.
Each disease still uses the standard single-endpoint API and provenance checks.

The existing `MMAIConfig` owns configuration. Shared `config.remote` selects the
endpoint, concurrency, timeout, and maximum request attempts (`max_retries`).
The new `config.guideline` stage uses the same `remote.model_name`,
`remote.request_params`, and `remote.extra_body` convention as other LLM stages.
Existing trial and patient defaults are unchanged.

| Guideline setting | Default / behavior |
| --- | --- |
| `remote.model_name` | `None`: discover the served model from `/v1/models` |
| `context_window` | `None`: use the full advertised `max_model_len`; require an explicit value if unavailable |
| `remote.request_params.max_tokens` | 100,000 reserved output tokens, including reasoning |
| `remote.request_params.temperature` / `top_p` | 1.0 / 0.95 |
| `remote.extra_body.top_k` | Vendor profile: Gemma 4 = 64, Qwen 3.8 = 20 |
| `remote.extra_body.chat_template_kwargs.enable_thinking` | `True` |
| `reasoning_effort` | `xhigh` for Qwen 3.8; Gemma uses its thinking switch |
| `safety_tokens` | 2,048 tokens for retry feedback |
| `packet_pages` | 8 PRIMARY pages per extraction call |
| `tokenizer_mode` | `endpoint`: count the actual chat template through vLLM `/tokenize` |
| `response_format` | `json_schema`; also supports `json_object` or `none` |
| `stream` | `True`, preserving raw content, reasoning, usage, and finish reasons |
| `api_key_env` | `OPENAI_API_KEY`; only the environment variable name is persisted |

The stage resolves the vendor generation profile from the actual served model.
See [model-aware sampling](../reference/configuration.md#model-aware-sampling-and-reasoning).
For another provider, explicitly select its model/context, sampling parameters,
supported response format, and reasoning controls. Set `sampling_profile="none"` before removing `top_k` and
`chat_template_kwargs` from `extra_body` to omit those vLLM extensions. If `/tokenize`
is unavailable, set `tokenizer_mode="bytes"` to use conservative UTF-8 byte
accounting; this is not a model-token measurement and is never selected silently.
Do not put credentials in URLs or request parameters; use the named environment
variable. This initial API supports one remote endpoint and does not launch a
server or support local vLLM generation.

Context packing keeps whole pages, prioritizes required source evidence, and uses
all available source context when it fits. Every omission is recorded. The output
reserve is never reduced to make input fit. Oversized mandatory evidence fails
with instructions to reduce packet size or choose a larger context. A response
ending at a token limit is rejected even if its JSON parses. Sustained repetitive
output is stopped, recorded, and retried within the configured attempt bound.

## Population ownership and field meanings

Each source page has one PRIMARY extraction owner; other calls may use it as
supporting CONTEXT. A candidate anchors its defining branch to owned source lines.
Routing text and discussion paraphrases should defer to the defining treatment
table's owner. Source addresses are the only persistent identifiers the model
must cite. Candidate/paradigm IDs, hashes, and batching bookkeeping stay in Python.
Consolidation returns clinical definitions rather than arbitrary ID mappings.

The prompts preserve one current treatment decision per space. Source-defined
treatment lines and prior exposure belong in the prior-treatment fields; burden
contains extent, stage, resectability, and disease-severity states. Visceral crisis
is burden and endocrine refractoriness is treatment response. Alternatives across
fields are split without changing OR to AND. Generic breast HR-negative expands
to ER-negative AND PR-negative; HR-positive expands to ER-positive OR PR-positive.
Detail generation freezes the canonical definition and supplies its current menu.

Prompts, including field descriptions, consolidation, selection, and citation
repair instructions, live in `src/matchminer_ai/prompts/guideline.*.txt` and
`structured.retry.txt`. These generation prompts are fingerprinted for resume checks.
Citation-only repair can change references and page accounting, never clinical
fields, option content, or defining branch ownership.

For failed structured generations, retries include the most recent complete JSON
draft and validation feedback, using `structured.revise_retry.txt`. Extraction
feedback reports every candidate that fails the field checks, rather than stopping
at the first candidate. The LLM revises its own draft against the original source;
code never guesses Boolean grouping or edits clinical criteria. Incomplete JSON
and reasoning traces are not reused. If the draft will not fit, retry with feedback
alone, keeping all original source context and the full output-token reserve.
Every actual retry message is saved in `request-attempt-*.json`. The revision-only
instruction does not invalidate already accepted generation checkpoints; accepted
responses still undergo the same grounding and field validation.

## Persistence and review

Use a separate external output directory for each disease, edition, and generation
configuration. The API rejects directories inside the source collection or code
repository. **Outputs contain source-derived and sometimes verbatim guideline
text**, including checkpoint prompts and raw responses. Keep the entire output
directory outside source control and public artifacts, regardless of debug mode.

The directory includes `paradigms.jsonl`, `trial_spaces.csv`, `report.md`, source
provenance, coverage, ownership, run settings, `status.json`, per-request
checkpoints, and `validation.json`. A failed stage raises rather than returning
a partial catalog. Repeat the same call to resume completed requests; token counts
and accepted responses are cached. Changes to concurrency, timeout, retry count,
or streaming transport are allowed without discarding work. Source, prompt, model,
and generation changes require a new directory. Use fresh directories when moving
from the standalone prototype; its old generated outputs are not imported.

```python
from matchminer_ai.trials import audit_guideline_catalog

audit = audit_guideline_catalog(
    source,
    disease="breast",
    output_dir="/path/to/private-derived-data/breast-run-1",
)
```

The offline audit verifies source hashes, ownership, exact quotations, raw-response
preservation, budgets, and agreement among exports. QC separately reports uncertain
source pages, states with uncertainties or omitted context, and empty treatment
menus. Neither audit nor QC certifies completeness or clinical appropriateness.

## Retrieve considerations from an existing catalog

Use the complete `paradigms.jsonl` export or its containing disease directory.
The earlier standalone prototype's complete JSONL catalogs are supported without
regenerating them. `trial_spaces.csv` omits the diagnostic/treatment menus and is
not sufficient for consideration lookup. A list of catalog paths can combine
several diseases or editions; space identifiers must remain unique.

```python
from matchminer_ai.trials import load_guideline_catalog, get_guideline_considerations

catalog_path = "/path/to/private-derived-data/breast-run-1"
catalog = load_guideline_catalog(catalog_path)
record = get_guideline_considerations(
    catalog,
    space_trial_id=catalog.iloc[0]["space_trial_id"],
)
print(record.iloc[0]["diagnostic_workup"])
print(record.iloc[0]["treatment_options"])
```

Lookup requires a catalog ID or the exact `clinical_space_summary` text. Unknown
spaces raise an error rather than silently substituting another population. Text
lookup may return multiple source editions; use an ID for one specific record.
Loading checks the stored definition, menus, and evidence structure. If a stored
status/audit exists, it must report completion/success and the exported JSONL hash
must match. These are checks of the stored catalog, not a new source or clinical
audit. Bare JSONL and DataFrame inputs explicitly have unavailable audit status.

Validated file records are cached in process memory by default. Subsequent lookup
and patient retrieval calls check the identity, size, modification time, and change
time of `paradigms.jsonl`, `status.json`, and `validation.json`; unchanged files
reuse their parsed, validated records without rereading or rehashing their contents.
Only new or changed catalogs are reloaded. Changes or failures never fall back to
an older cached version. Returned nested records are independent copies, and
combined catalogs still undergo identifier/space-number uniqueness checks.
`metadata["validation_cache"]` reports reused and loaded file counts (under
`metadata["catalog"]` for patient retrieval). Progress distinguishes checking
files, loading/validating changed catalogs, and reusing validated records.

The cache is bounded to 128 files and 512 MiB of source JSONL; Python records have
additional memory overhead. It disappears on process restart and contains no
patient summaries or results. File-change detection follows the filesystem's
metadata visibility, including any network filesystem delay. Call
`load_guideline_catalog(path, refresh=True)` to force a full read and validation
and replace a cached version. DataFrame inputs are always copied and validated.

## Retrieve considerations for patient summaries

```python
import pandas as pd
from matchminer_ai import load_default_preset
from matchminer_ai.matching import (
    retrieve_guideline_considerations,
    write_guideline_considerations_report,
)

patients = pd.DataFrame([
    {"patient_id": "example", "cancer_history_summary": "Your patient summary here"}
])
config = load_default_preset()
# Set both to "cpu" when running without a supported GPU.
config.embedding["device"] = "cuda"
config.raw["match_quality"]["device"] = "cuda"

ranked, metadata = retrieve_guideline_considerations(
    patients,
    catalog_path,
    top_n=5,
    candidate_k=20,
    config=config,
    return_metadata=True,
    progress_callback=print,
)
write_guideline_considerations_report(
    ranked,
    patients,
    "/path/to/private-derived-data/patient-review.md",
    metadata=metadata,
)
```

This composes the existing public functions in order:

1. `embed_for_matching` embeds the patient summaries and the catalog's TrialSpace
   definitions with the same configured TrialSpace model. Catalog vectors are
   computed once per call for all supplied patients, or reused from the optional
   disk cache described below.
2. `generate_candidate_matches` retrieves `candidate_k` spaces by cosine similarity.
   Set `candidate_k=None` to send all catalog spaces to the checker.
3. `score_match_quality` scores the retrieved patient-space pairs with TrialChecker.
4. The highest `top_n` spaces per patient are returned with their stored diagnostic
   and treatment considerations. `rank` is TrialChecker order; `retrieval_rank`
   retains the earlier TrialSpace order. Ties use cosine similarity, then space ID.

The unit of ranking is a **space**, so multiple populations from one guideline can
appear in the results. No disease-level/trial-ID deduplication occurs. `candidate_k`
must be at least `top_n`; a smaller catalog returns all available spaces. TrialChecker
cutoff flags are retained but do not remove candidates from the top-N result. Scores
use the existing package's `match_quality_score` semantics and are not calibrated
probabilities of patient-specific applicability.

No generative LLM rewrites the menus or adjudicates individual options. Stored
conditions, categories, evidence, and uncertainties are returned unchanged. A
high-ranking population does not establish that every listed therapy applies to
this patient. The report displays the actual summary, both ranks/scores, the
population definitions, menus, caveats, local PDF links when available, and stored
source excerpts for review. Reports contain patient text and source-derived
content and must stay outside the code repository and source collection.

The embedding and checker inference are local. On a first run the configured
model weights/metadata may be downloaded from Hugging Face; no patient text or
catalog text is used in those requests. To use existing model caches entirely
offline, set `HF_HUB_CACHE` and `HF_HUB_OFFLINE=1` before imports and point
`config.model_metadata_cache_dir` at matching cached model metadata. Both stages
must use compatible dated model releases; this workflow does not accept external
precomputed vectors that might have been produced by another TrialSpace model.

Pass `embedding_cache_dir="/path/to/private/guideline_embeddings"` to
`retrieve_guideline_considerations` to persist guideline vectors across calls and
process restarts. The default is `None` (no disk vector cache). The cache stores
vectors and input hashes, never patient summaries, patient vectors, or guideline
text. Keep it outside code repositories because the vectors are source-derived.
Identical text can reuse vectors across disease selections and catalog updates;
new or changed definitions are embedded individually. Catalog validation and
source-integrity checks still run on every call, and returned menus always come
from the current catalog.

Cache identity includes the actual loaded Hugging Face encoder revision,
tokenizer content, prompt, sequence cutoff, embedding implementation, and library
versions. A mutable local model directory or an encoder without a verifiable
loaded revision bypasses persistent caching. Corrupt vector entries are rebuilt;
an unavailable cache falls back to fresh computation. Metadata records cache
hits, newly embedded texts, and the cache identity under
`guideline_embedding_cache`; progress callbacks report reuse. Patient embedding
and TrialChecker scoring still run for every request, and model weights must
still load on the first request after restarting the process.

The source-checkout example `examples/retrieve_guideline_considerations.py` reads
one selected patient from a Parquet summary table and writes the review Markdown,
ranked JSONL, input summary, and run provenance to an external directory. Its
`--patient-id-column` and `--summary-column` options adapt training-era columns
such as `pseudo_mrn` and `patient_summary` to the public API names. Optional
`--synthetic-notes` verifies the selected ID occurs in a supplied synthetic note
table and records that provenance. No sample patient or NCCN catalog is bundled.
Its `--embedding-cache-dir` option enables the same persistent guideline-vector
cache as the public retrieval API.

## Process the remaining collection

The source-checkout runner composes `summarize_guidelines` across diseases while
sharing one endpoint request cap. Several diseases can prepare context or perform
serial consolidation while others fill available generation slots. At most 32
HTTP requests are active at once with the following configuration; streaming
responses hold their slots until completion or failure.

```bash
PYTHONPATH=src python examples/extract_guideline_collection.py \
  --source /path/to/local/guideline-pdfs/markdown \
  --output-dir /path/to/private-derived-data/remaining-run \
  --endpoint http://your-server:8001/v1 \
  --model Qwen/Qwen3.8-Flash-Next \
  --concurrency 32 --disease-workers 8 \
  --completed-catalog /path/to/private-derived-data/completed-disease
```

Repeat `--completed-catalog` for each existing catalog. Skipping requires a passing
stored audit, verified JSONL hash, and a source PDF hash matching the current
collection manifest. Model provenance remains attached to each catalog; the runner
does not relabel older outputs as Qwen-generated. Optional repeated `--disease`
arguments restrict the run. Missing source files or failed diseases are recorded
in external `collection.json`; other diseases continue. A collection is marked
complete only when every selected disease passes its extraction and audit.
Repeat the identical command to resume accepted checkpoints and retry failures.
The collection runner keeps all generated content outside the repository.

To recover only diseases currently marked failed, add `--retry-failed` to the
same command. This writes a separate `recovery.json` rather than competing with
a live batch's `collection.json`; each disease retains its original output and
accepted checkpoints. The original collection summary remains unchanged until
the normal collection command is resumed. During a concurrent recovery, use a
small `--concurrency` value: request limits are shared within each process, not
across separate runners. Extraction retries retain up to four recent validation
diagnostics, including when resuming saved responses, so fixing one field does
not discard guidance about earlier errors. Validation still must pass before
any response is accepted.
