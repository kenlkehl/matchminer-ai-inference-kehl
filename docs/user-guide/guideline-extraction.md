# Guideline extraction

A guideline catalog describes the patient populations of one disease guideline. Each population is a **space**, one
self-contained combination of patient criteria, the same unit as a trial cohort. A space carries:

- **Definition:** nine population fields (`space`), also written as one line in `clinical_space_summary`;
- **Key criteria:** the space written as a trial cohort's key criteria (`key_criteria`, loaded as
  `cohort_key_criteria`), which TrialSpace and the decision model read;
- **Considerations:** diagnostic/workup items and treatment/management options, each with its conditions and the
  source's category;
- **Evidence:** exact source quotations with page and line addresses, plus the uncertainties the model recorded.

`trials.summarize_guidelines` builds a catalog from a locally supplied NCCN library; this page describes that path.
[NCI PDQ catalogs](pdq-catalogs.md) use the same pipeline and file format with PDQ summaries as the source.
`matching.retrieve_guideline_considerations` then ranks a catalog's spaces for a patient summary and returns their
stored considerations.

The configured LLM does all clinical extraction. Code assigns source ownership and record IDs, checks structure,
quotations and provenance, and writes the exports. A space is not a clinical trial, a treatment recommendation or an
eligibility determination. The checks establish where text occurs in the source, not clinical correctness or
completeness; review the original PDF where graphical relationships or clinical applicability are uncertain.

## Source library

No NCCN source files or generated catalogs are distributed with this package. Supply your own permitted local
copies. The reader takes the page-preserving `nccn-pdf-to-markdown` format: a collection `markdown/manifest.json`,
one folder per disease with its own manifest and one Markdown file per PDF page (a fenced text block and a recorded
hash), and the original PDFs beside `markdown/`. PDF hashes, page coverage and Markdown hashes must agree before any
model call; an incomplete copy fails explicitly and can be retried once copying finishes.

To build a library from PDFs, install the `guideline-conversion` extra and Poppler's `pdftotext`, then run from a
source checkout:

```bash
python scripts/convert_guidelines.py \
  --input-dir /path/to/guideline-pdfs \
  --output-dir /path/to/guideline-pdfs/markdown
```

| Option | Meaning |
| --- | --- |
| `--input-dir`, `--output-dir` | Required. Both must be outside the code repository |
| `--pdf NAME` | Convert only this PDF file; repeatable |
| `--pdftotext PATH` | The `pdftotext` binary (default: found on `PATH`) |
| `--overwrite` | Rebuild guideline folders the converter generated earlier |
| `--validate-only` | Check an existing library's page inventories, file hashes and local links |

The converter makes no network or model calls. It keeps text, page addresses and links. It does no OCR and does not
recover every arrow, table relationship, superscript or graphical decision branch, so keep the PDFs for review.
Its provenance is in `scripts/README.md`.

`list_guidelines(source)` lists the library's diseases (folder name, title, edition, pages and whether every page
file is present) without contacting an endpoint. Front-matter pages (`00-front-matter/`, which hold change logs) are
kept and hashed but used only as context, never mined for populations.

## Configure the endpoint

Guideline extraction runs only against one remote OpenAI-compatible endpoint; it never starts a server or
substitutes one. It uses the shared `config.remote` settings and the `config.guideline` section:

```python
from matchminer_ai import load_default_preset

config = load_default_preset()
config.remote.update(
    enabled=True,
    server_urls=["http://localhost:8000/v1"],    # exactly one URL
    max_concurrent_requests=32,
    request_timeout=7200,                       # seconds, including time queued at the server
)
```

- **`config.remote`:** `enabled` must be true and `server_urls` must hold exactly one URL. `max_concurrent_requests`
  caps requests in flight, `request_timeout` bounds each request (the preset's 600 s is too short for long
  generations), and `max_retries` is the number of attempts per request.
- **Model and sampling:** `guideline.remote.model_name` is `None`, so the served model is discovered from
  `/v1/models`. With `sampling_profile: auto` the served model's profile sets temperature, top-p and the vLLM
  extensions; see [model profiles](../reference/llm-runtime.md#model-profiles).
- **Credentials:** the API key is read from the environment variable named by `api_key_env`; only the name is saved.
  Do not put credentials in URLs or request parameters.

| `config.guideline` key | Default | Meaning |
| --- | --- | --- |
| `remote.model_name` | `None` | Served model; `None` discovers it from `/v1/models` |
| `context_window` | `None` | Context size; `None` uses the endpoint's advertised `max_model_len` and fails if there is none |
| `remote.request_params.max_tokens` | 100,000 | Output tokens reserved per request, reasoning included; never reduced to fit input |
| `remote.request_params.temperature`, `top_p` | profile | From the model profile; 1.0 and 0.95 for a model without one |
| `remote.extra_body.top_k` | profile | For example -1 (off) for Nemotron 3.5 Super, 64 for Gemma 4, 20 for Qwen |
| `remote.extra_body.chat_template_kwargs.enable_thinking` | `True` | Reasoning on |
| `sampling_profile` | `auto` | Recognize the model's profile; `none` sends only the section's own settings |
| `reasoning_effort` | `xhigh` | Effort for Qwen 3.8; other models use their thinking switch |
| `safety_tokens` | 2,048 | Extra room kept free for retry feedback |
| `packet_pages` | 8 | Source pages each extraction request owns |
| `tokenizer_mode` | `endpoint` | Count prompt tokens with the server's chat template through vLLM `/tokenize`; `bytes` uses conservative UTF-8 byte counts |
| `response_format` | `json_schema` | Also `json_object` or `none` |
| `stream` | `True` | Stream responses, keeping raw content, reasoning, usage and finish reasons |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable that holds the API key |

For a provider without vLLM extensions, set its model, context, sampling and supported `response_format` explicitly,
set `sampling_profile: none`, and remove `top_k` and `chat_template_kwargs` from `extra_body`. Where `/tokenize` is
unavailable, set `tokenizer_mode: bytes`; it is never selected automatically.

## Extract one disease

```python
from matchminer_ai.trials import list_guidelines, summarize_guidelines

source = "/path/to/guideline-pdfs/markdown"
print(list_guidelines(source))    # use a disease folder name from this table

spaces, metadata, qc = summarize_guidelines(
    source,
    disease="breast",
    output_dir="/path/to/private/catalogs/breast",
    config=config,
    return_metadata=True,
    return_qc=True,
    progress_callback=print,
)
```

| Parameter | Meaning |
| --- | --- |
| `source_directory` | The library: the collection folder, its `markdown` folder, or a parent holding exactly one collection |
| `disease` | One disease folder name from `list_guidelines` |
| `output_dir` | Required. A separate directory for this disease, edition and configuration, outside the source library and the code repository |
| `config` | `MMAIConfig` (default preset when `None`) |
| `return_metadata` | Also return package, config, model and source provenance, run settings and the audit result |
| `return_qc` | Also return the QC table (see [Audit and QC](#audit-and-qc)) |
| `progress_callback` | Receives progress strings; its exceptions are logged and ignored |
| `allow_endpoint_change` | Resume on another server of the same model (see [Resume](#resume-and-change-endpoints)) |

The return conventions follow `extract_cohorts`: the DataFrame, then metadata and/or QC as requested. There is one
row per space (see [Catalog records](#catalog-records)). A failed stage raises; a partial catalog is never returned.

## How extraction works

1. **Extract.** The disease's pages are split into packets of `packet_pages` pages. Each page has exactly one owning
   request; other pages can be supplied as context. The LLM returns candidate populations, each anchored to the
   defining branch on a page it owns, plus a disposition for every owned page (`states_extracted`, `context_only`,
   `no_disease_states` or `uncertain`).
2. **Canonicalize.** Candidates are consolidated in batches and then across batches into one catalog of distinct
   spaces. Records combined because their definitions are identical keep their original names in
   `population_names`. A coverage review then checks that every candidate is represented: identical names and
   definitions pass in code, the rest are compared by the LLM, and missing or uncertain coverage rejects the
   catalog and retries it. A broader umbrella population cannot stand in for a treatment-specific one.
3. **Detail.** With each space's nine fields fixed, the LLM writes its workup and treatment menus. Every citation is
   an exact excerpt from a supplied page: code finds its unique occurrence (whitespace may differ), derives the line
   numbers and stores the quotation.
4. **Key criteria.** One request per space writes its cohort-style key criteria (see [Key criteria](#key-criteria)).
5. **Export and audit.** The catalog is written, then audited offline. The run is `complete` only after the audit
   passes.

The prompts are `src/matchminer_ai/prompts/guideline.*.txt` and `structured.*retry.txt`. They explain the purpose
(patient-summary matching and retrieval of source-supported considerations), define a space, and state that only the
nine fields enter matching, so names and menus cannot supply missing criteria.

**Population rules.** The prompts keep one current treatment decision per space and require a clause-by-clause field
check before returning.

- **Fields:** extent, stage, resectability and disease severity are burden; source-defined treatment lines and prior
  exposure are prior treatment; untreated status is a scoped prior-treatment exclusion; treatment ineligibility is
  an option condition. For breast cancer, HR-negative means ER-negative and PR-negative, and HR-positive means
  ER-positive or PR-positive.
- **Alternatives:** alternatives across fields are split into separate spaces without turning OR into AND, keeping
  the shared criteria.
- **Treatment history:** fields name the actual prior drugs or classes, setting, response and any counts the source
  defines. A heading such as "second-line" does not by itself establish one prior regimen or progression.
  Undefined source terms stay as cited uncertainties.
- **Quotations:** exact quotations keep the source wording.

**Checks and retries.** All attempts keep the full source context, output reserve, sampling and reasoning settings,
and are bounded by `max_retries`.

- **Field checks:** each space must stand alone (no "as above") and group mixed AND/OR logic explicitly. Numeric
  comparators ("less than or equal to 2 cm"), counts ("one or more lesions"), inclusive time bounds ("at or after 2
  years") and "head and neck" are read as single terms. `≥`/`>=` and `≤`/`<=` compare as equal. Stored text is never
  rewritten.
- **Retry feedback:** a rejected response is retried with its last complete JSON draft and up to four recent
  distinct validation errors, every failing population listed. Mixed-logic errors add `guideline.boolean_retry.txt`.
  The LLM revises its own draft; code never edits clinical criteria or guesses Boolean grouping.
- **Rejected output:** a response that stops at the token limit is rejected even if its JSON parses, and sustained
  repetitive output is stopped and retried.
- **Citation repair:** an excerpt that is missing or ambiguous on its page is repaired by a focused request that
  returns only replacement quotations, first with the cited pages and then with all pages of the original request.
  If that fails, the LLM selects supporting passages from a code-built menu of literal source fragments. If none
  supports the assertion, the detail request is regenerated (`guideline.unsupported_detail_retry.txt`) with the
  space unchanged. An unresolved citation is a failure, never an empty or dropped citation.
- **Extraction recovery:** an extraction packet that exhausts its attempts is retried with optional context pages
  dropped to leave 8,192 tokens for feedback (owned pages and the output reserve stay), and can be split into
  smaller page groups. One that cited table rows or wrapped display lines as line numbers is retried with addresses
  restricted to its owned pages and nonblank lines (`guideline.extraction_addresses.txt`).
- **Smaller consolidation tasks:** a consolidation batch that exhausts its attempts is split into source-backed
  groups of at most 12 inputs, down to one. Coverage reviews take at most 12 inputs each, always against the
  complete proposed catalog. If a final coverage review finds a missed distinction, its findings go back to the
  affected source batches.
- **Fixed definitions:** a detail response that keeps altering its space is retried with the nine fields constrained
  to their exact values.
- **Source review:** if a single input still cannot be reconciled, its extraction packet is re-extracted once (at
  most two packets per catalog), with both earlier answers presented as fallible.
- **Deduplication fallback:** when cross-batch deduplication cannot preserve every definition, all validated
  definitions are kept and the run records that semantic duplicates may remain.
- **Transport failures** never trigger splitting or regeneration.

**Context packing.** Requests keep whole pages, put required evidence first and use all source context that fits.
Every omission is recorded (for a space's detail request, in `omitted_source_page_ids`). Mandatory evidence that does
not fit fails with a message to reduce `packet_pages` or use a larger context.

**Request scheduling.** `max_concurrent_requests` is shared by all catalogs in one process and held until each
response stream ends. `/tokenize` and `/models` calls use a separate pool of at most four requests per endpoint, so
prompt preparation does not wait behind long generations. Separate processes have separate limits; keep their
combined load within what the server allows.

## Output directory

**The output directory contains source-derived and verbatim guideline text**, including checkpointed prompts and raw
responses. Keep all of it outside source control and public artifacts. The API refuses directories inside the
source library or the code repository.

| File | Content |
| --- | --- |
| `paradigms.jsonl` | The catalog: one complete record per space. Use this for loading and retrieval |
| `trial_spaces.csv` | `space_trial_id`, `trial_id`, `clinical_space_number`, `clinical_space_summary`, `general_exclusion_criteria`, `paradigm_id` only; no menus |
| `report.md` | Readable catalog: each space with its key criteria, menus, uncertainties and links to the source pages and PDF |
| `status.json` | `status` (`running`, `complete`, `failed`, `interrupted`), `stage`, counts and any error |
| `validation.json` | The offline audit |
| `run_config.json` | Settings, prompt hashes, stage versions and the request identity used for resume checks |
| `sources.json` | Source edition, hashes and every page's address and Markdown hash |
| `endpoint_history.json` | Servers used, when the run moved between endpoints |
| `checkpoints/` | Per request: the request, every attempt sent (`request-attempt-*.json`), raw responses (`attempt-*.json`) and the accepted result |
| Stage files | `extraction.json`, `extraction_ownership.json`, `candidates.json`, `coverage.json` (page dispositions), `canonical_groups.json` and the other `canonical_*.json` files, `canonical_coverage.json` (final coverage judgments), `population_coverage/`, `details.json` |
| Recovery receipts | `quote_repairs/`, `canonical_batch_splits.json`, `extraction_packet_splits.json`, `extraction_source_reviews.json`, `superseded_candidate_artifacts/` |

## Catalog records

| Column | Content |
| --- | --- |
| `space_trial_id`, `paradigm_id` | The space's ID, `nccn-<disease>-<hash>`, stable within a source edition and definition |
| `trial_id` | `guideline:<disease>:<version>`; not an NCT ID, so never send it to services that expect one |
| `clinical_space_number` | Position in the catalog |
| `name` | The population's name |
| `space` | The nine fields: `age_range_allowed`, `sex_allowed`, `cancer_type_allowed`, `histology_allowed`, `cancer_burden_allowed`, `prior_treatment_required`, `prior_treatment_excluded`, `biomarkers_required`, `biomarkers_excluded` (`NA` = no restriction) |
| `clinical_space_summary` | The nine fields as one line, `Label: value.` per field |
| `key_criteria`, `key_criteria_items` | Cohort-style key criteria (see [Key criteria](#key-criteria)) |
| `general_exclusion_criteria` | `NA`: a guideline has no trial boilerplate |
| `diagnostic_workup`, `treatment_options` | Lists of items with `name`, `conditions`, `category` (as the source states it) and `evidence` |
| `evidence` | Citations for the population: `page_id`, `line_ids`, `source_lines`, `quote`, `pdf_page`, `printed_label` |
| `uncertainties` | The model's recorded uncertainties |
| `population_names` | Original names of records combined into this space, when any were |
| `source` | Disease, title, edition, source PDF, hashes, fingerprint and conversion warnings |
| `context_page_ids`, `omitted_source_page_ids`, `context_packing` | Pages supplied to and left out of the detail request, and its token budget |
| `source_batch_numbers` | Consolidation batches the space came from |
| `review_required`, `notice` | Always true; the research-use notice |

For a two-column page, `quote` is the verified excerpt from one column, and `source_lines` keeps the full original
lines.

### Key criteria

Each space has cohort-style key criteria: what a trial protocol would state as inclusion and exclusion criteria for
that population. One request per space (jobs `cohort-criteria-NNNN`, prompts `guideline.cohort_criteria*.txt`) writes
them after canonicalization. It sees the space's name and nine fields, the guideline lines its evidence cites (as
context only), and style examples from trial protocols in other cancers. Code then checks that:

- every non-NA field is expressed by a criterion, and each criterion names the fields it expresses;
- no criterion cites an NA field or adds a generic trial requirement the fields do not contain, such as consent,
  performance status, organ function, contraception or measurable disease;
- each criterion is one self-contained sentence with its logic in words, without numbering or cross-references.

`key_criteria_items` holds the criteria (`[{type, text, fields}]`) and `key_criteria` the rendered text. For a
fabricated space:

```text
Inclusion criteria:
The participant is aged 18 years or older.
The participant has fictional cancer with metastatic disease.
The participant has Marker X-positive disease.
Exclusion criteria:
The participant has received a prior fictional inhibitor.
```

`load_guideline_catalog` returns this text as `cohort_key_criteria`, the column a trial cohort's key criteria use.
TrialSpace retrieval and the decision model's match questions read it, as they do for trial cohorts; guideline
spaces have no boilerplate criteria. The population workup review and the guideline fetcher also pass it to workup
review as the population context. `clinical_space_summary` is unchanged and remains the key for exact lookup.

**Catalogs without stored key criteria.** For a record without `key_criteria`, the loader lays out the nine fields
as key criteria instead: the seven required fields under "Inclusion criteria:", then `prior_treatment_excluded` and
`biomarkers_excluded` under "Exclusion criteria:", one `Label: value.` line per non-NA field:

```text
Inclusion criteria:
Age range allowed: Age >=18 years.
Cancer type allowed: Fictional cancer.
Cancer burden allowed: Metastatic disease.
Biomarkers required: Marker X positive.
Exclusion criteria:
Prior treatment excluded: Prior fictional inhibitor.
```

`add_guideline_cohort_criteria(catalog_dir, output_dir=..., config=...)` adds stored key criteria to such a catalog.
It writes a derived edition to a new directory: the same records plus the two fields, its own checkpoints, and an
audit that ties each record to its base record and an accepted response. The base catalog, which must be complete
with a passing audit, is only read. Rerun the same call to resume. For a whole run directory
(`<catalog-root>/<disease>/`):

```bash
python examples/backfill_guideline_cohort_criteria.py \
  --catalog-root /path/to/private/catalogs \
  --output-dir /path/to/private/catalogs-with-criteria \
  --endpoint http://localhost:8000/v1
```

Options: `--model` (default: discovered), `--concurrency` (requests in flight for all diseases, default 128),
`--disease-workers` (default 8) and `--disease` (repeatable subset). Progress is in `backfill.json`.

Adding stored criteria changes the text that is embedded, so vectors you stored from the field layout must be
regenerated; the [embedding cache](#retrieve-considerations-for-patients) keys on the exact text and does this itself.

## Audit and QC

```python
from matchminer_ai.trials import audit_guideline_catalog

audit = audit_guideline_catalog(source, disease="breast", output_dir="/path/to/private/catalogs/breast")
```

The audit makes no model calls. It checks source hashes, page ownership, exact quotations against their pages and
recorded raw responses, immutable fields across stages, the token budgets, the key criteria, and agreement among
the exports, then writes `validation.json`. It recognizes extraction and citation-review catalogs.

The QC table (`return_qc=True`) has the columns `metric`, `value`, `denominator`, `percent` and `ids`:

| Metric | Counts |
| --- | --- |
| `guideline_uncertain_source_pages` | Pages the extraction marked `uncertain` |
| `guideline_spaces_with_uncertainties` | Spaces with recorded uncertainties |
| `guideline_spaces_with_omitted_context` | Spaces whose detail request left out source pages |
| `guideline_spaces_without_treatment_options` | Spaces with an empty treatment menu |

Neither the audit nor QC establishes completeness or clinical appropriateness.

## Resume and change endpoints

Repeat the same call to resume: accepted responses and token counts are reused.

- **Free to change:** concurrency, timeout, retry count and streaming.
- **Also allowed:** adding or revising the retry-only prompts (`guideline.boolean_retry.txt`,
  `guideline.unsupported_detail_retry.txt`, `guideline.quote_select.txt`, `guideline.extraction_source_review.txt`,
  `guideline.extraction_addresses.txt`); their hashes are recorded and accepted responses are revalidated.
- **New directory required:** a change of source, core prompts, model or generation settings.

A completed, audited run with an identical configuration is reused as is; only the audit is repeated.

To move an unfinished run to another server of the **same model**, stop the original process and call again with
the new URL and `allow_endpoint_change=True`. The new server must advertise the same model ID and context size, and
sampling, reasoning and extraction settings must be unchanged. Accepted checkpoints keep their provenance:
`run_config.json` keeps the original `base_url` and the current `transport_base_url`, `endpoint_history.json` lists
the servers, and the audit checks each request against that history. The advertised model ID is a compatibility
check, not a verification of the weights. A different model or context size needs a new output directory.

## Extract a collection

`examples/extract_guideline_collection.py` runs `summarize_guidelines` for every disease of a library, writing each to
`<output-dir>/<disease>/` and the collection's progress to `<output-dir>/collection.json`:

```bash
python examples/extract_guideline_collection.py \
  --source /path/to/guideline-pdfs/markdown \
  --output-dir /path/to/private/catalogs \
  --endpoint http://localhost:8000/v1 \
  --concurrency 32
```

| Option | Meaning |
| --- | --- |
| `--source`, `--output-dir`, `--endpoint` | Required: the library, the collection directory and the primary server |
| `--model` | Served model ID; omit to discover it |
| `--concurrency` | Requests in flight per endpoint (default 32), shared by all of that endpoint's diseases |
| `--disease-workers` | Optional cap on diseases running at once per endpoint; default up to `--concurrency` |
| `--disease NAME` | Run only these diseases; repeatable |
| `--completed-catalog PATH` | Skip a disease with this completed catalog; repeatable |
| `--additional-endpoint URL` | Another server of the same model and context for unstarted diseases; repeatable |
| `--new-disease-endpoint URL` | Send every disease without checkpoints to this one of the configured servers |
| `--source-override DISEASE PATH` | Read one disease from another verified library of the identical source edition |
| `--defer-disease NAME` | Keep a disease in the collection but leave it to another worker; repeatable |
| `--retry-failed` | Run only diseases `collection.json` marks failed, writing progress to `recovery.json` |

**Scheduling.** Each endpoint's full request cap is shared by its active diseases rather than divided among them, so
a finishing disease frees its slots at once, including for the last catalog's detail requests. Failed diseases are
scheduled first on resume, then running and unstarted ones. A disease that fails is recorded in `collection.json`
and the others continue. Dependencies, packing and the number of unfinished jobs can still keep traffic below the
cap.

**Completed catalogs.** A `--completed-catalog` is skipped only when it has a passing stored audit, a verified
export hash, stored key criteria (add them first with the backfill script) and the same source PDF hash as the
library. Each catalog keeps its own model provenance.

**Endpoints.** Additional endpoints are checked for the same model ID and context size before any guideline text is
sent. A disease that already has checkpoints stays on its recorded server; unstarted diseases are balanced across
servers, or all go to `--new-disease-endpoint`. Each endpoint has its own disease pool, so an added server starts at
once. The runner does not move a started disease to another server; use `summarize_guidelines` with
`allow_endpoint_change=True` for that.

**Resuming.** Stop the running process first; two writers must never share a collection. Then repeat the same
command, keeping the source, output, primary endpoint, model, disease, completed-catalog and source-override
arguments, which form the collection's identity; `--concurrency` and `--disease-workers` may change. The collection
is `complete` only when every selected disease passes extraction and audit; with outstanding deferred diseases it is
`deferred`.

- **`--defer-disease`:** lets a restarted runner leave diseases to separate recovery workers without duplicate
  writers; omit it on a later resume to take them back.
- **`--retry-failed`:** recovers failed diseases in their original output directories while another runner owns
  `collection.json`. Request limits apply per process, so give a concurrent recovery a small `--concurrency`.

## Review citations

`trials.review_guideline_citations` re-checks the citations of a completed catalog. For every population, workup
item and treatment option, the LLM selects exact source excerpts that support it, code derives their addresses, and
clinical fields, IDs, item order and uncertainties stay unchanged. It uses the normal guideline settings, with the
cited pages kept and other pages filling the remaining context.

```python
from matchminer_ai.trials import review_guideline_citations

reviewed, metadata = review_guideline_citations(
    "/path/to/private/catalogs/breast",
    output_dir="/path/to/private/citation-reviews/breast",
    config=config,
    return_metadata=True,
)
```

- **Input:** a completed catalog directory or its `paradigms.jsonl`, of one disease and source edition. The source
  library defaults to the one recorded in the catalog; `source_directory=` points at a moved copy, which must have
  the same fingerprint.
- **Output:** a new directory, separate from the original catalog, with `original_paradigms.jsonl`, one review per
  space in `reviews/`, raw responses, the standard exports and an audit. The original catalog is never changed.
- **Resume:** repeat the same call. A change of source, original catalog, prompts or generation settings needs a new
  output directory.

Each population and menu item gains `citation_review`: `version`, `status` (`supported_by_model_review` or
`unresolved`) and `issues`. An unresolved item can have partial or empty evidence. **Show these issues beside the
affected assertion**, as the guideline fetcher and the Markdown reports do, rather than presenting partial support as
validation of the whole recommendation. A completed review can contain unresolved items; the audit counts them.

For several diseases, write a JSON object that maps disease folder names to completed catalog directories:

```bash
python examples/review_guideline_collection.py \
  --catalog-manifest /path/to/private/catalogs.json \
  --output-dir /path/to/private/citation-reviews \
  --endpoint http://localhost:8000/v1 --concurrency 32
```

`--model` defaults to the served model, `--concurrency` (default 32) is shared by all diseases, and
`--disease-workers` (default 4) sets how many run at once. `collection.json` and each disease's `status.json` record
progress, failures and unresolved findings. Repeating the command resumes accepted checkpoints and retries failures.

## Load and look up a catalog

```python
from matchminer_ai.trials import get_guideline_considerations, load_guideline_catalog

catalog = load_guideline_catalog("/path/to/private/catalogs/breast")
record = get_guideline_considerations(catalog, space_trial_id=catalog.iloc[0]["space_trial_id"])
print(record.iloc[0]["diagnostic_workup"])
print(record.iloc[0]["treatment_options"])
```

`load_guideline_catalog` accepts a completed catalog directory, its `paradigms.jsonl`, a list of them (to combine
diseases or editions; space IDs must stay unique), or a DataFrame from `summarize_guidelines`. `trial_spaces.csv`
lacks the menus and is rejected.

- **Checks:** a present `status.json` must say `complete`, and a present `validation.json` must report a passing
  audit whose export hash matches the JSONL. The stored definitions, menus and evidence are validated too. These
  check the stored catalog; they do not repeat the source audit. Bare JSONL and DataFrame inputs have no audit
  status.
- **Added columns:** `cohort_key_criteria` (see [Key criteria](#key-criteria)), set on every load and excluded from
  `catalog_sha256`; `clinical_space_number` and `general_exclusion_criteria` (`NA`) where a record lacks them.
- **Metadata** (`return_metadata=True`): `catalog_sha256`, the number of spaces, the source editions, each file's
  hash and recorded audit status, and `validation_cache`.

`get_guideline_considerations` looks up a space by `space_trial_id`, by exact `clinical_space_summary` text, or by
both. There is no fuzzy matching: an unknown space or a mismatched pair raises `KeyError`. Text lookup can return the
same definition from several source editions; use the ID for one record.

**Cache.** Validated records are kept in process memory. Later calls check each file's identity, size, modification
and change times (for `paradigms.jsonl`, `status.json` and `validation.json`) and reuse unchanged files without
rereading them. A changed or failing file is never answered from an older cached copy, and returned records are
independent copies. The cache holds at most 128 files and 512 MiB of JSONL, contains no patient data and is lost on
restart. Change detection follows the filesystem's metadata, including network filesystem delays;
`refresh=True` forces a full read. DataFrame inputs are always copied and validated.

## Retrieve considerations for patients

```python
import pandas as pd
from matchminer_ai import load_default_preset
from matchminer_ai.matching import retrieve_guideline_considerations, write_guideline_considerations_report

patients = pd.DataFrame([{"patient_id": "P001", "cancer_history_summary": "Fabricated summary text."}])
config = load_default_preset()
config.embedding["model_path"] = "/path/to/trialspace"    # your TrialSpace checkpoint
# Both models run on CUDA by default; set both to "cpu" without a supported GPU.
config.embedding["device"] = "cuda"
config.raw["decision_model"]["device"] = "cuda"

ranked, metadata = retrieve_guideline_considerations(
    patients,
    "/path/to/private/catalogs/breast",
    top_n=5,
    candidate_k=20,
    config=config,
    return_metadata=True,
    progress_callback=print,
)
write_guideline_considerations_report(
    ranked, patients, "/path/to/private/review/P001.md", metadata=metadata,
)
```

`patient_summaries` needs `patient_id` and `cancer_history_summary`; only those columns enter the models. The
catalog argument takes anything `load_guideline_catalog` accepts. The function composes the public matching steps:

1. `embed_for_matching` embeds the summaries and the spaces' `cohort_key_criteria` with the configured TrialSpace
   model. Catalog vectors are computed once per call for all patients, or reused from the disk cache below.
2. `generate_candidate_matches` takes each patient's `candidate_k` nearest spaces by cosine similarity
   (`candidate_k=None` scores every space). `candidate_k` must be at least `top_n`.
3. `score_match_quality` scores those pairs with the decision model's match questions, with
   `filter_low_quality=False`, so a cutoff cannot remove pairs from the top N.
4. The `top_n` spaces per patient by `match_quality_score` are returned, ties broken by similarity and then space ID.

Each row has `patient_id`, `rank` (match-quality order), `retrieval_rank` (TrialSpace order), `similarity_score`,
`match_quality_score`, `match_quality_pass`, the per-question `match_quality_*` scores, and the complete catalog
record with both menus. The unit is the space, so several spaces of one guideline can appear for a patient. Metadata
records the config, the catalog identity, both models and the retrieval counts.

- **No rewriting:** no generative LLM is used. Stored conditions, categories, evidence and uncertainties are returned
  unchanged, and a high-ranked space does not mean every listed option applies to the patient.
- **Scores:** `match_quality_score` ranks and screens for review; it is not a calibrated probability or an
  eligibility decision. Check prior treatment, biomarkers and missing facts against the source.
- **Same model:** patient and catalog vectors must come from the same TrialSpace model; retrieval checks the model
  metadata and refuses to mix them. It does not accept precomputed catalog vectors.
- **Offline use:** model weights are loaded locally; first use may download them from Hugging Face, without
  patient or catalog text. To run offline, set `HF_HUB_CACHE` and `HF_HUB_OFFLINE=1` before importing, and point
  `config.model_metadata_cache_dir` at the cached model metadata.

`write_guideline_considerations_report(matches, patient_summaries, output_path, metadata=None,
patient_provenance=None)` writes a Markdown review: each patient's summary, both ranks and scores, the population
definitions and key criteria, the menus with their conditions, caveats and citation-review issues, stored source
excerpts, and local PDF links where the files exist. It contains patient text and guideline content, so the path must
be outside the code repository and the source libraries.

**Embedding cache.** `embedding_cache_dir="/path/to/private/guideline-embeddings"` keeps catalog vectors across calls
and restarts (default `None`: no disk cache).

- **Contents:** vectors and hashes of the embedded text, never patient summaries, patient vectors or guideline text.
  The vectors are source-derived, so the directory must be outside the code repository.
- **Keys:** entries are keyed by the exact `cohort_key_criteria` text, so new or changed spaces are embedded and
  identical text is reused across diseases and catalog updates. Each encoder gets its own file, identified by its
  loaded revision (or a digest of the loaded weights for a local directory), tokenizer, prompt, sequence length,
  encoding settings and library versions. An encoder whose identity cannot be verified is not cached.
- **Failures:** a corrupt entry is recomputed, and an unreadable cache falls back to computing every vector.
- **Still per call:** catalog validation and patient embedding run every time, and the returned menus always come
  from the current catalog. Metadata reports hits and new embeddings under `guideline_embedding_cache`.

**Command line.** `examples/retrieve_guideline_considerations.py` retrieves for one patient from a Parquet summary
table and writes `patient_considerations.md`, `ranked_spaces.jsonl`, `retrieval_metadata.json`, `patient_source.json`
and `patient_summary.json` to an external `--output-dir`:

```bash
python examples/retrieve_guideline_considerations.py \
  --catalog /path/to/private/catalogs/breast \
  --patient-summaries /path/to/private/summaries.parquet --patient-id P001 \
  --trialspace /path/to/trialspace --output-dir /path/to/private/review/P001
```

| Option | Meaning |
| --- | --- |
| `--patient-id-column`, `--summary-column` | Input columns (default `patient_id`, `cancer_history_summary`); for example `pseudo_mrn` and `patient_summary` for training tables |
| `--trialspace` | The TrialSpace checkpoint (default: `embedding.model_path`, else the site config's `paths.trialspace_model`) |
| `--top-n`, `--candidate-k` | As in the API (default 5 and 20) |
| `--device` | Device for both models (default `cpu`); `--cpu-threads` sets torch threads (default 8) |
| `--embedding-cache-dir` | The persistent catalog-vector cache |
| `--model-cache-dir`, `--metadata-cache-dir`, `--offline` | Hugging Face cache, model-metadata cache and offline mode |
| `--synthetic-notes` | A synthetic note table in which the selected ID must occur; recorded as provenance |

To find the patients of a population for one space, see [Population workup review](population-workup.md)
(`retrieve_guideline_patients`).
