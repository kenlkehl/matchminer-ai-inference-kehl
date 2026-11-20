# Extraction-Based Workup Checks

!!! warning "Review aids, not determinations"
    Results say whether structured items extracted from the notes document a
    guideline workup step. An LLM wrote the query and an LLM extracted the items:
    check every cited quote against its note. *Not documented* means no extracted
    item matched, not that the step was not done. This does not establish overall
    guideline concordance.

This method checks guideline workup from the records of [Structured Clinical Extraction](extraction.md) instead
of the notes (the note-reading reviews are `review_patient_workup` and `review_population_workup`):

1. **Compile once per requirement.** `compile_workup_queries` asks the
   `workup_query_compile` LLM to translate each guideline `diagnostic_workup`
   item (name, conditions, category), with the guideline population's key
   criteria as context, into a declarative JSON query. The model's only power is
   answering with a query: it has no tools, browsing, files or shell, and it never
   sees patient data. Code validates the query; an invalid answer is retried with
   its problems named.
2. **Evaluate in code.** `check_workup_with_extractions` runs every query on every
   patient's extraction items. The evaluator is pure Python with no I/O and no
   model calls: the same query and records always give the same result.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.patients import (
    check_workup_with_extractions,
    compile_workup_queries,
    extract_document_records,
)
from matchminer_ai.trials import load_guideline_catalog

config = load_default_preset()
config.remote.update(enabled=True, server_urls=["http://localhost:8000/v1"],
                     request_timeout=14400)

space = load_guideline_catalog("catalogs/nsclc").iloc[0]     # one guideline space
query_cache = {}                                             # any mapping; reuse it across calls
compiled = compile_workup_queries(
    space["diagnostic_workup"], config=config,
    population_context=space["cohort_key_criteria"], cache=query_cache,
)
documents = extract_document_records(notes, config=config)
report = check_workup_with_extractions(documents, compiled)
```

`check_workup_with_extractions` also takes patient records (pass a list of record DataFrames), `patient_ids` (adds
patients without records, whose results are unknown, and fixes the order), `reference_dates` (one date, or dates
by patient id), `anchors` (a supplied `diagnosis_date` by patient id) and `coverage` (counts such as
`notes_without_records` by patient id). It returns `patients` (one review per patient, with an assessment per
requirement), `aggregate` and `notice`.

For a cohort, `matchminer_ai.population.check_population_workup_with_extractions` takes the cohort (with its
space), the compiled queries and the records. It links the records to the cohort's own notes by note id and returns
the cohort with a `report` in the shape of the full-note review (`method: extraction_queries`); see
[Population workup review](../user-guide/population-workup.md#extraction-based-checks).

[SQL-Agent Workup Checks](workup-sql.md) read the same records with SQL queries that an LLM agent writes after
exploring the extracted data.

## The query language

A query is one JSON object, decoded under `query_response_schema()` with vLLM
structured outputs and validated by `validate_workup_query()`. The schema is
strict and non-recursive (no `$ref`, `null`, `minLength` or `pattern`).

| Key | Meaning |
| --- | --- |
| `summary` | One sentence: what the query checks. |
| `expressible` | `false` when no item type can check the requirement: physical examination, history taking, organ function tests such as PFTs or ECG, multidisciplinary or tumor board evaluation. Then `criteria` is `[]`, every expression is `{"all_of": []}` and `not_expressible_reason` says why; the result is `unknown`. Laboratory tests are `lab_test` items and counseling, screenings and referrals are `counseling` items, so those are expressible. |
| `approximations` | Up to 8 notes on what was left out or approximated ("upper abdomen" by the abdomen region). Shown as limitations. |
| `criteria` | Up to 24 named matchers of one item type each: `id`, `label`, `item_type`, `status` (`performed`, `planned`, `any`), `filters`, `window`, `min_count` (1-20). |
| `requirement` | What must be documented. |
| `applies_when` | Documented facts that make a conditional step apply (`{"all_of": []}`: the whole population). |
| `not_applicable_when` | Documented facts that make it not apply. It needs at least one criterion that is not negated: a step is never not applicable because something is undocumented. |

Expressions are AND-of-OR, with up to 24 groups of up to 12 literals: `{"all_of": [{"any_of": [{"criterion": "id",
"negated": false}, ...]}, ...]}`. Alternatives are literals of one group;
components that are all required are separate groups; a negated literal holds
when its criterion does not. A requirement needs at least one criterion that is
not negated, and every criterion must be used by an expression or a window anchor.

A criterion's `status` matters for imaging studies, procedures, lab tests, counseling and treatments (a treatment
is planned when its `treatment_status` is `planned`). Items of other types are always performed, so `performed`
is read as `any` for them.

A **filter** is `{"field", "op", "values", "if_not_stated"}`; a criterion has up to 8, and all must hold. Fields
are the item type's non-date fields (`record_json_schema`), `level`, `document_type`, `relation` (document-level
imaging studies, procedures, lab tests and counseling) and `procedure_group`. Dates are tested by the window, not
by filters. Operators:

| Operator | Fields | Holds when |
| --- | --- | --- |
| `in`, `not_in` | codes | the code is (is not) one of `values` |
| `contains_any`, `contains_all`, `contains_none` | lists of codes (`body_regions`) | the list contains any, all or none of `values` |
| `text_contains_any`, `text_contains_none` | text, text lists | a value occurs as a whole word or phrase, ignoring case and punctuation (`ALK` does not match "alkaline"; `PD-L1` matches "PD L1") |
| `text_equals_any` | text, text lists | the text equals a value, ignoring case and punctuation |
| `stage_group_in` | `stage_group` | the stage's main group (`0`, `I`-`IV`, read from "IIIA", "Stage IVB" or "4") is in `values` |
| `number_at_least`, `number_at_most` | integers | compared with `values[0]` |
| `is_stated` | any | the field has a value (`values` is `[]`) |

Choice values must be the field's vocabulary codes, and a filter has at most 24 values. `if_not_stated` decides
what a filter means for an item whose field was not stated (`not_stated`, `""`, `[]`, or a `line_of_therapy` of
0): `unknown` (the usual choice), `no_match` or `match`. A stage text with no recognizable group is unknown.

A **window** is `{"anchor", "anchor_criterion", "days_before", "days_after",
"if_no_anchor"}`: the item's date must lie from `days_before` days before the
anchor to `days_after` days after it (`-1`: no limit; `0`: not after or not
before). Anchors are `any_time` (no limit; both days `-1`), `reference_date` (the
evaluation date, by default the patient's latest note date), `diagnosis_date`
(the earliest dated extracted diagnosis, or a supplied one), and
`criterion_earliest` / `criterion_latest` (the earliest or latest certain match of
another criterion, such as the first systemic therapy; anchors may not form a
cycle). `if_no_anchor` is `unknown`, or `reference_date` to use the evaluation
date when the anchor does not exist (a patient not yet treated). An anchor that
may exist but is undated or only possibly matched is unknown and never replaced
by the evaluation date; an undated certain match makes the earliest anchor's
interval open-ended.

Example: a brain MRI, or a contrast head CT, before the first systemic therapy,
when stage III-IV is documented:

```json
{
  "summary": "Brain MRI or head CT before the first systemic therapy, if stage III-IV.",
  "expressible": true, "not_expressible_reason": "",
  "approximations": ["Whether MRI was possible is not extracted; head CT is accepted."],
  "criteria": [
    {"id": "brain_mri", "label": "Brain MRI", "item_type": "imaging_study", "status": "performed",
     "filters": [{"field": "modality", "op": "in", "values": ["mri"], "if_not_stated": "no_match"},
                 {"field": "body_regions", "op": "contains_any", "values": ["brain"], "if_not_stated": "unknown"}],
     "window": {"anchor": "criterion_earliest", "anchor_criterion": "systemic", "days_before": -1,
                "days_after": 0, "if_no_anchor": "reference_date"}, "min_count": 1},
    {"id": "systemic", "label": "Systemic therapy", "item_type": "treatment", "status": "performed",
     "filters": [{"field": "treatment_type", "op": "in", "values": ["systemic_therapy"], "if_not_stated": "no_match"}],
     "window": {"anchor": "any_time", "anchor_criterion": "", "days_before": -1, "days_after": -1,
                "if_no_anchor": "unknown"}, "min_count": 1},
    "... head_ct, stage_3_4 and stage_0_2 criteria ..."
  ],
  "applies_when": {"all_of": [{"any_of": [{"criterion": "stage_3_4", "negated": false}]}]},
  "not_applicable_when": {"all_of": [{"any_of": [{"criterion": "stage_0_2", "negated": false}]},
                                     {"any_of": [{"criterion": "stage_3_4", "negated": true}]}]},
  "requirement": {"all_of": [{"any_of": [{"criterion": "brain_mri", "negated": false},
                                         {"criterion": "head_ct", "negated": false}]}]}
}
```

The compile prompt's worked examples (`matchminer_ai.patients.workup_queries.language.EXAMPLES`)
are complete, validated queries for CT chest within 30 days, CT
chest/abdomen/pelvis as one or several studies, core or excisional biopsy, this
brain MRI, EGFR and ALK testing, and a requirement that cannot be expressed.

## How results are evaluated

Each test is **yes**, **unknown** or **no**, combined with Kleene logic (AND takes
the lower, OR the higher, NOT swaps yes and no).

**Dates** are intervals of days:

| Item date | Interval |
| --- | --- |
| day precision (written, or a report's own study dated by its note) | that day |
| month or year precision | every day of that month or year |
| `evidence_note_date` (a performed patient-level item dated by its earliest evidence note) | on or before that date |
| unknown, performed (for example `previously_performed` with no date) | on or before the note's date (patient level: the earliest evidence note's) |
| unknown, planned | on or after that date |

Items with no date field of their own (disease assessments, performance status) are dated by their note.
An item is inside a window when its whole interval is, outside when none of it
is, and unknown otherwise; anchors are intervals too. So a month-precision CT is
unknown against "within 30 days" unless the whole month fits, and an undated
"outside CT was done" mention counts for "any time" but is unknown against a
recent window.

**Deduplication.** The same study, procedure, result or course documented in
several notes, or at both levels, is one cluster. Its members share the item type and kind, have compatible dates
and, except for biomarkers, the same performed or planned status. The kind is:

- imaging: the modality, with overlapping body regions;
- procedures: the procedure type;
- biomarkers: the marker, result and alteration;
- treatments: the type and the agents or regimen;
- lab tests: the test (and its text for `other`);
- counseling: the topic and kind (and the topic text for `other`);
- diagnoses: the site and histology; events: the event type; performance status: the scale and score.

An undated or less precise mention joins the most recent compatible dated cluster. A cluster is dated by its most
precise member, and passes a filter when a member does, except that the report's own study (`this_report`) decides
when it contradicts a mention. Criteria count clusters: a CT report and a later note mentioning that CT count once.

**Criteria.** A criterion holds when at least `min_count` clusters certainly
match (status, filters and window), is unknown when enough clusters might, and
does not hold otherwise. Planned items never satisfy `performed`.

**Status and applicability:**

| Status | When | `reason` |
| --- | --- | --- |
| `met` | the requirement holds | `documented` |
| `not_met` | no extracted item documents it | `not_documented`, or `planned_only` when only planned or ordered items match |
| `unknown` | a matching item's date or details leave it open, its anchor is missing, some of the patient's notes failed extraction or have no record, there are no records, the requirement is not expressible, or no valid query was compiled | `uncertain_dates`, `fields_not_stated`, `anchor_unavailable`, `incomplete_extraction`, `no_extraction`, `not_expressible`, `query_unavailable` |
| `not_applicable` | `not_applicable_when` holds | `documented_not_applicable` |

The status vocabulary (`STATUSES`) also has `done_not_recommended`, which only the
[SQL agent](workup-sql.md#results-and-overrides) reports.

`applicability` is `applies` (no condition, or `applies_when` holds),
`not_applicable` (documented contrary facts) or `uncertain` (the condition is not
documented, or the facts conflict). Not applicable always needs documented facts.

Each result also carries:

- a machine-made `bottom_line`, for example "Met: CT chest: CT (chest, abdomen, pelvis), with contrast, 2024-03-02
  (day precision, dated by its report, note note-1a2b).";
- every criterion's certain and possible counts and anchor, and its matching clusters with their dates and
  mentions, listed certain first, then precisely dated before undated, then most recent;
- each criterion's `closest_non_matches`: items of the same kind (such as the same modality or marker) that failed
  only some filters or the window, with what differed. A not-documented explanation names the closest one;
- the `evidence` quotes with their `note_id`, `note_date` and character offsets;
- `limitations` (the query's approximations, and an incomplete extraction).

`aggregate_extraction_workup` counts statuses, applicability and reasons per
requirement; *met / applicable assessed* counts met among patients it applies to
whose result is met or not met.

## Compile settings, retries and caching

The `workup_query_compile` preset section defaults to Nemotron 3.5 Super with `sampling_profile: auto`,
`reasoning_effort: xhigh`, thinking on and structured outputs, and 40,000 output tokens in a 131,072-token local
context. The prompt (data model, vocabularies, query language and examples) is about 40,000 characters, roughly
10,000-12,000 tokens. A remote backend keeps `batch_size * batches_in_flight` (16) requests in flight.

An answer that is not complete JSON or fails validation is retried with its problems named
(`max_attempts: 3`), then once with thinking off (`reasoning_off_fallback`): three retries in all. A request that
fails in transport fails its entry. Each entry has `requirement`, `cache_key`, `status` (`compiled` or `failed`),
`query`, `error`, `attempts` (finish reasons and problems, no reasoning), `attempt_count`,
`first_attempt_valid`, `cached`, `model` and the compile `contract`. With `return_metadata=True` the call also
returns counts of distinct, cached, compiled and first-attempt-valid queries and requests.

Queries cover the item types of the extraction extensions that `document_extraction` and `patient_extraction`
select, unless `extensions` is passed. Distinct requirements are compiled once per call. Pass `cache` (any mapping,
such as a dict or a folder of JSON files) to reuse compilations across calls. The key (`workup_query_cache_key`)
covers the requirement's name, conditions and category, the population context, the extraction schema version and
extensions, the query language version (`mmai-workup-query-v1`), and the prompt and response-schema hashes. Only
compiled queries are cached, and a cached query is validated again before use. Compiled queries contain guideline
text only.

::: matchminer_ai.patients.workup_queries
    options:
      members:
        - compile_workup_queries
        - check_workup_with_extractions
        - evaluate_workup_query
        - validate_workup_query
        - query_response_schema
        - workup_query_cache_key
        - aggregate_extraction_workup
