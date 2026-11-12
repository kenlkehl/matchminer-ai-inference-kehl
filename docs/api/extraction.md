# Structured Clinical Extraction

!!! warning "Research abstraction aids"
    These records are research data-abstraction aids. They are not clinical
    documentation, and every item needs human review against the source note
    before it is used. A missing item does not mean that something was not
    done. The note text reaches the configured LLM endpoint, so the endpoint
    must be appropriate for the data.

Two stages turn clinical notes into structured, evidence-linked records that use
one data model:

| Function | One record per | Content |
| --- | --- | --- |
| `extract_document_records` | note | The document type; for an imaging report the study that produced it; for a pathology report the procedure that produced its specimen; and the diagnoses, biomarkers, treatments, disease assessments and performance status the note states. |
| `extract_patient_records` | patient | Cancer diagnoses with site, histology, dates and stage; biomarkers; treatment lines and regimens with agents, dates and best response; tissue procedures; imaging history; key dated events. |

Both accept the note table `summarize_patients` takes: `patient_id`,
`note_text` and `note_date`, plus optional `note_id` (unique) and `note_type`
(the record system's label, shown to the model as a hint). Without `note_id`,
each note gets a stable id from a hash of its patient, date, type and text, and
exact duplicate notes are kept once. Dates become `YYYY-MM-DD`; a missing date
stays unknown.

```python
from matchminer_ai import load_default_preset
from matchminer_ai.patients import (
    extract_document_records,
    extract_patient_records,
    extraction_items,
)

config = load_default_preset()
config.remote["enabled"] = True
config.remote["server_urls"] = ["http://localhost:8000/v1"]

patients = extract_patient_records(notes, config=config, checkpoint_dir="ckpt/patients")
documents = extract_document_records(
    notes, config=config, patient_records=patients, checkpoint_dir="ckpt/documents"
)
items = extraction_items(documents)  # one row per extracted item
```

The two stages are independent. Passing `patient_records` to the document stage
lists each patient's known diagnoses (`D1`, `D2`, ...) in its prompts so that
items can name the cancer they concern (`cancer_ids`), as the lung abstraction
this generalizes did after its patient pass.

## Records

Every record carries `schema_version` (`mmai-clinical-extraction-v1`), its
`level`, `patient_id`, the selected `extensions`, and `items`: one list per item
type. A document record adds `note_id`, `note_date`, `note_type`,
`document_type` and `document_type_text`; a patient record adds `note_count`,
`first_note_date`, `last_note_date`, `chunk_count` and `failed_chunks`, and gives
each item an `item_id` (`D1`, `B2`, `T3`, ...). `record_json_schema(level)`
returns the JSON schema of a record.

Every item has:

- `item_type` and its fields (below). Text the note does not state is `""`, a
  choice it does not settle is `not_stated` (where the vocabulary has it), a
  list is `[]`, and an integer is `0`.
- dates as `{"value": "2024-03", "precision": "month", "source": ...}`, with
  `precision` `day`, `month`, `year` or `unknown` (value null). `source` is
  `document_text` (written in the note), `note_date` (a report's own study or
  specimen whose date is not written gets the note's date), `evidence_note_date`
  (a performed patient-level study or procedure without a written date gets the
  date of the earliest note quoted: it was done on or before then) or `unknown`.
- `evidence`: one to four exact quotes, each with the note's `note_id`,
  `note_date` and the `start`/`end` character offsets of the quote in the
  note's own text. The model writes the quotes; code locates each one in the
  note (exactly, or with different whitespace only) and keeps the note's own
  text. A quote that is not in the note is rejected.

### Item types

| List | Item type | Levels | Fields |
| --- | --- | --- | --- |
| `imaging_studies` | `imaging_study` | both | `modality`, `modality_text`, `body_regions`, `body_site_text`, `contrast`, `tracer`, `study_date` |
| `procedures` | `procedure` | both | `procedure_type`, `procedure_group` (derived), `procedure_text`, `image_guidance`, `specimen_site`, `laterality`, `procedure_date`, `specimen_result`, `result_text` |
| `diagnoses` | `diagnosis` | both | `primary_site`, `histology`, `grade`, `diagnosis_date`, `stage_group`, `t_category`, `n_category`, `m_category`, `stage_basis`, `staging_system`, `extent_at_diagnosis` |
| `biomarkers` | `biomarker` | both | `biomarker`, `biomarker_type`, `result`, `alteration`, `value`, `method`, `assay_name`, `sample_type`, `origin`, `test_date` |
| `treatments` | `treatment` | both | `treatment_type`, `regimen`, `agents`, `site`, `setting`, `line_of_therapy`, `treatment_status`, `start_date`, `end_date`, `best_response`, `end_reason`, `toxicities` |
| `disease_assessments` | `disease_assessment` | document | `disease_status`, `disease_sites` |
| `performance_status` | `performance_status` | document | `scale` (`ecog`, `karnofsky`), `score` |
| `events` | `event` | patient | `event_type`, `description`, `event_date` |

Imaging studies and procedures also record how they occurred. In a document
record, `relation` is `this_report` (the study an imaging report reads, or the
procedure that produced the specimen a pathology report examines),
`previously_performed` (an earlier or outside study or procedure the note says
was done) or `planned`; `source_of_document` is true for `this_report`. In both
levels `status` is `performed` or `planned`. Procedures, biomarkers,
treatments and events carry `cancer_ids` (patient level, and document level when
`patient_records` is given). Disease assessments and performance status are
properties of one note, so they are document-level only, as in the lung
abstraction; their table date is the note date.

An imaging report's record must include the study it reads, and a pathology
report's the procedure that produced its specimens (`relation` `this_report`),
or the answer is retried. Neither was extracted by the lung abstraction: it
used the record system's scan type only to select reports, and it recorded a
specimen's procedure only as biopsy or surgical excision.

### Controlled vocabularies

Each choice field has a closed vocabulary (`matchminer_ai.patients.extraction.vocabulary`)
that is rendered into the prompt and enforced by the JSON schema. Most end with
`other` (whose companion text field must then name the value) and `not_stated`.

| Vocabulary | Codes |
| --- | --- |
| Document type | `imaging_report`, `pathology_report`, `molecular_report`, `laboratory_report`, `clinic_note`, `hospital_note`, `procedure_note`, `radiation_oncology_note`, `treatment_record`, `other` |
| Imaging modality | `ct`, `mri`, `pet_ct`, `pet_mri`, `pet`, `xray`, `ultrasound`, `echocardiography`, `mammography`, `bone_scan`, `nuclear_medicine_other`, `fluoroscopy`, `bone_densitometry`, `other`, `not_stated` |
| Body region (list) | `brain`, `head_and_neck`, `spine`, `chest`, `breast`, `heart`, `abdomen`, `pelvis`, `extremity`, `whole_body`, `other` |
| Contrast | `with_contrast`, `without_contrast`, `with_and_without_contrast`, `not_stated` |
| Pathology procedure (group) | histologic biopsy: `core_needle_biopsy`, `needle_biopsy_not_specified`, `excisional_biopsy`, `incisional_biopsy`, `punch_or_shave_biopsy`, `endoscopic_biopsy`, `bronchoscopic_biopsy`, `mediastinoscopy`, `sentinel_lymph_node_biopsy`, `biopsy_not_specified`; aspiration or cytology: `fine_needle_aspiration`, `ebus_tbna`, `eus_fna`, `fluid_cytology`, `brushing_or_washing_cytology`; surgical resection: `surgical_resection`, `lymph_node_dissection`; `bone_marrow_biopsy`; `peripheral_blood`; `other`; `not_stated` |
| Image guidance | `ct`, `ultrasound`, `mri`, `fluoroscopy`, `stereotactic`, `none`, `not_stated` |
| Specimen result | `malignant`, `suspicious_or_atypical`, `negative`, `nondiagnostic`, `not_stated` |
| Biomarker type / result / method / sample / origin | `mutation`, `fusion_or_rearrangement`, `copy_number_change`, `protein_expression`, `genomic_signature`, `serum_tumor_marker`, `cytogenetic`, `other` / `positive`, `negative`, `equivocal`, `indeterminate`, `not_stated` / `ngs`, `pcr`, `fish`, `ihc`, `karyotype`, `flow_cytometry`, `serum_assay`, `other`, `not_stated` / `tissue`, `plasma`, `blood`, `bone_marrow`, `other`, `not_stated` / `somatic`, `germline`, `not_stated` |
| Treatment type / status / setting / best response | `systemic_therapy`, `surgery`, `radiation_therapy`, `local_therapy`, `stem_cell_transplant`, `other` / `planned`, `ongoing`, `completed`, `discontinued`, `not_stated` / `neoadjuvant`, `adjuvant`, `definitive`, `advanced_or_metastatic`, `maintenance`, `not_stated` / `complete_response`, `partial_response`, `stable_disease`, `mixed_response`, `progressive_disease`, `not_evaluable`, `not_stated` |
| Disease status | `no_evidence_of_disease`, `response`, `stable`, `mixed`, `progression`, `indeterminate` |
| Event type | `initial_diagnosis`, `recurrence`, `progression`, `new_metastasis`, `remission`, `hospitalization`, `hospice_enrollment`, `death`, `clinical_trial_enrollment`, `other` |

Sites, histologies, stages, regimens and biomarker names are text as written;
they are not normalized. With `ncit_agent_codes: true`, each treatment also
gets `agents_ncit`: every agent's NCIt concept from the bundled NCIt snapshot
(exact name or synonym, then the snapshot's spelling rules), or nulls.

## Querying records

`extraction_items` flattens document or patient records to one row per item
with the same columns at both levels: `patient_id`, `level`, `note_id`,
`note_date`, `document_type`, `item_type`, `item_id`, a uniform `date`,
`date_precision` and `date_source` (the item's primary date, or the note date
for a document's assessments), every item field (a date field `x` becomes `x`,
`x_precision` and `x_source`), `evidence` and the complete `item`. Questions a
workup review asks become DataFrame filters:

```python
import pandas as pd

items = extraction_items(documents)
studies = items[(items.item_type == "imaging_study") & (items.status == "performed")]
ct_chest = studies[(studies.modality == "ct")
                   & studies.body_regions.map(lambda regions: "chest" in regions)]
reference = pd.Timestamp("2024-04-01")
recent = ct_chest[(pd.to_datetime(ct_chest.date) <= reference)
                  & (pd.to_datetime(ct_chest.date) >= reference - pd.Timedelta(days=30))
                  & (ct_chest.date_precision == "day")]

procedures = items[items.item_type == "procedure"]
core_biopsies = procedures[procedures.procedure_type == "core_needle_biopsy"]
any_histology = procedures[procedures.procedure_group == "histologic_biopsy"]
```

Filter by `item_type` first: fields of other item types are missing in those
rows. A month- or year-precision date is not a day; a query must decide how to
treat it, and an undated study mentioned in a note (`previously_performed`
with an `unknown` date) was done on or before that note's date. These are
documentation signals for review, not determinations that a test was or was
not done.

## Validation, retries and failures

Each request's answer is decoded under its JSON schema when the remote backend
is enabled and `structured_outputs` is on, then validated in code: field types
and vocabularies, `other` companions, dates, ids, `cancer_ids`, the report's own
study or procedure, and every quote. A failed answer is retried with its
problems named, up to `max_attempts`, then once with thinking off
(`reasoning_off_fallback`). When the last attempt's only problems are quotes not
found in the note, the items left without a found quote are dropped and counted
(`partial`); any other problem on the last attempt fails the note (document
level) or skips the chunk (patient level, listed in `failed_chunks`). A request
that fails in transport fails the note, or stops the patient so that a rerun
resumes at that chunk.

Returned DataFrames have `extraction_status` (`complete`, `partial` or
`failed`), `error`, `item_count` and the `record` dict (None when failed).
`return_metadata=True` adds the schema version, prompt and response-schema
hashes, extensions, model metadata and every request's attempts (finish reason
and problem codes, without note text; answers and reasoning only in debug mode).
`return_qc=True` adds failed, partial and truncated counts.

## Patient-level chunking

A patient's notes are read in date order, packed into chunks of
`patient_extraction.chunk_tokens` tokens (a longer note is split with
`chunk_overlap_tokens` overlap), as serial summarization reads them, but
without sentence deduplication, so quotes stay exact. Each chunk's request
shows the record so far (items with their ids, fields and the numbers of the
notes that support them) and returns only the chunk's new and changed items;
code merges them, adding each update's evidence to the item's earlier evidence.
The lung abstraction instead asked for the whole running JSON again on every
chunk, so its cost grew with the square of the record's length, and a later
answer could silently drop an item. A chunk whose prompt, with the output
reserve, would exceed `context_tokens` fails the patient rather than being
truncated.

## Checkpoints

With `checkpoint_dir`, the document stage appends each finished note to
`document_records.jsonl` and the patient stage saves each patient after every
chunk under `patient_records/` (files named by a hash of the patient id). A
rerun with unchanged notes, context and extraction contract (schema version,
prompt, field guide, extensions and response schema) reuses complete and
partial results and retries the rest. Checkpoints hold patient-derived text.

## Extensions

Disease-specific item types and prompt guidance are added by extensions,
selected by name in `document_extraction.extensions` and
`patient_extraction.extensions`, without changing the core model. The packaged
`thoracic` extension adds a `smoking_history` item (`smoking_status`,
`pack_years`, `quit_date`) and the thoracic guidance of the lung abstraction:
histology and biomarker examples, lobe and nodal-station sites, thoracic
operations and EBUS-TBNA. New extensions are registered with
`register_extension(Extension(name, description, item_types, guidance))`; their
item types must not reuse a core name, list key or id prefix.

::: matchminer_ai.patients.extraction
    options:
      members:
        - extract_document_records
        - extract_patient_records
        - extraction_items
        - record_json_schema
        - document_response_schema
        - patient_response_schema
        - Extension
        - register_extension
        - ItemType
        - Field
        - Vocabulary
