"""Controlled vocabularies of the structured clinical extraction data model.

Each vocabulary is a closed list of codes with a one-line meaning. The meanings
are rendered into the extraction prompts and the codes into the JSON schemas
that constrain the model's answers, so the prompt, the schema and the validator
read the same list. Choice vocabularies end with an ``other`` escape (whose
field has a free-text companion that must then name the value) and, where a
document can leave the value unsaid, ``not_stated``.

The vocabularies are general across cancers. Disease-specific vocabularies
belong in an extension (:mod:`.extensions`), not here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    """One code of a vocabulary, its meaning, and an optional coarser group."""

    code: str
    description: str
    group: str | None = None


@dataclass(frozen=True)
class Vocabulary:
    """A closed list of codes.

    ``other`` names the escape code whose field needs a free-text value, and
    ``unknown`` the code a document that does not say uses (None when the
    vocabulary has no such code, as for list-valued fields whose empty list
    means "not stated").
    """

    name: str
    description: str
    terms: tuple[Term, ...]
    other: str | None = "other"
    unknown: str | None = "not_stated"

    def __post_init__(self):
        codes = [term.code for term in self.terms]
        if len(set(codes)) != len(codes):
            raise ValueError(f"Vocabulary {self.name} repeats a code")
        for special in (self.other, self.unknown):
            if special is not None and special not in codes:
                raise ValueError(f"Vocabulary {self.name} lacks its {special} code")

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(term.code for term in self.terms)

    def group(self, code: str) -> str | None:
        """The coarser group of ``code`` (None when the vocabulary has none)."""
        for term in self.terms:
            if term.code == code:
                return term.group
        raise KeyError(f"{code!r} is not a {self.name} code")

    def describe(self) -> str:
        """The codes and their meanings, one per line, for a prompt."""
        return "\n".join(f"      {t.code}: {t.description}" for t in self.terms)


def _vocabulary(name, description, terms, *, other="other", unknown="not_stated"):
    return Vocabulary(
        name=name,
        description=description,
        terms=tuple(Term(*term) for term in terms),
        other=other,
        unknown=unknown,
    )


DOCUMENT_TYPES = _vocabulary(
    "document_type",
    "What kind of document this is.",
    [
        ("imaging_report", "A radiology or nuclear medicine report of an imaging study."),
        (
            "pathology_report",
            "A surgical pathology, cytology, hematopathology or bone marrow report "
            "on a specimen.",
        ),
        (
            "molecular_report",
            "A genomic or molecular test report (tumor or germline sequencing, PCR, "
            "FISH, circulating tumor DNA).",
        ),
        ("laboratory_report", "Blood or other laboratory results."),
        ("clinic_note", "An outpatient visit, consultation or progress note."),
        (
            "hospital_note",
            "An inpatient note: admission, progress, consultation or discharge summary.",
        ),
        (
            "procedure_note",
            "An operative or procedure note (surgery, endoscopy, bronchoscopy, "
            "interventional radiology).",
        ),
        (
            "radiation_oncology_note",
            "A radiation oncology consultation, treatment summary or completion note.",
        ),
        (
            "treatment_record",
            "A chemotherapy, infusion or medication administration record or "
            "treatment plan.",
        ),
        ("other", "Any other document; name it in document_type_text."),
    ],
    unknown=None,
)

IMAGING_MODALITIES = _vocabulary(
    "imaging_modality",
    "The imaging technique of a study.",
    [
        ("ct", "Computed tomography, including CT angiography and low-dose CT."),
        ("mri", "Magnetic resonance imaging, including MR angiography and MRCP."),
        ("pet_ct", "PET/CT with any tracer (FDG, PSMA, DOTATATE and so on)."),
        ("pet_mri", "PET/MRI with any tracer."),
        ("pet", "PET with no CT or MRI component stated."),
        ("xray", "Plain radiograph, including chest X-ray and skeletal survey."),
        ("ultrasound", "Ultrasound other than echocardiography."),
        ("echocardiography", "Echocardiogram."),
        ("mammography", "Mammography or breast tomosynthesis."),
        ("bone_scan", "Nuclear medicine bone scan (skeletal scintigraphy)."),
        (
            "nuclear_medicine_other",
            "Another nuclear medicine study (MUGA, SPECT, thyroid or renal scan, "
            "lymphoscintigraphy).",
        ),
        ("fluoroscopy", "A fluoroscopic study such as a barium swallow."),
        ("bone_densitometry", "DXA bone density scan."),
        ("other", "Another imaging technique; name it in modality_text."),
        ("not_stated", "The document does not say which technique."),
    ],
)

BODY_REGIONS = _vocabulary(
    "body_region",
    "A body region an imaging study covers. A study can cover several.",
    [
        ("brain", "Brain or head (intracranial)."),
        (
            "head_and_neck",
            "Face, sinuses, orbits, neck, thyroid, pharynx or larynx.",
        ),
        ("spine", "Cervical, thoracic, lumbar or sacral spine."),
        ("chest", "Chest: lungs, mediastinum, pleura, chest wall."),
        ("breast", "One or both breasts."),
        ("heart", "Heart (cardiac imaging)."),
        ("abdomen", "Abdomen: liver, pancreas, kidneys, adrenal glands, bowel."),
        ("pelvis", "Pelvis: bladder, prostate, uterus, ovaries, rectum."),
        ("extremity", "Arm or leg."),
        (
            "whole_body",
            "Whole body or skull base to thigh (typical of PET/CT, bone scan, "
            "skeletal survey).",
        ),
        ("other", "Another region; name it in body_site_text."),
    ],
    unknown=None,
)

CONTRAST = _vocabulary(
    "contrast",
    "Whether intravenous contrast was given.",
    [
        ("with_contrast", "With intravenous contrast (iodinated or gadolinium)."),
        ("without_contrast", "Without intravenous contrast."),
        ("with_and_without_contrast", "Images both before and after contrast."),
        ("not_stated", "The document does not say."),
    ],
    other=None,
)

#: Tissue and specimen procedures, grouped so that a query can ask for one
#: technique ("core_needle_biopsy") or any histologic sample ("histologic_biopsy").
PATHOLOGY_PROCEDURES = _vocabulary(
    "pathology_procedure",
    "How a specimen was obtained.",
    [
        (
            "core_needle_biopsy",
            "Core needle biopsy, usually percutaneous and image guided.",
            "histologic_biopsy",
        ),
        (
            "needle_biopsy_not_specified",
            "A needle biopsy whose core or aspiration technique is not stated.",
            "histologic_biopsy",
        ),
        (
            "excisional_biopsy",
            "Excisional biopsy: a whole lesion or lymph node removed for diagnosis.",
            "histologic_biopsy",
        ),
        (
            "incisional_biopsy",
            "Incisional biopsy: part of a lesion removed surgically.",
            "histologic_biopsy",
        ),
        ("punch_or_shave_biopsy", "Punch or shave biopsy of skin.", "histologic_biopsy"),
        (
            "endoscopic_biopsy",
            "Forceps biopsy during GI endoscopy, colonoscopy, cystoscopy, "
            "hysteroscopy, laryngoscopy and similar.",
            "histologic_biopsy",
        ),
        (
            "bronchoscopic_biopsy",
            "Endobronchial or transbronchial biopsy at bronchoscopy, including "
            "navigational bronchoscopy (not EBUS-TBNA).",
            "histologic_biopsy",
        ),
        (
            "mediastinoscopy",
            "Mediastinoscopy or mediastinotomy lymph node sampling.",
            "histologic_biopsy",
        ),
        (
            "sentinel_lymph_node_biopsy",
            "Sentinel lymph node biopsy.",
            "histologic_biopsy",
        ),
        (
            "biopsy_not_specified",
            "A biopsy whose technique is not stated.",
            "histologic_biopsy",
        ),
        (
            "fine_needle_aspiration",
            "Fine needle aspiration (FNA) other than EBUS or EUS.",
            "aspiration_or_cytology",
        ),
        (
            "ebus_tbna",
            "Endobronchial ultrasound-guided transbronchial needle aspiration.",
            "aspiration_or_cytology",
        ),
        (
            "eus_fna",
            "Endoscopic ultrasound-guided fine needle aspiration or biopsy.",
            "aspiration_or_cytology",
        ),
        (
            "fluid_cytology",
            "Cytology of a body fluid: pleural, peritoneal or pericardial fluid, "
            "cerebrospinal fluid, urine.",
            "aspiration_or_cytology",
        ),
        (
            "brushing_or_washing_cytology",
            "Brushings, washings, lavage or a Pap test.",
            "aspiration_or_cytology",
        ),
        (
            "surgical_resection",
            "Surgical resection of a tumor or organ (lobectomy, mastectomy, "
            "colectomy, prostatectomy, wide local excision and so on).",
            "surgical_resection",
        ),
        (
            "lymph_node_dissection",
            "Lymph node dissection.",
            "surgical_resection",
        ),
        (
            "bone_marrow_biopsy",
            "Bone marrow aspirate, core (trephine) biopsy, or both.",
            "bone_marrow",
        ),
        (
            "peripheral_blood",
            "A peripheral blood sample (flow cytometry, circulating tumor DNA).",
            "blood",
        ),
        ("other", "Another procedure; name it in procedure_text.", "other"),
        ("not_stated", "The document does not say how the specimen was obtained.", None),
    ],
)

IMAGE_GUIDANCE = _vocabulary(
    "image_guidance",
    "The imaging that guided a procedure.",
    [
        ("ct", "CT guided."),
        ("ultrasound", "Ultrasound guided (transabdominal or other surface probe)."),
        ("mri", "MRI guided."),
        ("fluoroscopy", "Fluoroscopy guided."),
        ("stereotactic", "Stereotactic (mammographic) guidance."),
        ("none", "Stated to be without image guidance (for example palpation guided)."),
        ("not_stated", "Not stated, or not applicable."),
    ],
    other=None,
)

LATERALITY = _vocabulary(
    "laterality",
    "Side of the body.",
    [
        ("left", "Left."),
        ("right", "Right."),
        ("bilateral", "Both sides."),
        ("not_stated", "Not stated, midline, or not applicable."),
    ],
    other=None,
)

SPECIMEN_RESULTS = _vocabulary(
    "specimen_result",
    "What a specimen showed about malignancy.",
    [
        ("malignant", "Invasive or in situ malignancy present."),
        ("suspicious_or_atypical", "Suspicious for, or atypical but not diagnostic of, malignancy."),
        ("negative", "Benign, or negative for malignancy."),
        ("nondiagnostic", "Insufficient or nondiagnostic specimen."),
        ("not_stated", "The result is not stated (for example still pending)."),
    ],
    other=None,
)

STAGE_BASES = _vocabulary(
    "stage_basis",
    "Whether a stage is clinical or pathologic.",
    [
        ("clinical", "Clinical stage (cTNM), including imaging-based staging."),
        ("pathologic", "Pathologic stage (pTNM) from surgical specimens."),
        ("not_stated", "Not stated."),
    ],
    other=None,
)

DISEASE_EXTENTS = _vocabulary(
    "disease_extent",
    "Extent of a cancer at diagnosis, in summary-stage terms.",
    [
        ("localized", "Confined to the organ of origin."),
        ("regional", "Spread to regional lymph nodes or adjacent tissue."),
        ("distant", "Distant metastasis."),
        ("not_stated", "Not stated."),
    ],
    other=None,
)

BIOMARKER_TYPES = _vocabulary(
    "biomarker_type",
    "The kind of biomarker.",
    [
        ("mutation", "A sequence variant (point mutation, insertion, deletion)."),
        ("fusion_or_rearrangement", "A gene fusion or rearrangement."),
        ("copy_number_change", "An amplification or deletion of a gene."),
        ("protein_expression", "Protein expression, usually by IHC (PD-L1, HER2, ER)."),
        (
            "genomic_signature",
            "A genome-wide signature: tumor mutational burden, microsatellite "
            "instability, homologous recombination deficiency.",
        ),
        ("serum_tumor_marker", "A blood tumor marker (PSA, CEA, CA-125, AFP)."),
        ("cytogenetic", "A karyotype or cytogenetic abnormality."),
        ("other", "Another kind; describe it in the biomarker field."),
    ],
    unknown=None,
)

BIOMARKER_RESULTS = _vocabulary(
    "biomarker_result",
    "The result of a biomarker test.",
    [
        (
            "positive",
            "Detected, altered, amplified, rearranged, overexpressed or positive.",
        ),
        ("negative", "Not detected, wild type, not amplified or negative."),
        ("equivocal", "Equivocal or borderline."),
        ("indeterminate", "The test failed or could not be interpreted."),
        ("not_stated", "No result category is stated (for example a value alone)."),
    ],
    other=None,
)

TEST_METHODS = _vocabulary(
    "test_method",
    "The laboratory method of a biomarker test.",
    [
        ("ngs", "Next-generation sequencing panel, exome or genome."),
        ("pcr", "PCR-based assay."),
        ("fish", "Fluorescence in situ hybridization."),
        ("ihc", "Immunohistochemistry."),
        ("karyotype", "Conventional cytogenetics."),
        ("flow_cytometry", "Flow cytometry."),
        ("serum_assay", "A serum or plasma protein assay."),
        ("other", "Another method; it may be named in the alteration or value field."),
        ("not_stated", "Not stated."),
    ],
)

SAMPLE_TYPES = _vocabulary(
    "sample_type",
    "The sample a biomarker was measured in.",
    [
        ("tissue", "Tumor tissue or cytology material."),
        ("plasma", "Plasma, as for circulating tumor DNA (liquid biopsy)."),
        ("blood", "Whole blood or serum (including germline testing on blood)."),
        ("bone_marrow", "Bone marrow."),
        ("other", "Another sample, such as saliva or cerebrospinal fluid."),
        ("not_stated", "Not stated."),
    ],
    other=None,
)

BIOMARKER_ORIGINS = _vocabulary(
    "biomarker_origin",
    "Whether a variant is inherited.",
    [
        ("somatic", "Somatic (tumor) finding."),
        ("germline", "Germline (inherited) finding."),
        ("not_stated", "Not stated."),
    ],
    other=None,
)

TREATMENT_TYPES = _vocabulary(
    "treatment_type",
    "The kind of anticancer treatment.",
    [
        (
            "systemic_therapy",
            "Chemotherapy, targeted therapy, immunotherapy, hormonal therapy, "
            "antibody-drug conjugate or cellular therapy.",
        ),
        ("surgery", "Cancer surgery."),
        ("radiation_therapy", "External beam radiation, brachytherapy or radioligand."),
        ("local_therapy", "Ablation, embolization or another local treatment."),
        ("stem_cell_transplant", "Autologous or allogeneic stem cell transplant."),
        ("other", "Another anticancer treatment; describe it in regimen."),
    ],
    unknown=None,
)

TREATMENT_STATUSES = _vocabulary(
    "treatment_status",
    "Where a treatment stands as of the text.",
    [
        ("planned", "Decided on and scheduled, not yet started."),
        ("ongoing", "Started and continuing."),
        ("completed", "Finished as planned."),
        ("discontinued", "Stopped early."),
        ("not_stated", "Not stated."),
    ],
    other=None,
)

TREATMENT_SETTINGS = _vocabulary(
    "treatment_setting",
    "The clinical setting of a treatment.",
    [
        ("neoadjuvant", "Before definitive surgery or radiation."),
        ("adjuvant", "After definitive surgery or radiation, with no known residual disease."),
        ("definitive", "Curative-intent primary treatment, such as chemoradiation."),
        (
            "advanced_or_metastatic",
            "For advanced, metastatic or recurrent disease, including palliative care.",
        ),
        ("maintenance", "Maintenance after an earlier treatment."),
        ("not_stated", "Not stated."),
    ],
    other=None,
)

TREATMENT_RESPONSES = _vocabulary(
    "treatment_response",
    "The best response the text reports to a treatment.",
    [
        ("complete_response", "Complete response or no evidence of disease."),
        ("partial_response", "Partial response or improvement."),
        ("stable_disease", "Stable disease."),
        ("mixed_response", "Some sites improved and others worsened."),
        ("progressive_disease", "Progression."),
        ("not_evaluable", "Not evaluable (for example adjuvant treatment)."),
        ("not_stated", "Not stated."),
    ],
    other=None,
)

DISEASE_STATUSES = _vocabulary(
    "disease_status",
    "A document's overall assessment of a cancer.",
    [
        ("no_evidence_of_disease", "No evidence of disease."),
        ("response", "Smaller, improved or resolved."),
        ("stable", "Unchanged."),
        ("mixed", "Some sites improved and others worsened."),
        ("progression", "Larger, or new lesions."),
        ("indeterminate", "The document cannot tell."),
    ],
    other=None,
    unknown=None,
)

PERFORMANCE_SCALES = _vocabulary(
    "performance_scale",
    "The performance status scale.",
    [
        ("ecog", "ECOG (Zubrod/WHO) performance status, 0-5."),
        ("karnofsky", "Karnofsky performance status, 0-100."),
    ],
    other=None,
    unknown=None,
)

EVENT_TYPES = _vocabulary(
    "event_type",
    "The kind of key dated event.",
    [
        ("initial_diagnosis", "First diagnosis of a cancer."),
        ("recurrence", "Recurrence after a disease-free interval."),
        ("progression", "Progression of known disease."),
        ("new_metastasis", "A new site of metastasis."),
        ("remission", "Complete response or no evidence of disease."),
        ("hospitalization", "Hospital admission."),
        ("hospice_enrollment", "Enrollment in hospice, or a decision to stop anticancer therapy."),
        ("death", "Death."),
        ("clinical_trial_enrollment", "Enrollment in a clinical trial."),
        ("other", "Another key event; describe it."),
    ],
    unknown=None,
)

#: How an imaging study or procedure relates to the document that mentions it.
DOCUMENT_RELATIONS = _vocabulary(
    "document_relation",
    "How an imaging study or procedure relates to this document.",
    [
        (
            "this_report",
            "The study this imaging report reads, or the procedure that produced "
            "the specimen this pathology or molecular report examines.",
        ),
        (
            "previously_performed",
            "An earlier or outside study or procedure that this document says was done.",
        ),
        ("planned", "Ordered, scheduled or recommended, not yet done."),
    ],
    other=None,
    unknown=None,
)

#: Patient-level counterpart of DOCUMENT_RELATIONS.
OCCURRENCE_STATUSES = _vocabulary(
    "occurrence_status",
    "Whether an imaging study or procedure was done.",
    [
        ("performed", "Done (on the date given, when one is known)."),
        ("planned", "Ordered or scheduled and not documented as done."),
    ],
    other=None,
    unknown=None,
)

DATE_PRECISIONS = ("day", "month", "year", "unknown")

VOCABULARIES: dict[str, Vocabulary] = {
    vocabulary.name: vocabulary
    for vocabulary in (
        DOCUMENT_TYPES,
        IMAGING_MODALITIES,
        BODY_REGIONS,
        CONTRAST,
        PATHOLOGY_PROCEDURES,
        IMAGE_GUIDANCE,
        LATERALITY,
        SPECIMEN_RESULTS,
        STAGE_BASES,
        DISEASE_EXTENTS,
        BIOMARKER_TYPES,
        BIOMARKER_RESULTS,
        TEST_METHODS,
        SAMPLE_TYPES,
        BIOMARKER_ORIGINS,
        TREATMENT_TYPES,
        TREATMENT_STATUSES,
        TREATMENT_SETTINGS,
        TREATMENT_RESPONSES,
        DISEASE_STATUSES,
        PERFORMANCE_SCALES,
        EVENT_TYPES,
        DOCUMENT_RELATIONS,
        OCCURRENCE_STATUSES,
    )
}

__all__ = [
    "BIOMARKER_ORIGINS",
    "BIOMARKER_RESULTS",
    "BIOMARKER_TYPES",
    "BODY_REGIONS",
    "CONTRAST",
    "DATE_PRECISIONS",
    "DISEASE_EXTENTS",
    "DISEASE_STATUSES",
    "DOCUMENT_RELATIONS",
    "DOCUMENT_TYPES",
    "EVENT_TYPES",
    "IMAGE_GUIDANCE",
    "IMAGING_MODALITIES",
    "LATERALITY",
    "OCCURRENCE_STATUSES",
    "PATHOLOGY_PROCEDURES",
    "PERFORMANCE_SCALES",
    "SAMPLE_TYPES",
    "SPECIMEN_RESULTS",
    "STAGE_BASES",
    "TEST_METHODS",
    "TREATMENT_RESPONSES",
    "TREATMENT_SETTINGS",
    "TREATMENT_STATUSES",
    "TREATMENT_TYPES",
    "Term",
    "VOCABULARIES",
    "Vocabulary",
]
