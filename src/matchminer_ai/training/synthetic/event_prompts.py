"""Seed and patient-history prompts for synthetic patient generation.

Edit the clinical instruction blocks here. Generation, validation, and
checkpointing belong to ``synthetic.generate``, not this prompt module.
"""


SYSTEM_PROMPT = (
    "I'm a medical oncologist and data scientist. Your job is to help me create synthetic clinical data "
    "for cancer research. Realism is CRITICAL. Use all your oncology and biology knowledge to strive for accuracy."
)


DIAGNOSIS_INSTRUCTIONS = (
    'Diagnosis events should include primary site/cancer type, histology, and stage where defined. Include key data elements relevant for workup or treatment of that type of cancer, as long as your data elements are compatible with the clinical context provided.\n'
    'Give a stage only when staging is a defined concept for that cancer, using its standard system: TNM stage '
    'and summary stage for solid tumors staged that way, or the cancer\'s own system where one applies, such as '
    'Lugano (Ann Arbor) stage for lymphoma, Rai or Binet stage for CLL, or R-ISS stage for multiple myeloma. Give '
    'no stage for cancers without a defined staging system, such as acute leukemias, myelodysplastic syndromes, or '
    'most primary brain tumors; give the cancer\'s standard '
    'classification, grade, or risk group instead, such as WHO grade for a glioma or ELN risk for AML.\n'
    'For TNM, use clinical staging (cTNM) unless the primary tumor was surgically resected; give pathologic staging '
    '(pTNM, or ypTNM after neoadjuvant therapy) only for a resected primary. Patients with distant metastases '
    'at diagnosis usually do not have the primary resected; include that only when it is standard for the '
    'cancer and situation, such as cytoreductive nephrectomy or resection of limited colorectal liver metastases.\n'
)


DIVERSITY_INSTRUCTIONS = (
    'To ensure diversity in the generated data, vary patient age, gender (if it is a cancer type that occurs in both sexes), name, stage at diagnosis, biomarkers, '
    'treatment approaches, and disease course (e.g., stable disease, progression, remission, recurrence). However, this variation must remain within the bounds of the clinical context provided, and it MUST remain clinically realistic. All anatomy must be accurate, any therapy only given for a certain known biomarker in real life must also only be given for that known biomarker in synthetic data, and so on.\n'
)


CONSISTENCY_INSTRUCTIONS = (
    'VERY IMPORTANT: All events MUST be in chronological order and must be internally consistent in that context. '
    'For example, an event from a younger age must not follow an event from an older age, and an organ that has '
    'been removed must not later be described as present.\n'
    'For patients who have surgical resection for their cancer diagnosis, the diagnosis event line item should follow the resection event.\n'
    'CRITICAL: Do not add findings, including claims that disease is absent, for organs that would not have been '
    'visualized by the specified scan type and field of view; for example, a CT of the chest, abdomen, and '
    'pelvis cannot establish the presence or absence of brain metastases.\n'
    'CRITICAL: Do not mention screening or enrollment on the trial itself anywhere in your chronology!! That would '
    'contaminate the synthetic data and render them useless, since we will be using the synthetic data to derive '
    'clinical history models to match patients to clinical trials on which they have not yet enrolled!\n'
)


ELIGIBILITY_INSTRUCTIONS = (
    'Apply all supplied cohort-specific and common eligibility criteria together. Shared criteria can be '
    'inclusion or exclusion requirements; do not interpret them all as exclusions. '
    'For a single cohort, the supplied text is the trial\'s original eligibility criteria. '
    'Prior-treatment and biomarker requirements may each include both positive and '
    'negative conditions. Preserve each clause, its negation, AND/OR logic, and exceptions when interpreting '
    'eligibility.\n'
)


EVENT_INSTRUCTIONS = (
    'Use everything you know about cancer and clinical oncology.\n'
    'Types of events might include a diagnosis as recorded in a cancer registry; initiation of a systemic therapy; '
    'a surgery; a radiation treatment; an adverse event; a clinical progress note; an imaging report; a pathology '
    'report; and a next-generation sequencing (NGS) report. For progress notes, imaging reports, pathology '
    'reports, and NGS reports, include findings documented in the report in the description of the event. \n'
    'NGS reports should be very detailed; they should include both any key actionable alterations if present, and '
    'comutations/fusions/copy number alterations; most reports should describe alterations in many genes, even '
    'though only some of those will be clinically relevant. \n'
    'CRITICAL: Genomic findings should be realistic based on known mutation and comutation patterns. For example, '
    'remember that EGFR mutant lung cancers almost never have KRAS co-mutations. \n'
    'There should be one event per line of text in your output, and each event should be formatted as a sentence. \n'
    'Most patients will have many events along their disease trajectories (often 20-30), but never '
    'extend a history past its final clinical state to add events. \n'
    'Tag each event with an event type at the beginning of the line. Acceptable event types include '
    '<demographics>, <diagnosis>, <systemic>, <surgery>, <radiation>, <adverse_event>, <clinical_note>, '
    '<imaging_report>, <pathology_report>, and <ngs_report>.\n'
    'Each event should correspond only to one point in time, and each report should correspond only to one report '
    'that could have been written at that time.\n'
    + DIAGNOSIS_INSTRUCTIONS
    + (
        'Imaging report events must describe only one radiographic study and should specify the type of study. Imaging '
        'report events must also indicate whether cancer was present on the scan; if so, whether it was responding, '
        'progressing, or neither; and what metastatic sites were involved.\n'
        'Oncologist note events must indicate whether cancer was present at the time; and if so, whether it was '
        'responding, progressing, or neither. \n'
        'NGS report events should indicate the diagnosis, specimen site, and genomic findings.\n'
        'Here is an example of what your output should look like. This is hypothetical, just to illustrate the '
        "formatting. Don't use this text in your output.\n"
        'Do adhere closely to the formatting.\n'
        '(Beginning of example)\n'
        '<demographics> The patient is a male named John Smith, aged 70 years and 0 weeks at the beginning of the history.\n'
        '<imaging_report> At age 70 years and 0 weeks, a brain MRI obtained for new left-sided weakness showed a 15 mm enhancing right basal ganglia lesion suspicious for malignancy, with surrounding edema.\n'
        '<surgery> At age 70 years and 1 week, the patient underwent stereotactic biopsy of the right basal ganglia lesion.\n'
        '<pathology_report> At age 70 years and 2 weeks, the brain biopsy showed diffuse large B-cell lymphoma with CD20-positive malignant cells.\n'
        '<imaging_report> At age 70 years and 2 weeks, an FDG PET/CT from scalp to toes showed no extracranial lymphoma.\n'
        '<diagnosis> At age 70 years and 2 weeks, the patient was diagnosed with primary central nervous system diffuse large B-cell lymphoma involving the right basal ganglia, with disease confined to the CNS, positive histologic confirmation, a 15 mm lesion, and no B symptoms; TNM staging was not applicable.\n'
        '<clinical_note> At age 70 years and 2 weeks, the oncologist reviewed the biopsy and staging studies; active CNS lymphoma was present and treatment had not yet begun.\n'
        '<systemic> At age 70 years and 3 weeks, the patient began methotrexate, temozolomide, and rituximab for primary CNS lymphoma.\n'
        '<clinical_note> At age 70 years and 4 weeks, CNS lymphoma remained present; left-sided weakness had improved after treatment began, but radiographic response had not yet been assessed.\n'
        '<ngs_report> At age 70 years and 6 weeks, sequencing of the right basal ganglia biopsy showed MYD88 p.L265P, CD79B p.Y196F, and CDKN2A loss in the diffuse large B-cell lymphoma.\n'
        '<imaging_report> At age 70 years and 8 weeks, a brain MRI showed the right basal ganglia lesion had decreased to 8 mm with less surrounding edema, consistent with partial response and no new intracranial lesions.\n'
        '<clinical_note> At age 70 years and 8 weeks, the oncologist documented residual CNS lymphoma responding to treatment and improvement in left-sided weakness.\n'
        '<adverse_event> At age 70 years and 9 weeks, the patient developed transient grade 1 nausea during chemotherapy.\n'
        '<imaging_report> At age 70 years and 12 weeks, a brain MRI showed a residual 4 mm enhancing right basal ganglia lesion, continued response, and no new intracranial lesions.\n'
        '<clinical_note> At age 70 years and 12 weeks, the oncologist documented continued response with residual CNS lymphoma.\n'
        '<imaging_report> At age 70 years and 20 weeks, a brain MRI showed no residual enhancing lymphoma in the brain.\n'
        '<clinical_note> At age 70 years and 20 weeks, the oncologist documented a complete radiographic response and resolution of the left-sided weakness.\n'
        '<imaging_report> At age 71 years and 4 weeks, a surveillance brain MRI showed no recurrent intracranial lymphoma.\n'
        '<imaging_report> At age 71 years and 12 weeks, a brain MRI obtained for recurrent left-sided weakness showed a new enhancing lesion at the prior right basal ganglia site, concerning for recurrent CNS lymphoma.\n'
        '<clinical_note> At age 71 years and 12 weeks, the oncologist reviewed the new brain lesion and documented suspected recurrent CNS lymphoma requiring further evaluation.\n'
        '(End of example)\n'
        '\n'
    )
)


SEED_COMORBIDITY_INSTRUCTIONS = (
    'Give each patient realistic non-cancer comorbidities, usually one to four, chosen for the patient\'s age, sex, '
    'cancer type, and risk factors (for example, COPD in a heavy former smoker or type 2 diabetes with obesity), '
    'with each condition\'s current severity or control and its treatment. If the patient received prior cancer '
    'treatment, also give the toxicities that treatment realistically caused, such as oxaliplatin neuropathy, '
    'immune-related colitis or hypothyroidism after a checkpoint inhibitor, or anthracycline cardiomyopathy: name '
    'the causing treatment, the CTCAE grade, the management (for example, dose reduction, treatment hold, '
    'corticosteroids, or discontinuation), and the toxicity\'s status at the end of the history. A patient with no '
    'prior cancer treatment has no treatment toxicities. Comorbidities and toxicities must agree with the supplied '
    'criteria: for matching patients, none may violate an exclusion or an organ-function, performance-status, or '
    'washout requirement (for example, keep residual neuropathy within any allowed grade, and give no active '
    'autoimmune disease when the criteria exclude it). For near-miss patients, a comorbidity or toxicity may be the '
    'feature that narrowly misses a criterion.\n'
)


SECOND_PRIMARY_INSTRUCTION = (
    "In this particular group of patients, each patient have had a second primary prior unrelated cancer history before "
    "the cancer relevant to the trial. Incorporate events for the prior cancer as well as the current cancer into your output."
)


SEED_FORMAT_EXAMPLE = (
    'Marisol Vega, female, was diagnosed with sigmoid colon adenocarcinoma at '
    '54 years and 19 weeks: site sigmoid colon, histology adenocarcinoma, '
    'grade 2, tumor size 2.8 cm, pT2 pN0 cM0, stage I, summary '
    'stage localized, 18 regional lymph nodes examined with none positive, '
    'lymphovascular and perineural invasion absent, margins negative; '
    'microsatellite stable, KRAS p.G12D mutation detected, BRAF wild-type.'
)


def seed_messages(criteria, patient_class, rng):
    """Seed-stage messages; ``rng`` decides whether the cohort requests a prior second primary."""
    population = ('meet all of these criteria at the end of their histories'
                  if patient_class == 'positive' else
                  'would narrowly miss at least one of these criteria at the end of their histories')
    content = (
        f'Create five diverse patient seeds for patients with cancer who {population}.\n'
        'Eligibility criteria:\n' + criteria + '\n\n'
        'This is the seed stage only. Output exactly five patients as five short '
        'prose paragraphs, separated by <new_patient> on its own line. Each '
        'paragraph must give the patient\'s name, sex, diagnoses, and age at each '
        'diagnosis. Use no event tags, headings, numbering, or commentary.\n'
        'Express each age at diagnosis as whole years plus additional weeks, '
        'for example "50 years and 21 weeks". Use nonnegative integer years and '
        'integer weeks from 0 through 51; include both units even when weeks is '
        'zero. Give ages only for diagnoses; the later event-generation call '
        'will determine the age at the beginning of the complete history. '
        'Include relevant cancer site, histology, stage where staging is defined, '
        'and biomarkers in the prose. If a prior unrelated cancer is included, describe its diagnosis '
        'and its own age at diagnosis. Keep each seed brief and limited to these '
        'facts plus the concise clinical facts needed to establish the patient\'s '
        'state at the end of the history against these criteria, including prior '
        'treatment/response and its toxicities, performance status, organ function, '
        'and comorbidities. Do not write a full event chronology or reports here. '
        'A separate call will expand each seed into its clinical events.\n'
        + DIAGNOSIS_INSTRUCTIONS + SEED_COMORBIDITY_INSTRUCTIONS + ELIGIBILITY_INSTRUCTIONS
        + DIVERSITY_INSTRUCTIONS + CONSISTENCY_INSTRUCTIONS
    )
    if rng.randrange(10) == 5:
        content += '\n' + SECOND_PRIMARY_INSTRUCTION + '\n'
    content += (
        '\nKeep the five identities and their clinical facts distinct. Biological '
        'and anatomical consistency take precedence over diversity. '
        + ('For these positive examples, all eligibility criteria take precedence '
           'over diversity requests: omit a prior unrelated cancer when it would '
           'violate the supplied exclusions.\n' if patient_class == 'positive' else
           'Each negative seed must contain concrete clinical facts that violate '
           'at least one supplied inclusion or exclusion requirement while '
           'remaining a plausible near-miss case.\n')
        + 'Express these features as patient facts; do not mention eligibility, '
        'screening, enrollment, or positive/negative labels in the seed. Do not '
        'state a life expectancy or other prognosis estimate, such as "life '
        'expectancy greater than 3 months"; describe the clinical facts instead.\n'
        + '\nThe following is one hypothetical prose seed, provided only to '
        'illustrate formatting. Do not copy this example into your output or '
        'reuse its patient identity. Generate five new patients. '
        'It is not an additional patient or a template for the required cancer '
        'type or biomarkers; use the supplied criteria for those.\n'
        '(Beginning of seed prose format example)\n' + SEED_FORMAT_EXAMPLE
        + '\n(End of seed prose format example)\n'
        'Return only five new short prose seeds separated by <new_patient>.'
    )
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]


def patient_messages(seed):
    content = (
        'Generate the complete event chronology for exactly one patient based on the assigned prose seed.\n'
        + EVENT_INSTRUCTIONS + CONSISTENCY_INSTRUCTIONS
        + '\nThe assigned patient prose seed is:\n<patient_seed>\n' + seed
        + '\n</patient_seed>\n\n'
        'Expand only this patient. Use the prose seed as the factual basis for '
        'the complete tagged event list. Generate the wording of every event, '
        'including demographics and diagnoses. Preserve the seed\'s name, sex, '
        'diagnoses, ages at diagnosis, cancer sites, histologies, and biomarker '
        'facts, including any specified prior treatment/response, treatment '
        'toxicities, performance status, organ function, comorbidities, and final '
        'clinical state. '
        'Include each seed diagnosis as a <diagnosis> event at its assigned '
        'age, in its correct chronological position. '
        'Begin the output with exactly one <demographics> line giving name, sex, '
        'and age at the earliest event in the generated history. Keep this line '
        'first. Express that age as whole years plus additional weeks, for '
        'example "50 years and 21 weeks". '
        'The first clinical event does not have to be a diagnosis. Include '
        'clinically relevant symptoms, imaging, biopsy, or surgery before the '
        'diagnosis when appropriate, while preserving the seed diagnosis age. '
        'Place the workup establishing a diagnosis before its diagnosis entry, '
        'including events at the same age in causal order. Report findings '
        'only after the examination, procedure, or test that establishes them; '
        'for example, do not report surgical margins, lymph-node counts, or '
        'pathologic staging before the relevant surgery and pathology. '
        'The seed\'s final clinical state is the patient\'s state at the last '
        'event; end the history there. Add no treatment, surgery, radiation, '
        'recurrence, progression, or other change after that state, and add no '
        'treatment the seed neither states nor clearly implies. A seed that '
        'gives no prior treatment describes an untreated patient. Earlier events '
        'may include the symptoms, workup, staging, and stated treatment needed '
        'to reach that state.\n'
        'Incorporate the seed\'s comorbidities and treatment toxicities into the '
        'history. Record each toxicity as an <adverse_event> with its grade at a '
        'plausible time after its causing treatment began, and reflect its '
        'management, such as a dose reduction, treatment hold, corticosteroids, '
        'or discontinuation, in later events. Document comorbidities and ongoing '
        'toxicities, with their treatment and current status, in the clinical '
        'notes where an oncologist would record them, especially the first '
        'oncology consultation. Report imaging or pathology findings of these '
        'conditions only when that study would show them, such as emphysema or '
        'pneumonitis on a chest CT. Use <diagnosis> only for cancers, not for '
        'comorbidities. Supportive treatment of a stated comorbidity or toxicity, '
        'such as levothyroxine for hypothyroidism, is not new cancer treatment. '
        'Add no clinically significant comorbidity or toxicity beyond the seed; '
        'minor expected effects such as grade 1 nausea or fatigue are acceptable.\n'
        'Start each event line with its event tag. Immediately after the tag, '
        'every non-demographic event must begin with "At age N years and W weeks," '
        'using nonnegative integer years and integer weeks from 0 through 51. '
        'Include both units even when weeks is zero. Use weeks to distinguish '
        'events within the same year; do not round every event to a whole year. '
        'Output events in nondecreasing age order, comparing both years and weeks. Include '
        'all diagnoses and other events in one complete history, usually about 20–30 '
        'events; use fewer rather than extending past the final clinical state. Do not output <new_patient>, another patient, headings, closing '
        'tags, or an explanation.'
    )
    return [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': content}]
