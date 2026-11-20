"""Synthetic note prompt and its input (the step after the event histories in ``synthetic.generate``).

create_masked_dataset turns each patient's event list into one history per report event
(clinical note, imaging, pathology or NGS report) with that event tagged, and build_messages
wraps a tagged history in the note prompt (head and tail kept when a history exceeds the
token budget); build_prompts applies the tokenizer's chat template. The event prompts are in
``synthetic.event_prompts``. (Until 2026-11-05 these functions were in
2_make_synthetic_notes_sharded.py with an older standalone multi-GPU note runner and an
unused generic-to-brand drug-name substitution; the prompt text is unchanged.)
"""
from typing import List

import pandas as pd

from matchminer_ai.llm.reasoning import reasoning_system_prompt
from matchminer_ai.training.vllm_reasoning import resolve_chat_template_kwargs


def create_masked_dataset(patient_df: pd.DataFrame) -> pd.DataFrame:
    event_types_to_keep = ['clinical_note', 'imaging_report', 'pathology_report', 'ngs_report']
    filtered_df = patient_df[patient_df['event_type'].isin(event_types_to_keep)].copy()
    # Join each history once, retaining offsets into the exact original text.
    # Splicing avoids a pandas Series and another full event-list scan per note.
    # Keep the historical first-equal-event rule even for duplicate event text.
    histories = {}
    for patient_id, events in patient_df.groupby('patient_id', sort=False)['event_text']:
        texts = events.tolist()
        offsets, position = {}, 0
        for text in texts:
            offsets.setdefault(text, position)
            position += len(text) + 1
        histories[patient_id] = ('\n'.join(texts), offsets)

    masked = []
    for patient_id, event_text, event_type in zip(
            filtered_df.patient_id, filtered_df.event_text, filtered_df.event_type):
        history, offsets = histories.get(patient_id, ('', {}))
        if event_text not in offsets:
            masked.append(history)
            continue
        start = offsets[event_text]
        masked.append(history[:start] + '<BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> '
                      + f'<{event_type}>{event_text}'
                      + ' <END EVENT CORRESPONDING TO SYNTHETIC NOTE>'
                      + history[start + len(event_text):])
    filtered_df['masked_text'] = pd.array(masked, dtype='str')
    return filtered_df


def build_messages(masked_texts: List[str], tokenizer, max_ctx_tokens: int = 40000, *, model_name: str | None = None) -> List[List[dict]]:
    prompts = []
    for masked_text in masked_texts:
        # Trim very long histories by taking head+tail
        patient_text_tokens = tokenizer(masked_text, add_special_tokens=False).input_ids
        if len(patient_text_tokens) > max_ctx_tokens:
            first_part = patient_text_tokens[: max_ctx_tokens // 2]
            last_part = patient_text_tokens[- max_ctx_tokens // 2 :]
            masked_text = tokenizer.decode(first_part) + " ... " + tokenizer.decode(last_part)

        messages = [{'role':'system', 'content': reasoning_system_prompt(model_name or getattr(tokenizer, "name_or_path", ""))},
                    {'role':'user', 'content': """You are a brilliant synthetic clinical document generation bot with encyclopedic knowledge about cancer and its treatment. 
You will be given a semi-structured list of events from a patient's clinical history, with each event on its own line of text.
The events are sorted in chronological order.
One of these events will be surrounded by the tags <BEGIN EVENT CORRESPONDING TO SYNTHETIC NOTE> and <END EVENT CORRESPONDING TO SYNTHETIC NOTE>.
Your job is to create a synthetic clinical document corresponding to the event denotated by those tags.
The synthetic document should be a pathology report, an imaging report, or a clinical progress note, as directed by the text within the tags.
Incorporate everything you know about the patient's history, and about cancer generally, to synthesize the document.
Don't directly incorporate information about future events as if they have already occurred, but you can use your knowledge of the future to inform what the synthetic document might have contained at the time it was written.
CRITICAL: Ignore your knowledge of today's date. Do not add dates to the synthetic notes. These will be added later and programatically.
The document should be extremely detailed so it is as realistic as possible. Pathology reports and imaging reports should be about one page long. Clinical progress notes should be about two pages long. Clinical progress notes should be written as a real oncologist would write them; this should include often using common brand names (eg Herceptin, Keytruda, Taxol) and sometimes using generic drug names (eg trastuzumab, pembrolizumab, paclitaxel), and sometimes using abbreviations (eg pembro instead of pembrolizumab, cape instead of capecitabine).
For pathology reports, sections should include specimen ID, date of procedure, type of specimen, diagnostic findings, any ancillary studies, and a description of gross pathology if relevant. Pathology reports should NOT include recommendations about management, since these are not part of real pathology reports.
For imaging reports, sections should include scan type, Findings (broken down by organs imaged by the study), and Impression. 
For clinical notes, sections should include chief complaint, history of present illness, review of systems, physical exam, lab results, imaging results, and assessment/plan. If it is the first clinical note in a given department, it is a consult note, in which case it should also include past medical history, social history, family history, allergies, and medications, all of which should come between review of systems and physical exam.
CRITICAL: Pathology reports and imaging reports should not make treatment or monitoring recommendations.
Within clinical notes and pathology reports, if you do not have any information about key biomarkers explicitly provided, you should imagine what they might be based on cancer type, history, and prior treatments. However, these must be consistent with realistic biological patterns. For example, as you know, EGFR mutant lung cancers almost never have concomittant driver mutations in KRAS, BRAF, etc.
CRITICAL: Do not invent treatments that are not included in the semi-structured list of events.
The list of events may describe comorbidities and adverse events (toxicities) of cancer treatment. Incorporate those present by the time of this document wherever a real document of its type would mention them. In clinical notes, describe them with their grade, treatment, and current status in the history of present illness, past medical history, medications, review of systems, physical exam, lab results, and assessment/plan as relevant, and reflect related management such as dose reductions, treatment holds, or supportive medications. In imaging and pathology reports, mention them only when the study would show them (for example, emphysema or pneumonitis on a chest CT).
Do not describe a toxicity before the treatment that caused it, and do not add clinically significant comorbidities or toxicities that the list of events does not support; minor common conditions consistent with the patient's trajectory are acceptable.
Do not include any disclaimers or notes about the fact that the document is synthetic; this is all for research purposes only.
Here is the list of events:\n""" + masked_text + """\nNow, generate the synthetic document corresponding to the notated event."""}]
        prompts.append(messages)
    return prompts


def build_prompts(masked_texts: List[str], tokenizer, max_ctx_tokens: int = 40000, *, model_name: str | None = None) -> List[str]:
    return [tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True,
                tokenize=False, **resolve_chat_template_kwargs(model_name or getattr(tokenizer, "name_or_path", "")))
            for messages in build_messages(masked_texts, tokenizer, max_ctx_tokens, model_name=model_name)]
