"""Retry failed serial summaries from the original notes and last valid prior."""
from dataclasses import replace
import json
from pathlib import Path
import uuid

from matchminer_ai.llm.backends import LLMGenerationResult
from .summary_quality import check_patient_summary, summary_retry_codes


class PatientSummaryQualityError(ValueError):
    def __init__(self, rows, attempts):
        super().__init__(f"Patient summary quality retries exhausted for input rows {rows}")
        self.attempts = attempts


def generate_checked_summaries(backend, prompts, *, llm_config, metadata_cache_dir,
                               max_attempts=3, trace_dir=None):
    if max_attempts < 1:
        raise ValueError("summary_quality_max_attempts must be positive")
    pending = list(range(len(prompts)))
    accepted, attempts = {}, {i: [] for i in pending}
    metadata = {}
    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        # Backends index within their batch, not within the original patient table.
        batch = [replace(prompts[i], row_idx=j) for j, i in enumerate(pending)]
        generation = backend.generate_llm_outputs(prompt_list=batch, llm_config=llm_config,
                                                  model_metadata_cache_dir=metadata_cache_dir)
        # Remote chat backends expose parsed content/reasoning without a raw
        # completion string. Preserve that absence; do not invent a raw trace.
        raw_outputs = generation.raw_outputs or [None] * len(pending)
        fields = [generation.final_outputs, generation.reasoning_outputs,
                  raw_outputs, generation.finish_reasons]
        if any(len(values) != len(pending) for values in fields):
            raise ValueError("Patient summarization backend returned an incomplete batch")
        metadata = generation.model_metadata
        remaining = []
        for batch_index, (i, answer, reasoning, raw, finish) in enumerate(zip(pending, *fields, strict=True)):
            codes = summary_retry_codes(answer, finish)
            trace = dict(attempt=attempt, answer=answer, reasoning=reasoning, raw_output=raw,
                         finish_reason=finish, quality=check_patient_summary(answer), retry_codes=codes,
                         messages=prompts[i].messages, prompt_text=prompts[i].prompt_text,
                         model_metadata={k: v for k, v in metadata.items() if k != 'responses'})
            responses = metadata.get('responses', [])
            if len(responses) == len(pending):
                trace['response_metadata'] = responses[batch_index]
            if trace_dir:
                folder = Path(trace_dir)
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / f"summary_{uuid.uuid4().hex}.json"
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(trace))
                temporary.replace(path)
                trace['trace_path'] = str(path)
            attempts[i].append(trace)
            if codes:
                remaining.append(i)
            else:
                accepted[i] = (answer, reasoning, raw, finish)
        pending = remaining
    if pending:
        raise PatientSummaryQualityError(pending, attempts)
    values = [accepted[i] for i in range(len(prompts))]
    return LLMGenerationResult(final_outputs=[v[0] for v in values],
        reasoning_outputs=[v[1] for v in values],
        raw_outputs=[v[2] for v in values] if all(v[2] is not None for v in values) else [],
        finish_reasons=[v[3] for v in values], model_metadata=metadata), attempts
