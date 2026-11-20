#!/usr/bin/env python3
"""Resumable event lists and notes using the ``synthetic.event_prompts`` and ``synthetic.note_prompts`` prompts.

One shared live endpoint pool serves both classes of histories and all notes.
Bounded preparation and atomic shards avoid materializing millions of prompts.
This command does not train models, summarize patients, or stop serving machines.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import resource
import uuid

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from matchminer_ai.training.cohort_format import training_rows, patient_generation_criteria, read_clinical_csv, file_stamp
import matchminer_ai.training.synthetic.event_prompts as prompts

import matchminer_ai
from matchminer_ai.llm.teacher_pool import (
    CompletionSampling, OutputValidationError, RequestSession,
    add_remote_cli_args, build_registry_from_args, make_completion_work_fn,
)
from matchminer_ai.training.vllm_reasoning import resolve_parser_name, resolve_sampling_defaults, resolve_chat_template_kwargs
from matchminer_ai.training.vllm_reasoning import load_tokenizer as load_generation_tokenizer
from matchminer_ai.training.paths import module_file

# The teacher pool moved to the inference package on 2026-11-06; it stays part of this stage's identity.
POOL_SOURCES = ('llm/teacher_pool.py', 'llm/scheduler_control.py', 'llm/teacher_runtime.py')
EVENT_TYPES = {'demographics', 'diagnosis', 'systemic', 'surgery', 'radiation',
               'adverse_event', 'clinical_note', 'imaging_report', 'pathology_report', 'ngs_report'}
REPORT_TYPES = {'clinical_note', 'imaging_report', 'pathology_report', 'ngs_report'}
PATIENTS_PER_CLASS = 5
VALIDATION_REVISION = 'dated-seed-events-v3-ambiguous-recurrence-intervals'
# A stopped run may resume under this generator only with a registered upgrade
# receipt; existing seeds and patients are kept, as with a validation upgrade.
SEED_REPLACEMENT_REVISION = 'exhausted-patient-seed-replacement-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_frame(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


async def finish_checkpoint(coroutine):
    """Do not leave a thread writing a shard after its writer lock is released."""
    task = asyncio.create_task(coroutine)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def checkpoint_paths(directory):
    # Keep each directory small on NFS while retaining flat-checkpoint readability.
    return sorted([*directory.glob('shard_*.parquet'), *directory.glob('*/shard_*.parquet')],
                  key=lambda path: path.name)


def ensure_file_descriptor_capacity(pending_requests):
    """Reserve sockets plus checkpoint/idle-connection headroom before generation."""
    if pending_requests < 1:
        raise ValueError('--pending-requests must be positive')
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    required = 2 * pending_requests + 2048
    if soft == resource.RLIM_INFINITY or soft >= required:
        return soft
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError(
            f'--pending-requests={pending_requests} needs an open-file limit of at least '
            f'{required}, but the hard limit is {hard}. Raise the hard limit or reduce '
            '--pending-requests before starting generation.')
    resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    return required


def parse_histories(answer):
    """Validate the requested five patients without rewriting clinical content."""
    histories = []
    for section in answer.split('<new_patient>'):
        # A literal inequality (e.g. "margin < 1 mm") is clinical content;
        # only a complete tag begins the next event. Unknown tags still fail below.
        events = re.findall(r'<(\w+)>(.*?)(?=<\w+>|\Z)', section, re.DOTALL)
        if not events:
            continue  # A heading or a leading/trailing delimiter is not a patient.
        if any(kind not in EVENT_TYPES or not text.strip() for kind, text in events):
            raise OutputValidationError('Unknown event tag or empty event')
        kinds = {kind for kind, _ in events}
        if 'demographics' not in kinds or not kinds.intersection(REPORT_TYPES):
            raise OutputValidationError('Each patient needs demographics and report events')
        histories.append([(kind, text.strip().replace('\n', ' ')) for kind, text in events])
    if len(histories) != 5:
        raise OutputValidationError(f'Expected five patient histories; got {len(histories)}')
    return histories


def tagged_events(text):
    """Parse flat event tags, preserving inequalities and clinical content."""
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    # Balanced XML-style event closers are harmless formatting. Remove only
    # matched, non-nested pairs; mismatched or nested markup is still rejected.
    text = re.sub(r'<(\w+)>(.*?)</\1>',
                  lambda match: match[0] if re.search(r'</?\w+>', match[2]) else f'<{match[1]}>{match[2]}',
                  text, flags=re.DOTALL)
    if re.search(r'</\w+>', text):
        raise OutputValidationError('Use flat event tags without mismatched or nested closing tags')
    if not text.startswith('<'):
        raise OutputValidationError('Return tagged events only, without headings or commentary')
    events = re.findall(r'<(\w+)>(.*?)(?=<\w+>|\Z)', text, re.DOTALL)
    if not events or any(kind not in EVENT_TYPES or not value.strip() for kind, value in events):
        raise OutputValidationError('Unknown event tag or empty event')
    return [(kind, ' '.join(value.split())) for kind, value in events]


def check_event_order(events):
    """Validate event syntax; retain order problems as reviewable warnings."""
    previous = None
    warnings = []
    age_pattern = r'(\d+) years? and (\d+) weeks?'
    for kind, text in events:
        match = (re.search(r'\b' + age_pattern + r'\b', text, re.I) if kind == 'demographics'
                 else re.match(r'At age ' + age_pattern + ',', text, re.I))
        if not match:
            requirement = ('Give the demographics age in years and weeks' if kind == 'demographics'
                           else f'Every {kind} event must begin with At age N years and W weeks,')
            raise OutputValidationError(requirement + ' (for example, 50 years and 21 weeks)')
        age = tuple(map(int, match.groups()))
        if not 0 <= age[1] <= 51:
            raise OutputValidationError('Use integer weeks from 0 through 51 after the whole years')
        if kind == 'demographics':
            previous = age
            continue
        if previous is not None and age < previous:
            warnings.append(
                f'Events are out of chronological order: age {age[0]} years and {age[1]} weeks '
                f'follows {previous[0]} years and {previous[1]} weeks')
        previous = age
        if re.search(r'\b(?:study (?:entry|enrollment)|trial (?:screening|enrollment)|(?:this|the) trial)\b', text, re.I):
            raise OutputValidationError('Remove references to trial screening, study entry, or enrollment')
    return warnings


def seed_age_anchors(seed_text, *, warnings=None):
    """Require diagnosis tags only for explicit diagnostic clauses.

    Other dated seed facts still need an event at that age. Free prose cannot
    reliably assign a clinical event type from an age alone. In particular,
    current status, progression, procedures, and treatment are not new diagnoses.
    """
    # Seeds also mention durations such as "in remission for 4 years and 48
    # weeks" or "disease-free interval 1 year and 37 weeks". Only an age cue
    # makes a time expression a diagnosis anchor; small pediatric ages remain
    # valid, and malformed ages must still fail validation.
    age_cue = r'\b(?:at\s+(?:(?:the\s+)?age\s+(?:of\s+)?)?|aged\s+|ages?\s+(?:of\s+)?)'
    matches = list(re.finditer(age_cue + r'(-?\d+(?:\.\d+)?) years?(?: and (-?\d+(?:\.\d+)?) weeks?)?',
                              seed_text, re.I))
    ages = [match.groups(default='') for match in matches]
    if not ages or any(not years.isdigit() or not weeks.isdigit() or not 0 <= int(weeks) <= 51
                       for years, weeks in ages):
        raise OutputValidationError(
            'Give each diagnosis age in nonnegative whole years and weeks from 0 through 51 '
            '(for example, 50 years and 21 weeks)')
    diagnoses, other = set(), set()
    previous_end = 0
    cue_pattern = (r'\b(?P<diagnosis>diagnosed|diagnosis|diagnoses)\b|'
                   r'\b(?:completed|finished|began|started|received|underwent|stopped|discontinued|'
                   r'administered|initiated|treated|resected|implanted|performed|transplant(?:ation)?|'
                   r'\w*ectomy|current(?:ly)?|status|progress\w*|remission|response|menopause|'
                   r'recurr\w*|relaps\w*|scan|imaging|CT|MRI)\b')
    for index, (match, (years, weeks)) in enumerate(zip(matches, ages)):
        # Do not carry a diagnosis cue across a sentence/semicolon from a prior
        # fact. Decimal numbers and codes do not end a sentence.
        before = re.split(r'[;.!?](?=\s|$)', seed_text[previous_end:match.start()])[-1]
        cues = list(re.finditer(cue_pattern, before, re.I))
        cue = cues[-1] if cues else None
        if cue is None and not before.strip(' ,:'):
            # In "At age ..., she received ...", the action follows the age.
            end = matches[index + 1].start() if index + 1 < len(matches) else len(seed_text)
            after = re.split(r'[;.!?](?=\s|$)', seed_text[match.end():end])[0]
            cue = re.search(cue_pattern, after, re.I)
        explicit_diagnosis = cue is not None and cue.group('diagnosis') is not None
        age = (int(years), int(weeks))
        explicit_age = re.search(r'\b(?:age|ages|aged)\b', match[0], re.I)
        # "Diagnosed at 61 ... recurred at 2 years" can describe an elapsed
        # interval. Do not force an invented age-two event into the history.
        # Explicit age statements and pediatric ages remain exact anchors.
        ambiguous_interval = (not explicit_age and not explicit_diagnosis and diagnoses
            and age < max(diagnoses) and cue is not None
            and re.fullmatch(r'(?:recurr\w*|relaps\w*|progress\w*)', cue[0], re.I))
        if ambiguous_interval:
            if warnings is not None:
                warnings.append(f'Ambiguous recurrence interval in seed: {years} years and {weeks} weeks; '
                                'not enforced as an absolute patient age')
        else:
            (diagnoses if explicit_diagnosis else other).add(age)
        previous_end = match.end()
    return diagnoses, other


def seed_diagnosis_ages(seed_text):
    return seed_age_anchors(seed_text)[0]


def parse_seeds(answer):
    sections = [s.strip() for s in answer.split('<new_patient>') if s.strip()]
    # A single outer markdown fence is harmless; seed content remains prose.
    if answer.strip().startswith('```') and answer.strip().endswith('```'):
        return parse_seeds(answer.strip().split('\n', 1)[1].rsplit('```', 1)[0])
    if len(sections) != PATIENTS_PER_CLASS:
        raise OutputValidationError(f'Expected five patient seeds; got {len(sections)}')
    seeds = []
    seen = set()
    for section in sections:
        if re.search(r'</?\w+>', section):
            raise OutputValidationError('Each seed must be short prose without event tags')
        seed = ' '.join(section.split())
        if seed.casefold() in seen:
            raise OutputValidationError('The five patient seeds must be distinct')
        seen.add(seed.casefold())
        seed_diagnosis_ages(seed)
        seeds.append(seed)
    return seeds


def parse_patient_history(answer, seed_text, *, warnings=None):
    if '<new_patient>' in answer:
        raise OutputValidationError('Generate exactly one patient, without a new_patient separator')
    events = tagged_events(answer)
    if events[0][0] != 'demographics' or sum(kind == 'demographics' for kind, _ in events) != 1:
        raise OutputValidationError('Generate one patient with one initial demographics line')
    if not any(kind == 'diagnosis' for kind, _ in events):
        raise OutputValidationError('Include diagnosis events for the supplied prose seed')
    if not REPORT_TYPES.intersection(kind for kind, _ in events):
        raise OutputValidationError('The patient history needs report events for note generation')
    order_warnings = check_event_order(events)
    if warnings is not None:
        warnings.extend(order_warnings)
    diagnosis_ages = {tuple(map(int, re.match(r'At age (\d+) years? and (\d+) weeks?,', text, re.I).groups()))
                     for kind, text in events if kind == 'diagnosis'}
    required_diagnoses, required_events = seed_age_anchors(seed_text, warnings=warnings)
    missing = required_diagnoses - diagnosis_ages
    if missing:
        ages = ', '.join(f'{years} years and {weeks} weeks' for years, weeks in sorted(missing))
        raise OutputValidationError(f'Include the seed diagnoses at their assigned ages: {ages}')
    event_ages = {tuple(map(int, re.match(r'At age (\d+) years? and (\d+) weeks?,', text, re.I).groups()))
                 for kind, text in events if kind != 'demographics'}
    missing = required_events - event_ages
    if missing:
        ages = ', '.join(f'{years} years and {weeks} weeks' for years, weeks in sorted(missing))
        raise OutputValidationError(f'Include the other dated seed events at their assigned ages: {ages}')
    # Free prose has no reliable exact-text identity or clinical-fact comparison.
    # Those facts are preserved by the prompt and assessed in output review.
    return events


def validate_completion(answer, metadata, stage, seed_text=None):
    warnings = []
    if metadata.get('finish_reason') != 'stop':
        raise OutputValidationError(f"Incomplete completion: {metadata.get('finish_reason')}")
    if not answer.strip() or answer.lstrip().startswith('ERROR:'):
        raise OutputValidationError('Empty or error output')
    if stage == 'events':
        parse_histories(answer)
    elif stage == 'seeds':
        parse_seeds(answer)
    elif stage == 'patient_events':
        parse_patient_history(answer, seed_text, warnings=warnings)
    return warnings


def recover_saved_history(candidates, seed_text, model):
    """Revalidate original LLM attempts; never synthesize or retag clinical facts."""
    acceptable = []
    for path, trace in candidates:
        try:
            warnings = validate_completion(trace['answer'], trace, 'patient_events', seed_text)
        except OutputValidationError:
            continue
        events = parse_patient_history(trace['answer'], seed_text)
        record = dict(reasoning=trace['reasoning'],
                      answer='\n'.join(f'<{kind}> {text}' for kind, text in events),
                      raw_response=(json.dumps(trace['response_json']) if 'response_json' in trace
                                    else trace['raw_text']),
                      transport=trace.get('transport', 'completions'), finish_reason=trace['finish_reason'],
                      model=model, endpoint=trace['endpoint'], generated_at=trace['at'],
                      prompt_sha256=trace['prompt_sha256'], validation_revision=VALIDATION_REVISION,
                      validation_warnings=json.dumps(warnings), recovered_from=str(path))
        acceptable.append((len(warnings), trace['at'], record))
    # Prefer an attempt without chronology warnings, then the earliest one:
    # later attempts may have received misleading feedback from the old validator.
    return min(acceptable, key=lambda row: row[:2])[2] if acceptable else None


def compatible_validation_upgrade(previous, current, receipt):
    """Allow only an explicitly registered validator change, retaining patient IDs."""
    expected = {**previous, 'source_sha256': {
        **previous['source_sha256'],
        'synthetic.generate': current['source_sha256']['synthetic.generate']}}
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return (current == expected and receipt.get('kind') in (VALIDATION_REVISION, SEED_REPLACEMENT_REVISION)
            and receipt.get('original_manifest_sha256') == digest(previous)
            and receipt.get('approved_manifest_sha256') == digest(current))


def event_rows(records, run_identity):
    """Stable global patient IDs shared by event lists, notes, and resume."""
    rows = []
    for record in records:
        space = int(record['space_index'])
        kind_offset = 0 if record['patient_class'] == 'positive' else 5
        for patient_number, events in enumerate(parse_histories(record['answer'])):
            mrn = 10 * space + kind_offset + patient_number + 1
            patient_id = 'patient_' + str(uuid.uuid5(uuid.NAMESPACE_URL, f'{run_identity}/{mrn}'))
            for event_number, (event_type, event_text) in enumerate(events):
                rows.append(dict(pseudo_mrn=mrn, patient_id=patient_id, space_index=space,
                                 nct_id=record['nct_id'], patient_class=record['patient_class'],
                                 event_number=event_number, event_type=event_type, event_text=event_text))
    return pd.DataFrame(rows)


def compose_parquet(paths, destination, expected_rows=None):
    """Stream assembly with an explicit row-count check; never skip unreadable files."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    count, writer = 0, None
    try:
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                table = pa.Table.from_batches([batch])
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
                writer.write_table(table)
                count += len(table)
        if writer is None:
            raise ValueError('No rows to assemble')
    finally:
        if writer is not None:
            writer.close()
    if expected_rows is not None and count != expected_rows:
        raise ValueError(f'Assembly has {count} rows, expected {expected_rows}')
    temporary.replace(destination)
    return count


def note_preparation_workers(requested=0):
    """Bound CPU/RAM use by default; zero selects up to eight available CPUs."""
    if requested < 0:
        raise ValueError('--note-preparation-workers must be nonnegative')
    return requested or min(8, os.process_cpu_count() or 1)


def initialize_note_preparation_worker():
    # Each process already owns one CPU task. Avoid multiplying Arrow's thread
    # pools by the worker count, and never inherit the parent's tokenizer/sockets.
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


@lru_cache(maxsize=1)
def note_prompts():
    """synthetic.note_prompts, loaded on first use (it imports the package's prompt rendering)."""
    return importlib.import_module('matchminer_ai.training.synthetic.note_prompts')


def prepare_note_chunk(job):
    """Write one independent chunk with IDs assigned in input order by the parent."""
    patients = event_rows(job['records'], job['identity'])
    promptframe = note_prompts().create_masked_dataset(patients).reset_index(drop=True)
    chunk = job['chunk']
    if len(promptframe) != chunk['notes']:
        raise ValueError('Prepared note count differs from event-tag count')
    promptframe['row_id'] = range(chunk['first_row_id'], chunk['last_row_id'] + 1)
    promptframe['key'] = promptframe.row_id.map(lambda value: f'note:{value}')
    save_frame(chunk['events'], patients)
    save_frame(chunk['path'], promptframe)
    return chunk


class Generator:
    def __init__(self, args, tokenizer):
        self.args, self.tokenizer = args, tokenizer
        self.out = Path(args.output_dir)
        self.notes = note_prompts()
        self.sampling = CompletionSampling(
            model=args.model, request_timeout=args.request_timeout,
            **resolve_sampling_defaults(args.tokenizer_model or args.model,
                fallback=dict(temperature=1.0, top_p=.95, top_k=64, repetition_penalty=1.0)),
        )
        self.status = {'started_at': now(), 'stage': 'preparing', 'automatic_vm_shutdown': False}
        self.write_status()

    def write_status(self, **values):
        self.status.update(updated_at=now(), **values)
        save_json(self.out / 'status.json', self.status)

    def manifest(self):
        sources = ['synthetic.generate', 'synthetic.note_prompts', 'synthetic.event_prompts', 'vllm_reasoning',
                   'cohort_format']
        value = dict(version='synthetic-cohort-seeds-patient-events-notes-v5', input_stamp=file_stamp(self.args.input_csv),
                     source_sha256={**{name: sha_file(module_file(name)) for name in sources},
                                    **{'matchminer_ai/' + name: sha_file(Path(matchminer_ai.__file__).parent / name)
                                       for name in POOL_SOURCES}},
                     sampling=asdict(self.sampling), tokenizer=self.args.tokenizer_model or self.args.model,
                     chat_template_kwargs=resolve_chat_template_kwargs(self.args.tokenizer_model or self.args.model),
                     transport='completions',
                     chat_template_sha256=hashlib.sha256(str(self.tokenizer.chat_template).encode()).hexdigest(),
                     seed=self.args.seed, enable_thinking=True, patients_per_space_per_class=5,
                     seed_max_tokens=self.args.seed_max_tokens,
                     event_max_tokens=self.args.event_max_tokens, note_max_tokens=self.args.note_max_tokens,
                     context_limit=self.args.max_model_len, spaces_per_note_chunk=self.args.spaces_per_note_chunk)
        path = self.out / 'manifest.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous != value:
                receipt_path = self.out / 'event_validation_upgrade.json'
                receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                if not compatible_validation_upgrade(previous, value, receipt):
                    raise ValueError('Generation manifest changed; use a fresh output directory or register the validation-only upgrade')
            # The original run identity also defines patient IDs. Preserve it
            # across the recorded validation-only upgrade.
            value = previous
        else:
            save_json(path, value)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def build_payload(self, item, stage):
        if stage == 'seeds':
            messages = prompts.seed_messages(patient_generation_criteria(item), item['patient_class'],
                                             random.Random(self.args.seed + int(item['space_index'])))
            max_tokens = self.args.seed_max_tokens
        elif stage == 'patient_events':
            messages = prompts.patient_messages(item['seed_text'])
            max_tokens = self.args.event_max_tokens
        else:
            # Preserve the exact note prompt. The larger limit prevents its
            # legacy head/tail truncation; reject an overlong context below.
            messages = self.notes.build_messages([item['masked_text']], self.tokenizer,
                                             max_ctx_tokens=self.args.max_model_len,
                                             model_name=self.args.tokenizer_model or self.args.model)[0]
            max_tokens = self.args.note_max_tokens
        template_kwargs = resolve_chat_template_kwargs(self.args.tokenizer_model or self.args.model)
        prompt = self.tokenizer.apply_chat_template(
            conversation=messages, add_generation_prompt=True, tokenize=False, **template_kwargs)
        length = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        if length + max_tokens + 256 > self.args.max_model_len:
            raise ValueError(f'Full {stage} prompt and output budget exceed context for {item["key"]}')
        return dict(prompt=prompt, max_tokens=max_tokens, enable_thinking=True,
                    include_completion_metadata=True, include_raw_text=True, stage=stage,
                    key=item['key'], source_messages=messages, messages=messages,
                    chat_template_kwargs=template_kwargs, seed_text=item.get('seed_text'))

    def completion_work(self, directory):
        parser = resolve_parser_name(self.args.tokenizer_model or self.args.model, self.args.reasoning_parser)
        base_work = make_completion_work_fn(self.sampling, parser, self.tokenizer)

        async def work(client, payload):
            reasoning, answer, metadata = await base_work(client, payload)
            try:
                warnings = await asyncio.to_thread(validate_completion, answer, metadata,
                                                   payload['stage'], payload.get('seed_text'))
            except OutputValidationError as exc:
                failure = dict(at=now(), stage=payload['stage'], key=payload['key'],
                               endpoint=str(client.base_url), error=str(exc),
                               prompt_sha256=hashlib.sha256(payload['prompt'].encode()).hexdigest(),
                               reasoning=reasoning, answer=answer, **metadata)
                await asyncio.to_thread(save_json, directory / ('invalid_' + str(uuid.uuid4()) + '.json'), failure)
                if payload.get('source_messages'):
                    # Pool retries the same payload. Supply the precise validation
                    # failure without accumulating prior patients or invalid drafts.
                    messages = payload['source_messages'] + [{'role': 'user', 'content':
                        'The previous response failed validation: ' + str(exc) + '\n'
                        'Regenerate the complete requested output from the original instructions. '
                        'Preserve the facts and diagnosis ages in the assigned prose seed if one was supplied.'}]
                    payload['prompt'] = await asyncio.to_thread(
                        self.tokenizer.apply_chat_template, conversation=messages,
                        add_generation_prompt=True, tokenize=False, **resolve_chat_template_kwargs(self.args.tokenizer_model or self.args.model))
                    payload['messages'] = messages
                raise
            if payload['stage'] == 'seeds':
                answer = '\n<new_patient>\n'.join(parse_seeds(answer))
            elif payload['stage'] == 'patient_events':
                answer = '\n'.join(f'<{kind}> {text}' for kind, text in
                                   parse_patient_history(answer, payload['seed_text']))
            raw_response = (json.dumps(metadata['response_json']) if 'response_json' in metadata
                            else metadata['raw_text'])
            return dict(reasoning=reasoning, answer=answer, raw_response=raw_response,
                        transport=metadata.get('transport', 'completions'),
                        finish_reason=metadata['finish_reason'], model=self.args.model,
                        endpoint=str(client.base_url), generated_at=now(),
                        validation_revision=VALIDATION_REVISION,
                        validation_warnings=json.dumps(warnings),
                        prompt_sha256=hashlib.sha256(payload['prompt'].encode()).hexdigest())
        return work

    async def generate(self, items, stage, directory):
        directory.mkdir(parents=True, exist_ok=True)
        done = set()
        files = checkpoint_paths(directory)
        for path in files:
            keys = pq.read_table(path, columns=['key']).column('key').to_pylist()
            if done.intersection(keys) or len(set(keys)) != len(keys):
                raise ValueError('Duplicate generation checkpoint keys')
            done.update(keys)
        next_shard = max((int(path.stem.split('_')[-1]) for path in files), default=-1) + 1
        work = self.completion_work(directory)

        buffer, lock = [], asyncio.Lock()
        completed, expected = len(done), 0

        async def commit_buffer():
            nonlocal next_shard
            if buffer:
                frame = pd.DataFrame(buffer)
                await asyncio.to_thread(save_frame, directory / f'{next_shard // 1024:04d}' / f'shard_{next_shard:08d}.parquet', frame)
                next_shard += 1
                buffer.clear()
                self.write_status(stage=stage, stage_completed=completed, stage_expected=expected)

        async def flush():
            await finish_checkpoint(commit_buffer())

        registry = build_registry_from_args(self.args)
        iterator = iter(items)
        async with RequestSession(registry, work, max_attempts=self.args.max_attempts) as session:
            async def worker():
                nonlocal completed, expected
                while True:
                    # No await while advancing the shared iterator. Preparation
                    # is bounded by --pending-requests, not the full note count.
                    item = next(iterator, None)
                    if item is None:
                        return
                    expected += 1
                    if item['key'] in done:
                        continue
                    payload = await asyncio.to_thread(self.build_payload, item, stage)
                    result = await session.submit(payload)
                    if not isinstance(result, dict):
                        raise RuntimeError(f"Generation failed for {item['key']}: {result}")
                    async with lock:
                        buffer.append({**item, **result})
                        completed += 1
                        if len(buffer) >= self.args.results_per_shard:
                            await flush()

            tasks = [asyncio.create_task(worker()) for _ in range(self.args.pending_requests)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await flush()
        if completed != expected:
            raise ValueError(f'Checkpoint coverage mismatch: {completed} != {expected}')
        return expected

    def seed_path(self, item):
        return self.out / 'patient_seeds' / f'{int(item["space_index"]) // 256:04d}' / (
            f'{item["patient_class"]}_{int(item["space_index"]):08d}.json')

    async def generate_patient_events(self, spaces):
        """Persist five seeds, then immediately expand their patients in one pool.

        Groups overlap: an unfinished seed batch never holds up another group's
        expansions. The pending-call semaphore also bounds prompt preparation.
        """
        directory = self.out / 'patient_events'
        directory.mkdir(parents=True, exist_ok=True)
        files = checkpoint_paths(directory)
        done = {}
        for path in files:
            for row in pq.read_table(path, columns=['key', 'seed_sha256']).to_pylist():
                if row['key'] in done:
                    raise ValueError('Duplicate patient generation checkpoint key')
                done[row['key']] = row['seed_sha256']
        expected_keys = {f'{kind}:{int(index)}:{patient}' for index in spaces.space_index
                         for kind in ('positive', 'negative') for patient in range(PATIENTS_PER_CLASS)}
        if set(done) - expected_keys:
            raise ValueError('Patient checkpoint contains keys outside the frozen input cohort')
        next_shard = max((int(path.stem.split('_')[-1]) for path in files), default=-1) + 1
        completed, seeds_completed, recovered = len(done), 0, 0
        failure_directory = self.out / 'generation_failures'
        failure_directory.mkdir(exist_ok=True)
        failure_records = {record['key']: record for path in failure_directory.glob('*.json')
                           if (record := json.loads(path.read_text()))['key'] not in done}
        failed = {key: record['error'] for key, record in failure_records.items()}
        recovery_directory = self.out / 'event_recoveries'
        # New seeds for patients whose histories failed validation on every
        # attempt; the last is the patient's seed. Seed batches are not rewritten.
        replacement_directory = self.out / 'patient_seed_replacements'
        replacements = {record['key']: record['replacements'] for path in replacement_directory.glob('*.json')
                        if (record := json.loads(path.read_text()))}

        def cached_failures():
            candidates = {}
            if failed:
                for path in directory.glob('invalid_*.json'):
                    trace = json.loads(path.read_text())
                    if trace.get('stage') == 'patient_events' and trace['key'] in failed:
                        candidates.setdefault(trace['key'], []).append((path, trace))
            return candidates

        cached = await asyncio.to_thread(cached_failures)
        buffer, lock = [], asyncio.Lock()
        pending = asyncio.Semaphore(self.args.pending_requests)
        iterator = iter((dict(row, patient_class=kind, key=f'{kind}:{int(row["space_index"])}')
                         for row in spaces.to_dict('records') for kind in ('positive', 'negative')))

        def progress():
            self.write_status(stage='events', event_phase='seeds_and_individual_histories',
                              seeds_completed=seeds_completed, expected_seed_batches=2 * len(spaces),
                              stage_completed=completed - len(buffer), stage_expected=len(expected_keys),
                              completed_patient_histories=completed,
                              recovered_patient_histories=recovered,
                              buffered_patient_histories=len(buffer), unresolved_items=len(failed),
                              replaced_patient_seeds=sum(map(len, replacements.values())))

        def failure_path(key):
            return failure_directory / (key.replace(':', '_') + '.json')

        async def record_failure(item, exc):
            failed[item['key']] = str(exc)
            await asyncio.to_thread(save_json, failure_path(item['key']),
                dict(key=item['key'], error=str(exc), at=now(),
                     seed_text=item.get('seed_text'), nct_id=item['nct_id']))
            print(f'[generation] isolated failed item {item["key"]}: {exc}', flush=True)
            progress()

        async def commit_buffer():
            nonlocal next_shard
            if buffer:
                await asyncio.to_thread(save_frame,
                    directory / f'{next_shard // 1024:04d}' / f'shard_{next_shard:08d}.parquet',
                    pd.DataFrame(buffer))
                next_shard += 1
                buffer.clear()
            progress()

        async def flush():
            await finish_checkpoint(commit_buffer())

        registry = build_registry_from_args(self.args)
        async with RequestSession(registry, self.completion_work(directory),
                                  max_attempts=self.args.max_attempts) as session:
            async def request(item, stage):
                async with pending:
                    payload = await asyncio.to_thread(self.build_payload, item, stage)
                    result = await session.submit(payload)
                if not isinstance(result, dict):
                    raise RuntimeError(f'Generation failed for {item["key"]}: {result}')
                return result

            def current_seed(item, seeds, patient):
                replaced = replacements.get(f'{item["key"]}:{patient}')
                return replaced[-1]['seed_text'] if replaced else seeds[patient]

            async def replace_seed(item, seeds, patient, seed, error):
                """Draw a new seed batch with the unchanged seed prompt and keep its first seed, from
                the patient's own position on, that no patient of this cohort has or this one had."""
                key = f'{item["key"]}:{patient}'
                history = replacements.setdefault(key, [])
                batch = await request(item, 'seeds')
                used = {s.casefold() for s in [*(current_seed(item, seeds, i) for i in range(len(seeds))),
                                               seeds[patient], *(r['seed_text'] for r in history)]}
                drawn = parse_seeds(batch['answer'])
                new = next((s for s in drawn[patient:] + drawn[:patient] if s.casefold() not in used), None)
                if new is None:
                    raise RuntimeError(f'Seed replacement for {key} only repeated existing seeds')
                history.append(dict(number=len(history) + 1, at=now(), seed_text=new,
                                    seed_sha256=hashlib.sha256(new.encode()).hexdigest(),
                                    replaced_seed_sha256=hashlib.sha256(seed.encode()).hexdigest(),
                                    replaced_because=str(error), batch_answer=batch['answer'],
                                    **{k: batch[k] for k in ('reasoning', 'model', 'endpoint', 'generated_at',
                                                             'prompt_sha256', 'validation_revision')}))
                replacement_directory.mkdir(exist_ok=True)
                await asyncio.to_thread(save_json, replacement_directory / (key.replace(':', '_') + '.json'),
                                        dict(key=key, replacements=history))
                print(f'[generation] replaced seed {len(history)} for {key} after: {error}', flush=True)
                progress()
                return new

            async def expand(item, seeds, patient):
                nonlocal completed, recovered
                key = f'{item["key"]}:{patient}'
                seed = current_seed(item, seeds, patient)
                seed_hash = hashlib.sha256(seed.encode()).hexdigest()
                if key in done:
                    if done[key] != seed_hash:
                        raise ValueError(f'Saved patient seed changed: {key}')
                    failure_path(key).unlink(missing_ok=True)
                    return
                child = dict(item, key=key, seed_key=item['key'], patient_number=patient,
                             seed_text=seed, seed_sha256=seed_hash)
                # Saved attempts made for an earlier seed of this patient are not candidates.
                replaced_at = replacements[key][-1]['at'] if replacements.get(key) else ''
                candidates = [(path, trace) for path, trace in cached.pop(key, []) if trace['at'] > replaced_at]
                if candidates and failure_records[key].get('seed_text') != seed:
                    raise ValueError(f'Saved failed-patient seed changed: {key}')
                result = (await asyncio.to_thread(recover_saved_history, candidates, seed, self.args.model)
                          if candidates else None)
                if result is not None:
                    recovery_directory.mkdir(exist_ok=True)
                    await asyncio.to_thread(save_json, recovery_directory / (key.replace(':', '_') + '.json'),
                        dict(key=key, seed_sha256=seed_hash, recovered_at=now(),
                             source=result['recovered_from'], source_sha256=sha_file(result['recovered_from']),
                             answer_sha256=hashlib.sha256(result['answer'].encode()).hexdigest(),
                             original_error=failed.get(key), validation_revision=VALIDATION_REVISION,
                             validation_warnings=json.loads(result['validation_warnings'])))
                    recovered += 1
                    print(f'[generation] recovered saved attempt for {key}', flush=True)
                else:
                    # A seed that already failed validation on every attempt is replaced
                    # before more attempts; other failures (e.g. transport) are retried.
                    previous = failure_records.get(key, {})
                    error = previous['error'] if (previous.get('seed_text') == seed
                                                  and 'OutputValidationError' in previous.get('error', '')) else None
                    while True:
                        if error is None:
                            try:
                                result = await request(child, 'patient_events')
                                break
                            except RuntimeError as exc:
                                error = str(exc)
                        if ('OutputValidationError' not in error
                                or len(replacements.get(key, [])) >= self.args.max_seed_replacements):
                            await record_failure(child, error)
                            return
                        try:
                            seed = await replace_seed(item, seeds, patient, seed, error)
                        except RuntimeError as exc:
                            await record_failure(child, exc)
                            return
                        child = dict(child, seed_text=seed, seed_sha256=hashlib.sha256(seed.encode()).hexdigest())
                        error = None
                failed.pop(key, None)
                failure_path(key).unlink(missing_ok=True)
                # Avoid copying the full trial text into every patient checkpoint.
                record = {key: child[key] for key in ('key', 'seed_key', 'patient_number', 'seed_text',
                          'seed_sha256', 'space_index', 'patient_class', 'nct_id')}
                record.update({key: child[key] for key in ('cohort_trial_id', 'trial_representation') if key in child})
                record['seed_replacement'] = len(replacements.get(key, []))
                async with lock:
                    buffer.append({**record, **result})
                    completed += 1
                    if len(buffer) >= self.args.results_per_shard:
                        await flush()

            async def worker():
                nonlocal seeds_completed
                while (item := next(iterator, None)) is not None:
                    path = self.seed_path(item)
                    if path.exists():
                        saved = await asyncio.to_thread(lambda: json.loads(path.read_text()))
                        identity_keys = ['key', 'space_index', 'patient_class', 'nct_id']
                        identity_keys += [k for k in ('cohort_trial_id', 'trial_representation') if k in item]
                        if any(saved.get(k) != item[k] for k in identity_keys):
                            raise ValueError(f'Seed checkpoint identity mismatch: {path}')
                    else:
                        if any(f'{item["key"]}:{i}' in done for i in range(PATIENTS_PER_CLASS)):
                            raise ValueError(f'Missing parent seed checkpoint: {path}')
                        try:
                            result = await request(item, 'seeds')
                        except RuntimeError as exc:
                            await record_failure(item, exc)
                            continue
                        saved = {**{k: item[k] for k in ('key', 'space_index', 'patient_class', 'nct_id')}, **result}
                        saved.update({k: item[k] for k in ('cohort_trial_id', 'trial_representation') if k in item})
                        path.parent.mkdir(parents=True, exist_ok=True)
                        await asyncio.to_thread(save_json, path, saved)
                    failure_path(item['key']).unlink(missing_ok=True)
                    failed.pop(item['key'], None)
                    seeds = parse_seeds(saved['answer'])
                    seeds_completed += 1
                    if seeds_completed % 32 == 0:
                        progress()
                    # TaskGroup cancels siblings on interruption/failure before
                    # the shared pool closes; completed patients are flushed below.
                    async with asyncio.TaskGroup() as group:
                        for patient in range(len(seeds)):
                            group.create_task(expand(item, seeds, patient))

            tasks = [asyncio.create_task(worker()) for _ in range(self.args.pending_requests)]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await flush()
        if failed:
            raise RuntimeError(f'{len(failed)} generation items remain unresolved; other items were completed and saved. See generation_failures/')
        if completed != len(expected_keys) or seeds_completed != 2 * len(spaces):
            raise ValueError('Seed/patient checkpoint coverage mismatch')

    def assembled_event_records(self, spaces):
        """Join five independently saved histories without mixing patient order.

        Every shard is read once. A cohort's patients finish at different times
        and land in different shards, so a small shard cache re-read about one
        shard per patient (hundreds of GB for a 77,000-patient run).
        """
        saved = {}
        for path in checkpoint_paths(self.out / 'patient_events'):
            for record in pd.read_parquet(path).to_dict('records'):
                if record['key'] in saved:
                    raise ValueError('Duplicate patient checkpoint key')
                saved[record['key']] = record

        for row in spaces.to_dict('records'):
            for kind in ('positive', 'negative'):
                key = f'{kind}:{int(row["space_index"])}'
                # Released once joined, so memory shrinks as the joined records grow.
                patients = [saved.pop(f'{key}:{i}') for i in range(PATIENTS_PER_CLASS)]
                provenance = [{**{k: patient[k] for k in ('key', 'patient_number', 'endpoint', 'model',
                              'prompt_sha256', 'seed_sha256', 'generated_at')},
                              **{k: patient[k] for k in ('validation_revision', 'validation_warnings', 'recovered_from')
                                 if k in patient},
                              # Shards from before seed replacement have no such column.
                              **({'seed_replacement': int(patient['seed_replacement'])}
                                 if patient.get('seed_replacement', 0) > 0 else {})} for patient in patients]
                yield dict(key=key, space_index=row['space_index'], patient_class=kind,
                           answer='\n<new_patient>\n'.join(p['answer'] for p in patients),
                           raw_response='\n<new_patient>\n'.join(p['raw_response'] for p in patients),
                           reasoning='\n<new_patient>\n'.join(p['reasoning'] for p in patients),
                           model=self.args.model, endpoint=json.dumps([p['endpoint'] for p in patients]),
                           generated_at=max(p['generated_at'] for p in patients),
                           prompt_sha256=hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest(),
                           patient_provenance_json=json.dumps(provenance),
                           seed_checkpoint=str(self.seed_path(dict(row, patient_class=kind))))

    def export_events(self, spaces, directory):
        frame = pd.DataFrame(self.assembled_event_records(spaces))
        expected = {f'{kind}:{int(index)}' for kind in ('positive', 'negative') for index in spaces.space_index}
        if len(frame) != len(expected) or set(frame.key) != expected:
            raise ValueError('Event coverage does not match every sampled space and both classes')
        for kind in ('positive', 'negative'):
            subset = frame.loc[frame.patient_class.eq(kind)].sort_values('space_index')
            result = spaces.merge(subset[['space_index', 'answer', 'raw_response', 'reasoning',
                                         'model', 'endpoint', 'generated_at', 'prompt_sha256',
                                         'patient_provenance_json', 'seed_checkpoint']],
                                  on='space_index', validate='one_to_one')
            result['criteria_index'] = result.space_index
            result = result.rename(columns={'answer': 'synthetic_patient_prompts',
                                             'raw_response': 'synthetic_patient_prompt_generation_reasoning'})
            path = self.out / f'trial_spaces_with_{kind}_prompts.csv'
            temporary = path.with_suffix('.csv.tmp')
            result.to_csv(temporary, index=False)
            temporary.replace(path)
        frame = spaces[['space_index', 'nct_id']].merge(frame, on='space_index', validate='one_to_many')
        return frame.sort_values(['space_index', 'patient_class']).reset_index(drop=True)

    def prepare_notes(self, spaces, events, identity):
        directory = self.out / 'note_inputs'
        directory.mkdir(exist_ok=True)
        chunk_size = self.args.spaces_per_note_chunk
        if chunk_size < 1:
            raise ValueError('--spaces-per-note-chunk must be positive')
        total = (len(spaces) + chunk_size - 1) // chunk_size
        workers = min(note_preparation_workers(getattr(self.args, 'note_preparation_workers', 0)), max(1, total))
        groups = events.groupby('space_index', sort=False).indices
        completed = {}

        def progress():
            self.write_status(stage='preparing_notes', note_preparation_workers=workers,
                              note_preparation_completed_chunks=len(completed),
                              note_preparation_expected_chunks=total,
                              prepared_note_inputs=sum(chunk['notes'] for chunk in completed.values()))

        def jobs():
            row_offset = 0
            for index, start in enumerate(range(0, len(spaces), chunk_size)):
                selected = spaces.iloc[start:start + chunk_size].space_index
                # Select only the columns needed by event_rows; do not send raw
                # model responses/reasoning to each child or scan the full frame.
                positions = sorted(position for space in selected for position in groups[space])
                rows = events.iloc[positions][['space_index', 'patient_class', 'nct_id', 'answer']].to_dict('records')
                # Counting complete report tags is cheap; parsing/validation and
                # construction happen in the worker. Prefix counts make row IDs
                # independent of process completion order, retries, and resume.
                nrows = sum(row['answer'].count(f'<{kind}>') for row in rows for kind in REPORT_TYPES)
                chunk = dict(path=str(directory / f'promptframe_{index:06d}.parquet'),
                             events=str(directory / f'events_{index:06d}.parquet'), notes=nrows,
                             first_row_id=row_offset, last_row_id=row_offset + nrows - 1)
                row_offset += nrows
                if Path(chunk['path']).exists() and Path(chunk['events']).exists():
                    if pq.ParquetFile(chunk['path']).metadata.num_rows != nrows:
                        raise ValueError(f'Saved note input count changed: {chunk["path"]}')
                    completed[index] = chunk
                    progress()
                    continue
                yield index, dict(records=rows, identity=identity, chunk=chunk)

        def collect(pending, futures):
            for future in futures:
                completed[pending.pop(future)] = future.result()
                progress()

        progress()
        if workers == 1:
            for index, job in jobs():
                completed[index] = prepare_note_chunk(job)
                progress()
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize_note_preparation_worker) as pool:
                pending = {}
                try:
                    for index, job in jobs():
                        if len(pending) >= workers:
                            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                            collect(pending, finished)
                        pending[pool.submit(prepare_note_chunk, job)] = index
                    while pending:
                        finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                        collect(pending, finished)
                finally:
                    # On failure, finish active atomic writes and cancel queued
                    # chunks before releasing the generation lock.
                    for future in pending:
                        future.cancel()
        records = [completed[index] for index in range(total)]
        save_json(self.out / 'note_input_manifest.json', dict(chunks=records, notes=sum(c['notes'] for c in records)))
        return records

    async def run(self):
        identity = self.manifest()
        spaces = training_rows(read_clinical_csv(self.args.input_csv))
        if spaces.space_index.tolist() != list(range(len(spaces))):
            raise ValueError('Expected contiguous 0c space_index values')
        spaces['trial_boilerplate_text'] = spaces.trial_boilerplate_text.fillna('')
        self.write_status(stage='events', sampled_trials=spaces.nct_id.nunique(), sampled_spaces=len(spaces),
                          expected_seed_batches=2 * len(spaces), expected_patients=10 * len(spaces))
        await self.generate_patient_events(spaces)
        events = await asyncio.to_thread(self.export_events, spaces, self.out / 'patient_events')
        if self.args.stop_after_events:
            self.write_status(stage='events_completed', all_outputs_local=True)
            return
        self.write_status(stage='preparing_notes')
        chunks = await asyncio.to_thread(self.prepare_notes, spaces, events, identity)
        del events
        events_path = self.out / 'all_synthetic_events.parquet'
        event_count = await asyncio.to_thread(compose_parquet, [chunk['events'] for chunk in chunks], events_path)
        note_count = sum(chunk['notes'] for chunk in chunks)
        self.write_status(stage='notes', expected_notes=note_count)
        # All chunks share one persistent pool. Reading at most 128 rows at a
        # time bounds memory even for millions of notes.
        def note_items():
            for chunk in chunks:
                for batch in pq.ParquetFile(chunk['path']).iter_batches(batch_size=128):
                    yield from batch.to_pylist()
        await self.generate(note_items(), 'notes', self.out / 'note_shards')
        self.write_status(stage='assembling')
        # Schema names consumed by patient summarization and training.
        paths = []
        for source in checkpoint_paths(self.out / 'note_shards'):
            target_dir = self.out / 'notes_for_training'
            target_dir.mkdir(exist_ok=True)
            target = target_dir / source.relative_to(self.out / 'note_shards')
            if not target.exists():
                frame = await asyncio.to_thread(pd.read_parquet, source)
                frame = frame.rename(columns={'answer': 'synthetic_note', 'raw_response': 'synth_note_reasoning_and_note'})
                await asyncio.to_thread(save_frame, target, frame)
            paths.append(target)
        notes_path = self.out / 'all_synthetic_notes.parquet'
        await asyncio.to_thread(compose_parquet, paths, notes_path, note_count)
        self.write_status(stage='completed', completed_at=now(), notes=note_count, events=event_count,
                          outputs=[str(notes_path), str(events_path)], all_outputs_local=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--tokenizer-model')
    parser.add_argument('--download-dir')
    parser.add_argument('--reasoning-parser', default='auto')
    parser.add_argument('--seed-max-tokens', type=int, default=16000)
    parser.add_argument('--event-max-tokens', type=int, default=20000)
    parser.add_argument('--note-max-tokens', type=int, default=15000)
    parser.add_argument('--max-model-len', type=int, default=250000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--spaces-per-note-chunk', type=int, default=64)
    parser.add_argument('--note-preparation-workers', type=int, default=0,
                        help='CPU processes preparing note inputs: 0 uses up to 8 available CPUs; 1 is serial')
    parser.add_argument('--pending-requests', type=int, default=1536)
    parser.add_argument('--max-seed-replacements', type=int, default=3,
                        help='New seeds drawn for a patient whose history fails validation on every attempt')
    parser.add_argument('--stop-after-events', action='store_true', help='Save seeds and patient histories without starting notes')
    add_remote_cli_args(parser)
    parser.set_defaults(results_per_shard=128, max_attempts=16, request_timeout=1800,
                        scheduler_adaptive_concurrency=True)
    args = parser.parse_args()
    if args.max_seed_replacements < 0:
        parser.error('--max-seed-replacements must be nonnegative')
    if args.note_preparation_workers < 0 or args.spaces_per_note_chunk < 1:
        parser.error('--note-preparation-workers must be nonnegative and --spaces-per-note-chunk must be positive')
    if args.scheduler_adaptive_concurrency and not args.scheduler_metrics_file:
        args.scheduler_metrics_file = str(Path(args.output_dir) / 'concurrency_status.json')
    return args


def main():
    args = parse_args()
    ensure_file_descriptor_capacity(args.pending_requests)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'generation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tokenizer = load_generation_tokenizer(args.tokenizer_model or args.model,
                                              cache_dir=args.download_dir)
        generator = Generator(args, tokenizer)
        try:
            asyncio.run(generator.run())
        except Exception as exc:
            generator.write_status(stage='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
