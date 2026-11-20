"""Shared pieces of the dual-judge evaluation: privacy, files, requests, schemas, units and agreement.

Standard library only, so the judge worker, the comparison, the adjudication app and the metrics run on any machine
that can read the run directory; only build_items.py needs the data and the package's dependencies.

A run directory is self-contained:
  build.json                        what was sampled, from where, with which prompts
  items/<component>.jsonl           the frozen items (PHI for summary and decision)
  requests/<component>.jsonl        one provider-neutral request per item, the same bytes for both judges
  responses/<judge>/<component>.jsonl   the judges' answers, append-only (latest final attempt per request wins)
  compare/units.jsonl, compare/summary.json   every unit with both judges' values and its status
  adjudications.jsonl               the user's decisions, append-only (latest per unit wins)
  metrics/metrics.json, metrics/report.md
Everything is created under umask 077 (files 600, directories 700); logs carry counts only.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

COMPONENTS = ('extraction', 'summary', 'decision')
JUDGES = ('vertex', 'azure')  # Claude Opus 5.5 on Vertex AI; GPT-6.1 Sol on Azure
HERE = Path(__file__).resolve().parent


# --- privacy and logging -------------------------------------------------------------------------------------------

def private():
    """umask 077 for this process and its children: new files 600, new directories 700."""
    os.umask(0o077)


def private_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def mask(value):
    """Text safe to print: any run of 6 or more digits becomes #s."""
    return re.sub(r'\d{6,}', lambda m: '#' * len(m.group()), str(value))


def log(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S %Z")}] {mask(message)}', flush=True)


# --- files ---------------------------------------------------------------------------------------------------------

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    private_dir(path.parent)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def write_json(path, value):
    path = Path(path)
    private_dir(path.parent)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


class Journal:
    """An append-only JSONL file shared by threads: one line per write, flushed and fsynced."""

    def __init__(self, path):
        self.path = Path(path)
        private_dir(self.path.parent)
        self.lock = threading.Lock()

    def append(self, row):
        line = json.dumps(row, ensure_ascii=False) + '\n'
        with self.lock, self.path.open('a') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def sha256(value):
    data = value if isinstance(value, (bytes, bytearray)) else (
        value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)).encode()
    return hashlib.sha256(data).hexdigest()


# Prices per million tokens (Claude Opus 5.5 list price; GPT-6.1 Sol is a placeholder until set with --usd-in and
# --usd-out). Anthropic reports cache writes and reads apart from input_tokens and bills them at 1.25x and 0.1x the
# input price; OpenAI counts cached tokens inside input_tokens, so they are billed here at the full price.
PRICES = dict(vertex=dict(usd_in=4.0, usd_out=20.0), azure=dict(usd_in=5.0, usd_out=30.0))


def usage_cost(usage, prices):
    """US dollars of one response's usage at ``prices`` ({usd_in, usd_out})."""
    usage = usage or {}
    tokens_in = (usage.get('input_tokens', 0) + 1.25 * usage.get('cache_write_tokens', 0)
                 + 0.1 * usage.get('cache_read_tokens', 0))
    return (tokens_in * prices['usd_in'] + usage.get('output_tokens', 0) * prices['usd_out']) / 1e6


def prompt_text(name):
    """A judge prompt (``prompts/judge.*.txt``) exactly as stored, final newline included."""
    from importlib import resources
    return resources.files('matchminer_ai.prompts').joinpath(name).read_text(encoding='utf-8')


# --- requests ------------------------------------------------------------------------------------------------------

ANSWER_FORMAT = ('ANSWER FORMAT\nReturn only one JSON object, with no other text and no Markdown fences, that '
                 'matches this JSON schema exactly (every field present, no other fields):\n')


def make_request(component, item_id, system, user, schema, *, max_output_tokens, expect=None):
    """One provider-neutral request. Both judges receive exactly these bytes; ``expect`` lets the worker validate an
    answer without the item (for example the line numbers every answer must cover).

    The schema is also written into the system text: the Vertex project's organization policy forbids structured
    outputs for partner models on Vertex (``constraints/vertexai.allowedPartnerModelFeatures``, 2026-11-18), so
    Claude answers to the written schema while Azure also enforces it; the worker validates every answer."""
    schema = strict_schema(schema)
    system = f'{system}\n\n{ANSWER_FORMAT}{json.dumps(schema, ensure_ascii=False)}'
    body = dict(system=system, user=user, schema=schema)
    return dict(request_id=f'{component}:{item_id}:{sha256(body)[:12]}', component=component, item_id=item_id,
                system=system, user=user, schema=schema, schema_name=f'mmai_judge_{component}',
                max_output_tokens=max_output_tokens, expect=expect or {}, prompt_sha256=sha256(body))


def strict_schema(schema):
    """The schema as both strict structured-output modes take it (Anthropic ``output_config.format`` and OpenAI
    strict ``json_schema``): every object closed and requiring all its properties, every enum typed. A schema that
    leaves a property optional is an error, not silently changed."""
    def walk(node, where):
        if isinstance(node, list):
            return [walk(item, where) for item in node]
        if not isinstance(node, dict):
            return node
        node = {key: walk(value, f'{where}.{key}') for key, value in node.items()}
        if 'enum' in node and 'type' not in node:
            kinds = {type(value) for value in node['enum']}
            if len(kinds) != 1:
                raise ValueError(f'{where}: an enum needs values of one type')
            node['type'] = {str: 'string', bool: 'boolean', int: 'integer'}[kinds.pop()]
        if node.get('type') == 'object':
            properties = node.get('properties', {})
            if set(node.get('required', [])) != set(properties):
                raise ValueError(f'{where}: every property must be required')
            node['additionalProperties'] = False
        return node
    return walk(schema, '$')


def parse_answer(text):
    """The JSON object of an answer. A complete outer Markdown fence is removed; JSON is never repaired."""
    text = (text or '').strip()
    fence = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, flags=re.S)
    value = json.loads(fence.group(1) if fence else text)
    if not isinstance(value, dict):
        raise ValueError('The answer must be one JSON object')
    return value


def check_enums(value, schema, where='$'):
    """Errors where ``value`` breaks ``schema`` (types, enums, required and unknown keys): a schema check for answers
    a provider returned without enforcing the schema."""
    errors = []
    kind = schema.get('type')
    if 'enum' in schema and value not in schema['enum']:
        errors.append(f'{where}: {value!r} is not one of {schema["enum"]}')
    if kind == 'object':
        if not isinstance(value, dict):
            return [f'{where}: expected an object']
        properties = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                errors.append(f'{where}: missing "{key}"')
        for key, item in value.items():
            if key not in properties:
                errors.append(f'{where}: unknown field "{key}"')
            else:
                errors += check_enums(item, properties[key], f'{where}.{key}')
    elif kind == 'array':
        if not isinstance(value, list):
            return [f'{where}: expected an array']
        for i, item in enumerate(value):
            errors += check_enums(item, schema.get('items', {}), f'{where}[{i}]')
    elif kind == 'string' and not isinstance(value, str):
        errors.append(f'{where}: expected a string')
    elif kind == 'integer' and (not isinstance(value, int) or isinstance(value, bool)):
        errors.append(f'{where}: expected an integer')
    elif kind == 'boolean' and not isinstance(value, bool):
        errors.append(f'{where}: expected true or false')
    return errors


# --- evidence ------------------------------------------------------------------------------------------------------

def _normal(text):
    return re.sub(r'\s+', ' ', str(text)).strip().lower()


def quote_found(quote, source):
    """True when a quote appears in the source, ignoring case and whitespace; '...' may join pieces in order."""
    pieces = [_normal(piece) for piece in re.split(r'\.\.\.|…', str(quote))]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return False
    text, start = _normal(source), 0
    for piece in pieces:
        found = text.find(piece, start)
        if found < 0:
            return False
        start = found + len(piece)
    return True


# --- units and agreement -------------------------------------------------------------------------------------------

def compared(kinds, kind, value):
    """The value two judges are compared on: the kind's collapse of ``value`` (for example minor omissions counted
    with none), or the value itself."""
    collapse = kinds[kind].get('collapse')
    return collapse.get(value, value) if collapse else value


def unit_status(values):
    """``agreed`` / ``disputed`` from the compared values of the judges that answered; ``single`` with one answer,
    ``missing`` with none."""
    present = [value for value in values.values() if value is not None]
    if not present:
        return 'missing'
    if len(present) == 1:
        return 'single'
    return 'agreed' if len(set(present)) == 1 else 'disputed'


def latest_decisions(rows):
    """The user's latest decision per unit; a later "skip" withdraws an earlier decision."""
    decisions = {}
    for row in sorted(rows, key=lambda r: r.get('at', '')):
        if row.get('choice') == 'skip':
            decisions.pop(row['unit_id'], None)
        else:
            decisions[row['unit_id']] = row
    return decisions


def gold(unit, decisions):
    """The reference value of a unit: the user's decision, else the judges' agreed value, else None (pending)."""
    decision = decisions.get(unit['unit_id'])
    if decision is not None:
        return decision['value'], 'adjudicated'
    if unit['status'] == 'agreed':
        return next(value for value in unit['compared'].values() if value is not None), 'agreed'
    return None, 'pending'


def cohen_kappa(pairs):
    """Cohen's kappa of (a, b) pairs; None when undefined (no pairs, or chance agreement 1)."""
    pairs = list(pairs)
    if not pairs:
        return None
    n = len(pairs)
    labels = sorted({value for pair in pairs for value in pair}, key=str)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == label for a, _ in pairs) / n) * (sum(b == label for _, b in pairs) / n)
                   for label in labels)
    return None if expected >= 1 else (observed - expected) / (1 - expected)


def component_module(name):
    """The component's rubric module (rubric_extraction, rubric_summary, rubric_decision)."""
    import importlib
    return importlib.import_module(f'matchminer_ai.evaluation.dual_judge.rubric_{name}')


__all__ = ['COMPONENTS', 'HERE', 'JUDGES', 'Journal', 'check_enums', 'cohen_kappa', 'component_module', 'compared',
           'gold', 'latest_decisions', 'log', 'make_request', 'mask', 'parse_answer', 'PRICES', 'private', 'private_dir',
           'prompt_text', 'quote_found', 'read_jsonl', 'sha256', 'strict_schema', 'unit_status', 'usage_cost', 'write_json',
           'write_jsonl']
