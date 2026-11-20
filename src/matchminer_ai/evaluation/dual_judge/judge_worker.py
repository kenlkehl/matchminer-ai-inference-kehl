#!/usr/bin/env python3
"""Send a run directory's requests to one judge and journal the answers (resumable; safe to rerun).

  vertex      Claude Opus 5.5 on Vertex AI (``anthropic[vertex]``, Google ADC; --project, ANTHROPIC_VERTEX_PROJECT_ID
              or the site config's vertex.project)
  databricks  Claude Opus 5.5 through a Databricks AI Gateway (``anthropic`` and ``databricks-sdk``; --model, else the
              site config's databricks.endpoints.opus; the site config's databricks.host and databricks.profile)
  azure       GPT-6.1 Sol on Azure (Responses API over HTTPS; an Entra token from ``az`` or AZURE_OPENAI_API_KEY)

Both judges receive the same bytes (``requests/<component>.jsonl``), whose system text ends with the answer's JSON
schema, with no sampling parameters. Azure also enforces the schema (strict ``json_schema``); Vertex cannot (the
project's organization policy forbids structured outputs for partner models, 2026-11-18). An answer that
breaks the schema or the component's coverage rules is retried once with the errors appended; refusals,
truncation and twice-invalid answers are final and go to the adjudication queue as single-judge units. A judge
never falls back to another model. Answers go to ``responses/<judge>/<component>.jsonl``, one row per attempt.

Needs only the standard library, plus ``anthropic[vertex]`` for the vertex judge, so it runs on any machine that
can read and write the run directory (mode 600/700: run it as the owner).

  python judge_worker.py --run-dir RUN --judge vertex --dry-run            # price the remaining requests
  python judge_worker.py --run-dir RUN --judge vertex --limit 1            # one live request per component
  python judge_worker.py --run-dir RUN --judge azure \\
      --endpoint https://<resource>.openai.azure.com/openai/v1 --deployment <gpt-6.1-sol deployment>
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time
import urllib.error
import urllib.request

from matchminer_ai.evaluation.dual_judge.judging import (COMPONENTS, PRICES, Journal, check_enums, component_module, log, mask, parse_answer,  # noqa: E402
                     private, read_jsonl, usage_cost)

FINAL = ('ok', 'refusal', 'max_tokens', 'invalid')
DEFAULTS = dict(
    vertex=dict(model='claude-opus-5-5', concurrency=8, min_interval=0.0, **PRICES['vertex']),
    databricks=dict(model=None, concurrency=16, min_interval=0.0, **PRICES['databricks']),  # the gateway endpoint name
    azure=dict(model=None, concurrency=3, min_interval=2.0, **PRICES['azure']),  # Azure deployments are named by the user
)
# --dry-run estimates, from the Opus smoke run (2026-11-18): output tokens per request with some margin for a judge
# that reasons longer, and characters per input token (clinical notes and criteria tokenize densely: ~2.0)
EXPECTED_OUTPUT = dict(extraction=5000, summary=10000, decision=1000)
CHARS_PER_TOKEN = 2.0


class Refused(Exception):
    pass


# --- providers -----------------------------------------------------------------------------------------------------

class VertexJudge:
    name = 'vertex'

    def __init__(self, args):
        from anthropic import AnthropicVertex
        self.model, self.effort = args.model, args.effort
        self.client = AnthropicVertex(project_id=args.project, region=args.region, max_retries=8, timeout=3600)

    def body(self, request, user):
        return dict(model=self.model, max_tokens=request['max_output_tokens'],
                    system=[dict(type='text', text=request['system'], cache_control=dict(type='ephemeral'))],
                    messages=[dict(role='user', content=user)],
                    # no output_config.format: the organization policy forbids structured outputs for partner
                    # models; the schema is in the system text and every answer is validated
                    output_config=dict(effort=self.effort))

    def call(self, request, user):
        with self.client.messages.stream(**self.body(request, user)) as stream:
            message = stream.get_final_message()
        usage = message.usage
        meta = dict(served_model=message.model, provider_id=message.id, stop_reason=message.stop_reason,
                    usage=dict(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                               cache_read_tokens=getattr(usage, 'cache_read_input_tokens', 0) or 0,
                               cache_write_tokens=getattr(usage, 'cache_creation_input_tokens', 0) or 0))
        text = ''.join(block.text for block in message.content if block.type == 'text')
        status = {'end_turn': 'ok', 'max_tokens': 'max_tokens', 'refusal': 'refusal'}.get(message.stop_reason, 'ok')
        if status == 'refusal' and getattr(message, 'stop_details', None):
            meta['refusal_category'] = getattr(message.stop_details, 'category', None)
        return status, text, meta


class DatabricksJudge(VertexJudge):
    """Claude through a Databricks AI Gateway's Messages route: the Vertex judge's requests, signed with a
    Databricks token (no refusal fallback: a judge never falls back to another model)."""
    name = 'databricks'

    def __init__(self, args):
        import anthropic
        from anthropic.lib.credentials import AccessToken
        from matchminer_ai.llm.anthropic_messages import GATEWAY_PATH, gateway_url_error
        from matchminer_ai.llm.remote_auth import databricks_token_provider
        from matchminer_ai.site import load_site_config
        site = load_site_config()
        base_url = str(site.get('databricks.host') or '').rstrip('/') + GATEWAY_PATH
        if gateway_url_error(base_url):
            raise SystemExit('databricks needs the site config\'s databricks.host (the workspace URL)')
        self.model = args.model or site.get('databricks.endpoints.opus')
        if not self.model:
            raise SystemExit('databricks needs --model (the gateway endpoint name) or databricks.endpoints.opus')
        self.effort = args.effort
        tokens = databricks_token_provider(str(site.get('databricks.profile') or ''))

        def credentials(*, force_refresh=False):
            return AccessToken(token=tokens.token(), expires_at=0)

        self.client = anthropic.Anthropic(base_url=base_url, credentials=credentials, max_retries=8, timeout=3600)


class AzureJudge:
    name = 'azure'

    def __init__(self, args):
        if not args.endpoint or not args.deployment:
            raise SystemExit('azure needs --endpoint (https://<resource>.openai.azure.com/openai/v1) and --deployment')
        self.url = args.endpoint.rstrip('/') + '/responses'
        self.model, self.effort, self.az = args.deployment, args.effort, args.az
        self.key = os.environ.get('AZURE_OPENAI_API_KEY')
        self.token, self.token_at, self.lock = None, 0.0, threading.Lock()

    def headers(self, refresh=False):
        if self.key:
            return {'api-key': self.key}
        with self.lock:
            if refresh or self.token is None or time.time() - self.token_at > 1800:
                result = subprocess.run([self.az, 'account', 'get-access-token',
                                         '--resource=https://cognitiveservices.azure.com/', '--query', 'accessToken',
                                         '--output', 'tsv'], capture_output=True, text=True, timeout=120)
                token = result.stdout.strip()
                if result.returncode != 0 or not token or any(c.isspace() for c in token):
                    raise RuntimeError('Azure token refresh failed (run "az login"); no other credential is tried')
                self.token, self.token_at = token, time.time()
            return {'Authorization': f'Bearer {self.token}'}

    def body(self, request, user):
        return {'model': self.model,
                'input': [{'role': 'system', 'content': request['system']}, {'role': 'user', 'content': user}],
                'reasoning': {'effort': self.effort}, 'max_output_tokens': request['max_output_tokens'],
                'text': {'format': {'type': 'json_schema', 'name': request['schema_name'], 'strict': True,
                                    'schema': request['schema']}},
                'store': False, 'stream': False, 'truncation': 'disabled'}

    def call(self, request, user):
        data = json.dumps(self.body(request, user)).encode()
        refreshed, last = False, '429/5xx'
        for attempt in range(10):
            http = urllib.request.Request(self.url, data=data, method='POST',
                                          headers={'Content-Type': 'application/json', **self.headers(refreshed)})
            try:
                with urllib.request.urlopen(http, timeout=3600) as response:
                    return self.parse(json.load(response))
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403) and not refreshed and not self.key:
                    refreshed = True
                    continue
                if exc.code == 429 or exc.code >= 500:
                    hint = exc.headers.get('retry-after-ms')
                    wait = float(hint) / 1000 if hint else float(exc.headers.get('retry-after') or 0)
                    time.sleep(max(wait, min(300, 5 * 2 ** attempt)) + random.random())
                    last = f'HTTP {exc.code}'
                    continue
                try:  # keep only the service's error message, never request content or headers
                    message = json.loads(exc.read()).get('error', {}).get('message', '')
                except Exception:  # noqa: BLE001
                    message = ''
                raise RuntimeError(f'HTTP {exc.code}: {mask(message)[:300]}') from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = type(exc).__name__
                time.sleep(min(300, 5 * 2 ** attempt) + random.random())
        raise RuntimeError(f'Azure request failed after 10 attempts ({last})')

    @staticmethod
    def parse(raw):
        usage = raw.get('usage') or {}
        meta = dict(served_model=raw.get('model'), provider_id=raw.get('id'), stop_reason=raw.get('status'),
                    usage=dict(input_tokens=usage.get('input_tokens', 0), output_tokens=usage.get('output_tokens', 0),
                               cached_input_tokens=(usage.get('input_tokens_details') or {}).get('cached_tokens', 0),  # inside input_tokens
                               reasoning_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0)))
        text, refused = [], False
        for item in raw.get('output', []):
            if item.get('type') != 'message':
                continue
            for block in item.get('content', []):
                if block.get('type') == 'refusal':
                    refused = True
                elif block.get('type') == 'output_text':
                    text.append(block.get('text', ''))
        if refused:
            return 'refusal', '', meta
        if raw.get('status') == 'incomplete':
            meta['incomplete_reason'] = (raw.get('incomplete_details') or {}).get('reason')
            return 'max_tokens', ''.join(text), meta
        if raw.get('status') != 'completed':
            raise RuntimeError(f'Azure response status {raw.get("status")}')
        return 'ok', ''.join(text), meta


# --- the run -------------------------------------------------------------------------------------------------------

def cost(usage, args):
    return usage_cost(usage, dict(usd_in=args.usd_in, usd_out=args.usd_out))


def done_ids(rows, retry_failed):
    latest = {}
    for row in rows:
        if row.get('final'):
            latest[row['request_id']] = row['status']
    keep = ('ok',) if retry_failed else FINAL
    return {rid for rid, status in latest.items() if status in keep}


def feedback(user, errors):
    return (user + '\n\nA previous answer to this request was rejected for these reasons; return a complete, corrected '
            'answer:\n' + '\n'.join(f'- {e}' for e in errors[:20]))


def judge_one(judge, request, module, journal, args, spend):
    """One request: an attempt, then one retry with the validation errors appended if the answer was invalid.
    Every attempt is journaled; ``spend`` is told each attempt's cost."""
    user, previous = request['user'], None
    for attempt in (1, 2):
        started = time.time()
        row = dict(request_id=request['request_id'], item_id=request['item_id'], component=request['component'],
                   judge=judge.name, model=judge.model, effort=args.effort, attempt=attempt,
                   started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(started)))
        try:
            status, text, meta = judge.call(request, user)
        except Exception as exc:  # noqa: BLE001  transport or provider failure: not final, a rerun resends it
            row.update(final=False, status='error', error=mask(f'{type(exc).__name__}: {exc}')[:400],
                       seconds=round(time.time() - started, 1))
            if previous is not None:  # the first answer was invalid and its retry failed: the invalid answer stands
                row.update(final=True, status='invalid', errors=previous)
            journal.append(row)
            return row
        row.update(meta, text=text, seconds=round(time.time() - started, 1),
                   prices=dict(usd_in=args.usd_in, usd_out=args.usd_out), cost_usd=round(cost(meta['usage'], args), 4))
        spend(row['cost_usd'])
        if status != 'ok':
            row.update(final=True, status=status, answer=None)
            journal.append(row)
            return row
        try:
            answer = parse_answer(text)
            errors = check_enums(answer, request['schema']) or module.validate(answer, request.get('expect', {}))
        except ValueError as exc:
            answer, errors = None, [f'not one valid JSON object: {exc}']
        if not errors:
            row.update(final=True, status='ok', answer=answer)
            journal.append(row)
            return row
        row.update(final=attempt == 2, status='invalid', answer=answer, errors=errors)
        journal.append(row)
        previous, user = errors, feedback(request['user'], errors)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--judge', required=True, choices=('vertex', 'databricks', 'azure'))
    p.add_argument('--components', nargs='+', default=list(COMPONENTS), choices=COMPONENTS)
    p.add_argument('--limit', type=int, help='at most this many new requests per component (a live test)')
    p.add_argument('--concurrency', type=int)
    p.add_argument('--min-interval', type=float, help='seconds between request starts')
    p.add_argument('--max-usd', type=float, default=75.0, help='stop sending once this judge has spent this much')
    p.add_argument('--usd-in', type=float, help='price per million input tokens')
    p.add_argument('--usd-out', type=float, help='price per million output tokens')
    p.add_argument('--effort', default='high')
    p.add_argument('--retry-failed', action='store_true', help='also resend refused, truncated and invalid requests')
    p.add_argument('--dry-run', action='store_true', help='price the remaining requests; send nothing')
    p.add_argument('--model', help='vertex model (default claude-opus-5-5), or the databricks gateway endpoint name')
    p.add_argument('--project', default=os.environ.get('ANTHROPIC_VERTEX_PROJECT_ID'),
                   help="Vertex project (default: ANTHROPIC_VERTEX_PROJECT_ID, else the site config's vertex.project)")
    p.add_argument('--region', default=os.environ.get('CLOUD_ML_REGION', 'global'))
    p.add_argument('--endpoint', default=os.environ.get('AZURE_OPENAI_ENDPOINT'))
    p.add_argument('--deployment', default=os.environ.get('AZURE_OPENAI_DEPLOYMENT'))
    p.add_argument('--az', default='az', help='Azure CLI executable for Entra tokens')
    args = p.parse_args(argv)
    defaults = DEFAULTS[args.judge]
    for key in ('usd_in', 'usd_out', 'concurrency', 'min_interval'):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    args.model = args.model or defaults['model']
    if args.judge == 'vertex' and not args.project:
        from matchminer_ai.site import load_site_config  # needs PyYAML only when a site config is used
        args.project = load_site_config().require('vertex.project')
    private()

    work, spent = [], 0.0
    for component in args.components:
        requests = read_jsonl(args.run_dir / 'requests' / f'{component}.jsonl')
        journal_path = args.run_dir / 'responses' / args.judge / f'{component}.jsonl'
        rows = read_jsonl(journal_path)
        spent += sum(usage_cost(row.get('usage'), row.get('prices') or PRICES[args.judge]) for row in rows)
        done = done_ids(rows, args.retry_failed)
        todo = [r for r in requests if r['request_id'] not in done][:args.limit]
        finished = sum(r['request_id'] in done for r in requests)
        log(f'{args.judge} {component}: {len(requests)} requests, {finished} done, {len(todo)} to send')
        work += [(component, r) for r in todo]
    if args.dry_run:
        chars = {c: sum(len(r['system']) + len(r['user']) for k, r in work if k == c) for c in args.components}
        total = 0.0
        for c in args.components:
            n = sum(k == c for k, _ in work)
            tokens_in, tokens_out = chars[c] / CHARS_PER_TOKEN, n * EXPECTED_OUTPUT[c]
            usd = (tokens_in * args.usd_in + tokens_out * args.usd_out) / 1e6
            total += usd
            log(f'{args.judge} {c}: {n} requests, ~{tokens_in / 1e6:.2f}M input and ~{tokens_out / 1e6:.2f}M output '
                f'tokens, ~${usd:.2f}')
        log(f'{args.judge} total ~${total:.2f} (prices ${args.usd_in}/${args.usd_out} per M tokens; already spent '
            f'${spent:.2f}; cap ${args.max_usd})')
        return 0
    if not work:
        return 0
    judge = dict(vertex=VertexJudge, databricks=DatabricksJudge, azure=AzureJudge)[args.judge](args)
    modules = {c: component_module(c) for c in args.components}
    journals = {c: Journal(args.run_dir / 'responses' / args.judge / f'{c}.jsonl') for c in args.components}
    budget, gate = dict(spent=spent), threading.Lock()
    next_start = [0.0]
    counts = {}

    def task(component, request):
        with gate:
            if budget['spent'] >= args.max_usd:
                return None
            wait = next_start[0] - time.time()
            next_start[0] = max(time.time(), next_start[0]) + args.min_interval
        if wait > 0:
            time.sleep(wait)
        def spend(usd):
            with gate:
                budget['spent'] += usd
        return judge_one(judge, request, modules[component], journals[component], args, spend)

    with ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(task, c, r) for c, r in work]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            key = 'skipped (budget)' if row is None else f'{row["component"]} {row["status"]}'
            counts[key] = counts.get(key, 0) + 1
            if i % 5 == 0 or i == len(futures):
                log(f'{args.judge}: {i}/{len(futures)} {dict(sorted(counts.items()))}, ${budget["spent"]:.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
