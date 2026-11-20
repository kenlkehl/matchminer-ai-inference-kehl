#!/usr/bin/env python3
"""The adjudication app: decide the units the two judges disagree on (standard library only; loopback only).

  python3 adjudicate.py --run-dir RUN [--port 8791]     then open http://127.0.0.1:8791 on the same machine

It reads the run directory's items and ``compare/units.jsonl`` (rerun compare.py to pick up new answers; the app
reloads it when it changes) and appends every decision to ``adjudications.jsonl`` (the latest decision per unit
wins; "skip" withdraws one). Judges appear as Judge 1 and Judge 2 in an order fixed per item and revealed only
after a decision, and for decision questions the teacher's answer and the decision model's P(yes) are shown only
after a decision too, unless the user ticks "show teacher and model before deciding". The data are PHI: the server binds to 127.0.0.1, refuses other hosts and Host headers, and
logs nothing about requests.
"""
import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import threading
import time
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
from matchminer_ai.evaluation.dual_judge.judging import COMPONENTS, JUDGES, Journal, component_module, latest_decisions, log, private, read_jsonl  # noqa: E402

JUDGE_NAMES = dict(vertex='Claude Opus 5.5 (Vertex)', azure='GPT-6.1 Sol (Azure)')


class Store:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.lock = threading.Lock()
        self.modules = {c: component_module(c) for c in COMPONENTS}
        self.items = {c: {i['item_id']: i for i in read_jsonl(self.run_dir / 'items' / f'{c}.jsonl')} for c in COMPONENTS}
        self.journal = Journal(self.run_dir / 'adjudications.jsonl')
        self.units_path = self.run_dir / 'compare' / 'units.jsonl'
        self.units_mtime, self.units, self.by_item = None, {}, {}
        self.reload()

    def reload(self):
        mtime = self.units_path.stat().st_mtime if self.units_path.exists() else None
        if mtime == self.units_mtime:
            return
        units = read_jsonl(self.units_path)
        self.units = {u['unit_id']: u for u in units}
        self.by_item = {}
        for u in units:
            self.by_item.setdefault((u['component'], u['item_id']), []).append(u['unit_id'])
        self.units_mtime = mtime

    def decisions(self):
        return latest_decisions(read_jsonl(self.run_dir / 'adjudications.jsonl'))

    @staticmethod
    def order(component, item_id):
        judges = list(JUDGES)
        random.Random(f'{component}:{item_id}:judge-order').shuffle(judges)
        return judges

    def state(self):
        with self.lock:
            self.reload()
            decisions = self.decisions()
            out = {}
            for component in COMPONENTS:
                module, rows = self.modules[component], []
                for item_id, item in self.items[component].items():
                    ids = self.by_item.get((component, item_id), [])
                    queued = [i for i in ids if self.units[i]['queued']]
                    view = module.view(item)
                    rows.append(dict(item_id=item_id, title=view['title'], stratum=item.get('stratum'),
                                     units=len(ids), queued=len(queued),
                                     decided=sum(i in decisions for i in queued)))
                out[component] = dict(items=rows, queued=sum(r['queued'] for r in rows),
                                      decided=sum(r['decided'] for r in rows))
            return dict(components=out, compared_at=self.units_mtime)

    def unit_view(self, unit, item, decisions, order, show_reference=False):
        module = self.modules[unit['component']]
        kind = module.KINDS[unit['kind']]
        collapse = kind.get('collapse') or {}
        choices = list(dict.fromkeys(collapse.get(v, v) for v in kind['values']))
        judges = [dict(slot=n, value=unit['values'][j], compared=unit['compared'][j], detail=unit['details'][j])
                  for n, j in enumerate(order, 1)]
        row = dict(unit_id=unit['unit_id'], key=unit['key'], kind=unit['kind'], label=kind['label'], choices=choices,
                   target=module.unit_target(item, unit['key']), status=unit['status'], queued=unit['queued'],
                   queue_reason=unit['queue_reason'], judges=judges, decision=None,
                   question=module.question(item, unit['key']) if hasattr(module, 'question') else None)
        decision = decisions.get(unit['unit_id'])
        if hasattr(module, 'reveal') and (decision is not None or show_reference):
            row['reference'] = module.reveal(item, unit['key'])
        if decision is not None:
            row['decision'] = dict(value=decision['value'], note=decision.get('note', ''), at=decision['at'],
                                   stale=decision.get('seen') != unit['compared'])
            row['reveal'] = dict(judges={n: JUDGE_NAMES[j] for n, j in enumerate(order, 1)})
        return row

    def item(self, component, item_id, show_reference=False):
        with self.lock:
            self.reload()
            item = self.items[component][item_id]
            decisions, order = self.decisions(), self.order(component, item_id)
            units = [self.unit_view(self.units[i], item, decisions, order, show_reference)
                     for i in self.by_item.get((component, item_id), [])]
            return dict(component=component, item_id=item_id, view=self.modules[component].view(item), units=units)

    def decide(self, body):
        with self.lock:
            self.reload()
            unit = self.units.get(body.get('unit_id'))
            if unit is None:
                raise KeyError('unknown unit')
            choice, value = body.get('choice'), body.get('value')
            if choice != 'skip':
                module = self.modules[unit['component']]
                kind = module.KINDS[unit['kind']]
                allowed = {(kind.get('collapse') or {}).get(v, v) for v in kind['values']}
                if value not in allowed:
                    raise ValueError('value is not one of the unit\'s choices')
            order = self.order(unit['component'], unit['item_id'])
            picked = order[int(choice[-1]) - 1] if choice in ('judge1', 'judge2') else None
            self.journal.append(dict(unit_id=unit['unit_id'], choice=choice, value=value, picked_judge=picked,
                                     note=str(body.get('note', ''))[:4000], seen=unit['compared'],
                                     at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), user=os.environ.get('USER', '')))
        return self.item(unit['component'], unit['item_id'], bool(body.get('show_reference')))


def make_handler(store, port):
    allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
    page = (HERE / 'static' / 'adjudicate.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # nothing about requests is logged (item ids, PHI)
            pass

        def send(self, status, body, kind='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def guard(self):
            if self.headers.get('Host') not in allowed_hosts:
                self.send(HTTPStatus.FORBIDDEN, dict(error='open the app at http://127.0.0.1:%d' % port))
                return False
            return True

        def do_GET(self):  # noqa: N802
            if not self.guard():
                return
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == '/':
                    return self.send(HTTPStatus.OK, page, 'text/html; charset=utf-8')
                if url.path == '/api/state':
                    return self.send(HTTPStatus.OK, store.state())
                if url.path == '/api/item':
                    return self.send(HTTPStatus.OK, store.item(query['component'], query['item_id'],
                                                               query.get('reference') == '1'))
                return self.send(HTTPStatus.NOT_FOUND, dict(error='not found'))
            except KeyError:
                return self.send(HTTPStatus.NOT_FOUND, dict(error='unknown item'))

        def do_POST(self):  # noqa: N802
            if not self.guard():
                return
            if urlparse(self.path).path != '/api/decide':
                return self.send(HTTPStatus.NOT_FOUND, dict(error='not found'))
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
                return self.send(HTTPStatus.OK, store.decide(body))
            except (KeyError, ValueError, TypeError) as exc:
                return self.send(HTTPStatus.BAD_REQUEST, dict(error=str(exc)))

    return Handler


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8791)
    args = p.parse_args(argv)
    if args.host not in ('127.0.0.1', 'localhost'):
        raise SystemExit('The app serves PHI and binds to 127.0.0.1 only; reach it from elsewhere with ssh -L.')
    private()
    store = Store(args.run_dir)
    if not store.units:
        raise SystemExit('No compare/units.jsonl yet: run compare.py first')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store, args.port))
    log(f'adjudication app on http://127.0.0.1:{args.port} ({len(store.units)} units)')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
