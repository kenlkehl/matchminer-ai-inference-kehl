#!/usr/bin/env python3
"""Sample and freeze the items of a dual-judge run and render one request per item (where the data lives, with the package installed).

Reads the fidelity evaluation's outputs (``--fidelity-dir``: extraction, corpus sample, teacher labels, decision
scores), the validation SOC notes and summaries (``--denominators``) and training's frozen extraction prompts, and
writes ``items/<component>.jsonl``, ``requests/<component>.jsonl`` and ``build.json`` into ``--run-dir``. A component
already built with the same settings is skipped; one built under other settings is refused (use a new run
directory). Counts only in the log.
"""
import argparse
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
from matchminer_ai.evaluation.dual_judge.judging import (COMPONENTS, component_module, log, make_request, private, private_dir,  # noqa: E402
                     sha256, write_json, write_jsonl)
from matchminer_ai.evaluation.dual_judge import judge_worker
from matchminer_ai.training.paths import PACKAGE, no_phi



def frozen_prompts():
    """The prompt directories of the code snapshots that extracted the evaluation's cohorts (under no_phi)."""
    return dict(
        step1=no_phi('training_run_screened_sample5000_20261104/source_cohorts_fast/matchminer-ai-inference-kehl'
                     '/src/matchminer_ai/prompts'),
        step2=no_phi('synthetic_dataset_n5000_screened_20261106/source/matchminer-ai-inference-kehl/src'
                     '/matchminer_ai/prompts'))


VERSION = 'dual-judge-build-v1'


def code_digest(name):
    """The component's own code: its rubric, its judge prompt and the shared request builder."""
    files = [HERE / f'rubric_{name}.py', PACKAGE / 'prompts' / f'judge.{name}.txt', HERE / 'judging.py']
    return sha256({f.name: f.read_text() for f in files})


def build_component(name, args):
    module = component_module(name)
    if name == 'extraction':
        items, instructions, design = module.build(args.fidelity_dir, frozen_prompts(), args.extraction_trials, args.seed)
    elif name == 'summary':
        items, instructions, design = module.build(args.fidelity_dir, args.denominators, args.summary_events, args.seed)
    else:
        items, instructions, design = module.build(
            args.fidelity_dir, dict(match=args.match_pairs, boilerplate=args.boilerplate_pairs,
                                    criteria=args.criteria_lines), args.seed)
    requests = []
    for item in items:
        system, user = module.render(item, instructions)
        schema = module.schema(item) if hasattr(module, 'schema') else module.SCHEMA
        requests.append(make_request(name, item['item_id'], system, user, schema,
                                     max_output_tokens=module.MAX_OUTPUT_TOKENS, expect=module.expect(item)))
    return items, requests, design


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--fidelity-dir', required=True, type=Path, help='the retro_fidelity run (E)')
    p.add_argument('--denominators', required=True, type=Path, help='real_eval_denominators_20261114 (R)')
    p.add_argument('--components', nargs='+', default=list(COMPONENTS), choices=COMPONENTS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--extraction-trials', type=int, default=10)
    p.add_argument('--summary-events', type=int, default=8)
    p.add_argument('--match-pairs', type=int, default=16)
    p.add_argument('--boilerplate-pairs', type=int, default=12)
    p.add_argument('--criteria-lines', type=int, default=24)
    p.add_argument('--rebuild', nargs='+', default=[], choices=COMPONENTS,
                   help='rebuild these components: their items and requests move to superseded/<time>/, and answers to '
                        'requests that did not change still count (answers are matched by request id)')
    args = p.parse_args(argv)
    private()
    private_dir(args.run_dir)
    manifest_path = args.run_dir / 'build.json'
    manifest = __import__('json').loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        version=VERSION, components={})
    settings = dict(seed=args.seed, fidelity_dir=args.fidelity_dir.name, denominators=args.denominators.name,
                    sizes=dict(extraction=args.extraction_trials, summary=args.summary_events,
                               decision=dict(match=args.match_pairs, boilerplate=args.boilerplate_pairs,
                                             criteria=args.criteria_lines)))
    for name in args.components:
        identity = dict(version=VERSION, seed=args.seed, fidelity_dir=settings['fidelity_dir'],
                        denominators=settings['denominators'], size=settings['sizes'][name], code=code_digest(name))
        previous = manifest['components'].get(name)
        if previous and previous['identity'] == identity and (args.run_dir / 'requests' / f'{name}.jsonl').exists():
            log(f'[skip] {name} already built')
            continue
        if previous and name not in args.rebuild:
            raise SystemExit(f'{name} was built under other settings or code; pass --rebuild {name} or use a new run '
                             'directory')
        old_ids = set()
        if previous:
            archive = private_dir(args.run_dir / 'superseded' / time.strftime('%Y%m%dT%H%M%S') / name)
            for folder in ('items', 'requests'):
                path = args.run_dir / folder / f'{name}.jsonl'
                if path.exists():
                    if folder == 'requests':
                        old_ids = {r['request_id'] for r in map(__import__('json').loads, path.open())}
                    path.replace(archive / f'{folder}.jsonl')
            write_json(archive / 'build.json', previous)
        started = time.time()
        items, requests, design = build_component(name, args)
        write_jsonl(args.run_dir / 'items' / f'{name}.jsonl', items)
        write_jsonl(args.run_dir / 'requests' / f'{name}.jsonl', requests)
        chars = sum(len(r['system']) + len(r['user']) for r in requests)
        manifest['components'][name] = dict(
            identity=identity, design=design, items=len(items), requests=len(requests), request_chars=chars,
            max_request_chars=max((len(r['system']) + len(r['user']) for r in requests), default=0),
            built_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), seconds=round(time.time() - started))
        write_json(manifest_path, manifest)
        kept = sum(r['request_id'] in old_ids for r in requests)
        log(f'{name}: {len(items)} items, {len(requests)} requests ({kept} unchanged from the previous build), '
            f'{chars / 1e6:.2f}M characters (largest {manifest["components"][name]["max_request_chars"] / 1e6:.2f}M)')
    for judge in ('vertex', 'azure'):
        judge_worker.main(['--run-dir', str(args.run_dir), '--judge', judge, '--dry-run',
                           '--components', *[c for c in COMPONENTS if (args.run_dir / 'requests' / f'{c}.jsonl').exists()]])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
