#!/usr/bin/env python3
"""Summarize the retrospective trial and SOC cohorts' pre-index notes with Nemotron on an on-prem server.

User, 2026-11-14: "queue summarization of these retrospective enrollment and soc cohort patients behind the ongoing OA
patient summarization work".

Same summarizer and settings as the real-time (OA) summarization run:
- the inference package's serial summarizer with its default preset (matchminer_ai.patients.summarize_patients:
  50,000-token chunks, 500 overlap, continuous scheduling, the model's own sampling profile; Nemotron gets NVIDIA's
  sampling with thinking on);
- fixed per-process concurrency from servers.json;
- on-prem servers approved for PHI only (--server-url, default the site config's endpoints.teacher.url). Never a
  cloud VM.

Unit: one index event (``pseudo_mrn``), because a patient's notes are cut at a different date for each of their
events. ``patient_id`` in the summaries is the pseudo_mrn, which joins to the denominator CSVs. Notes come from the
note-level datasets built by build_note_level_dataset.py (docs/ANCHOR_DATES.md):
- trials: cut before note_anchor_dt - 14 days, with the own protocol number, NCT ID and title drug codes masked;
- SOC: cut before plan start - 14 days.
Events without notes have nothing to summarize.

Cohorts: ``enrollments`` and ``soc`` are the evaluation cohorts (SOC = the test split, as in v22). ``soc_validation``
(user, 2026-11-16: "add to the code a run for patients in the validation set for soc ... i want to run some evals
without committing to eval on my held out test set yet") is the same SOC definition on the validation split, pulled into
its own folder (<run>/soc_validation/, pull_denominators.py --cohorts soc --soc-split-filter validation) with its own
pseudo_mrn numbering; never join its summaries to the test-split CSV.

Outputs (PHI, under --out): <cohort>/summaries/<batch>.parquet, <cohort>/qc/<batch>.csv and
<cohort>/metadata/<batch>.json. Round checkpoints live in --work (a private local directory) and are removed per finished
batch.
- Batches are claimed with an exclusive lock, so several processes (--shard i --shards N) never work on one batch.
- Finished batches are skipped on rerun, and an interrupted batch continues from its checkpoints.

``--launch-after-oa RESUME_JSON``: wait until every OA shard listed there, and every orphaned OA worker, has exited,
then start --shards processes of this script (recorded in <out>/<--launch-record>, default launch.json) and exit.
``--launch``: start them now, once every cohort's note dataset is complete (its .summary.json exists; at most
--notes-wait-hours). Shard logs are <out>/run.shard<i>.log for the default cohorts and
run.<cohorts>.shard<i>.log otherwise. While any shard runs, the OA resume waiter (data/phi/real_time/
resume_oa_after_retro.py) keeps OA from starting, since it waits for every summarize_retro_cohorts.py process.

Prints counts and batch names only, never clinical text or identifiers. Research summaries for evaluation, not
clinical documentation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from matchminer_ai.site import load_site_config
from matchminer_ai.training.paths import child_env, phi

NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
NOTE_FILES = {  # under --run
    "enrollments": "enrollments_note_level_dataset_minus14_masked.parquet",
    "soc": "soc_note_level_dataset_minus14.parquet",
    "soc_validation": "soc_validation/soc_validation_note_level_dataset_minus14.parquet",
}
NOTES: dict[str, Path] = {}  # cohort -> its note-level dataset, set from --run by parse_args
DEFAULT_COHORTS = "enrollments,soc"
logger = logging.getLogger("summarize_retro")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohorts", default=DEFAULT_COHORTS, help=", ".join(NOTE_FILES))
    p.add_argument("--run", type=Path,
                   help="The denominators directory (default: <data root>/phi/real_eval_denominators_20261114)")
    p.add_argument("--out", type=Path, help="Default: RUN/summaries")
    p.add_argument("--work", type=Path, default=Path("/tmp/mmai-retro-phi/summaries"),
                   help="private local directory for round checkpoints (mode 700)")
    p.add_argument("--server-url", help="the first servers.json row (default: the site config's endpoints.teacher.url)")
    p.add_argument("--model", default=NEMOTRON)
    p.add_argument("--concurrency", type=int, default=64, help="each process's cap on the first server (OA: 64)")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--parallel-batches", type=int, default=5)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--prompt-workers", type=int, default=4)
    p.add_argument("--request-timeout", type=float, default=4 * 3600)
    p.add_argument("--max-retries", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="load notes and print the batch plan; no requests")
    p.add_argument("--launch-after-oa", type=Path, help="OA resume_*.json listing the running OA shard PIDs")
    p.add_argument("--launch", action="store_true", help="start --shards processes now and exit")
    p.add_argument("--launch-record", default="launch.json", help="launch record file name under --out")
    p.add_argument("--notes-wait-hours", type=float, default=6,
                   help="launch: wait this long for each cohort's note dataset (its .summary.json, written last)")
    p.add_argument("--poll-seconds", type=int, default=300)
    args = p.parse_args(argv)
    args.run = args.run or phi("real_eval_denominators_20261114")
    args.out = args.out or args.run / "summaries"
    args.server_url = args.server_url or load_site_config().require("endpoints.teacher.url")
    NOTES.update({cohort: args.run / name for cohort, name in NOTE_FILES.items()})
    return args


def ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def oa_running(resume_json: Path) -> int:
    """OA processes still alive: the recorded shards plus any summarize_oa_patients.py process (orphaned workers)."""
    rec = json.loads(resume_json.read_text())
    alive = sum(ticks(pid) == t for pid, t in zip(rec["pids"], rec["start_ticks"]))
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    others = sum("summarize_oa_patients.py --out" in line and "python" in line for line in ps)
    return max(alive, others)


def launch(args) -> int:
    """Start --shards processes (after OA has finished, with --launch-after-oa) and record them."""
    args.out.mkdir(parents=True, exist_ok=True)
    os.chmod(args.out, 0o700)
    status = args.out / args.launch_record
    record = dict(queued_at=datetime.now(timezone.utc).isoformat(timespec="seconds"), cohorts=args.cohorts,
                  launcher_pid=os.getpid())
    if args.launch_after_oa:
        record.update(waits_for=str(args.launch_after_oa), state="waiting for OA summarization to finish")
        status.write_text(json.dumps(record, indent=2) + "\n")
        while oa_running(args.launch_after_oa):
            time.sleep(args.poll_seconds)
        oa_out = args.launch_after_oa.parent
        record.update(oa_finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      oa_summary_batches=len(list((oa_out / "summaries").glob("*.parquet"))))
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    pending = [c for c in cohorts if not NOTES[c].with_suffix(".summary.json").exists()]
    if pending:
        record.update(state=f"waiting for the note datasets of {', '.join(pending)}")
        status.write_text(json.dumps(record, indent=2) + "\n")
        deadline = time.time() + args.notes_wait_hours * 3600
        while any(not NOTES[c].with_suffix(".summary.json").exists() for c in cohorts):
            if time.time() > deadline:
                record.update(state="not started: the note datasets were not ready in time",
                              gave_up_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                status.write_text(json.dumps(record, indent=2) + "\n")
                return 1
            time.sleep(min(args.poll_seconds, 60))
        record.update(notes_ready_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    tag = "" if args.cohorts == DEFAULT_COHORTS else args.cohorts.replace(",", "+") + "."
    children = []
    for i in range(args.shards):
        log = open(args.out / f"run.{tag}shard{i}.log", "a")
        command = [sys.executable, "-m", "matchminer_ai.evaluation.retro_cohorts.summarize_retro_cohorts", "--cohorts", args.cohorts, "--run", str(args.run),
                   "--out", str(args.out), "--work", str(args.work),
                   "--server-url", args.server_url, "--model", args.model, "--concurrency", str(args.concurrency),
                   "--batch-size", str(args.batch_size), "--parallel-batches", str(args.parallel_batches),
                   "--shard", str(i), "--shards", str(args.shards), "--prompt-workers", str(args.prompt_workers)]
        child = subprocess.Popen(command, cwd=args.out, env=child_env(), stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        children.append(child.pid)
    time.sleep(3)
    record.update(state="started", started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  pids=children, start_ticks=[ticks(p) for p in children],
                  stop_note=("kill the pids, then orphaned workers: ps -eo pid,ppid,args | awk '$2==1 && "
                             "/[s]ummarize_retro_cohorts.py/ {print $1}' | xargs -r kill"))
    status.write_text(json.dumps(record, indent=2) + "\n")
    return 0


def servers(args) -> Path:
    path = args.out / "servers.json"
    if not path.exists():
        path.write_text(json.dumps({"servers": [dict(url=args.server_url, upstream_concurrency=args.concurrency,
                                                     model=args.model)]}, indent=2) + "\n")
    for row in json.loads(path.read_text())["servers"]:
        with urllib.request.urlopen(row["url"].rstrip("/") + "/models", timeout=60) as response:
            served = [m["id"] for m in json.load(response)["data"]]
        if row.get("model", args.model) not in served:
            raise SystemExit(f"{row['url']} serves {served}, not {row.get('model', args.model)}")
    return path


def config(args, registry):
    from matchminer_ai.config import load_default_preset
    value = load_default_preset()
    value.remote.update(enabled=True, server_urls_file=str(registry), request_timeout=args.request_timeout,
                        max_retries=args.max_retries)
    value.remote["adaptive_concurrency"].update(enabled=False, metrics_interval=10)  # as in the OA run (2026-11-12)
    value.model_metadata_cache_dir = str(args.out / "model_metadata")
    value.patient["remote"].update(model_name=args.model, tokenizer_name=args.model)
    value.patient["prompt_build_workers"] = args.prompt_workers
    return value


def load_notes(cohort: str) -> pd.DataFrame:
    notes = pd.read_parquet(NOTES[cohort], columns=["pseudo_mrn", "date", "RPT_ID", "text"])
    return notes.sort_values(["pseudo_mrn", "date", "RPT_ID"], kind="stable").reset_index(drop=True)


def plan(cohorts, notes_by_cohort, batch_size):
    batches = []
    for cohort in cohorts:
        events = notes_by_cohort[cohort].pseudo_mrn.drop_duplicates().tolist()
        for i in range(0, len(events), batch_size):
            batches.append((cohort, f"{cohort}-{i // batch_size:04d}", events[i:i + batch_size]))
    return batches


def summarize(cohort, name, events, notes, args, registry):
    target = args.out / cohort / "summaries" / f"{name}.parquet"
    if target.exists():
        return 0
    lock = args.work / "checkpoints" / f"{name}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if target.exists():
            return 0
        from matchminer_ai.patients import summarize_patients
        started = time.time()
        rows_in = notes[notes.pseudo_mrn.isin(set(events))]
        frame = pd.DataFrame(dict(patient_id=rows_in.pseudo_mrn.astype(str), note_text=rows_in.text,
                                  note_date=rows_in.date))
        checkpoints = args.work / "checkpoints" / name
        rows, metadata, qc = summarize_patients(frame, config=config(args, registry), checkpoint_dir=checkpoints,
                                                return_metadata=True, return_qc=True)
        for folder in ("summaries", "qc", "metadata"):
            (args.out / cohort / folder).mkdir(parents=True, exist_ok=True)
        qc.to_csv(args.out / cohort / "qc" / f"{name}.csv", index=False)
        (args.out / cohort / "metadata" / f"{name}.json").write_text(json.dumps(dict(
            batch=name, cohort=cohort, index_events=len(events), notes=len(rows_in), summarized=len(rows),
            seconds=round(time.time() - started), finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            notes_file=str(NOTES[cohort]), **metadata), indent=2, default=str) + "\n")
        rows.to_parquet(target.with_suffix(".tmp"), index=False)
        target.with_suffix(".tmp").replace(target)
        for path in (target, args.out / cohort / "qc" / f"{name}.csv", args.out / cohort / "metadata" / f"{name}.json"):
            path.chmod(0o600)
        shutil.rmtree(checkpoints, ignore_errors=True)
        logger.info("Batch %s: %d of %d index events summarized in %.0f min.", name, len(rows), len(events),
                    (time.time() - started) / 60)
        return len(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.umask(0o077)
    if args.launch_after_oa or args.launch:
        return launch(args)
    cohorts = [c.strip() for c in args.cohorts.split(",") if c.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    for path in (args.out, args.work.parent, args.work):
        os.chmod(path, 0o700)
    notes = {c: load_notes(c) for c in cohorts}
    everything = plan(cohorts, notes, args.batch_size)
    mine = everything[args.shard::args.shards] + [b for i, b in enumerate(everything) if i % args.shards != args.shard]
    done = {p.stem for c in cohorts for p in (args.out / c / "summaries").glob("*.parquet")}
    todo = [b for b in mine if b[1] not in done]
    logger.info("Process %d of %d: %d batches (%s); %d done, %d to go; %d at a time.", args.shard, args.shards,
                len(everything), ", ".join(f"{c} {sum(1 for b in everything if b[0] == c)} batches / "
                                           f"{notes[c].pseudo_mrn.nunique()} events / {len(notes[c])} notes" for c in cohorts),
                len(everything) - len(todo), len(todo), args.parallel_batches)
    if args.dry_run:
        return 0
    registry = servers(args)
    failed = []
    with ThreadPoolExecutor(args.parallel_batches) as pool:
        futures = {name: pool.submit(summarize, cohort, name, events, notes[cohort], args, registry)
                   for cohort, name, events in todo}
        for name, future in futures.items():
            try:
                future.result()
            except Exception as error:
                failed.append(name)
                where = "; ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}"
                                  for f in traceback.extract_tb(error.__traceback__)[-6:])
                logger.error("Batch %s failed (%s: %s) at %s; a rerun continues it from its checkpoints.", name,
                             type(error).__name__, re.sub(r"\d{5,}", "N", str(error))[:300], where)
    logger.info("Finished: %d batches failed%s.", len(failed), f" ({', '.join(failed)})" if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
