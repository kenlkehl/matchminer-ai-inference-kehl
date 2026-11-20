"""Spot-check the SQL workup agent on a synthetic extraction store (non-PHI only).

For each guideline space given, this matches a cohort from the population view's default
synthetic sample (the same retrieval as the population view), has the SQL agent write one
query per workup item (cached in the app cache, exploring the whole store), runs each query
once over the cohort, and runs the compiled-query method on the same records for
comparison. It writes, per space:

  queries.md     each item's SQL, the agent's summary, approximations and exploration steps,
                 and the compiled JSON query
  results.csv    patient x item: both methods' status, reason, applicability and bottom line
  agreement.md   per item: status agreement between the two methods
  spot_check.md  a stratified sample (K patients per SQL status, plus disagreements) with
                 each result's evidence quotes and the patient's generation-time synthetic
                 event history, for checking by eye

No reference labels are made and no LLM labels the sample. The agent's exploration results
(extracted values from the synthetic store) go to the selected connection: Gemini 4 Hydrogen
on Vertex by default, or any OpenAI-compatible endpoint with --backend local --endpoint.
--sql-backend gives the SQL agent a connection of its own, such as Claude through the
Databricks AI Gateway (claude_databricks, with --sql-model or
GUIDELINE_DATABRICKS_SONNET_ENDPOINT naming the gateway endpoint); compiled queries keep
--backend.

Examples:
  python examples/workup_sql_spot_check.py --space nccn-nscl-df052cc3b953c3b7 --patients 30
  python examples/workup_sql_spot_check.py --space S --items 2,8,13 --backend local \\
      --endpoint http://localhost:8000/v1 --model served/model
  python examples/workup_sql_spot_check.py --space S --sql-backend claude_databricks
  python examples/workup_sql_spot_check.py --build-db /tmp/synthetic_extractions.duckdb
"""

import argparse
import copy
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from matchminer_ai.apps.guideline_fetcher.settings import (  # noqa: E402
    CLAUDE_BACKENDS, CLAUDE_ENDPOINT_NAME, SQL_TASK, VERTEX_PROFILES, Settings, data_path,
    mode_side_error,
)

STATUSES = ("met", "not_met", "unknown", "not_applicable")
MAX_HISTORY_CHARS = 6000


def parse(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", action="append", default=[], metavar="SPACE_TRIAL_ID",
                        help="a guideline space to check (repeatable)")
    parser.add_argument("--items", help="comma-separated workup item indexes (default: all)")
    parser.add_argument("--patients", type=int, default=30,
                        help="matched patients per space (candidate_k; default %(default)s)")
    parser.add_argument("--min-reasonableness", type=float, default=0.75,
                        help="decision-model P(reasonable) a matched patient needs (default %(default)s)")
    parser.add_argument("--store", help="extraction record store (default: the app's default)")
    parser.add_argument("--out", help="output folder (default: a dated folder under "
                        "data/no_phi/guideline_fetcher_extractions/spot_checks)")
    parser.add_argument("--backend", choices=[*VERTEX_PROFILES, "local"],
                        help="LLM connection (default: the app's default, Gemini 4 Hydrogen)")
    parser.add_argument("--endpoint", help="OpenAI-compatible /v1 URL for --backend local")
    parser.add_argument("--model", help="served model for --backend local")
    parser.add_argument("--tokenizer", help="tokenizer for --backend local (default: --model)")
    parser.add_argument("--sql-backend", choices=[*VERTEX_PROFILES, *CLAUDE_BACKENDS],
                        help="the SQL agent's own connection (default: --backend's)")
    parser.add_argument("--sql-model", help="the Databricks gateway endpoint name for "
                        "--sql-backend claude_databricks (default: "
                        "GUIDELINE_DATABRICKS_SONNET_ENDPOINT)")
    parser.add_argument("--no-compiled-queries", action="store_true",
                        help="skip the compiled-query comparison")
    parser.add_argument("--per-status", type=int, default=3,
                        help="spot-check patients per SQL status and item (default %(default)s)")
    parser.add_argument("--max-disagreements", type=int, default=6,
                        help="disagreeing patients shown per item (default %(default)s)")
    parser.add_argument("--no-events", action="store_true",
                        help="omit the generation-time event histories")
    parser.add_argument("--seed", type=int, default=23, help="spot-check sampling seed")
    parser.add_argument("--build-db", metavar="PATH",
                        help="only write the store's exploration database to PATH and exit")
    return parser.parse_args(argv)


def connection(args):
    values = {}
    if args.backend:
        values["llm_backend"] = args.backend
    if args.backend == "local":
        if not (args.endpoint and args.model):
            sys.exit("error: --backend local needs --endpoint and --model")
        values.update(endpoint=args.endpoint.rstrip("/"), model=args.model,
                      tokenizer=args.tokenizer or args.model)
    if args.sql_backend in CLAUDE_BACKENDS:
        profile = CLAUDE_BACKENDS[args.sql_backend]
        model = args.sql_model or profile["model"]
        if not CLAUDE_ENDPOINT_NAME.fullmatch(model or ""):
            sys.exit("error: --sql-backend claude_databricks needs the full three-part gateway "
                     "endpoint name (--sql-model or GUIDELINE_DATABRICKS_SONNET_ENDPOINT)")
        values["task_connections"] = {SQL_TASK: {
            "llm_backend": args.sql_backend, "endpoint": profile["endpoint"], "model": model,
            "tokenizer": profile["tokenizer"]}}
    elif args.sql_backend:
        values["task_connections"] = {SQL_TASK: {"llm_backend": args.sql_backend}}
    settings = Settings(**values)
    if settings.phi:
        sys.exit("error: spot checks run in the non-PHI app only")
    return settings


def event_histories(settings, ids):
    """Each patient's generation-time synthetic event history (masked_text), by id."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(settings.population_notes, format="parquet")
    if "masked_text" not in dataset.schema.names:
        return {}
    key = "pseudo_mrn" if "pseudo_mrn" in dataset.schema.names else "patient_id"
    typed = dataset.to_table(columns=[key]).column(key).unique().to_pylist()
    wanted = [v for v in typed if str(v) in set(ids)]
    table = dataset.to_table(columns=[key, "masked_text"], filter=ds.field(key).isin(wanted))
    histories = {}
    for pid, text in zip(table.column(key).to_pylist(), table.column("masked_text").to_pylist()):
        if str(pid) in histories or not isinstance(text, str):
            continue
        text = re.sub(r"<(?:BEGIN|END) EVENT CORRESPONDING TO SYNTHETIC NOTE>", "", text)
        histories[str(pid)] = re.sub(r"\n{3,}", "\n\n", text).strip()
    return histories


def assessments(report):
    return {p["patient_id"]: p["review"]["assessments"] for p in report["patients"]}


def quotes(assessment, limit=4):
    lines = []
    for evidence in assessment.get("evidence", [])[:limit]:
        what = " · ".join(x for x in (evidence.get("item"), evidence.get("date_text")) if x)
        lines.append(f"  - [{evidence.get('state')}] {what}: \"{evidence.get('quote', '')}\"")
    return lines


def write_queries(path, space, sql, queries):
    lines = [f"# {space['name']}", "", f"`{space['space_trial_id']}`", "",
             "Population (key criteria):", "", "```", (space.get("cohort_key_criteria") or ""
                                                     ).strip(), "```", ""]
    for number, item in enumerate(space["diagnostic_workup"]):
        entry = sql[number]
        lines += [f"## {number}. {item['name']}", "",
                  f"Conditions: {item.get('conditions') or '(none)'} · "
                  f"{item.get('category') or ''}", "",
                  f"**SQL agent**: {entry['status']}, {entry.get('termination')}"
                  f"{' (accepted by ' + entry['accepted_by'] + ')' if entry.get('accepted_by') else ''}"
                  f", {entry.get('action_count')} actions, {entry.get('submit_count')} submissions"
                  f"{', cached' if entry.get('cached') else ''}; "
                  f"{entry.get('usage', {}).get('prompt_tokens', 0):,} prompt tokens", ""]
        if entry.get("expressible") is False:
            lines += [f"Not expressible: {entry.get('not_expressible_reason')}", ""]
        if entry.get("error"):
            lines += [f"Error: {entry['error']}", ""]
        if entry.get("summary"):
            lines += [f"Summary: {entry['summary']}", ""]
        lines += [f"- Approximation: {a}" for a in entry.get("approximations") or []]
        if entry.get("distribution"):
            lines += ["", "Exploration population: " + ", ".join(
                f"{k} {n}" for k, n in entry["distribution"]["status_reason"].items())]
        if entry.get("sql"):
            lines += ["", "```sql", entry["sql"], "```"]
        if entry.get("steps"):
            lines += ["", "<details><summary>Exploration steps</summary>", ""]
            for step in entry["steps"]:
                lines += [f"{step['step']}. **{step['action']}** — {step['purpose']} → "
                          f"{step['outcome']}", "", "```sql", step["sql"], "```", ""]
            lines += ["</details>"]
        if queries is not None:
            q = queries[number]
            lines += ["", f"**Compiled query**: {q['status']}"
                      f"{' — ' + q['query']['summary'] if q.get('query') else ''}", "",
                      "<details><summary>JSON</summary>", "", "```json",
                      json.dumps(q.get("query"), indent=1), "```", "</details>"]
        lines.append("")
    path.write_text("\n".join(lines))


def compare(space, sql_report, query_report, histories, out, args):
    items = space["diagnostic_workup"]
    by_sql = assessments(sql_report)
    by_query = assessments(query_report) if query_report else {}
    rows, agreement = [], []
    for pid, results in by_sql.items():
        for number, a in enumerate(results):
            b = by_query.get(pid, [None] * len(items))[number]
            rows.append({
                "patient_id": pid, "item": number, "workup": items[number]["name"],
                "sql_status": a["status"], "sql_reason": a["reason"],
                "sql_applicability": a["applicability"], "sql_bottom_line": a["bottom_line"],
                "query_status": b["status"] if b else "", "query_reason": b["reason"] if b else "",
                "query_applicability": b["applicability"] if b else "",
                "query_bottom_line": b["bottom_line"] if b else "",
                "agree": "" if b is None else a["status"] == b["status"],
            })
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# Agreement: SQL agent vs compiled queries · {space['name']}", "",
             f"{len(by_sql)} patients. Rows: SQL agent status; columns: compiled-query status.",
             ""]
    for number, item in enumerate(items):
        pairs = Counter((r["sql_status"], r["query_status"]) for r in rows if r["item"] == number)
        sql_counts = Counter(r["sql_status"] for r in rows if r["item"] == number)
        same = sum(n for (a, b), n in pairs.items() if a == b)
        total = sum(pairs.values())
        agreement.append((number, same, total))
        lines += [f"## {number}. {item['name']}", "",
                  "SQL agent: " + ", ".join(f"{s} {sql_counts[s]}" for s in STATUSES), ""]
        if query_report:
            lines += [f"Agreement {same}/{total} ({same / total:.0%})" if total else "", "",
                      "| SQL \\ compiled | " + " | ".join(STATUSES) + " |",
                      "|---|" + "---|" * len(STATUSES)]
            for a in STATUSES:
                lines.append(f"| {a} | " + " | ".join(str(pairs[(a, b)]) for b in STATUSES)
                             + " |")
        lines.append("")
    (out / "agreement.md").write_text("\n".join(lines))
    rng = random.Random(args.seed)
    lines = [f"# Spot checks · {space['name']}", "",
             "Each case shows both methods' results, the SQL result's evidence quotes and the "
             "patient's generation-time synthetic event history (what the notes were written "
             "from; labs, examinations and counseling were never generator events). Check by "
             "eye; no labels were made.", ""]
    for number, item in enumerate(items):
        lines += [f"## {number}. {item['name']}", f"Conditions: {item.get('conditions') or ''}",
                  ""]
        chosen = []
        for status in STATUSES:
            ids = [pid for pid, results in by_sql.items() if results[number]["status"] == status]
            chosen += [(pid, f"SQL {status}") for pid in rng.sample(ids, min(args.per_status,
                                                                             len(ids)))]
        if by_query:
            differ = [pid for pid in by_sql if by_query[pid][number]["status"]
                      != by_sql[pid][number]["status"] and pid not in {p for p, _ in chosen}]
            chosen += [(pid, "disagreement") for pid in differ[:args.max_disagreements]]
        for pid, why in chosen:
            a = by_sql[pid][number]
            lines += [f"### Patient {pid} · {why}", "",
                      f"- **SQL agent**: {a['status']} / {a['reason']} / {a['applicability']} — "
                      f"{a['bottom_line']}", *quotes(a)]
            if by_query:
                b = by_query[pid][number]
                lines += [f"- **Compiled query**: {b['status']} / {b['reason']} / "
                          f"{b['applicability']} — {b['bottom_line']}"]
            history = histories.get(pid)
            if history:
                cut = history if len(history) <= MAX_HISTORY_CHARS else (
                    history[:MAX_HISTORY_CHARS] + "\n[history cut]")
                lines += ["", "<details><summary>Generation-time event history</summary>", "",
                          "```", cut, "```", "</details>"]
            lines.append("")
    (out / "spot_check.md").write_text("\n".join(lines))
    return agreement


def main(argv=None):
    args = parse(argv)
    settings = connection(args)
    from matchminer_ai.apps.guideline_fetcher.extraction_workup import (
        compile_space_queries, compile_space_sql, open_store, sql_summary,
    )
    from matchminer_ai.patients.workup_sql import (
        build_extraction_database, write_extraction_database,
    )
    from matchminer_ai.apps.guideline_fetcher.population import PopulationInference, load_population_sample
    from matchminer_ai.apps.guideline_fetcher.services import Catalogs, InputError

    def progress(message):
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    try:
        store = open_store(settings, args.store)
        if args.build_db:
            target = Path(args.build_db).expanduser().resolve()
            error = mode_side_error(False, target)
            if error:
                sys.exit(f"error: {error}")
            documents, records = store.all_records()
            database = build_extraction_database(
                [f for f in (documents, records) if f is not None])
            write_extraction_database(target, database)
            print(f"Wrote {target}: {database.summary()}")
            return
        if not args.space:
            sys.exit("error: give at least one --space (or --build-db)")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = Path(args.out).expanduser().resolve() if args.out else (
            data_path("no_phi/guideline_fetcher_extractions/spot_checks") / stamp)
        error = mode_side_error(False, out)
        if error:
            sys.exit(f"error: {error}")
        out.mkdir(parents=True, exist_ok=True)
        config = settings.extraction_config()
        paths = Catalogs(settings.catalog_root).resolve([])
        source = load_population_sample(settings, settings.population_sample_size,
                                        settings.population_seed, progress)
        histories = {} if args.no_events else event_histories(
            settings, source.summaries.patient_id.astype(str).tolist())
        runs = []
        for space_id in args.space:
            started = time.monotonic()
            cohort = PopulationInference(settings).retrieve(
                source, paths, space_id, args.patients, args.min_reasonableness, progress)
            space = copy.deepcopy(cohort.space)
            if args.items:
                keep = [int(i) for i in args.items.split(",")]
                space["diagnostic_workup"] = [space["diagnostic_workup"][i] for i in keep]
                cohort = replace(cohort, space=space)
            ids = cohort.summaries.patient_id.tolist()
            progress(f"{space_id}: {len(ids)} matched patients, "
                     f"{len(space['diagnostic_workup'])} workup items")
            if not ids:
                continue
            documents, records = store.records(ids)
            from matchminer_ai.population import (
                check_population_workup_with_extractions, check_population_workup_with_sql,
            )

            sql, sql_meta = compile_space_sql(space, settings, config, store, progress)
            sql_report = check_population_workup_with_sql(
                cohort, sql, document_records=documents, patient_records=records,
                config=config, progress_callback=progress).report
            query_report = queries = None
            if not args.no_compiled_queries:
                queries, _ = compile_space_queries(space, settings, config, progress)
                query_report = check_population_workup_with_extractions(
                    cohort, queries, document_records=documents, patient_records=records,
                    progress_callback=progress).report
            folder = out / re.sub(r"[^A-Za-z0-9_.-]", "_", space_id)
            folder.mkdir(exist_ok=True)
            write_queries(folder / "queries.md", space, sql, queries)
            agreement = compare(space, sql_report, query_report, histories, folder, args)
            summary = {k: v for k, v in sql_summary(sql, sql_meta).items() if k != "queries"}
            runs.append({"space_trial_id": space_id, "name": space["name"], "patients": ids,
                         "items": len(space["diagnostic_workup"]), "sql_agent": summary,
                         "agreement": agreement, "seconds": round(time.monotonic() - started)})
            progress(f"{space_id}: wrote {folder}")
        model = VERTEX_PROFILES.get(settings.llm_backend, {}).get("model", settings.model)
        (out / "run.json").write_text(json.dumps({
            "created_utc": stamp, "backend": settings.llm_backend, "model": model,
            "sql_agent_model": config.workup_sql_agent["remote"]["model_name"],
            "store": store.path.name, "spaces": runs,
        }, indent=1, default=str))
        print(f"Spot checks written to {out}")
    except InputError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
