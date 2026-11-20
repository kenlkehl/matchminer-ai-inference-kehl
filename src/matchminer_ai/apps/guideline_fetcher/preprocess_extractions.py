"""Build structured extraction records for a population, offline, for extraction-based
workup checks.

Runs the inference package's ``extract_document_records`` (one request per note) and,
with ``--levels document_and_patient``, ``extract_patient_records`` (one record per
patient) on a population's notes, then writes an extraction record store that the app's
population view reads for matched patients (see extraction_workup.py for the layout).

Every note is sent to the LLM endpoint given here: use only an endpoint authorized for
the notes. Checkpoints (``--checkpoint-dir``, kept for resuming) hold patient-derived
text like the store itself.

* ``--phi``: notes from a PHI-mode preprocessed population cache (``notes.parquet``);
  the store and checkpoints are private (0700/0600) and never in a no_phi folder.
  Logs show counts only.
* default: synthetic notes from the non-PHI population Parquet (or ``--notes``, which
  must not be in a phi folder); the store never goes in a phi folder.

Requests are long: Nemotron 3.5 Super often reasons 10,000-15,000 tokens per note.
Keep ``--request-timeout`` above a whole completion on a busy server.

``--server-registry FILE`` spreads one build over several servers, which may serve
different builds of a model (for example NVFP4 and FP8): a JSON file
``{"servers": [{"url": ..., "model": ..., "upstream_concurrency": ...}]}``. Each server
must report its model at ``/v1/models``; each takes at most its
``upstream_concurrency`` requests (default ``--concurrency``). Checkpoints do not
record the model, so a build can move between servers and resume.

Example (synthetic, 20 patients):
  python -m matchminer_ai.apps.guideline_fetcher.preprocess_extractions --limit 20 --endpoint http://localhost:8000/v1
"""

import argparse
import os
import sys
from pathlib import Path

from .settings import Settings, mode_side_error
from matchminer_ai.schema import note_columns


def read_notes(settings, args):
    """The selected patients' notes as patient_id, note_text, note_date, note_type.

    Only the patient-id column is read to choose patients; then only their notes.
    """
    import pyarrow.dataset as ds

    from .services import InputError

    if settings.phi:
        from .population_cache import _manifest, resolve_cache

        path = resolve_cache(settings, args.cache)
        manifest = _manifest(path)
        dataset = ds.dataset(path / "notes.parquet", format="parquet")
        key, columns = note_columns(("patient_id",))
        source = {"kind": "population_cache", "name": path.name,
                  "created_utc": manifest["created_utc"], "notes": manifest["counts"]["notes"]}
    else:
        path = Path(args.notes or settings.population_notes).expanduser().resolve()
        error = mode_side_error(False, path)
        if error:
            raise InputError(error)
        dataset = ds.dataset(path, format="parquet")
        key, columns = note_columns(dataset.schema.names)
        source = {"kind": "synthetic_notes", "name": path.name,
                  "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    typed = dataset.to_table(columns=[key]).column(key).unique().to_pylist()
    by_text = {str(v): v for v in typed}
    patients = sorted(by_text)
    if args.patients:
        wanted = [line.strip() for line in Path(args.patients).read_text().splitlines()
                  if line.strip()]
        unknown = [p for p in wanted if p not in by_text]
        if unknown:
            raise InputError(f"{len(unknown)} listed patients have no notes in the source.")
        patients = list(dict.fromkeys(wanted))
    elif args.limit:
        import random

        patients = sorted(random.Random(args.seed).sample(patients, min(args.limit, len(patients))))
    names = [c for c in columns if c in dataset.schema.names]
    notes = dataset.to_table(
        columns=names, filter=ds.field(key).isin([by_text[p] for p in patients])
    ).to_pandas().rename(columns=columns)
    if "note_type" not in notes:
        notes["note_type"] = None
    notes["patient_id"] = notes.patient_id.astype(str)
    notes = notes.loc[notes.note_text.map(lambda t: isinstance(t, str) and bool(t.strip()))]
    source.update(patients=len(patients), sample_limit=args.limit, seed=args.seed,
                  patient_list=bool(args.patients))
    return notes.reset_index(drop=True), source


def read_registry(path):
    """The servers of a ``--server-registry`` file: [{url, model, upstream_concurrency}]."""
    import json

    from .services import InputError

    try:
        data = json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError) as exc:
        raise InputError(f"Cannot read the server registry: {exc}") from None
    rows = data.get("servers") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise InputError('The server registry needs a nonempty "servers" list.')
    servers = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("url"), str) \
                or not isinstance(row.get("model"), str) or not row["model"].strip():
            raise InputError("Each registry server needs a url and the model it serves.")
        limit = row.get("upstream_concurrency")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 1024):
            raise InputError("upstream_concurrency must be an integer from 1 to 1024.")
        url = row["url"].strip().rstrip("/")
        servers.append({"url": url if url.endswith("/v1") else f"{url}/v1",
                        "model": row["model"].strip(),
                        **({"upstream_concurrency": limit} if limit else {})})
    if len({s["url"] for s in servers}) != len(servers):
        raise InputError("The server registry lists a server twice.")
    return servers


def served_models(url, timeout=30):
    """The model ids an OpenAI-compatible server reports at /v1/models."""
    import httpx

    response = httpx.get(f"{url}/models", timeout=timeout, trust_env=False)
    response.raise_for_status()
    return [m["id"] for m in response.json()["data"]]


def check_registry(servers):
    """Refuse servers that are unreachable or do not serve the model the registry names."""
    from .services import InputError

    for server in servers:
        try:
            models = served_models(server["url"])
        except Exception as exc:  # noqa: BLE001 -- any failure means not usable
            raise InputError(f"{server['url']} is not reachable: {type(exc).__name__}") from None
        if server["model"] not in models:
            raise InputError(f"{server['url']} serves {models}, not {server['model']}.")


def build_config(settings, args, servers=None):
    config = settings.extraction_config()
    config.remote.update(request_timeout=args.request_timeout,
                         max_concurrent_requests=args.concurrency)
    if servers:
        # The endpoint pool reads the registry: each server's model and ceiling.
        config.remote.update(server_urls=[s["url"] for s in servers],
                             server_urls_file=str(Path(args.server_registry).expanduser()
                                                  .resolve()))
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phi", action="store_true",
                        help="read a PHI-mode population cache and write a private PHI store")
    parser.add_argument("--cache", help="PHI mode: population cache folder (default: the app's)")
    parser.add_argument("--notes", help="non-PHI mode: synthetic notes Parquet (default: the app's)")
    parser.add_argument("--output", help="store folder to create (default: the mode's "
                        "GUIDELINE_EXTRACTION_RECORDS)")
    parser.add_argument("--levels", choices=["document", "document_and_patient"],
                        default="document", help="document records only, or also patient "
                        "records (default: %(default)s)")
    parser.add_argument("--patients", help="a file of patient ids, one per line")
    parser.add_argument("--limit", type=int, help="a seeded sample of this many patients")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--endpoint", help="OpenAI-compatible /v1 URL (default: MMAI_ENDPOINT)")
    parser.add_argument("--server-registry", metavar="FILE",
                        help="JSON registry of several servers and the model each serves "
                        "(instead of --endpoint); see above")
    parser.add_argument("--model", help="served model (default: the extraction preset's model)")
    parser.add_argument("--tokenizer", help="tokenizer for patient-level chunks (default: --model)")
    parser.add_argument("--request-timeout", type=int, default=14400,
                        help="seconds per completion (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="requests in flight (default: %(default)s)")
    parser.add_argument("--checkpoint-dir",
                        help="resume folder (default: .<output name>.checkpoints beside it)")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing store after the new one is complete")
    args = parser.parse_args(argv)
    if args.phi:
        os.umask(0o077)
    from matchminer_ai import load_default_preset

    from .extraction_workup import (
        extraction_metadata, resolve_store, write_store,
    )
    from .services import InputError

    servers = None
    if args.server_registry:
        if args.endpoint:
            sys.exit("error: use --endpoint or --server-registry, not both")
        try:
            servers = read_registry(args.server_registry)
            check_registry(servers)
        except InputError as exc:
            sys.exit(f"error: {exc}")
        args.endpoint = servers[0]["url"]
    model = args.model or (servers[0]["model"] if servers else
                           load_default_preset().document_extraction["remote"]["model_name"])
    values = {"llm_backend": "local", "model": model, "tokenizer": args.tokenizer or model}
    if args.endpoint:
        values["endpoint"] = args.endpoint.rstrip("/")
    settings = Settings(phi=args.phi, **values)
    try:
        output = resolve_store(settings, args.output, must_exist=False)
        if output.exists() and not args.overwrite:
            raise InputError(f"{output} exists. Pass --overwrite to replace it.")
        checkpoints = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir \
            else output.parent / f".{output.name}.checkpoints"
        error = mode_side_error(settings.phi, checkpoints)
        if error:
            raise InputError(error)
        checkpoints.mkdir(mode=0o700, parents=True, exist_ok=True)
        notes, source = read_notes(settings, args)
        print(f"{source['patients']:,} patients, {len(notes):,} notes", flush=True)
        config = build_config(settings, args, servers)
        import matchminer_ai.patients as patients
        from matchminer_ai.patients.extraction.notes import note_ids

        patient_records = patient_meta = None
        if args.levels == "document_and_patient":
            patient_records, patient_meta = patients.extract_patient_records(
                notes, config=config, checkpoint_dir=checkpoints, return_metadata=True,
                progress_callback=lambda done, total: print(
                    f"Patient records: {done:,}/{total:,} chunks", flush=True),
            )
            ids = notes.assign(note_id=note_ids(notes)).groupby("patient_id").note_id.apply(
                lambda v: sorted(set(v)))
            patient_records["note_ids"] = [ids.get(p, []) for p in patient_records.patient_id]
        documents, document_meta = patients.extract_document_records(
            notes, config=config, patient_records=patient_records, checkpoint_dir=checkpoints,
            return_metadata=True,
            progress_callback=lambda done, total: print(
                f"Document records: {done:,}/{total:,} notes", flush=True),
        )
        manifest = write_store(
            settings, output, documents, patient_records, source=source,
            extraction=extraction_metadata(
                document_meta, patient_meta,
                ", ".join(dict.fromkeys(s["model"] for s in servers)) if servers else model,
                servers=servers),
            overwrite=args.overwrite,
        )
    except InputError as exc:
        sys.exit(f"error: {exc}")
    counts = manifest["counts"]
    print(f"Saved records for {counts['patients']:,} patients to {output}: "
          f"documents {counts.get('documents')}, patients {counts.get('patient_records')}",
          flush=True)


if __name__ == "__main__":
    main()
