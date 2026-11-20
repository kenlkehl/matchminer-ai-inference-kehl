"""Thin adapters over the public MatchMiner-AI APIs."""

import copy
import hashlib
import io
import json
import re
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


class InputError(ValueError):
    pass


class Catalogs:
    """Discover complete editions; validate their export hashes in the public loader."""

    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.rows = []
        self.expires = 0

    def list(self, refresh=False):
        with self.lock:
            if not refresh and time.monotonic() < self.expires:
                return list(self.rows)
            candidates = {}
            # The supported layout is ROOT/run/disease. Never traverse checkpoints.
            folders = list(self.root.glob("*/*/validation.json"))
            folders += list(self.root.glob("*/validation.json"))
            for audit_file in folders:
                folder = audit_file.parent
                try:
                    audit = json.loads(audit_file.read_text())
                    status = json.loads((folder / "status.json").read_text())
                    source = json.loads((folder / "sources.json").read_text())
                    if audit.get("status") != "passed" or status.get("status") != "complete":
                        continue
                    if not (folder / "paradigms.jsonl").is_file():
                        continue
                    relative = str(folder.relative_to(self.root))
                    row = dict(
                        id=hashlib.sha256(relative.encode()).hexdigest()[:20],
                        disease=source["disease"],
                        title=source["title"],
                        version=source["version"],
                        spaces=audit["paradigms"],
                        updated=status.get("updated_utc", ""),
                        path=str(folder),
                    )
                    # One completed catalog per disease; keep the latest completed run.
                    old = candidates.get(row["disease"])
                    if old is None or row["updated"] > old["updated"]:
                        candidates[row["disease"]] = row
                except (OSError, ValueError, KeyError):
                    continue  # A copying/in-progress artifact is not a selectable catalog.
            self.rows = sorted(candidates.values(), key=lambda row: row["title"].casefold())
            self.expires = time.monotonic() + 60
            return list(self.rows)

    def resolve(self, selected):
        rows = self.list()
        available = {r["id"]: r for r in rows}
        if selected and any(key not in available for key in selected):
            raise InputError("A selected catalog is no longer available. Refresh the library.")
        paths = (
            [available[key]["path"] for key in selected] if selected else [r["path"] for r in rows]
        )
        if not paths:
            raise InputError("No completed catalogs are available yet.")
        return paths


def pdf_pages(content):
    """Each page's text, through the package's PDF reader (``matchminer_ai.documents.ocr_pdf``: the embedded text
    layer, else local RapidOCR; nothing leaves this machine). The upload is written to a private temporary folder
    that is removed afterwards."""
    from tempfile import TemporaryDirectory

    from matchminer_ai.documents import PDFOCRError, PDFOCRNoTextError, ocr_pdf
    from matchminer_ai.documents.ocr import _PAGE_SEPARATOR

    with TemporaryDirectory(prefix="guideline-fetcher-pdf-") as folder:
        source = Path(folder) / "upload.pdf"
        source.write_bytes(content)
        try:
            text = ocr_pdf(source, Path(folder) / "upload.txt").read_text(encoding="utf-8")
        except PDFOCRNoTextError:
            raise InputError("PDF page 1 has no readable text, even with OCR; no pages were imported.") from None
        except PDFOCRError:
            raise InputError("This PDF could not be opened. If it is password protected, upload an unlocked "
                             "copy.") from None
    return [page.strip() for page in text.rstrip("\n").split(_PAGE_SEPARATOR)]


def parse_upload(name, content):
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        pages = pdf_pages(content)
        for i, text in enumerate(pages, 1):
            if not text:
                raise InputError(
                    f"PDF page {i} has no readable text, even with OCR; no pages were imported."
                )
        blocks = [f"[PDF page {i}; note date unavailable]\n{text}" for i, text in enumerate(pages, 1)]
        return {
            "text": "\n\n".join(blocks),
            "notes": None,
            "notice": "PDF order preserved. Dates were not inferred from file metadata. Pages without a text "
                      "layer were read with local OCR; verify tables, layout and OCR text.",
        }
    if suffix in {".txt", ".md"}:
        return {
            "text": content.decode("utf-8-sig"),
            "notes": None,
            "notice": "Free text imported. Structured note dates are unavailable.",
        }
    if suffix != ".csv":
        raise InputError("Choose a PDF, CSV, TXT, or Markdown file.")
    frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    frame = frame.rename(columns={
        "text": "note_text", "date": "note_date", "type": "note_type", "pseudo_mrn": "patient_id"
    })
    if frame.columns.duplicated().any():
        raise InputError(
            "CSV columns are ambiguous; provide one note_text and one note_date column."
        )
    if "note_text" not in frame:
        raise InputError("CSV needs a note_text column; note_date and patient_id are optional.")
    if "patient_id" in frame and frame.patient_id.nunique() != 1:
        raise InputError("Upload notes for exactly one patient.")
    if "note_date" not in frame:
        frame["note_date"] = ""
    supplied_dates = frame.note_date.str.strip()
    dates = pd.to_datetime(supplied_dates, errors="coerce", format="mixed")
    if (dates.isna() & supplied_dates.ne("")).any():
        raise InputError(
            "Some note dates are invalid. Use YYYY-MM-DD, or leave unknown dates blank."
        )
    frame["note_date"] = dates
    frame = frame.loc[frame.note_text.str.strip().ne("")].sort_values("note_date", kind="stable")
    if frame.empty:
        raise InputError("The file contains no nonempty notes.")
    records = [
        {
            "note_text": r.note_text,
            "note_date": None if pd.isna(r.note_date) else r.note_date.isoformat(),
            **({"note_type": r.note_type} if "note_type" in frame and r.note_type.strip() else {}),
        }
        for r in frame.itertuples()
    ]
    text = "\n\n".join(
        f"[Note date: {r['note_date'] or 'unavailable'}]\n{r['note_text']}" for r in records
    )
    return {
        "text": text,
        "notes": records,
        "notice": f"{len(records)} notes imported in date order. Undated notes follow dated notes.",
    }


# A workup item that is a history and physical examination, however qualified ("with complete
# skin examination", "surveillance", "including performance status"), and names nothing else.
_HISTORY_PHYSICAL = re.compile(r"\bphysical\b|\bh\s*&\s*p\b", re.IGNORECASE)
_OTHER_WORKUP = re.compile(
    r"laborator|\blabs?\b|goals?[- ]of[- ](?:care|therapy)|shared decision|endoscop|imaging|\b(?:ct|mri|pet|ecg)\b|"
    r"ultrasound|biops|photograph|family history|genetic|counsel|\btests?\b|\bblood\b|"
    r"marker|\bscan\b|x-?ray|echocardio|transfusion|medication|\bcbc\b|metabolic|chemistr|"
    r"microglobulin|distress|\bpsa\b|patholog|smear|marrow|cytopenia|fiberoptic|"
    r"laryngoscop", re.IGNORECASE)


def is_history_and_physical(item):
    """Whether a workup item only asks for a history and physical examination.

    Such items are always done and are left out of workup checks (user instruction,
    2026-11-17). An item that also names another step ("history, physical and laboratory
    evaluation") stays; the reviews ignore its history and physical part.
    """
    name = str(item.get("name") or "")
    return bool(_HISTORY_PHYSICAL.search(name)) and not _OTHER_WORKUP.search(name)


def without_history_and_physical(space):
    """A copy of ``space`` whose diagnostic_workup leaves out history-and-physical items."""
    if not space or not space.get("diagnostic_workup"):
        return space
    return {**space, "diagnostic_workup": [
        item for item in space["diagnostic_workup"] if not is_history_and_physical(item)]}


def display_considerations(frame):
    """Clarify absent recommendation categories in the UI/report, preserving source data.

    History-and-physical workup items are left out (``is_history_and_physical``).
    """
    rows = copy.deepcopy(frame.to_dict("records"))
    for row in rows:
        row["diagnostic_workup"] = [item for item in row["diagnostic_workup"]
                                    if not is_history_and_physical(item)]
        for menu in ("diagnostic_workup", "treatment_options"):
            for option in row[menu]:
                category = " ".join(option["category"].casefold().rstrip(".").split())
                if category in {
                    "not specified in supplied evidence",
                    "not specified in the supplied evidence",
                }:
                    option["category"] = "NCCN recommendation category not specified"
    return pd.DataFrame(rows, columns=frame.columns)


def synthetic_note_sources(settings):
    """Expose named synthetic samples, without accepting client-supplied file paths."""
    sources = []
    if settings.biliary_synthetic_notes.is_file():
        sources.append({"id": "biliary", "label": "Biliary cancer · cholangiocarcinoma"})
    sources.append({
        "id": "demo",
        "label": "Configured synthetic case" if settings.synthetic_notes
                 else "Small-cell lung cancer · HF demo",
    })
    return sources


def synthetic_notes(settings, source="demo"):
    """Load a named external synthetic CSV, retaining raw notes and their dates."""
    sources = {item["id"]: item for item in synthetic_note_sources(settings)}
    if source not in sources:
        raise InputError("That synthetic case is unavailable. Refresh the page and choose a case.")
    if source == "biliary":
        path = settings.biliary_synthetic_notes
    elif settings.synthetic_notes:
        path = Path(settings.synthetic_notes)
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(
            repo_id="ksg-dfci/mmai-synthetic", repo_type="dataset",
            filename="sample_patient_notes.csv",
            revision="b1845aaed70b8106868e0369c4c013ad24b48e72",
        ))
    if path.stat().st_size > settings.max_upload:
        raise InputError("Synthetic sample exceeds the upload limit.")
    result = parse_upload("synthetic.csv", path.read_bytes())
    result["source"] = source
    result["label"] = sources[source]["label"]
    result["notice"] = f"Synthetic patient: {result['label']}. " + result["notice"]
    return result


class Inference:
    def __init__(self, settings, *, compression_cache=None, compression_settings=None,
                 workup_backups=None, extractions=None):
        self.settings = settings
        self.compression_cache = compression_cache
        self.compression_settings = compression_settings
        self.workup_backups = workup_backups or {}
        # (SessionExtractions, owner): on-demand extraction records for this session.
        self.extractions = extractions

    def _copy(self, **changes):
        values = dict(settings=self.settings, compression_cache=self.compression_cache,
                      compression_settings=self.compression_settings,
                      workup_backups=self.workup_backups, extractions=self.extractions)
        values.update(changes)
        settings = values.pop("settings")
        return type(self)(settings, **values)

    def with_settings(self, settings):
        """Bind a job to an immutable choice of workspace settings."""
        return self._copy(settings=settings)

    def with_compression_cache(self, cache, *, settings=None):
        return self._copy(compression_cache=cache,
                          compression_settings=settings or self.compression_settings)

    def with_workup_backups(self, *, full, agent):
        return self._copy(workup_backups={"full_notes": full, "agentic": agent})

    def with_extractions(self, sessions, owner):
        return self._copy(extractions=(sessions, owner))

    def extraction_cost(self, text, notes, space, levels, *, method="extractions"):
        """What an extraction-based review of this patient would send (no model calls)."""
        from .extraction_workup import (
            LEVELS, SessionExtractions, cached_query_count, cached_sql_count, estimate_cost,
            exploration_store, notes_frame,
        )

        if levels not in LEVELS:
            raise InputError("Choose document-level or document and patient-level extraction.")
        config = self.settings.extraction_config()
        frame = notes_frame(text, notes)
        sql = {}
        if method == "extraction_sql":
            from matchminer_ai.patients.workup_sql import agent_limits

            store = exploration_store(self.settings)
            distinct, cached, built = cached_sql_count(space, self.settings, config, store)
            actions = agent_limits(config).max_actions
            compile_requests = (distinct - cached) * actions
            sql = {"sql_agent": {
                "store": store.path.name,
                "patients": store.manifest["counts"].get("patients", 0),
                "database_built": built,
                "items_to_explore": distinct - cached,
                "max_actions_per_item": actions,
                "model": config.workup_sql_agent["remote"]["model_name"],
            }}
        else:
            distinct, cached = cached_query_count(space, self.settings, config)
            compile_requests = distinct - cached
        sessions, owner = self.extractions or (None, None)
        saved = sessions is not None and sessions.get(
            owner, SessionExtractions.key(frame, levels, config)) is not None
        cost = estimate_cost(frame, levels, compile_requests=compile_requests,
                             cached_queries=cached, items=len(space["diagnostic_workup"]))
        if saved:
            cost.update(document_requests=0, patient_requests=0,
                        requests=compile_requests, reused_extraction=True)
        return {**cost, **sql, "method": method, "reused_extraction": saved,
                "model": config.document_extraction["remote"]["model_name"]}

    def _compress_notes(self, text, notes, progress, config):
        from matchminer_ai.patients import compress_patient_notes

        if self.compression_cache is not None:
            saved = self.compression_cache.lookup(text, notes, config)
            if saved is not None:
                progress(f"Reusing saved compressed notes · {len(saved['notes'])} notes")
                frame = pd.DataFrame(saved["notes"])
                metadata = {**saved["metadata"], "reused": True}
                return frame.drop(columns="note_text").rename(
                    columns={"compressed_note_text": "note_text"}
                ), metadata
        frame = pd.DataFrame(notes or [{"note_text": text, "note_date": None}])
        compressed = compress_patient_notes(
            frame, config=config, thinking="on" if self.settings.reasoning else "off",
            max_concurrent_requests=8,
            progress_callback=lambda completed, total: progress(
                f"Compressing clinical notes · {completed}/{total} complete"
            ),
        )
        metadata = {
            "note_count": len(frame),
            "input_characters": int(frame.note_text.str.len().sum()),
            "output_characters": int(compressed.compressed_note_text.str.len().sum()),
            "max_concurrent_requests": 8,
            "thinking_requested": "on" if self.settings.reasoning else "off",
            "thinking": compressed.attrs.get("note_compression", {}).get(
                "thinking", "on_if_supported" if self.settings.reasoning else "off_if_supported"
            ),
            "losslessness_verified": False,
            "reused": False,
        }
        if self.compression_cache is not None:
            rows = notes or [{"note_text": text, "note_date": None}]
            self.compression_cache.save(
                text, notes, config,
                [{**row, "compressed_note_text": value}
                 for row, value in zip(rows, compressed.compressed_note_text, strict=True)],
                metadata,
            )
        # Preserve row metadata and keep the original inputs in job restore state.
        return compressed.drop(columns="note_text").rename(
            columns={"compressed_note_text": "note_text"}
        ), metadata

    def compress(self, text, notes, progress):
        """Compress and save notes independently of any clinical workflow."""
        _, metadata = self._compress_notes(text, notes, progress, self.settings.inference_config())
        return {"compression": metadata}

    def _selected_notes(self, text, notes, config, note_source, progress):
        if note_source not in {"original", "compressed"}:
            raise InputError("Choose original or saved compressed notes.")
        compression_config = (self.compression_settings.inference_config()
                              if self.compression_settings is not None else config)
        saved = (self.compression_cache.lookup(text, notes, compression_config)
                 if self.compression_cache is not None else None)
        if note_source == "original":
            return pd.DataFrame(notes) if notes else text, None
        if saved is None:
            raise InputError(
                "Saved compressed notes are unavailable for this record. "
                "Run Compress notes first, or select Original notes."
            )
        progress(f"Using saved compressed notes · {len(saved['notes'])} notes")
        frame = pd.DataFrame(saved["notes"])
        return frame.drop(columns="note_text").rename(
            columns={"compressed_note_text": "note_text"}
        ), {**saved["metadata"], "reused": True}

    def summarize(self, text, notes, progress, *, note_source="original"):
        from matchminer_ai.patients import summarize_patients

        progress("Preparing notes for serial summarization")
        config = self.settings.inference_config()
        frame, compression = self._selected_notes(text, notes, config, note_source, progress)
        if isinstance(frame, str):
            frame = pd.DataFrame([{"note_text": frame, "note_date": None}])
        frame["patient_id"] = "workspace-patient"
        # NaT stays unavailable; never substitute a made-up clinical date.
        frame["note_date"] = pd.to_datetime(frame["note_date"], errors="coerce", format="mixed")
        progress("Summarizing the cancer history · this may take several minutes")
        result = summarize_patients(frame, config=config)
        if len(result) != 1 or not str(result.iloc[0].cancer_history_summary).strip():
            raise InputError("The summarizer did not return one usable patient summary.")
        return {
            "summary": str(result.iloc[0].cancer_history_summary),
            "exclusion_evidence": str(
                result.iloc[0].get("general_exclusion_criteria_evidence", "")
            ),
            "compression": compression,
        }

    def review_workup(
        self, text, notes, space, progress, *, method="full_notes",
        patient_summary=None, note_source="original", extraction_levels="document",
    ):
        from matchminer_ai.patients import (
            review_patient_workup,
            review_patient_workup_with_note_search,
        )

        if method in {"extractions", "extraction_sql"}:
            from .extraction_workup import patient_review

            if note_source != "original":
                raise InputError(
                    "Structured extraction quotes the original notes; select Original notes.")
            sessions, owner = self.extractions or (None, None)
            return patient_review(
                text, notes, space, settings=self.settings,
                config=self.settings.extraction_config(), sessions=sessions, owner=owner,
                levels=extraction_levels, progress=progress,
                method="extraction_sql" if method == "extraction_sql" else "extraction_queries",
            )
        config = self.settings.inference_config()
        options = {}
        if method not in {"full_notes", "agentic"}:
            raise InputError("Choose full-note review, agentic search, extraction queries or the "
                             "SQL agent.")
        raw, compression = self._selected_notes(text, notes, config, note_source, progress)
        if method == "agentic":
            review = review_patient_workup_with_note_search
            # All workup questions are eligible to run at once in this job.
            # Summarization and full-note jobs retain their one-request limit.
            config.remote["max_concurrent_requests"] = len(space["diagnostic_workup"])
            options.update(
                history=text if notes and note_source == "original" else None,
                patient_summary=patient_summary,
                # Inherit this task's known-model switch; avoid the public
                # agent's off default overriding the user's thinking choice.
                thinking=None,
            )
        else:
            review = review_patient_workup
        backup = self.workup_backups.get(method)
        if backup is not None and backup.backup_enabled:
            backup_config = backup.inference_config()
            if method == "agentic":
                backup_config.remote["max_concurrent_requests"] = len(space["diagnostic_workup"])
            options.update(backup_config=backup_config,
                           max_consecutive_failures=backup.failure_threshold)
        review_progress = progress
        if compression is not None:
            def review_progress(message):
                progress(message.replace("raw-note", "compressed-note")
                         .replace("raw notes", "compressed notes"))
        result = review(
            raw, space["diagnostic_workup"], config=config,
            progress_callback=review_progress,
            population_context=space.get("cohort_key_criteria") or space.get("clinical_space_summary"),
            **options,
        )
        if compression is not None:
            result["metadata"].update(
                scope=("searched_compressed_excerpts" if method == "agentic"
                       else "all_supplied_compressed_notes"),
                evidence_source="llm_compressed_notes", compression=compression,
            )
            result["notice"] += (
                " This review uses LLM-compressed notes. Losslessness is unverified. "
                "Quoted excerpts match the compressed text and may differ from the "
                "original notes; confirm findings against the original record."
            )
        return result

    def retrieve(self, summary, paths, top_n, candidate_k, progress):
        import torch
        from matchminer_ai.matching import (
            retrieve_guideline_considerations,
            write_guideline_considerations_report,
        )

        torch.set_num_threads(8)
        patients = pd.DataFrame(
            [{"patient_id": "workspace-patient", "cancer_history_summary": summary}]
        )
        result, metadata = retrieve_guideline_considerations(
            patients,
            paths,
            top_n=top_n,
            candidate_k=candidate_k,
            config=self.settings.inference_config(),
            embedding_cache_dir=self.settings.cache / "guideline_embeddings",
            return_metadata=True,
            progress_callback=progress,
        )
        result = display_considerations(result)
        with TemporaryDirectory(prefix="guideline-fetcher-report-") as tmp:
            report = write_guideline_considerations_report(
                result,
                patients,
                Path(tmp) / "considerations.md",
                metadata=metadata,
            ).read_text()
        return {
            "spaces": json.loads(result.to_json(orient="records", force_ascii=False)),
            "summary": summary,
            "report": report,
            "provenance": {
                "catalog": metadata.get("catalog", {}),
                "retrieval": metadata.get("retrieval", {}),
                "models": metadata.get("model_metadata", {}),
                "guideline_embedding_cache": metadata.get("guideline_embedding_cache", {}),
            },
        }
