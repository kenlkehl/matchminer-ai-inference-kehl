"""Where the dashboard reads its trial data, synthetic patients and models, and where it keeps its store.

Patients, the TrialSpace round, the encoder and the decision model are those of ``core.settings`` (the synthetic
n=5000 run), except two decision-model defaults (user, 2026-11-12): the newest interim export of the
run's decision model, taken from a Trainer checkpoint while it trains, comes before the fit-2 checkpoint, and the
model runs on ``cuda:0`` when CUDA is available. The trials are the OA trials being extracted in ``trial_data``
with the OA DuckDB copy's trial table (``pipeline.trial_table``). The store (``dashboard.json``), the real cohorts'
vectors and the score cache hold confidential protocol-derived text, so they live in ``store_dir`` beside the trial
data (owner only), not in the repository or the home directory. Each trial's drugs and mechanisms come from the
GoodOption catalog (``good_options_catalog``), read in place.

Defaults come from the site config's ``dashboard`` section, else folders under the data root's
``phi/real_time`` (``matchminer_ai.site``); ``--site-config`` and ``--data-root`` name them.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

from matchminer_ai.apps.common.security import LOOPBACK_HOSTS
from matchminer_ai.site import DATA_ROOT_ENV, SITE_CONFIG_ENV, data_root, load_site_config

from .core.settings import RUN_DECISION_MODEL, Settings, complete_decision_model
from .llm import DEFAULT_BACKEND, NEMOTRON, default_endpoint

# The bundled trial-group rules: a small example. A site's own rules: the site config's dashboard.grouping.
DEFAULT_GROUPING = Path(__file__).with_name("grouping.yaml")


def site_default(key: str, relative: str) -> Path:
    """The site config's ``dashboard.<key>``, else ``relative`` under the data root's ``phi/real_time``."""
    value = load_site_config().get(f"dashboard.{key}")
    return Path(value).expanduser() if value else data_root() / "phi" / "real_time" / relative


def default_grouping() -> Path:
    value = load_site_config().get("dashboard.grouping")
    return Path(value).expanduser() if value else DEFAULT_GROUPING
ENV_PREFIX = "MMAI_DASH_"
ACCESS_TOKEN_ENV = "MMAI_DASH_ACCESS_TOKEN"
ACCESS_TOKEN_FILE = "~/.config/mmai-dashboard/access_token"
# Interim exports of the run's decision model while it trains (relative to the data root): step-<N>/, each with
# interim.json written last, listing every file's size. The run's final model replaces them once its stage receipt is
# complete (core.settings.complete_decision_model). Since 2026-11-15 (user) these are the relabeled retrain's
# exports; the old-label run's (stopped at step 31,181 on 2026-11-13) stay in trialspace/models/decision_model_interim.
INTERIM_DECISION_MODELS = "trialspace/models/decision_model_interim_relabel_20261114"
INTERIM_FORMAT = "mmai-decision-interim-v1"
DEFAULT_DEVICE = "cuda:0"  # the decision model's default when CUDA is available (user, 2026-11-12); else cpu
AUTO_DEVICE = "auto"


@dataclass
class DashSettings(Settings):
    port: int = 8781
    cache_dir: Path | None = None  # default: <store_dir>/cache
    trial_data: Path = field(default_factory=lambda: site_default("trial_data", "trial_data"))
    # The newest OA snapshot (a glob).
    oa_db: Path = field(default_factory=lambda: site_default("oa_db", "oa_db_*.duckdb"))
    store_dir: Path = field(default_factory=lambda: site_default("store_dir", "dashboard"))
    grouping: Path = field(default_factory=default_grouping)
    # Trials' drugs (public trial and drug information only, built from NCT IDs and drug names, read in place);
    # a missing catalog: no drugs.
    good_options_catalog: Path = field(default_factory=lambda: site_default("good_options_catalog",
                                                                            "good_options_catalog/catalog"))
    # The LLM checks' endpoint (llm.py): by default the Vertex profile Gemini 4 Hydrogen (user, 2026-11-17); with
    # llm_backend "openai", an OpenAI-compatible server (llm_endpoint, default the site config's
    # endpoints.teacher.url; "none" turns the LLM checks off). Changeable on Data and Models for the server's session.
    llm_backend: str = DEFAULT_BACKEND
    llm_endpoint: str = field(default_factory=default_endpoint)
    llm_model: str = NEMOTRON
    llm_tokenizer: str = ""
    rebuild_poll_seconds: float = 30  # how often new extraction results are looked for; 0: never
    auto_score_n: int = 10  # pairs the result pages score with the decision model as they open
    default_k: int = 50
    device: str = AUTO_DEVICE  # the decision model: cuda:0 when CUDA is available, else cpu (device_notice says so)
    # Host headers accepted (loopback names; a forwarded port's Host is localhost or 127.0.0.1).
    allowed_hosts: tuple[str, ...] = tuple(LOOPBACK_HOSTS)
    # Require an access token (MMAI_DASH_ACCESS_TOKEN, else a private token file) to keep out other accounts on
    # this machine, which can reach a loopback port.
    require_token: bool = False
    device_notice: str | None = None

    def __post_init__(self):
        if self.device in (None, "", AUTO_DEVICE):
            self.device, self.device_notice = default_device()
        self.trial_data = Path(self.trial_data).expanduser()
        self.oa_db = Path(self.oa_db).expanduser()
        self.store_dir = Path(self.store_dir).expanduser()
        self.grouping = Path(self.grouping).expanduser()
        self.good_options_catalog = Path(self.good_options_catalog).expanduser()
        if self.cache_dir is None:
            self.cache_dir = self.store_dir / "cache"
        super().__post_init__()

    @property
    def store_path(self) -> Path:
        return self.store_dir / "dashboard.json"

    def resolved_decision_model(self) -> tuple[str, str | None, str]:
        """(model name or directory, Hub revision or None, why it was chosen): an explicit --decision-model, else the
        run's trained decision model once its stage receipt is complete, else the newest complete interim export
        (``INTERIM_DECISION_MODELS/step-<N>``), else fit 2, else base d1-2B (as ``core.settings``,
        with the interim export inserted before fit 2)."""
        if self.decision_model:
            return super().resolved_decision_model()
        pending = complete_decision_model(self.synthetic_run / RUN_DECISION_MODEL)
        if pending is None:
            return super().resolved_decision_model()
        found = newest_interim_decision_model(self.synthetic_run / INTERIM_DECISION_MODELS)
        if found is None:
            return super().resolved_decision_model()
        path, info = found
        return str(path), None, (f"INTERIM checkpoint of the run's decision model, not the final model (step "
                                 f"{info['checkpoint_step']:,} of {info['max_steps']:,}, {info['epoch']:.3f} epoch); "
                                 f"{RUN_DECISION_MODEL}: {pending}")


def default_device() -> tuple[str, str | None]:
    """(device, notice): cuda:0 when CUDA is available, else cpu and why."""
    try:
        available = cuda_available()
    except Exception as exc:  # a broken torch or driver: the model still runs on CPU
        return "cpu", (f"Could not check for CUDA ({type(exc).__name__}): the decision model runs on cpu, "
                       f"not {DEFAULT_DEVICE}")
    if available:
        return DEFAULT_DEVICE, None
    return "cpu", f"CUDA is not available: the decision model runs on cpu, not {DEFAULT_DEVICE}"


def cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def complete_interim_decision_model(directory: Path) -> dict | None:
    """interim.json of a complete interim export (every file it lists has its size), else None."""
    path = Path(directory) / "interim.json"
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    outputs = info.get("outputs") or {}
    if (info.get("format") != INTERIM_FORMAT or info.get("interim") is not True
            or not isinstance(info.get("checkpoint_step"), int) or not isinstance(info.get("max_steps"), int)
            or not all(name in outputs for name in ("config.json", "decision_contract.json", "model.safetensors"))):
        return None
    for name, size in outputs.items():
        file = Path(directory) / name
        if not file.is_file() or file.stat().st_size != size:
            return None
    return info


def newest_interim_decision_model(directory: Path) -> tuple[Path, dict] | None:
    """The complete ``step-<N>`` export with the highest N, with its interim.json."""
    steps = []
    for child in Path(directory).glob("step-*") if Path(directory).is_dir() else ():
        number = child.name[len("step-"):]
        if child.is_dir() and number.isdigit():
            steps.append((int(number), child))
    for _, child in sorted(steps, reverse=True):
        info = complete_interim_decision_model(child)
        if info is not None:
            return child, info
    return None


def _env(name: str):
    return os.environ.get(ENV_PREFIX + name.upper())


def parse_settings(argv: list[str] | None = None) -> DashSettings:
    """Settings from the command line, then MMAI_DASH_* variables, then the defaults."""
    parser = argparse.ArgumentParser(prog="mmai-dashboard",
                                     description="MatchMiner-AI dashboard over the OA trials (research use)")
    parser.add_argument("command", nargs="?", default="serve", choices=("serve", "build"),
                        help="serve (default) or build the dashboard store once and exit")
    parser.add_argument("--site-config", help=f"site config YAML (default ${SITE_CONFIG_ENV})")
    parser.add_argument("--data-root", help=f"data root holding phi/ and no_phi/ (default ${DATA_ROOT_ENV} or the "
                        "site config's paths.data_root)")
    parser.add_argument("--trial-data", type=Path, help="process_oa_trials.py output (default: site config "
                        "dashboard.trial_data, else phi/real_time/trial_data under the data root)")
    parser.add_argument("--oa-db", type=Path, help="OA DuckDB copy, or a glob for its newest (default: dashboard.oa_db, "
                        "else phi/real_time/oa_db_*.duckdb)")
    parser.add_argument("--store-dir", type=Path, help="dashboard store, vectors and caches (default: "
                        "dashboard.store_dir, else phi/real_time/dashboard)")
    parser.add_argument("--grouping", type=Path, help="trial-group rules (default: the site config's dashboard.grouping, else the bundled rules)")
    parser.add_argument("--good-options-catalog", type=Path, help="GoodOption catalog directory with the trials' drugs "
                        "(default: dashboard.good_options_catalog; missing: no drugs)")
    parser.add_argument("--rebuild-poll-seconds", type=float, help="how often to look for new extraction results (default 30)")
    parser.add_argument("--llm-backend", help=f"LLM checks: a Vertex profile (default {DEFAULT_BACKEND}; also "
                        "gemini_vertex, gemma_vertex) or openai (an OpenAI-compatible URL, --llm-endpoint)")
    parser.add_argument("--llm-endpoint", help="with --llm-backend openai: the endpoint (default: the site config's "
                        "endpoints.teacher.url; none: the LLM checks are off)")
    parser.add_argument("--llm-model", help=f"the endpoint's model (default {NEMOTRON})")
    parser.add_argument("--llm-tokenizer", help="tokenizer for local prompt budgets (default: the model)")
    parser.add_argument("--synthetic-run", type=Path, help="synthetic run (patients, TrialSpace rounds, decision model; "
                        "default: dashboard.synthetic_run, else no_phi/synthetic_run under the data root)")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--host")
    parser.add_argument("--allowed-hosts", help="comma-separated Host header names to accept (default: "
                        f"{','.join(LOOPBACK_HOSTS)})")
    parser.add_argument("--require-token", action="store_true", help="require an access token "
                        f"(MMAI_DASH_ACCESS_TOKEN, else {ACCESS_TOKEN_FILE}, created on first use)")
    parser.add_argument("--port", type=int)
    parser.add_argument("--decision-model", help="checkpoint directory or Hub id (default: the run's trained decision "
                        "model once complete, else its newest interim export, else fit 2, else base d1-2B)")
    parser.add_argument("--decision-revision")
    parser.add_argument("--device", help="decision-model device, e.g. cpu or cuda:1 (default cuda:0 when CUDA is "
                                         "available, else cpu)")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--embedding-python", type=Path)
    parser.add_argument("--embedding-cache", type=Path, action="append")
    parser.add_argument("--embedding-device", help="TrialSpace encoder device for the trial cohorts (default cpu)")
    parser.add_argument("--auto-score-n", type=int, help="pairs scored automatically per search (default 10)")
    parser.add_argument("--round-poll-seconds", type=float)
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args(argv)
    # Before any default reads them.
    if args.site_config:
        os.environ[SITE_CONFIG_ENV] = args.site_config
    if args.data_root:
        os.environ[DATA_ROOT_ENV] = args.data_root
    names = {"trial_data": Path, "oa_db": Path, "store_dir": Path, "grouping": Path, "good_options_catalog": Path,
             "rebuild_poll_seconds": float, "llm_backend": str, "llm_endpoint": str, "llm_model": str,
             "llm_tokenizer": str,
             "synthetic_run": Path, "cache_dir": Path, "host": str, "port": int, "decision_model": str,
             "decision_revision": str, "device": str, "dtype": str, "threads": int, "embedding_python": Path,
             "embedding_device": str, "auto_score_n": int, "round_poll_seconds": float}
    values: dict = {}
    for name, kind in names.items():
        raw = getattr(args, name)
        if raw is None and _env(name) is not None:
            raw = kind(_env(name))
        if raw is not None:
            values[name] = raw
    if args.embedding_cache:
        values["embedding_caches"] = tuple(args.embedding_cache)
    values["allow_downloads"] = bool(args.allow_downloads or (_env("allow_downloads") or "").lower() in ("1", "true"))
    values["require_token"] = bool(args.require_token or (_env("require_token") or "").lower() in ("1", "true"))
    hosts = args.allowed_hosts or _env("allowed_hosts")
    if hosts:
        values["allowed_hosts"] = tuple(h.strip() for h in hosts.split(",") if h.strip())
    settings = DashSettings(**values)
    settings.command = args.command
    return settings
