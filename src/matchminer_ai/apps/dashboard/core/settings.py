"""Where the app reads its synthetic data and models, and how it runs them.

Every path defaults to the synthetic training run (``synthetic_run``: the site config's ``dashboard.synthetic_run``,
else ``no_phi/synthetic_run`` under the data root; see ``matchminer_ai.site``).
``matchminer_ai.apps.dashboard.settings.DashSettings`` extends these settings with the dashboard's own and parses the
command line and ``MMAI_DASH_*`` variables (``--synthetic-run`` moves the run). The dataset is read-only; the app
writes only to ``cache_dir``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys

from matchminer_ai.site import data_root, load_site_config


def default_synthetic_run() -> Path:
    """The synthetic run the dashboard's patients, TrialSpace rounds and decision model come from."""
    value = load_site_config().get("dashboard.synthetic_run")
    return Path(value).expanduser() if value else data_root() / "no_phi" / "synthetic_run"


def default_embedding_python() -> Path:
    """The Python that runs the query encoder (site config ``dashboard.embedding_python``, else this one)."""
    value = load_site_config().get("dashboard.embedding_python")
    return Path(value).expanduser() if value else Path(sys.executable)


def default_embedding_caches() -> tuple[Path, ...]:
    """Hugging Face caches searched for the encoder's weights (site config ``dashboard.embedding_caches``, else the
    standard cache)."""
    values = load_site_config().get("dashboard.embedding_caches")
    if values:
        return tuple(Path(v).expanduser() for v in values)
    from huggingface_hub.constants import HF_HUB_CACHE

    return (Path(HF_HUB_CACHE),)

# Relative to the data root. The run's own decision model (train_from_summaries.py --models-dir trialspace/models)
# is used once its stage receipt shows it complete; until then the fit-2 smoke checkpoint (user, 2026-11-08).
RUN_DECISION_MODEL = "trialspace/models/decision_model"
DEFAULT_DECISION_MODEL = "smoke_tests/round1_labels_fit2_20261107/decision_fit/model"
RECEIPT_FORMAT = "mmai-stage-receipt-v1"
BASE_DECISION_MODEL = "LiquidAI/d1-2B"
BASE_DECISION_REVISION = "652f920e544ea37f81e4b0ce89b81d4ed0da90b1"
# EmbeddingGemma 2 needs sentence-transformers 6. The query encoder runs in its own process, by default with this
# Python (checked 2026-11-20: the app's environment with sentence-transformers 6.1 embeds a round-3 query exactly
# as the former separate environment did); a site may name another environment and Hub caches
# (dashboard.embedding_python, .embedding_caches).


@dataclass
class Settings:
    synthetic_run: Path = field(default_factory=default_synthetic_run)
    cache_dir: Path = Path("~/.cache/mmai-app").expanduser()
    host: str = "127.0.0.1"
    port: int = 8781
    # The decision model: a local checkpoint directory, or a Hub id pinned by decision_revision.
    decision_model: str | None = None  # None: the run's decision model, else fit 2, else base d1-2B
    decision_revision: str = BASE_DECISION_REVISION
    device: str = "cpu"
    dtype: str = "bfloat16"
    threads: int = 32
    decision_batch_size: int = 8
    decision_token_budget: int = 16384
    # Query embedding (custom texts only; corpus entries reuse the round's stored vectors).
    embedding_python: Path = field(default_factory=default_embedding_python)
    embedding_caches: tuple[Path, ...] = field(default_factory=default_embedding_caches)
    embedding_device: str = "cpu"
    allow_downloads: bool = False
    default_k: int = 50
    max_k: int = 500
    round_poll_seconds: float = 60  # how often a running app looks for a later TrialSpace round; 0: never
    max_score_n: int = 100
    paths: dict = field(default_factory=dict)

    def __post_init__(self):
        self.synthetic_run = Path(self.synthetic_run).expanduser()
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.embedding_python = Path(self.embedding_python).expanduser()
        self.embedding_caches = tuple(Path(p).expanduser() for p in self.embedding_caches)
        L = self.synthetic_run
        defaults = {
            "corpus": L / "trialspace" / "trialspace_corpus",
            "trialspace": L / "trialspace",
            "patient_metadata": L / "patient_summaries" / "synthetic_patient_metadata.parquet",
            "notes": L / "patient_summaries" / "all_synthetic_notes.parquet",
        }
        self.paths = {key: Path(self.paths.get(key, value)) for key, value in defaults.items()}

    def path(self, key: str) -> Path:
        return self.paths[key]

    def resolved_decision_model(self) -> tuple[str, str | None, str]:
        """(model name or directory, Hub revision or None, why it was chosen)."""
        if self.decision_model:
            local = Path(self.decision_model).expanduser()
            if local.is_dir():
                return str(local), None, "configured checkpoint directory"
            return self.decision_model, self.decision_revision, "configured Hub model"
        run = self.synthetic_run / RUN_DECISION_MODEL
        pending = complete_decision_model(run)
        if pending is None:
            return str(run), None, f"the run's trained decision model ({RUN_DECISION_MODEL})"
        smoke = self.synthetic_run / DEFAULT_DECISION_MODEL
        if (smoke / "config.json").exists():
            return str(smoke), None, f"fine-tuned smoke checkpoint (fit 2); {RUN_DECISION_MODEL}: {pending}"
        return BASE_DECISION_MODEL, BASE_DECISION_REVISION, f"fallback: {smoke} not found, using base d1-2B"

    def describe(self) -> dict:
        value = asdict(self)
        value["paths"] = {k: str(v) for k, v in self.paths.items()}
        return {k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if isinstance(v, tuple) else v)
                for k, v in value.items()}


def complete_decision_model(directory: Path) -> str | None:
    """None when ``train_decision_model.py train`` finished in ``directory`` (its stage receipt is written last and
    every file it lists has the listed size), else why not."""
    path = Path(directory) / "stage_receipt.json"
    if not path.exists():
        return "not trained yet" if not Path(directory).exists() else "training not finished (no stage receipt)"
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return f"unreadable stage receipt ({type(exc).__name__})"
    if receipt.get("format") != RECEIPT_FORMAT or receipt.get("kind") != "decision-model":
        return "stage receipt is not a decision-model receipt"
    outputs = receipt.get("outputs") or {}
    for name in ("decision_contract.json", "model.safetensors"):
        if name not in outputs:
            return f"stage receipt does not list {name}"
    for name, size in outputs.items():
        file = Path(directory) / name
        if not file.is_file() or file.stat().st_size != size:
            return f"{name} is missing or differs in size from its stage receipt (still copying?)"
    return None

