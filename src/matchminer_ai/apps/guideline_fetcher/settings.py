"""Server-side configuration; no clinical data or model weights live in this package.

One app, two deployments. The default is the non-PHI app. PHI mode (`mmai-guideline-fetcher --phi`,
Settings(phi=True)) reads real patient records from an operator-selected DuckDB file
or preprocessed population cache, binds only to loopback, requires an access token
and keeps its own cache under the data root's phi/ folder.

Paths default to folders under the data root (``matchminer_ai.site.data_root``: ``--data-root``,
``MMAI_DATA_ROOT`` or the site config's ``paths.data_root``). Institution-specific values come from the site
config (``matchminer_ai.site``), under ``guideline_fetcher``, ``vertex`` and ``databricks``; see the
site-config reference in the docs.
"""

import os
import re
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from matchminer_ai.site import data_root, load_site_config

ROOT = Path(__file__).resolve().parent
SITE = load_site_config()
DEFAULT_PORT = 8034
# PHI mode only.
PHI_PORT = 8044


def data_path(relative):
    """``relative`` under the data root (which has no default; see ``matchminer_ai.site``)."""
    return data_root() / relative


def site_path(key, relative):
    """The site config's ``guideline_fetcher.<key>``, else ``relative`` under the data root."""
    value = SITE.get(f"guideline_fetcher.{key}")
    return Path(value).expanduser() if value else data_path(relative)


# Each mode keeps model metadata and guideline embeddings on its own side of the data root.
NO_PHI_CACHE = "no_phi/guideline_fetcher_cache"
PHI_CACHE = "phi/guideline_fetcher_phi_cache"


def default_population_duckdb():
    """PHI mode's real-patient database copy (site config ``guideline_fetcher.population_duckdb``)."""
    return site_path("population_duckdb", "phi/guideline_fetcher_population.duckdb")


def default_population_cache():
    """Built by the preprocess_population command; contains PHI, so it stays beside the database."""
    duckdb = default_population_duckdb()
    return duckdb.parent / "guideline_population_cache" / duckdb.stem


def default_duckdb_roots():
    """Folders PHI mode may browse for DuckDB files (site config ``guideline_fetcher.duckdb_roots``)."""
    roots = SITE.get("guideline_fetcher.duckdb_roots")
    return os.pathsep.join(str(Path(r).expanduser()) for r in roots) if roots else str(data_root())


def default_extraction_records(phi):
    """Structured extraction records built offline by the preprocess_extractions command. PHI mode's sit beside its
    population cache; the non-PHI app's hold synthetic notes only."""
    if phi:
        duckdb = default_population_duckdb()
        return duckdb.parent / "guideline_extraction_records" / duckdb.stem
    return data_path("no_phi/guideline_fetcher_extractions/synthetic")


DEFAULT_ACCESS_TOKEN_FILE = "~/.config/guideline-fetcher-phi/access_token"
MIN_ACCESS_TOKEN_LENGTH = 32
# The Google Cloud project of the Vertex (Agent Platform) profiles: site config ``vertex.project``.
GEMINI_PROJECT = SITE.get("vertex.project") or ""
GEMINI_PROFILE = {
    "endpoint": f"https://aiplatform.googleapis.com/v1/projects/{GEMINI_PROJECT}/locations/global/endpoints/openapi",
    "model": "google/gemini-3.8-flash",
    # Local chunk boundaries only; this is not Gemini's tokenizer or weights.
    "tokenizer": "Qwen/Qwen3-0.6B",
}
GEMMA_PROFILE = {
    "endpoint": GEMINI_PROFILE["endpoint"],
    "model": "google/gemma-4-26b-a4b-it-maas",
    # Chunk boundaries only; hosted prompt budgets use conservative UTF-8 bytes.
    "tokenizer": GEMINI_PROFILE["tokenizer"],
}
# Gemini 4 Hydrogen (GA 2026-11-12) is the default since 2026-11-14 (user: the Gemma 4 26B MaaS endpoint "is very
# janky"). Same fixed endpoint and project; 1,048,576-token window shared by input and output, up to 65,536 output
# tokens per request.
GEMINI4_PROFILE = {
    "endpoint": GEMINI_PROFILE["endpoint"],
    "model": "google/gemini-4-hydrogen",
    "tokenizer": GEMINI_PROFILE["tokenizer"],
}
VERTEX_PROFILES = {"gemini4_vertex": GEMINI4_PROFILE, "gemini_vertex": GEMINI_PROFILE, "gemma_vertex": GEMMA_PROFILE}
VERTEX_LABELS = {"gemini4_vertex": "Gemini 4 Hydrogen", "gemini_vertex": "Gemini 3.8 Flash",
                 "gemma_vertex": "Gemma 4 26B MaaS"}
# Hosted Gemini models manage their own sampling and thinking (reasoning_effort, no chat-template switches).
GEMINI_BACKENDS = frozenset({"gemini4_vertex", "gemini_vertex"})
# Claude through the Databricks Unity AI Gateway's Anthropic Messages route, for the SQL agent only (2026-11-19, user
# instruction; Gemini stays available). The server fixes the gateway URL. The model is the full three-part endpoint
# name from the gateway onboarding email (GUIDELINE_DATABRICKS_SONNET_ENDPOINT fills it in). Access tokens come
# from the Databricks CLI login profile (`databricks auth login --host <host> --profile unity-gateway`).
DATABRICKS_HOST = (os.getenv("GUIDELINE_DATABRICKS_HOST") or SITE.get("databricks.host") or "").rstrip("/")
DATABRICKS_PROFILE = (os.getenv("GUIDELINE_DATABRICKS_PROFILE") or SITE.get("databricks.profile")
                      or "unity-gateway")
CLAUDE_PROFILE = {
    "endpoint": DATABRICKS_HOST + "/ai-gateway/anthropic",
    "model": (os.getenv("GUIDELINE_DATABRICKS_SONNET_ENDPOINT") or SITE.get("databricks.sonnet_endpoint")
              or "").strip(),
    # Unused: hosted prompt budgets use conservative UTF-8 bytes.
    "tokenizer": GEMINI_PROFILE["tokenizer"],
}
CLAUDE_BACKENDS = {"claude_databricks": CLAUDE_PROFILE}
CLAUDE_LABELS = {"claude_databricks": "Claude · Databricks AI Gateway"}
# Sonnet 5.5's default. On two colon items (2026-11-19) medium took the same time (~25 s per item) but skipped
# reference-date bounds and planned-only reasons that high effort checked.
CLAUDE_EFFORT = "high"
CLAUDE_CONTEXT = 1_000_000
# Gateway endpoint names have three dot-separated parts, the last ending in the Claude model ID.
CLAUDE_ENDPOINT_NAME = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*claude-[a-z0-9-]+")
# Backends that serve only this task.
SQL_ONLY_BACKENDS = frozenset(CLAUDE_BACKENDS)
SQL_TASK = "workup_sql"
LLM_TASKS = {
    "summarize": "Patient summarization",
    "compress": "Note compression",
    "workup": "Guideline concordance checks",
    SQL_TASK: "Guideline concordance checks: SQL agent",
    "workup_full_backup": "Guideline concordance checks: Full note backup",
    "workup_agent_backup": "Guideline concordance checks: Agent backup",
}


def normalize_endpoint(value):
    """Accept a server address or an OpenAI-compatible API base URL."""
    value = value.strip()
    if not value or any(ch.isspace() or ord(ch) < 32 for ch in value):
        raise ValueError("Enter an HTTP or HTTPS LLM server address.")
    if "://" not in value:
        value = "http://" + value
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"} and parsed.hostname
            and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
        )
        parsed.port  # Validate malformed/out-of-range ports before making a request.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Use an HTTP(S) server URL without credentials, query, or fragment.")
    path = parsed.path.rstrip("/") or "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def mode_side_error(phi, path):
    """Why ``path`` is on the other mode's side of data/ (None when it is not).

    PHI-mode data and caches never sit in a no_phi folder, and the non-PHI app
    never reads or writes a phi folder.
    """
    parts = Path(path).expanduser().resolve().parts
    if phi and "no_phi" in parts:
        return "PHI mode keeps patient records out of no_phi folders."
    if not phi and "phi" in parts:
        return "The non-PHI app never uses a phi folder."
    return None


def access_token_file():
    return Path(os.getenv("GUIDELINE_ACCESS_TOKEN_FILE", DEFAULT_ACCESS_TOKEN_FILE)).expanduser()


def load_access_token():
    """PHI mode: GUIDELINE_ACCESS_TOKEN, else a private token file created on first use.

    The loopback bind keeps other machines out; the token keeps out other
    accounts on this machine. Never pass the token on a command line (ps shows it).
    """
    token = os.getenv("GUIDELINE_ACCESS_TOKEN", "").strip()
    if not token:
        path = access_token_file()
        if not path.exists():
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "w") as file:
                    file.write(secrets.token_urlsafe(32) + "\n")
            except FileExistsError:
                pass
        if path.stat().st_mode & 0o077:
            raise PermissionError(f"Restrict the access token file to your account: chmod 600 {path}")
        token = path.read_text().strip()
    if len(token) < MIN_ACCESS_TOKEN_LENGTH or any(ch.isspace() for ch in token):
        raise ValueError(
            f"The access token needs at least {MIN_ACCESS_TOKEN_LENGTH} characters and no spaces."
        )
    return token


def default_device():
    """Use CUDA when PyTorch can access it; honor an operator's explicit choice."""
    requested = os.getenv("MMAI_DEVICE", "auto").strip() or "auto"
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Settings:
    # The PHI deployment (mmai-guideline-fetcher --phi); see the module docstring.
    phi: bool = False
    llm_backend: str = field(
        default_factory=lambda: os.getenv("MMAI_LLM_BACKEND", "gemini4_vertex")
    )
    # None takes GUIDELINE_ALLOWED_HOSTS, else the mode's default (see __post_init__).
    allowed_hosts: list[str] | None = None
    # Both modes use the same finalized NCCN catalogs.
    catalog_root: Path = field(
        default_factory=lambda: Path(
            os.getenv("GUIDELINE_CATALOG_ROOT") or str(data_path("no_phi/space_paradigms_nccn"))
        ).resolve()
    )
    endpoint: str = field(
        default_factory=lambda: os.getenv("MMAI_ENDPOINT", "http://localhost:8000/v1").rstrip(
            "/"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv("MMAI_LLM_MODEL", "Qwen/Qwen3.8-27B-FP8")
    )
    tokenizer: str = field(
        default_factory=lambda: os.getenv("MMAI_TOKENIZER", "Qwen/Qwen3.8-27B-FP8")
    )
    device: str = field(default_factory=default_device)
    # None takes GUIDELINE_CACHE, else NO_PHI_CACHE or, in PHI mode, PHI_CACHE.
    cache: Path | None = None
    example: Path = field(
        default_factory=lambda: Path(
            os.getenv("GUIDELINE_SYNTHETIC_EXAMPLE")
            or str(data_path("no_phi/guideline_fetcher_examples/patient_summary.json"))
        )
    )
    # The non-PHI synthetic population sample.
    population_summaries: Path = field(default_factory=lambda: Path(
        os.getenv("GUIDELINE_POPULATION_SUMMARIES")
        or site_path("population_summaries", "no_phi/patient_summaries.parquet")))
    population_notes: Path = field(default_factory=lambda: Path(
        os.getenv("GUIDELINE_POPULATION_NOTES")
        or site_path("population_notes", "no_phi/all_synthetic_notes.parquet")))
    # PHI mode only: the real-patient database, its preprocessed cache and browsable folders.
    population_duckdb: Path = field(default_factory=lambda: Path(
        os.getenv("GUIDELINE_POPULATION_DUCKDB") or default_population_duckdb()))
    population_cache: Path = field(default_factory=lambda: Path(
        os.getenv("GUIDELINE_POPULATION_CACHE") or default_population_cache()))
    duckdb_roots: list[Path] = field(default_factory=lambda: [
        Path(p) for p in (os.getenv("GUIDELINE_DUCKDB_ROOTS") or default_duckdb_roots()).split(os.pathsep)
        if p.strip()
    ])
    # None takes GUIDELINE_EXTRACTION_RECORDS, else the mode's default (see __post_init__).
    extraction_records: Path | None = None
    population_sample_size: int = field(default_factory=lambda: int(os.getenv("GUIDELINE_POPULATION_SAMPLE_SIZE", "1000")))
    population_seed: int = field(default_factory=lambda: int(os.getenv("GUIDELINE_POPULATION_SEED", "23")))
    population_upload_limit: int = 512 * 1024 * 1024
    population_archive_limit: int = field(default_factory=lambda: int(os.getenv(
        "GUIDELINE_POPULATION_ARCHIVE_MAX_BYTES", str(8 * 1024**3))))
    ttl: int = 3600
    synthetic_notes: str | None = field(
        default_factory=lambda: os.getenv("GUIDELINE_SYNTHETIC_NOTES")
    )
    biliary_synthetic_notes: Path = field(
        default_factory=lambda: Path(
            os.getenv("GUIDELINE_BILIARY_SYNTHETIC_NOTES")
            or str(data_path("no_phi/guideline_fetcher_examples/biliary_patient_notes.csv"))
        )
    )
    max_upload: int = 20 * 1024 * 1024
    reasoning: bool = False
    backup_enabled: bool = False
    failure_threshold: int = 3
    task_connections: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.allowed_hosts is None:
            # PHI mode accepts loopback names only, rejecting DNS-rebinding Host headers.
            default = "127.0.0.1,localhost" if self.phi else "*"
            self.allowed_hosts = [
                host.strip()
                for host in os.getenv("GUIDELINE_ALLOWED_HOSTS", default).split(",")
                if host.strip()
            ]
        if self.cache is None:
            self.cache = Path(os.getenv("GUIDELINE_CACHE") or data_path(PHI_CACHE if self.phi else NO_PHI_CACHE))
        # A cache override must stay on the mode's own side of data/.
        parts = Path(self.cache).expanduser().resolve().parts
        if self.phi and "no_phi" in parts:
            raise ValueError("PHI mode never writes to a no_phi folder. Choose a PHI GUIDELINE_CACHE.")
        if not self.phi and "phi" in parts:
            raise ValueError("The non-PHI app never writes to a phi folder. Choose another GUIDELINE_CACHE.")
        if self.extraction_records is None:
            self.extraction_records = Path(os.getenv("GUIDELINE_EXTRACTION_RECORDS")
                                           or default_extraction_records(self.phi))
        side_error = mode_side_error(self.phi, self.extraction_records)
        if side_error:
            raise ValueError(side_error + " Choose another GUIDELINE_EXTRACTION_RECORDS.")

    @property
    def mode(self):
        return "phi" if self.phi else "standard"

    def extraction_config(self):
        """The inference configuration for extraction-based workup checks.

        This connection serves the query compile step and on-demand document and
        patient extraction (their preset sections, with its endpoint, model and
        thinking switch). The SQL workup agent uses the SQL-agent task's connection,
        which is this one until set, and may be Claude through Databricks.
        Hosted Vertex models answer free JSON that the package validates and
        retries, instead of vLLM structured outputs; the SQL agent's small action
        schema keeps JSON-schema responses there too.
        """
        c = self.inference_config()
        # On-demand extraction sends one request per note; let several run at once.
        c.remote["max_concurrent_requests"] = 8
        for name in ("document_extraction", "patient_extraction", "workup_query_compile"):
            self._connect_section(getattr(c, name), name)
        self.for_task(SQL_TASK)._connect_section(c.workup_sql_agent, "workup_sql_agent",
                                                 own_connection=True)
        return c

    def _connect_section(self, section, name, *, own_connection=False):
        """Point one preset section at this connection.

        ``own_connection``: the section names its endpoint and provider itself, so it
        can differ from the shared ``remote`` the other sections use.
        """
        remote = dict(section["remote"])
        local = dict(section.get("local", {}))
        gemini = self.llm_backend in GEMINI_BACKENDS
        if self.llm_backend in CLAUDE_BACKENDS:
            if name != "workup_sql_agent":
                raise ValueError("Claude through Databricks serves only the SQL agent.")
            profile = CLAUDE_BACKENDS[self.llm_backend]
            remote.update(
                provider="databricks_anthropic", server_urls=[profile["endpoint"]],
                databricks_profile=DATABRICKS_PROFILE, model_name=self.model,
                tokenizer_name=profile["tokenizer"], extra_body={},
                request_params={"max_tokens": min(
                    65536, remote.get("request_params", {}).get("max_tokens", 32768)),
                    "reasoning_effort": CLAUDE_EFFORT},
            )
            local["chat_template_kwargs"] = {}
            local["engine"] = {**local.get("engine", {}), "max_model_len": CLAUDE_CONTEXT}
            section.update(structured_outputs=True, reasoning_effort=CLAUDE_EFFORT,
                           sampling_profile="none")
        elif self.llm_backend in VERTEX_PROFILES:
            profile = VERTEX_PROFILES[self.llm_backend]
            template = {} if gemini else {"enable_thinking": self.reasoning}
            remote.update(
                model_name=profile["model"], tokenizer_name=profile["tokenizer"],
                request_params={"max_tokens": min(
                    65536, remote.get("request_params", {}).get("max_tokens", 32768)),
                    **({"reasoning_effort": "high"} if gemini else {})},
                extra_body={} if gemini else {"chat_template_kwargs": template.copy()},
            )
            if own_connection:
                remote.update(provider="google_agent_platform", google_project_id=GEMINI_PROJECT,
                              server_urls=[profile["endpoint"]])
            local["chat_template_kwargs"] = template.copy()
            local["engine"] = {**local.get("engine", {}),
                               "max_model_len": 1048576 if gemini else 262144}
            section.update(structured_outputs=name == "workup_sql_agent",
                           reasoning_effort="high",
                           sampling_profile="none" if gemini else "auto")
        else:
            remote.update(model_name=self.model, tokenizer_name=self.tokenizer)
            if own_connection:
                remote.update(provider="openai", server_urls=[self.endpoint])
            if name == "workup_sql_agent":
                # Read the served context from /v1/models rather than assume the preset's.
                local["engine"] = {k: v for k, v in local.get("engine", {}).items()
                                   if k != "max_model_len"}
            if self.thinking_supported:
                template = {"enable_thinking": self.reasoning}
                remote["extra_body"] = {**remote.get("extra_body", {}),
                                        "chat_template_kwargs": template.copy()}
                local["chat_template_kwargs"] = template.copy()
        section.update(remote=remote, local=local)

    def for_task(self, task):
        if task not in LLM_TASKS:
            raise ValueError("Unsupported LLM task")
        values = self.task_connections.get(task)
        if values:
            return replace(self, **values)
        if task == SQL_TASK:
            # Until set, the SQL agent uses the concordance checks' connection.
            return self.for_task("workup")
        return replace(self, backup_enabled=False) if task.endswith("_backup") else self

    @property
    def thinking_supported(self):
        return self.llm_backend == "gemma_vertex" or (
            self.llm_backend == "local"
            and bool(re.search(r"(?:qwen[-_ ]?3|gemma[-_ ]?4)", self.model, re.IGNORECASE))
        )

    def inference_config(self):
        from matchminer_ai import load_default_preset

        c = load_default_preset()
        c.debug_mode = False
        c.model_metadata_cache_dir = str(self.cache)
        # TrialSpace and the decision model (match-quality scoring) run locally.
        c.embedding["device"] = self.device
        c.raw["decision_model"]["device"] = self.device
        c.remote.update(
            enabled=True,
            server_urls=[self.endpoint],
            max_concurrent_requests=1,
            request_timeout=7200,
        )
        c.patient["remote"].update(model_name=self.model, tokenizer_name=self.tokenizer)
        c.patient["sampling_profile"] = "auto"
        c.patient["reasoning_effort"] = "xhigh"
        c.patient["prompt_build_workers"] = 1
        # Only send model-specific thinking controls for models with a known
        # switch. Unknown models retain their own defaults; compression also
        # retries without a rejected disable-thinking argument.
        template = {"enable_thinking": self.reasoning} if self.thinking_supported else {}
        c.patient.setdefault("local", {})["chat_template_kwargs"] = template.copy()
        c.patient["remote"]["extra_body"] = (
            {"chat_template_kwargs": template.copy()} if template else {}
        )
        if self.llm_backend in VERTEX_PROFILES:
            profile = VERTEX_PROFILES[self.llm_backend]
            gemini = self.llm_backend in GEMINI_BACKENDS
            context_window = 1048576 if gemini else 262144
            template = {} if gemini else {"enable_thinking": self.reasoning}
            c.remote.update(
                provider="google_agent_platform",
                google_project_id=GEMINI_PROJECT,
                server_urls=[profile["endpoint"]],
            )
            c.patient.update(
                sampling_profile="none" if gemini else "auto", reasoning_effort="high",
                context_window=context_window, tokenizer_mode="bytes",
            )
            c.patient["local"]["engine"]["max_model_len"] = context_window
            c.patient["local"]["chat_template_kwargs"] = template.copy()
            c.patient["remote"].update(
                model_name=profile["model"],
                tokenizer_name=profile["tokenizer"],
                request_params={"max_tokens": 65536, **({"reasoning_effort": "high"} if gemini else {})},
                extra_body={} if gemini else {"chat_template_kwargs": template.copy()},
            )
            if not gemini:
                c.remote.update(
                    request_start_interval_seconds=2.0,
                    capacity_retry_initial_seconds=15.0,
                    capacity_retry_max_seconds=60.0,
                )
        elif self.llm_backend in CLAUDE_BACKENDS:
            raise ValueError("Claude through Databricks serves only the SQL agent; choose another "
                             "connection for this task.")
        elif self.llm_backend != "local":
            raise ValueError("Unsupported LLM backend")
        return c
