"""Where training and evaluation read and write, and what identifies the code that ran.

Data paths come from the data root (``matchminer_ai.site``: ``--data-root``, ``MMAI_DATA_ROOT`` or the site config),
never from the checkout's location. ``core_sources`` hashes the inference core only (not ``training``,
``evaluation`` or ``apps``), so a run identity that covers the package's prompts and code is not invalidated by
edits to the training stages or the apps.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import matchminer_ai
from matchminer_ai.site import data_root, source_checkout

PACKAGE = Path(matchminer_ai.__file__).resolve().parent
NOT_CORE = ("training", "evaluation", "apps")


def no_phi(*parts: str, root=None) -> Path:
    """``<data root>/no_phi/<parts>`` (synthetic and public data)."""
    return data_root(root).joinpath("no_phi", *parts)


def phi(*parts: str, root=None) -> Path:
    """``<data root>/phi/<parts>`` (real clinical data)."""
    return data_root(root).joinpath("phi", *parts)


def commit(path: Path | None = None) -> str | None:
    """The git commit of the checkout holding ``path`` (default: this package), or None outside a checkout."""
    where = Path(path) if path else (source_checkout() or PACKAGE)
    result = subprocess.run(["git", "-C", str(where if where.is_dir() else where.parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() or None if result.returncode == 0 else None


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_files(package: Path = PACKAGE) -> list[Path]:
    """Every core ``.py``/``.txt``/``.yaml`` file of the package (not training, evaluation or apps)."""
    return [p for p in sorted(package.rglob("*")) if p.suffix in (".py", ".txt", ".yaml")
            and p.relative_to(package).parts[0] not in NOT_CORE and "__pycache__" not in p.parts]


def core_sources(package: Path = PACKAGE) -> dict[str, str]:
    """sha256 of every core file (``core_files``), keyed by its path in the package."""
    return {str(p.relative_to(package)): sha_file(p) for p in core_files(package)}


def models_root(explicit=None) -> Path:
    """Where trained models go: the site config's ``paths.models_root``, else ``<data root>/../models``."""
    from matchminer_ai.site import load_site_config, outside_checkout

    chosen = explicit or load_site_config().get("paths.models_root")
    return outside_checkout(chosen, "models root") if chosen else data_root().parent / "models"


def module_name(name: str) -> str:
    """``trials.screen_trials`` -> ``matchminer_ai.training.trials.screen_trials``; full names pass through."""
    return name if name.startswith("matchminer_ai.") else f"matchminer_ai.training.{name}"


def module_command(name: str, python: str | None = None) -> list[str]:
    """The command that runs a stage module: ``python -m <module>``."""
    return [python or sys.executable, "-m", module_name(name)]


def module_file(name: str) -> Path:
    """The source file of a stage module (for run identities)."""
    return PACKAGE.joinpath(*module_name(name).split(".")[1:]).with_suffix(".py")


def child_env(**extra: str) -> dict[str, str]:
    """os.environ with this package's ``src`` first on PYTHONPATH, so stage subprocesses run this same code."""
    src = str(PACKAGE.parent)
    rest = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and p != src]
    return {**os.environ, "PYTHONPATH": os.pathsep.join([src, *rest]), **extra}
