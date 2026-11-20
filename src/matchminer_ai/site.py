"""Site configuration and the data root: values that belong to one installation, kept outside this repository.

A site config is a YAML file named by ``--site-config`` or ``MMAI_SITE_CONFIG``. It holds what differs between
institutions and machines and is never committed here: endpoint URLs, cloud projects and gateways, database schema
maps, site and group names, intranet links and local paths. Code reads it with dotted keys::

    site = load_site_config()
    site.get("endpoints.teacher.url")       # None when the site does not set it
    site.require("endpoints.teacher.url")   # an error naming the key, the file and how to set it

The data root (``--data-root``, ``MMAI_DATA_ROOT`` or the site config's ``paths.data_root``) holds ``phi/`` and
``no_phi/``. It has no default, and a data path inside this package's source checkout is refused, so clinical data
cannot land in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

SITE_CONFIG_ENV = "MMAI_SITE_CONFIG"
DATA_ROOT_ENV = "MMAI_DATA_ROOT"
_MISSING = object()


class SiteConfig:
    """Read-only view of a site config (empty when none is configured)."""

    def __init__(self, values: dict | None = None, path: Path | None = None):
        self.values = values or {}
        self.path = path

    def __bool__(self) -> bool:
        return bool(self.values)

    def get(self, key: str, default: Any = None) -> Any:
        value: Any = self.values
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def require(self, key: str) -> Any:
        value = self.get(key, _MISSING)
        if value is _MISSING or value is None:
            where = f"site config {self.path}" if self.path else f"a site config (--site-config or {SITE_CONFIG_ENV})"
            raise KeyError(f"{key!r} is not set; add it to {where}.")
        return value

    def section(self, key: str) -> dict:
        value = self.get(key, {})
        return dict(value) if isinstance(value, dict) else {}


_CACHE: dict[tuple[str, float], SiteConfig] = {}


def load_site_config(path: str | os.PathLike | None = None) -> SiteConfig:
    """The site config at ``path``, else at ``$MMAI_SITE_CONFIG``, else an empty one."""
    chosen = path or os.environ.get(SITE_CONFIG_ENV)
    if not chosen:
        return SiteConfig()
    resolved = Path(chosen).expanduser().resolve()
    key = (str(resolved), resolved.stat().st_mtime)
    if key not in _CACHE:
        import yaml

        values = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(values, dict):
            raise ValueError(f"site config {resolved} must be a YAML mapping")
        _CACHE[key] = SiteConfig(values, resolved)
    return _CACHE[key]


def source_checkout() -> Path | None:
    """The repository root when this package runs from a source checkout (src layout), else None."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").is_file() and (root / "src" / "matchminer_ai").is_dir() else None


def outside_checkout(path: str | os.PathLike, what: str = "data path") -> Path:
    """``path`` resolved, refusing one inside the source checkout."""
    resolved = Path(path).expanduser().resolve()
    root = source_checkout()
    if root is not None and resolved.is_relative_to(root):
        raise ValueError(f"{what} {resolved} is inside the source checkout {root}; keep data outside the repository.")
    return resolved


def data_root(explicit: str | os.PathLike | None = None, *, site: SiteConfig | None = None) -> Path:
    """The data root from ``explicit``, ``$MMAI_DATA_ROOT`` or the site config, outside the source checkout."""
    chosen = explicit or os.environ.get(DATA_ROOT_ENV) or (site or load_site_config()).get("paths.data_root")
    if not chosen:
        raise ValueError(f"No data root: pass --data-root, set {DATA_ROOT_ENV} or the site config's paths.data_root.")
    return outside_checkout(chosen, "data root")
