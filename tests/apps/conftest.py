"""The apps' tests need the ``apps`` extra (FastAPI); without it they are not collected."""

import importlib.util


def _installed(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


collect_ignore_glob = [] if _installed("fastapi") else ["*"]
