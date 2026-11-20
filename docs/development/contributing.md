# Contributing

Contributions are welcome through GitHub pull requests: fork
[`kenlkehl/matchminer-ai-kehl`](https://github.com/kenlkehl/matchminer-ai-kehl) and open a pull request from your
fork.

## Setup

Clone your fork and install the package in editable mode with [`uv`](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/<your-username>/matchminer-ai-kehl.git
cd matchminer-ai-kehl
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,training,evaluation,apps]"
pre-commit install
```

- **`dev`:** pytest, pytest-cov, pre-commit, ruff, mypy and `types-PyYAML`.
- **`training`, `evaluation`, `apps`:** needed by the tests of those parts. CI installs the same three.
- **`sql`, `databricks`:** optional; tests that need DuckDB, `anthropic` or the Databricks SDK skip without them.

## Pre-commit hooks

`pre-commit install` runs these hooks on every commit (`.pre-commit-config.yaml`):

| Hook | Checks |
| --- | --- |
| `ruff`, `ruff-format` | Linting and formatting |
| `mypy` | Type checks |
| `check-added-large-files`, `end-of-file-fixer`, `trailing-whitespace` | Large files and whitespace |
| `gitleaks`, `trufflehog` | Secrets (`.gitleaks.toml`; trufflehog reports verified secrets only) |

Run them on the whole tree with `pre-commit run --all-files`.

## Tests

Run the suite from the repository root:

```shell
pytest
```

`pytest.ini` puts `src/` on the path and skips tests marked `resource_heavy`, which need GPUs and full model
downloads. Run those on a GPU machine with `pytest -m resource_heavy`.

| Directory | Covers |
| --- | --- |
| `tests/` | The runtime package: trials, patients, embedding, matching, the LLM clients, labeling, guideline catalogs |
| `tests/training/` | `matchminer_ai.training`; stages resolve their paths under a temporary data root |
| `tests/evaluation/` | `matchminer_ai.evaluation` |
| `tests/apps/` | The dashboard and the guideline fetcher; not collected without the `apps` extra |
| `tests/integration/` | GPU smoke tests of trial and patient summarization (`resource_heavy`) |
| `tests/data/` | Fixtures, such as the embedding regression vectors |

The guideline fetcher's browser modules have Node tests (Node 18 or later):

```shell
node --test tests/apps/guideline_fetcher/*.mjs
```

Tests use synthetic or fabricated records only.

## Documentation

```shell
uv pip install -e ".[docs]"
mkdocs serve                # live preview
mkdocs build --strict       # fails on broken links and warnings
```

The `::: module.path` blocks in the API pages render docstrings, so public functions need complete numpy-style
docstrings.

## Pull requests

!!! warning
    Do not include PHI, PII, credentials, secrets, generated cache files, or model outputs containing sensitive data.

The pull request template asks you to confirm that you ran the tests and the pre-commit hooks and checked for
sensitive data. GitHub Actions runs on every push:

- the pre-commit hooks on all files;
- a gitleaks scan;
- the install with `.[training,evaluation,apps]`, an import check and `pytest` on Python 3.12, 3.13 and 3.14.

These checks must pass before merge. Pushes to `main` also deploy the documentation site.
