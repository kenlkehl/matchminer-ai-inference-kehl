# Installation

`matchminer-ai` needs Python 3.12 or later on Linux; see the [requirements](requirements.md). Install it from a
checkout with [`uv`](https://docs.astral.sh/uv/):

```shell
git clone https://github.com/kenlkehl/matchminer-ai-kehl.git
cd matchminer-ai-kehl
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
```

Check the install:

```shell
python -c "import matchminer_ai; print(matchminer_ai.__version__)"
```

## Extras

The base install covers the matching workflow. Optional parts are extras:

| Extra | Adds |
| --- | --- |
| `training` | The training pipeline (`mmai-train`) |
| `evaluation` | The evaluations (install together with `training`) |
| `apps` | The dashboard and the guideline fetcher |
| `sql` | SQL-agent workup checks over extraction records |
| `databricks` | Claude through a Databricks AI Gateway |
| `guideline-conversion` | PDF guideline conversion |
| `dev`, `docs` | Tests, linters and the documentation site |

For example:

```shell
uv pip install -e ".[training,evaluation,apps]"
```

The dashboard's frontend is built separately with `pnpm`; see [Dashboard](../apps/dashboard.md). For a development
setup, see [contributing](../development/contributing.md).
