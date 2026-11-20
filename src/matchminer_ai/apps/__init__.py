"""Web apps built on the package: the guideline fetcher and the dashboard.

Each app is a subpackage started with its console script (``mmai-guideline-fetcher``, ``mmai-dashboard``) or
``python -m matchminer_ai.apps.<name>``. They need the ``apps`` extra. Institution-specific values come from the
site config (``matchminer_ai.site``), and data paths from the data root; neither has a default in this repository.
"""
