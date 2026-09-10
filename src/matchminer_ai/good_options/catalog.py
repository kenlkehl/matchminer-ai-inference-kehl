"""Compatibility alias for :mod:`matchminer_ai.trials.drug_catalog`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("matchminer_ai.trials.drug_catalog")
