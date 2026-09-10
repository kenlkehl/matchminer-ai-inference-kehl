"""Compatibility alias for :mod:`matchminer_ai.matching.good_options`."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("matchminer_ai.matching.good_options")
