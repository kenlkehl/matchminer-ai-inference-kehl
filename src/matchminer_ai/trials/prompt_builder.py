"""Packaged prompt text for the guideline modules."""

from __future__ import annotations

from importlib import resources


def load_prompt_text(filename: str) -> str:
    """Load a prompt text file from the package."""
    prompt_path = resources.files("matchminer_ai.prompts").joinpath(filename)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()
