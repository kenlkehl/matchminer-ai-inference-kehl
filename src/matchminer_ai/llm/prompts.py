"""Load versioned prompt resources shared by trial research and matching."""

from importlib import resources


def load_prompt_text(filename: str) -> str:
    """Read a bundled template, removing only its file-terminating newline.

    Preserve all other whitespace: checker inputs and distilled label prompts
    are versioned interfaces. Format templates once so braces in source data
    are never interpreted as further template fields.
    """
    return (
        resources.files("matchminer_ai.prompts")
        .joinpath(filename)
        .read_text(encoding="utf-8")
        .removesuffix("\n")
    )
