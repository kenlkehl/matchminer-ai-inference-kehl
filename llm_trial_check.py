"""Run the MatchMiner-AI terminal LLM trial checker from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Load the source package and delegate to its packaged CLI."""
    source_dir = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source_dir))

    from matchminer_ai.cli.llm_trial_check import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
