"""Checkpoint selection that tolerates a Spot shutdown during saving."""
from pathlib import Path
import json


def latest_complete_checkpoint(directory):
    # Trainer writes its state after model, optimizer, scheduler and RNG saves.
    # A directory alone is insufficient evidence of a completed save.
    candidates = []
    for path in Path(directory).glob("checkpoint-*"):
        if not path.is_dir() or not path.name.removeprefix("checkpoint-").isdigit():
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_text())
        except (OSError, ValueError):
            continue
        step = int(path.name.removeprefix("checkpoint-"))
        if state.get("global_step") == step:
            candidates.append((step, path))
    return str(max(candidates)[1]) if candidates else None
