"""Embedding pasted texts with exactly the encoder that made the round's vectors.

EmbeddingGemma 2 (and TrialSpace fitted from it) needs sentence-transformers 6, which the app's own Python
does not have, so query texts go to a long-lived worker process under the embedding Python
(``encoder_worker.py``), which runs the inference package's ``open_encoder``. Before any text is embedded,
``EncoderSettings`` are rebuilt from the round manifest and their ``encoder_identity`` must equal the
manifest's encoder, both here and again in the worker.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import subprocess
import threading

import numpy as np

from .vectors import RoundInfo, encoder_settings, find_hub_cache

WORKER = Path(__file__).with_name("encoder_worker.py")


class EncoderUnavailable(RuntimeError):
    pass


class EncoderClient:
    def __init__(self, info: RoundInfo | None, settings, log_path: Path):
        self.info, self.settings, self.log_path = info, settings, Path(log_path)
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.state = "not started"
        self.problem = self._check()

    def _check(self) -> str | None:
        """Why pasted texts cannot be embedded, or None."""
        if self.info is None:
            return "no valid TrialSpace round"
        if not Path(self.settings.embedding_python).exists():
            return f"embedding Python not found: {self.settings.embedding_python}"
        if self.info.model.kind == "hub":
            self.cache = find_hub_cache(self.settings.embedding_caches, self.info.model.model, self.info.model.revision)
            if self.cache is None and not self.settings.allow_downloads:
                return (f"{self.info.model.model}@{str(self.info.model.revision)[:8]} weights not found in "
                        f"{[str(c) for c in self.settings.embedding_caches]} (downloads are off)")
        else:
            self.cache = None
        try:
            self.encoder_settings, same = encoder_settings(
                self.info, cache_dir=str(self.cache) if self.cache else None, device=self.settings.embedding_device)
        except Exception as exc:  # e.g. an unknown backend
            return f"cannot rebuild the encoder settings: {exc}"
        if not same:
            return "the rebuilt encoder identity differs from the round manifest's (custom search disabled)"
        return None

    @property
    def available(self) -> bool:
        return self.problem is None

    def describe(self) -> dict:
        return {"available": self.available, "problem": self.problem, "state": self.state,
                "python": str(self.settings.embedding_python), "cache": str(getattr(self, "cache", None)),
                "device": self.settings.embedding_device}

    def _start(self):
        from dataclasses import asdict

        import matchminer_ai

        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
        env["PYTHONPATH"] = str(Path(matchminer_ai.__file__).resolve().parent.parent)  # the same inference source
        env["PYTHONNOUSERSITE"] = "1"
        if not self.settings.allow_downloads:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        threads = str(max(1, min(16, self.settings.threads)))
        env.update(OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads, TOKENIZERS_PARALLELISM="false")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "ab")
        # -P: the worker's directory is not put on sys.path (nothing of the app shadows a library); -s: no user site.
        self.process = subprocess.Popen([str(self.settings.embedding_python), "-P", "-s", str(WORKER)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env,
                                        cwd=str(self.log_path.parent), text=True, bufsize=1)
        self.state = "loading"
        payload = asdict(self.encoder_settings)
        payload["devices"] = list(payload["devices"])
        reply = self._request({"op": "init", "settings": payload, "identity": self.info.encoder,
                               "threads": int(threads)})
        self.state = "ready"
        return reply

    def _request(self, message: dict) -> dict:
        assert self.process is not None and self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            self.state = "stopped"
            raise EncoderUnavailable(f"encoder worker exited (log: {self.log_path})")
        reply = json.loads(line)
        if not reply.get("ok"):
            self.state = "error"
            raise EncoderUnavailable(reply.get("error") or "encoder worker error")
        return reply

    def encode(self, texts: list[str], prompt_name: str) -> np.ndarray:
        if not self.available:
            raise EncoderUnavailable(self.problem)
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start()
            reply = self._request({"op": "encode", "texts": list(texts), "prompt_name": prompt_name})
        return np.frombuffer(base64.b64decode(reply["vectors"]), dtype=np.float32).reshape(reply["shape"])

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
        self.process = None
