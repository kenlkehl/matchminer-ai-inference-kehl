"""Sentence-transformers encoding on one or more devices, or over vLLM embedding servers.

:func:`open_encoder` returns an encoder whose ``encode(texts, prompt_name)`` gives float32, L2-normalized
vectors in input order:

- one device: the model's own ``encode`` (it sorts the texts by length and encodes batches of
  ``batch_size``);
- several devices (``devices=("cuda:0", "cuda:1", ...)``): one worker process per device. The
  coordinator takes the model's own length order and deals out the same batches the one-device encoder
  would form, so every text is encoded in the same batch as on one device and the vectors match it
  (on the same GPU type);
- ``backend="openai"``: OpenAI-compatible ``/v1/embeddings`` servers (vLLM >= 0.32 serves
  EmbeddingGemma 2 with ``vllm serve <model> --runner pooling``; see
  :func:`matchminer_ai.llm.vllm_server.build_embedding_server_command`). The client applies the prompt
  and truncation exactly as sentence-transformers does, sends token ids, and normalizes; vLLM's kernels
  and bf16 weights give vectors close to (about 0.999 cosine), not equal to, the local backends'.

:func:`encoder_identity` names everything that changes the vectors and no paths, hosts or devices, so
vectors made by different backends or settings are never mixed. Training's mining encodes with this module,
and ``embed_for_matching`` loads and encodes with the same functions (one device), its ``embedding``
configuration section giving the settings.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
import traceback
from typing import Any, Iterator, Sequence

import numpy as np

IMPLEMENTATION = "mmai-encoder-v1"


@dataclass(frozen=True)
class EncoderSettings:
    """What is loaded and how it encodes.

    ``model`` is a Hub id or a local directory; ``model_identity`` replaces it in the identity (for a local
    checkpoint, the id of the stage that wrote it). ``prompts`` overrides or adds named prompts (for
    example the TrialSpace profile, ``trialspace_contract.model_prompts("trialspace")``). ``text_only``
    loads EmbeddingGemma 2 without its vision and audio towers. ``dtype`` sets the weights' precision and
    ``autocast`` the CUDA autocast precision; None keeps the model's and disables autocast.
    """

    model: str
    revision: str | None = None
    cache_dir: str | None = None
    devices: tuple[str, ...] = ("cpu",)
    batch_size: int = 32
    max_seq_length: int | None = None
    dtype: str | None = None
    autocast: str | None = None
    prompts: dict[str, str] = field(default_factory=dict)
    default_prompt_name: str | None = None
    include_prompt_in_pooling: bool | None = None
    text_only: bool = False
    model_identity: str | None = None
    backend: str = "sentence_transformers"
    # backend "openai": server_urls or server_urls_file (a {"servers": [{"url": ...}]} registry),
    # model_name (served name; default the first /v1/models entry), max_concurrent_per_server,
    # request_batch_size, request_timeout, max_retries, api_key_env.
    remote: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in ("sentence_transformers", "openai"):
            raise ValueError("Encoder backend must be sentence_transformers or openai")
        if self.batch_size < 1 or not self.devices:
            raise ValueError("Encoders need a positive batch size and at least one device")
        if self.dtype not in (None, "float32", "bfloat16") or self.autocast not in (None, "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16 and autocast bfloat16, or None")
        object.__setattr__(self, "devices", tuple(self.devices))


def encoder_identity(settings: EncoderSettings) -> dict[str, Any]:
    """Everything that changes the vectors (backend, model, revision, prompts, length, precision, batching)."""
    identity = dict(implementation=IMPLEMENTATION, backend=settings.backend,
                    model=settings.model_identity or settings.model, revision=settings.revision,
                    max_seq_length=settings.max_seq_length, prompts=dict(sorted(settings.prompts.items())),
                    default_prompt_name=settings.default_prompt_name, text_only=settings.text_only)
    if settings.backend == "sentence_transformers":
        identity.update(batch_size=settings.batch_size, dtype=settings.dtype, autocast=settings.autocast,
                        include_prompt_in_pooling=settings.include_prompt_in_pooling)
    return identity


def load_sentence_transformer(settings: EncoderSettings, device: str):
    """The model on ``device``, with the settings' prompts, length and pooling applied."""
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = dict(device=device)
    if settings.revision:
        kwargs["revision"] = settings.revision
    if settings.cache_dir:
        kwargs["cache_folder"] = settings.cache_dir
    if settings.text_only:
        kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
    if settings.dtype:
        kwargs["model_kwargs"] = {"dtype": getattr(torch, settings.dtype)}
    model = SentenceTransformer(settings.model, **kwargs)
    if settings.max_seq_length:
        model.max_seq_length = settings.max_seq_length
    model.prompts.update(settings.prompts)
    if settings.default_prompt_name:
        model.default_prompt_name = settings.default_prompt_name
    if settings.include_prompt_in_pooling is not None:
        model.set_pooling_include_prompt(settings.include_prompt_in_pooling)
    return model


def _encode_on(model, texts: Sequence[str], prompt_name: str | None, settings: EncoderSettings, device: str,
               batch_size: int) -> np.ndarray:
    import torch

    cuda = str(device).startswith("cuda")
    autocast = (torch.autocast("cuda", dtype=getattr(torch, settings.autocast))
                if settings.autocast and cuda else nullcontext())
    with torch.inference_mode(), autocast:
        values = model.encode(list(texts), prompt_name=prompt_name, batch_size=batch_size,
                              normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(values, dtype=np.float32)


def batch_order(model, texts: Sequence[str]) -> list[int]:
    """The order in which the model's own ``encode`` batches ``texts`` (sentence-transformers 5 and 6)."""
    length = getattr(model, "_input_length", None) or model._text_length
    order = np.argsort([-length(text) for text in texts])
    can_flatten = getattr(model, "_can_flatten_inputs", None)
    if can_flatten is not None and can_flatten():
        order = model._interleave_sorted_indices(order)
    return [int(i) for i in order]


class LocalEncoder:
    """One device: the model's own batching."""

    def __init__(self, settings: EncoderSettings):
        self.settings, self.device = settings, settings.devices[0]
        self.model = load_sentence_transformer(settings, self.device)

    def encode(self, texts: Sequence[str], prompt_name: str | None = None) -> np.ndarray:
        return _encode_on(self.model, texts, prompt_name, self.settings, self.device, self.settings.batch_size)

    def close(self):
        self.model = None


def _worker(settings: dict, device: str, tasks, results, index: int):
    try:
        settings = EncoderSettings(**settings)
        model = load_sentence_transformer(settings, device)
    except BaseException:
        results.put(("error", index, traceback.format_exc()))
        return
    results.put(("ready", index, None))
    while (task := tasks.get()) is not None:
        kind, key, payload = task
        try:
            if kind == "order":
                value = batch_order(model, payload)
            else:
                texts, prompt_name = payload
                value = _encode_on(model, texts, prompt_name, settings, device, batch_size=len(texts))
            results.put(("done", key, value))
        except BaseException:
            results.put(("error", key, traceback.format_exc()))


class MultiDeviceEncoder:
    """One spawned worker per device; the one-device encoder's batches, dealt out to the workers."""

    def __init__(self, settings: EncoderSettings, *, start_timeout: float = 3600):
        import torch.multiprocessing as mp

        self.settings = settings
        context = mp.get_context("spawn")
        self.results = context.Queue()
        self.tasks = [context.Queue() for _ in settings.devices]
        self.workers = [context.Process(target=_worker, args=(asdict(settings), device, tasks, self.results, i),
                                        daemon=True, name=f"mmai-encoder-{i}")
                        for i, (device, tasks) in enumerate(zip(settings.devices, self.tasks))]
        for worker in self.workers:
            worker.start()
        ready, deadline = set(), time.monotonic() + start_timeout
        while len(ready) < len(self.workers):
            kind, index, detail = self._get(deadline)
            if kind != "ready":
                self.close()
                raise RuntimeError(f"Encoder worker {index} on {settings.devices[index]} failed to load:\n{detail}")
            ready.add(index)

    def _get(self, deadline=None):
        import queue

        while True:
            try:
                return self.results.get(timeout=5)
            except queue.Empty:
                dead = [w.name for w in self.workers if not w.is_alive()]
                if dead or (deadline is not None and time.monotonic() > deadline):
                    raise RuntimeError(f"Encoder workers stopped or timed out: {dead}") from None

    def encode(self, texts: Sequence[str], prompt_name: str | None = None) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self.tasks[0].put(("order", "order", texts))
        kind, _, order = self._get()
        if kind != "done":
            raise RuntimeError("Encoder worker could not order the texts:\n" + order)
        size = self.settings.batch_size
        batches = [order[start:start + size] for start in range(0, len(order), size)]
        pending, outstanding, done = iter(enumerate(batches)), {}, {}

        def submit(worker):
            item = next(pending, None)
            if item is not None:
                number, indices = item
                outstanding[number] = worker
                self.tasks[worker].put(("encode", number, ([texts[i] for i in indices], prompt_name)))

        for worker in range(len(self.workers)):
            for _ in range(2):  # two batches queued per worker keep its device busy between results
                submit(worker)
        while outstanding:
            kind, number, value = self._get()
            if kind != "done":
                raise RuntimeError(f"Encoder worker failed on batch {number}:\n{value}")
            done[number] = value
            submit(outstanding.pop(number))
        vectors = np.empty((len(texts), done[0].shape[1]), dtype=np.float32)
        for number, indices in enumerate(batches):
            vectors[indices] = done[number]
        return vectors

    def close(self):
        for tasks in self.tasks:
            tasks.put(None)
        for worker in self.workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()


class OpenAIEmbeddingsEncoder:
    """OpenAI-compatible ``/v1/embeddings`` servers, the prompt and truncation applied as sentence-transformers does."""

    def __init__(self, settings: EncoderSettings):
        import os
        from concurrent.futures import ThreadPoolExecutor
        import threading

        import httpx
        from transformers import AutoTokenizer

        if settings.include_prompt_in_pooling is False:
            raise ValueError("vLLM pools over every token, prompt included; include_prompt_in_pooling must be True")
        remote = dict(settings.remote)
        urls = list(remote.get("server_urls") or [])
        if remote.get("server_urls_file"):
            registry = json.loads(Path(remote["server_urls_file"]).read_text())
            urls += [row["url"] for row in registry.get("servers", [])]
        if not urls:
            raise ValueError("The openai encoder backend needs server_urls or server_urls_file")
        self.settings, self.urls = settings, [url.rstrip("/") for url in dict.fromkeys(urls)]
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model, revision=settings.revision,
                                                       cache_dir=settings.cache_dir)
        self.prompts = {**_saved_prompts(settings), **settings.prompts}
        key = os.environ.get(remote.get("api_key_env", "OPENAI_API_KEY"), "EMPTY")
        self.client = httpx.Client(timeout=float(remote.get("request_timeout", 600)),
                                   headers={"Authorization": f"Bearer {key}"}, trust_env=False)
        self.model_name = remote.get("model_name") or self.client.get(
            self.urls[0] + "/models").raise_for_status().json()["data"][0]["id"]
        self.request_batch_size = int(remote.get("request_batch_size", 32))
        self.max_retries = int(remote.get("max_retries", 6))
        per_server = int(remote.get("max_concurrent_per_server", 8))
        self.slots = {url: threading.Semaphore(per_server) for url in self.urls}
        self.load = {url: 0 for url in self.urls}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=per_server * len(self.urls))

    def _post(self, token_ids):
        last = None
        for attempt in range(self.max_retries):
            with self.lock:
                url = min(self.urls, key=lambda u: self.load[u])
                self.load[url] += 1
            try:
                with self.slots[url]:
                    response = self.client.post(url + "/embeddings", json=dict(
                        model=self.model_name, input=token_ids, encoding_format="float"))
                response.raise_for_status()
                data = sorted(response.json()["data"], key=lambda item: item["index"])
                if [item["index"] for item in data] != list(range(len(token_ids))):
                    raise ValueError("Embedding response does not cover the request in order")
                return np.asarray([item["embedding"] for item in data], dtype=np.float32)
            except Exception as error:  # a failing server cools down; the request moves on
                last = error
                time.sleep(min(30, 2 ** attempt))
            finally:
                with self.lock:
                    self.load[url] -= 1
        raise RuntimeError(f"Embedding request failed after {self.max_retries} attempts: {last!r}")

    def encode(self, texts: Sequence[str], prompt_name: str | None = None) -> np.ndarray:
        prompt = self.prompts.get(prompt_name or self.settings.default_prompt_name or "", "")
        encoded = self.tokenizer([prompt + text for text in texts], add_special_tokens=True, truncation=True,
                                 max_length=self.settings.max_seq_length or self.tokenizer.model_max_length)
        ids = list(encoded["input_ids"])
        size = self.request_batch_size
        parts = list(self.pool.map(self._post, [ids[i:i + size] for i in range(0, len(ids), size)]))
        if not parts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = np.concatenate(parts)
        return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)

    def close(self):
        self.pool.shutdown(wait=True)
        self.client.close()


def _saved_prompts(settings: EncoderSettings) -> dict[str, str]:
    """The prompts saved with a local model or Hub snapshot (config_sentence_transformers.json)."""
    path = Path(settings.model) / "config_sentence_transformers.json"
    if not path.exists():
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(settings.model, "config_sentence_transformers.json",
                                        revision=settings.revision, cache_dir=settings.cache_dir))
        except Exception:
            return {}
    return dict(json.loads(path.read_text()).get("prompts") or {})


@contextmanager
def open_encoder(settings: EncoderSettings) -> Iterator[Any]:
    """An encoder for ``settings``: vLLM servers, several worker processes, or this process."""
    if settings.backend == "openai":
        encoder = OpenAIEmbeddingsEncoder(settings)
    elif len(settings.devices) > 1:
        encoder = MultiDeviceEncoder(settings)
    else:
        encoder = LocalEncoder(settings)
    try:
        yield encoder
    finally:
        encoder.close()


__all__ = ["EncoderSettings", "LocalEncoder", "MultiDeviceEncoder", "OpenAIEmbeddingsEncoder", "batch_order",
           "encoder_identity", "load_sentence_transformer", "open_encoder"]
