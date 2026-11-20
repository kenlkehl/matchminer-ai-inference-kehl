"""Query-embedding worker, run under the embedding Python (by default the app's own; sentence-transformers 6, for
EmbeddingGemma 2).

Reads one JSON request per stdin line and writes one JSON reply per stdout line:

- ``{"op": "init", "settings": {...EncoderSettings fields}, "identity": {...}, "threads": 8}``: load the
  encoder; refuse unless ``encoder_identity(settings)`` equals the expected identity (the round manifest's).
- ``{"op": "encode", "texts": [...], "prompt_name": "patient_summary" | "this_space"}``: base64 float32
  vectors (normalized, input order), from the inference package's ``open_encoder``.

Started and fed only by ``encoder.EncoderClient``; it makes no network requests (Hub offline unless the app
allows downloads) and writes nothing.
"""
import base64
import json
import sys
import traceback


def main():
    encoder = None
    context = None
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if request["op"] == "init":
                import torch
                from matchminer_ai.embedding.encoder import EncoderSettings, encoder_identity, open_encoder

                torch.set_num_threads(int(request.get("threads") or 8))
                settings = EncoderSettings(**request["settings"])
                identity = encoder_identity(settings)
                if identity != request["identity"]:
                    raise ValueError("encoder identity differs from the round manifest's; refusing to embed")
                context = open_encoder(settings)
                encoder = context.__enter__()
                reply = {"ok": True, "identity": identity}
            elif request["op"] == "encode":
                import numpy as np

                if encoder is None:
                    raise RuntimeError("encoder not initialized")
                vectors = np.ascontiguousarray(encoder.encode(request["texts"], request.get("prompt_name")),
                                               dtype=np.float32)
                reply = {"ok": True, "shape": list(vectors.shape),
                         "vectors": base64.b64encode(vectors.tobytes()).decode()}
            else:
                raise ValueError(f"unknown op {request.get('op')!r}")
        except Exception as exc:  # reported to the app, which shows it
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:]}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
    if context is not None:
        context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
