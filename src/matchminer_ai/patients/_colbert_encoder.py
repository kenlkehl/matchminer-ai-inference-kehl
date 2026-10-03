"""Load a GTE/ModernColBERT Transformer + trained token projection checkpoint.

Uses the checkpoint's query/document markers, punctuation mask and expansion
settings, matching its PyLate encoding contract without changing this package's
Transformers/Sentence Transformers dependency versions.
"""

import hashlib
import json
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from matchminer_ai.cancellation import check_cancelled


class TokenEncoder:
    def __init__(self, config):
        local = Path(config.model_name).is_dir()
        folder = (
            Path(config.model_name)
            if local
            else Path(
                snapshot_download(
                    config.model_name,
                    revision=config.revision,
                    allow_patterns=[
                        "*.json",
                        "*.txt",
                        "*.safetensors",
                        "1_Dense/*.json",
                        "1_Dense/*.safetensors",
                    ],
                )
            )
        )
        modules = json.loads((folder / "modules.json").read_text())
        if (
            len(modules) != 2
            or modules[0]["type"] != "sentence_transformers.models.Transformer"
            or modules[0]["path"] != ""
            or modules[1]["path"] != "1_Dense"
            or modules[1]["type"]
            not in {"pylate.models.Dense.Dense", "sentence_transformers.models.Dense"}
        ):
            raise ValueError(
                "Expected a ColBERT Transformer plus trained 1_Dense token projection checkpoint."
            )
        self.settings = json.loads(
            (folder / "config_sentence_transformers.json").read_text()
        )
        projection = json.loads((folder / "1_Dense/config.json").read_text())
        if projection.get("activation_function") != "torch.nn.modules.linear.Identity":
            raise ValueError("Unsupported ColBERT projection activation.")
        if self.settings.get("prompts") or self.settings.get("default_prompt_name"):
            raise ValueError(
                "Custom checkpoint prompts are not supported by this ColBERT encoder."
            )
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            folder, use_fast=True, local_files_only=True
        )
        if not self.tokenizer.is_fast or self.tokenizer.padding_side != "right":
            raise ValueError(
                "ColBERT encoding requires a fast tokenizer with right padding."
            )
        self.model = (
            AutoModel.from_pretrained(
                folder, local_files_only=True, use_safetensors=True
            )
            .to(self.device)
            .eval()
        )
        self.projection = torch.nn.Linear(
            projection["in_features"],
            projection["out_features"],
            bias=projection["bias"],
        )
        weights = load_file(folder / "1_Dense/model.safetensors")
        self.projection.load_state_dict(
            {k.removeprefix("linear."): v for k, v in weights.items()}, strict=True
        )
        self.projection.to(
            device=self.device, dtype=next(self.model.parameters()).dtype
        ).eval()
        self.query_length, self.document_length = (
            config.query_length,
            config.chunk_size + 8,
        )
        if (
            max(self.query_length, self.document_length)
            > self.model.config.max_position_embeddings
        ):
            raise ValueError(
                "ColBERT query/document lengths exceed the model's position capacity."
            )
        self.prefixes = {}
        for name in ("query_prefix", "document_prefix"):
            prefix = self.settings[name]
            if prefix and prefix not in self.tokenizer.get_vocab():
                raise ValueError(
                    "The checkpoint is missing its trained ColBERT prefix token."
                )
            self.prefixes[name] = (
                self.tokenizer.convert_tokens_to_ids(prefix) if prefix else None
            )
        self.tokenizer.pad_token_id = (
            self.tokenizer.mask_token_id
            or self.tokenizer.eos_token_id
            or self.tokenizer.pad_token_id
        )
        self.skiplist = [
            self.tokenizer.convert_tokens_to_ids(word)
            for word in self.settings["skiplist_words"]
        ]
        self.expansion = self.settings["do_query_expansion"]
        # Hash weights as well as configuration: local edits must not mix with
        # vectors from another checkpoint. Cached once per loaded encoder.
        hasher = hashlib.sha256()
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".safetensors", ".txt"}:
                check_cancelled()
                hasher.update(str(path.relative_to(folder)).encode())
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(block)
        self.signature = dict(
            model_name=config.model_name,
            revision="local" if local else folder.name,
            artifact_sha256=hasher.hexdigest(),
            encoder_version=1,
            transformers_version=transformers.__version__,
            settings=config.encoding_settings(),
        )

    def encode(self, texts, *, is_query, **kwargs):
        """One caller-bounded batch, with no pooling and no silent truncation."""
        check_cancelled()
        prefix = self.prefixes["query_prefix" if is_query else "document_prefix"]
        limit = self.query_length if is_query else self.document_length
        inputs = self.tokenizer(
            [text.strip() for text in texts],
            return_tensors="pt",
            truncation=False,
            padding="max_length" if is_query and self.expansion else True,
            **(
                {"max_length": limit - (prefix is not None)}
                if is_query and self.expansion
                else {}
            ),
        )
        if prefix is not None:
            for key, value in list(inputs.items()):
                marker = (
                    prefix
                    if key == "input_ids"
                    else 1
                    if key == "attention_mask"
                    else 0
                )
                inputs[key] = torch.cat(
                    (
                        value[:, :1],
                        torch.full((len(texts), 1), marker, dtype=value.dtype),
                        value[:, 1:],
                    ),
                    dim=1,
                )
        if inputs["input_ids"].shape[1] > limit:
            raise ValueError(
                "ColBERT input exceeds its configured length; no text was truncated."
            )
        if is_query and self.expansion and self.settings["attend_to_expansion_tokens"]:
            inputs["attention_mask"].fill_(1)
        inputs = {k: value.to(self.device) for k, value in inputs.items()}
        with torch.inference_mode():
            embeddings = self.projection(self.model(**inputs).last_hidden_state)
            masks = inputs["attention_mask"].bool()
            if is_query and self.expansion:
                masks = torch.ones_like(masks)
            elif not is_query:
                for token_id in self.skiplist:
                    masks &= inputs["input_ids"] != token_id
            result = [
                torch.nn.functional.normalize(value[mask], p=2, dim=1)
                .float()
                .cpu()
                .numpy()
                for value, mask in zip(embeddings, masks, strict=True)
            ]
        check_cancelled()
        return result
