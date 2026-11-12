"""Encoders on one or several devices or over embedding servers, and batched exact search, on fabricated text."""
import http.server
import json
import threading

import numpy as np
import pytest

from matchminer_ai.embedding import EncoderSettings, encoder_identity, open_encoder
from matchminer_ai.matching.retrieval import score_batches, stable_top_k

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "patient", "cohort", "lung", "breast", "cancer", "metastatic",
         "stage", "four", "prior", "therapy", "query", ":"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A random two-layer BERT with a word-level tokenizer, saved for sentence-transformers (mean pooling)."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny-encoder")
    torch.manual_seed(7)
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                             special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                            sep_token="[SEP]", model_max_length=64).save_pretrained(path)
    BertModel(BertConfig(vocab_size=len(WORDS), hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=64)).save_pretrained(path)
    return str(path)


def texts(count=37):
    rng = np.random.default_rng(3)
    return [" ".join(rng.choice(WORDS[4:], size=int(rng.integers(1, 12)))) for _ in range(count)]


def test_several_devices_give_the_one_device_vectors(tiny_model):
    prompts = {"patient_summary": "patient query : "}
    one = EncoderSettings(tiny_model, devices=("cpu",), batch_size=5, prompts=prompts)
    two = EncoderSettings(tiny_model, devices=("cpu", "cpu"), batch_size=5, prompts=prompts)
    with open_encoder(one) as encoder:
        expected = encoder.encode(texts(), "patient_summary")
    with open_encoder(two) as encoder:
        actual = encoder.encode(texts(), "patient_summary")
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    assert np.array_equal(actual, expected)  # the same batches, so the same numbers
    assert np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-5)
    # Devices never enter the identity; everything that changes the vectors does.
    assert encoder_identity(one) == encoder_identity(two)
    assert encoder_identity(one) != encoder_identity(EncoderSettings(tiny_model, batch_size=6, prompts=prompts))


def test_embedding_servers_get_the_prompted_truncated_token_ids_in_order(tiny_model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    seen = []

    class Server(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply(dict(data=[dict(id="tiny")]))

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.server.server_port, body))
            vectors = [[float(len(ids)), float(sum(ids)), 1.0] for ids in body["input"]]
            self.reply(dict(data=[dict(index=i, embedding=v) for i, v in reversed(list(enumerate(vectors)))]))

        def reply(self, value):
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    servers = [http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server) for _ in range(2)]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        settings = EncoderSettings(tiny_model, backend="openai", max_seq_length=6, prompts={"q": "query : "},
                                   remote=dict(server_urls=[f"http://127.0.0.1:{s.server_port}/v1" for s in servers],
                                               request_batch_size=4, max_concurrent_per_server=2))
        sample = texts(10)
        with open_encoder(settings) as encoder:
            vectors = encoder.encode(sample, "q")
    finally:
        for server in servers:
            server.shutdown()
    expected_ids = tokenizer(["query : " + t for t in sample], add_special_tokens=True, truncation=True,
                             max_length=6)["input_ids"]
    sent = [ids for _, body in sorted(seen, key=lambda item: item[1]["input"][0] != expected_ids[0]) for ids in body["input"]]
    assert sorted(map(tuple, sent)) == sorted(map(tuple, expected_ids))
    assert all(body["model"] == "tiny" for _, body in seen) and len({port for port, _ in seen}) == 2
    raw = np.asarray([[len(ids), sum(ids), 1.0] for ids in expected_ids])
    assert np.allclose(vectors, raw / np.linalg.norm(raw, axis=1, keepdims=True))  # in input order, normalized


def test_batched_search_over_devices_equals_one_matmul_with_stable_ties():
    rng = np.random.default_rng(1)
    queries, documents = rng.normal(size=(23, 8)).astype(np.float32), rng.normal(size=(40, 8)).astype(np.float32)
    one = list(score_batches(queries, documents, batch_size=5))
    two = list(score_batches(queries, documents, batch_size=5, devices=("cpu", "cpu")))
    assert [s for s, _ in one] == [s for s, _ in two] == list(range(0, 23, 5))
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(one, two))
    assert np.allclose(np.concatenate([b for _, b in one]), queries @ documents.T, atol=1e-5)
    assert stable_top_k([0.5, 0.9, 0.5, 0.9, 0.1], 3) == [1, 3, 0]


def test_embed_for_matching_encodes_exactly_as_training_mines(tiny_model):
    """The embedding section's settings give training's encoder vectors, and a local model gets an identity."""
    from matchminer_ai.embedding.inference import _embedding_cache_identity, generate_embeddings
    from matchminer_ai.embedding.trialspace_contract import model_prompts

    config = dict(model_path=tiny_model, device="cpu", prompt_file="embedding.txt", max_seq_length=16,
                  dtype="float32", autocast="bfloat16", prompt_profile="trialspace",
                  default_prompt_name="SentenceSimilarity", include_prompt_in_pooling=True, batch_size=5)
    mining = EncoderSettings(tiny_model, devices=("cpu",), batch_size=5, max_seq_length=16, dtype="float32",
                             autocast="bfloat16", prompts=model_prompts("trialspace"),
                             default_prompt_name="SentenceSimilarity", include_prompt_in_pooling=True)
    with open_encoder(mining) as encoder:
        for entity_type, prompt in (("patient", "patient_summary"), ("trial", "this_space")):
            expected = encoder.encode(texts(), prompt)
            actual, _ = generate_embeddings(texts(), embedding_config=config, entity_type=entity_type)
            assert np.array_equal(np.asarray(actual, dtype=np.float32), expected)
    identity = _embedding_cache_identity(config, {"model_name": tiny_model})
    assert identity["encoder"] == encoder_identity(mining) and len(identity["loaded_weights_sha256"]) == 64
