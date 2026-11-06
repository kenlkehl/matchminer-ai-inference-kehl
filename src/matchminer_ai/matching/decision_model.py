"""The decision model: one System One model (LiquidAI d1) answering seven yes/no question families.

A decision model reads a *state* and named questions and returns, for each ``noul`` (yes/no)
question, P(yes) from one forward pass, without generating text. The pinned base model is
``LiquidAI/d1-2B``; the training repository fine-tunes it on the teacher's labels of the same
seven families (``train_decision_model.py``), and ``decision_model.model_name`` in the
configuration selects either.

The seven families, and what **yes** means for each:

1-5. ``reasonableness``, ``cancer_type``, ``burden``, ``treatment_history``, ``biomarker``: the match
     labels of :mod:`matchminer_ai.matching.match_labels`, five questions about one state, the
     patient summary's key section (cancer history) and the cohort's key criteria
     (:func:`match_state`, :func:`match_questions`). ``reasonableness`` yes: the patient is
     potentially eligible after review of these texts; no: clearly ineligible. The other four yes:
     the cohort specifically requires that dimension and the patient documentedly satisfies it.
6.   ``boilerplate``: one question about the patient summary's boilerplate section and the
     cohort's boilerplate criteria (:func:`boilerplate_state`, :func:`boilerplate_question`). Yes:
     potentially eligible; no: clearly fails at least one boilerplate criterion.
7.   ``criterion``: one question per line of the cohort's eligibility criteria, about the complete
     patient summary and the cohort's complete numbered criteria (:func:`criteria_state`,
     :func:`criterion_question`; the question restates the line with its number and its
     inclusion/exclusion type). **Yes: the patient does not clearly fail this line** (it is met,
     or its information is missing, or the patient could meet it by waiting). **No: the patient
     clearly fails it: an inclusion line clearly not met, or an exclusion line clearly met.**

These builders are the only definition of the states and questions: the training repository
renders its training examples with them, so inference reads exactly what the model learned.
Patient text reaches only the configured decision model, which runs locally by default.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from matchminer_ai.llm.prompts import load_prompt_text

from .match_labels import LABELS, match_labels_prompt

BASE_MODEL = "LiquidAI/d1-2B"
BASE_REVISION = "652f920e544ea37f81e4b0ce89b81d4ed0da90b1"
BOILERPLATE = "boilerplate"
CRITERION = "criterion"
FAMILIES = (*LABELS, BOILERPLATE, CRITERION)
CRITERION_TYPES = ("inclusion", "exclusion")
CRITERION_STATUSES = ("met", "not_met", "insufficient_information", "not_applicable")
# The per-criterion teacher's answer, for structured (constrained) decoding after its reasoning.
CRITERION_LABEL_JSON_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["answer", "criterion_status", "reason"],
    "properties": {"answer": {"enum": ["yes", "no"]}, "criterion_status": {"enum": list(CRITERION_STATUSES)},
                   "reason": {"type": "string"}},
}
PROMPTS = {
    "match_state": "decision.match.state.txt",
    "reasonableness": "decision.match.reasonableness.txt",
    "boilerplate_state": "decision.boilerplate.state.txt",
    "boilerplate_question": "decision.boilerplate.question.txt",
    "criterion_state": "decision.criterion.state.txt",
    "criterion_question": "decision.criterion.question.txt",
    "criterion_teacher": "criterion_check.labels.user.txt",
}
DEFAULTS = {"model_name": BASE_MODEL, "revision": BASE_REVISION, "backend": "transformers", "device": "cuda",
            "dtype": "bfloat16", "max_length": 10000, "set_size": None, "token_budget": 65536, "batch_size": 256,
            "compile": False}


@lru_cache(maxsize=None)
def _prompt(key: str) -> str:
    return load_prompt_text(PROMPTS[key]).strip()


def _criterion_label(criterion_type: str) -> str:
    if criterion_type not in CRITERION_TYPES:
        raise ValueError(f"criterion_type must be one of {CRITERION_TYPES}, not {criterion_type!r}")
    return criterion_type.capitalize()


def _fill(template: str, **values: str) -> tuple[str, dict[str, tuple[int, int]]]:
    """``template`` with each ``{name}`` replaced by its value, and each value's character range."""
    text, ranges, position = "", {}, 0
    for match in re.finditer(r"\{([a-z_]+)\}", template):
        text += template[position:match.start()]
        value = values[match.group(1)]
        ranges[match.group(1)] = (len(text), len(text) + len(value))
        text += value
        position = match.end()
    return text + template[position:], ranges


# --------------------------------------------------------------------------- #
# states and questions
# --------------------------------------------------------------------------- #


def match_state(patient_key_summary: str, cohort_key_criteria: str) -> str:
    """The state of questions 1-5: the patient's key section and the cohort's key criteria."""
    return match_state_with_ranges(patient_key_summary, cohort_key_criteria)[0]


def match_state_with_ranges(patient_key_summary: str, cohort_key_criteria: str):
    return _fill(_prompt("match_state"), patient_key_summary=patient_key_summary.strip(),
                 cohort_key_criteria=cohort_key_criteria.strip())


@lru_cache(maxsize=1)
def _match_rules() -> dict[str, tuple[str, str]]:
    """Each match label's question and rules, cut from the teacher's five-answer prompt."""
    prompt = match_labels_prompt()
    body = prompt.split("\n\n1. reasonableness: ", 1)[1].split("\n\nOUTPUT FORMAT", 1)[0]
    sections = re.split(r"(?m)^\n?(\d)\. ([a-z_]+): ", "1. reasonableness: " + body)
    rules = {}
    for index in range(1, len(sections), 3):
        question, rule = sections[index + 2].strip().split("\n", 1)
        rules[sections[index + 1]] = (question.strip(), rule.strip())
    if tuple(rules) != LABELS:
        raise ValueError("The match-label prompt's structure changed; review the decision-model questions")
    return rules


def _decision_rule(text: str) -> str:
    """A teacher rule without the teacher's evidence_status bookkeeping, which the decision model
    does not answer."""
    text = re.sub(r" with evidence_status \w+", "", text)
    return re.sub(r" \((?:supported|contradicted|insufficient_information|not_applicable)\)", "", text)


@lru_cache(maxsize=1)
def _match_questions() -> tuple[tuple[str, str], ...]:
    rules = _match_rules()
    questions = []
    for label in LABELS:
        question, rule = rules[label]
        if label == "reasonableness":  # potentially eligible -> yes, clearly ineligible -> no
            question = _prompt("reasonableness")
        questions.append((label, f"{question}\n{_decision_rule(rule)}"))
    return tuple(questions)


def match_questions() -> dict[str, str]:
    """Questions 1-5, by match label: each label's instructions (its question and rules)."""
    return dict(_match_questions())


def boilerplate_state(patient_boilerplate_summary: str, cohort_boilerplate_criteria: str) -> str:
    """The state of question 6: the patient's boilerplate section and the cohort's boilerplate criteria."""
    return boilerplate_state_with_ranges(patient_boilerplate_summary, cohort_boilerplate_criteria)[0]


def boilerplate_state_with_ranges(patient_boilerplate_summary: str, cohort_boilerplate_criteria: str):
    return _fill(_prompt("boilerplate_state"), patient_boilerplate_summary=patient_boilerplate_summary.strip(),
                 cohort_boilerplate_criteria=cohort_boilerplate_criteria.strip())


def boilerplate_question() -> str:
    """Question 6: yes when potentially eligible, no when clearly failing a boilerplate criterion."""
    return _prompt("boilerplate_question")


def _numbered(items: Sequence[Mapping[str, Any]]) -> str:
    from matchminer_ai.trials.cohort_boilerplate import number_criteria
    numbers = [int(item["number"]) for item in items]
    if numbers != list(range(1, len(items) + 1)):
        raise ValueError("criteria items must be numbered 1..n in order")
    return number_criteria([(item.get("heading") or "", item["text"]) for item in items])


def criteria_state(patient_summary: str, items: Sequence[Mapping[str, Any]]) -> str:
    """The state of question 7: the complete patient summary (key section, then ``Other relevant
    history:``) and the cohort's complete criteria, numbered as the boilerplate step numbers them."""
    return _fill(_prompt("criterion_state"), patient_summary=patient_summary.strip(), numbered_criteria=_numbered(items))[0]


def criterion_question(item: Mapping[str, Any]) -> str:
    """Question 7 for one criteria line (an item of ``cohort_criteria_items``). Yes: the patient does
    not clearly fail it; no: an inclusion line clearly not met or an exclusion line clearly met."""
    return _fill(_prompt("criterion_question"), number=str(int(item["number"])),
                 criterion_type=_criterion_label(item["criterion_type"]), criterion=str(item["text"]).strip())[0]


def criterion_name(item: Mapping[str, Any]) -> str:
    return f"{CRITERION}_{int(item['number'])}"


def checkable_items(items: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The items question 7 asks about: those with an inclusion or exclusion type (an item whose
    classification never validated has none and is skipped)."""
    return [item for item in items if item.get("criterion_type") in CRITERION_TYPES and str(item["text"]).strip()]


# --------------------------------------------------------------------------- #
# the per-criterion teacher (training labels for question 7)
# --------------------------------------------------------------------------- #


def criterion_teacher_messages(patient_summary: str, items: Sequence[Mapping[str, Any]],
                               number: int) -> list[dict[str, str]]:
    """The teacher request for line ``number``: the same state and question the decision model reads,
    then the answer format. Everything before the line is shared by a pair's lines (prefix caching)."""
    item = items[number - 1]
    if int(item["number"]) != number:
        raise ValueError("criteria items must be numbered 1..n in order")
    content, _ = _fill(_prompt("criterion_teacher"), state=criteria_state(patient_summary, items),
                       question=criterion_question(item))
    return [{"role": "user", "content": content}]


def parse_criterion_label(text: str, criterion_type: str, finish_reason: str = "stop") -> dict[str, str]:
    """The teacher's answer for one line: ``answer`` ("yes" = does not clearly fail the line, the
    question-7 target), ``criterion_status`` and ``reason``. An answer of "no" must name the failing
    status (``not_met`` for an inclusion line, ``met`` for an exclusion line); anything else raises."""
    from matchminer_ai.trials.cohort_schema import _unique_keys, unwrap_json_fence
    if finish_reason != "stop":
        raise ValueError(f"Incomplete response: {finish_reason}")
    try:
        value = json.loads(unwrap_json_fence(text.strip()), object_pairs_hook=_unique_keys)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Response must be valid JSON: {exc}") from exc
    fields = ["answer", "criterion_status", "reason"]
    if not isinstance(value, dict) or list(value) != fields:
        raise ValueError(f"Response must be a JSON object with exactly the fields {fields}")
    if value["answer"] not in ("yes", "no"):
        raise ValueError('answer must be "yes" or "no"')
    if value["criterion_status"] not in CRITERION_STATUSES:
        raise ValueError(f"criterion_status must be one of {list(CRITERION_STATUSES)}")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("reason must be a nonempty explanation")
    failing = "not_met" if _criterion_label(criterion_type) == "Inclusion" else "met"
    if value["answer"] == "no" and value["criterion_status"] != failing:
        raise ValueError(f'answer "no" (clearly fails) on an {criterion_type} criterion requires criterion_status '
                         f'"{failing}"')
    return value


def questions_digest() -> str:
    """SHA-256 of every text that defines the states and questions (the decision contract)."""
    texts = {key: _prompt(key) for key in PROMPTS} | {"match_labels": match_labels_prompt()}
    return hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #

_MODELS: dict[tuple, Any] = {}


def decision_settings(config: Any = None) -> dict[str, Any]:
    """The ``decision_model`` configuration section over :data:`DEFAULTS`."""
    if config is None:
        from matchminer_ai.config import load_default_preset
        config = load_default_preset()
    settings = {**DEFAULTS, **dict(config.raw.get("decision_model") or {})}
    if settings["backend"] != "transformers":
        raise ValueError(f"Unknown decision_model.backend {settings['backend']!r}; use 'transformers'")
    local = Path(str(settings["model_name"])).expanduser().is_dir()
    if not local and not settings.get("revision"):
        # The checkpoint's own code runs (trust_remote_code): a Hub model must be pinned.
        raise ValueError("decision_model.revision must pin a Hub model's commit; or give a local checkpoint directory")
    return settings


def clear_decision_model_cache() -> None:
    _MODELS.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def load_decision_model(settings: Mapping[str, Any]):
    """The decision model (cached): its Hugging Face class with the System One API."""
    name, revision = str(settings["model_name"]), settings.get("revision")
    local = Path(name).expanduser().is_dir()
    key = (name, None if local else revision, settings["device"], settings["dtype"], settings["token_budget"],
           bool(settings["compile"]))
    if key not in _MODELS:
        import torch
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(Path(name).expanduser()) if local else name,
                                          revision=None if local else revision, trust_remote_code=True,
                                          dtype=getattr(torch, settings["dtype"]))
        model = model.to(settings["device"]).eval()
        model.engine.token_budget = int(settings["token_budget"])
        if settings["compile"]:
            model.compile()
        _MODELS[key] = model
    return _MODELS[key]


def d1_module(model, name: str):
    """A module of the checkpoint's own code (``prompt``, ``attention``, ``slots``...)."""
    return importlib.import_module(type(model).__module__.rsplit(".", 1)[0] + "." + name)


def _model_sha(settings: Mapping[str, Any]) -> str | None:
    path = Path(str(settings["model_name"])).expanduser()
    if path.is_dir():
        contract = path / "decision_contract.json"
        return json.loads(contract.read_text()).get("fingerprint") if contract.exists() else None
    return settings.get("revision")


def noul(instructions: str) -> dict[str, str]:
    return {"type": "noul", "instructions": instructions}


def run_decisions(
    requests: Sequence[tuple[str, Mapping[str, str]]],
    *,
    config: Any = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """P(yes) of each named question of each ``(state, {name: instructions})`` request.

    Every question is a ``noul`` question; a request's questions are read together after its state
    (``decision_model.set_size`` at a time, the model's trained size by default). Requests longer
    than ``decision_model.max_length`` tokens (the fine-tuning length) still run, with no
    truncation, and are counted in the metadata's ``over_max_length``.
    """
    settings = decision_settings(config)
    model = load_decision_model(settings)
    prompt = d1_module(model, "prompt")
    answers: list[dict[str, float]] = []
    over, tokens = 0, 0
    batch = max(1, int(settings["batch_size"]))
    for start in range(0, len(requests), batch):
        chunk = requests[start:start + batch]
        calls = [(state, {name: noul(text) for name, text in questions.items()}) for state, questions in chunk]
        for state, questions in calls:
            if model.engine.tokens(state, [prompt.as_question(q) for q in questions.values()]) > settings["max_length"]:
                over += 1
        for out in model.system_one_batch(calls, set_size=settings["set_size"]):
            answers.append({name: float(answer["noul"]) for name, answer in out["answers"].items()})
            tokens += int(out["usage"]["input_tokens"])
        if progress is not None:
            progress(min(start + batch, len(requests)), len(requests))
    local = Path(str(settings["model_name"])).expanduser().is_dir()
    metadata = {"model_name": str(settings["model_name"]), "revision": None if local else settings.get("revision"),
                "model_sha": _model_sha(settings), "backend": settings["backend"], "questions_sha256": questions_digest(),
                "set_size": settings["set_size"] or getattr(model.config, "d1_set_size", None),
                "max_length": settings["max_length"], "requests": len(requests),
                "questions": sum(len(q) for _, q in requests), "input_tokens": tokens, "over_max_length": over}
    return answers, metadata


def attribute_decision(
    state: str,
    questions: Mapping[str, str],
    target: str,
    component_ranges: Mapping[str, tuple[int, int]],
    *,
    config: Any = None,
    top_k: int | None = 30,
) -> dict[str, Any]:
    """Gradient-times-input token attribution of one question's answer.

    The request is read as :func:`run_decisions` serves it (the same packed sequence); the target is
    the question's yes-minus-no logit at its answer slot, so positive scores support yes. Scores are
    the gradient times the input embedding, summed over the hidden dimension, for the state's tokens,
    mapped back to ``component_ranges`` (character ranges of the state).
    """
    import torch

    from .inference import _component_token_attributions

    settings = decision_settings(config)
    model = load_decision_model(settings)
    prompt, attention, slots = (d1_module(model, n) for n in ("prompt", "attention", "slots"))
    engine = model.engine
    names = list(questions)
    qs = [prompt.as_question(noul(questions[name])) for name in names]
    sequences = engine._sequences([(state, qs, [])], set_size=settings["set_size"])
    k = names.index(target)
    _, first, ids, state_len, question, ends, _ = next(s for s in sequences if s[1] <= k < s[1] + len(s[5]))
    slot = ends[k - first][-1]
    prefix, _ = engine._prefix(state, ())
    head = prefix[:prefix.index(state)] if state in prefix else None
    text = prefix + engine._suffixes([qs[first]])[0][0]
    encoded = engine.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [tuple(o) for o in encoded["offset_mapping"][:state_len]]
    yes = prompt.readout_ids(engine.tokenizer, prompt.Noul(""))
    device = model.get_input_embeddings().weight.device
    with torch.enable_grad():
        input_ids = torch.tensor([ids], device=device)
        embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        mask = torch.ones_like(input_ids)
        q = None if question is None else torch.tensor([question])
        s = None if state_len is None else torch.tensor([state_len])
        with attention.state_attention(model, s, q):
            out = attention.text_backbone(model)(inputs_embeds=embeddings, attention_mask=mask, use_cache=False)
        hidden = getattr(out, "last_hidden_state", None)
        hidden = out[0] if hidden is None else hidden
        logits = slots.output_head(model)(hidden[0, slot]).float()
        value = logits[yes[0]].max() - logits[yes[1]].max()
        value.backward()
    with torch.no_grad():
        scores = (embeddings.grad[0] * embeddings[0]).sum(-1).float().tolist()[:state_len]
    if head is None:  # the state's text was escaped (it held a special token): no character mapping
        return {"method": "gradient_x_input", "target": target, "token_attributions": {}, "coverage": {}}
    shift = len(head)
    shifted = {name: (a + shift, b + shift) for name, (a, b) in component_ranges.items()}
    tokens, coverage = _component_token_attributions(prompt=text, offsets=offsets, token_scores=scores,
                                                     component_ranges=shifted, top_k=top_k)
    return {"method": "gradient_x_input", "target": target, "token_attributions": tokens, "coverage": coverage}


def oriented(explanation: dict[str, Any], decision: bool) -> dict[str, Any]:
    """An :func:`attribute_decision` result (whose scores support yes) oriented to support ``decision``."""
    if decision:
        return explanation
    for tokens in explanation["token_attributions"].values():
        for token in tokens:
            for key in ("raw_score", "normalized_score"):
                token[key] = -token[key]
            token["direction"] = {"supports_prediction": "opposes_prediction",
                                  "opposes_prediction": "supports_prediction"}.get(token["direction"], "neutral")
    return explanation


__all__ = ["BASE_MODEL", "BASE_REVISION", "BOILERPLATE", "CRITERION", "CRITERION_LABEL_JSON_SCHEMA", "CRITERION_STATUSES",
           "CRITERION_TYPES",
           "FAMILIES", "attribute_decision", "boilerplate_question", "boilerplate_state", "checkable_items",
           "clear_decision_model_cache", "criteria_state", "criterion_name", "criterion_question",
           "criterion_teacher_messages", "decision_settings", "load_decision_model", "match_questions", "match_state",
           "noul", "oriented", "parse_criterion_label", "questions_digest", "run_decisions"]
