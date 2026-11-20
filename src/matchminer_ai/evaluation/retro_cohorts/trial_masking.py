"""Per-event masking of trial identifiers in note text, and the leakage regexes it shares with the measurement.

User, 2026-11-14: cut notes at note_anchor_dt - 14 days and mask the trial's identifiers. ``measure_note_leakage.py``
found that at that cutoff 13.1% of trial starts still have their own protocol number in kept notes, 3.9% an
investigational code from their trial's title and 0.2% their NCT ID. Masking every NN-NNN pattern would hit 27% of
kept notes (lab ranges, doses, ...), so masking is specific to each index event: an event masks the identifiers of
ALL its own trials (every denominator row sharing its pseudo_mrn), and nothing else.

- Protocol numbers (OnCore ``17-123``) in every form ``PROTOCOL`` recognizes (any dash character, one optional
  space on each side), with an optional "DFCI", "DFHCC", "protocol" or "#" prefix -> ``[TRIAL PROTOCOL]``.
- NCT IDs in every form ``NCT`` recognizes (any case, optional space after NCT, leading zero optional) ->
  ``[NCT ID]``.
- Since the 2026-11-14 rerun (user: "do both and rerun"), also the event's own protocol number in three spellings the
  leakage regex does not recognize, again only for the event's own numbers (``LOOSE_PROTOCOL_MASKS``):
  - no dash after a prefix ("DFCI 17123", "protocol #17123", "#17123");
  - a suffix ("17-123-01", "protocol 17-123/A"; the suffix is masked too);
  - after a letter and a slash, dot or dash ("DFCI/17-123", "DFHCC-17-123").
  Bare "17 123" and numeric contexts such as "4.17-123" stay unmasked: too ambiguous.
- Investigational codes from the trial's ctgov title (``title_codes``), any case, with an optional hyphen, other
  dash character or space between letters and digits -> ``[STUDY DRUG CODE]``. CD-number antigens (CD123, CD137) are
  not codes: masking them erased flow-cytometry results (2026-11-14).

``PROTOCOL``, ``NCT``, ``CODE`` and ``title_codes`` are the definitions ``measure_note_leakage.py`` measures with
(moved here unchanged so both use one copy). ``residual_flags`` re-runs that measurement on masked text. Nothing in
this module prints or logs text.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping

import pandas as pd

DASHES = "-‐‑‒–—"  # hyphen-minus, hyphen, non-breaking hyphen, figure dash, en dash, em dash
_DASH_NAMES = dict(zip(DASHES, ["hyphen", "unicode_hyphen", "non_breaking_hyphen", "figure_dash", "en_dash", "em_dash"]))
_NCT_HINT = re.compile("nct", re.I)
PROTOCOL = re.compile(r"(?<![\d.\-/])(\d{2})\s?[-‐‑‒–—]\s?(\d{3})(?![\d\-/])")
NCT = re.compile(r"NCT\s?0?\d{7,8}", re.I)
CODE = re.compile(r"\b([A-Z]{2,6})[- ]?(\d{3,7}[A-Z]?)\b")
CODE_ANY_CASE = re.compile(CODE.pattern, re.I)

PROTOCOL_TOKEN = "[TRIAL PROTOCOL]"
NCT_TOKEN = "[NCT ID]"
CODE_TOKEN = "[STUDY DRUG CODE]"

# PROTOCOL plus up to three prefix tokens ("DFCI", "DFHCC", "protocol", "#"), e.g. "DFCI protocol #17-123". The prefix
# only widens the replaced span: which numbers are masked is decided by PROTOCOL alone.
_PREFIX = r"(?P<prefix>(?:\b(?:DFCI|DFHCC|protocol)\s?[/:]?\s?|#:?\s?){0,3})"
PROTOCOL_MASK = re.compile(_PREFIX + PROTOCOL.pattern, re.I)
_CODE_KEY = re.compile(r"([A-Z]{2,6})(\d{3,7}[A-Z]?)")

# Masked since the 2026-11-14 rerun, for the event's own protocol numbers only (groups 1-2 are the digits).
LOOSE_PROTOCOL_MASKS = {
    "dashless_after_prefix": re.compile(r"(?:\b(?:DFCI|DFHCC|protocol)\s?[#:]?|#)\s?(\d{2})(\d{3})(?!\d)", re.I),
    "suffixed": re.compile(r"(?:(?:\b(?:DFCI|DFHCC|protocol)\s?[/:]?\s?|#:?\s?){0,3})(?<![\d.\-/])(\d{2})\s?[" + DASHES
                           + r"]\s?(\d{3})(?:[\-/][A-Za-z0-9]{1,6})+(?![\dA-Za-z])", re.I),
    "after_letter_and_separator": re.compile(r"(?<=[A-Za-z][.\-/])(\d{2})\s?[" + DASHES + r"]\s?(\d{3})(?![\d\-/])"),
}

# Own identifiers in forms the leakage regexes do not recognize (reported after masking). Since the 2026-11-14 rerun
# the first three are masked by LOOSE_PROTOCOL_MASKS where unambiguous; the rest are reported only.
_LOOSE_PROTOCOL = {
    "protocol_dashless_after_prefix": re.compile(r"(?:\b(?:DFCI|DFHCC|protocol)\s?[#:]?|#)\s?(\d{2})(\d{3})(?!\d)", re.I),
    "protocol_followed_by_dash_or_slash": re.compile(r"(?<![\d.\-/])(\d{2})\s?[-‐‑‒–—]\s?(\d{3})(?=[\-/])"),
    "protocol_after_dot_dash_or_slash": re.compile(r"(?<=[.\-/])(\d{2})\s?[-‐‑‒–—]\s?(\d{3})(?![\d\-/])"),
    "protocol_other_separator": re.compile(
        r"(?<![\d.\-/])(\d{2})(?:\s{2,}[-‐‑‒–—]\s*|\s*[-‐‑‒–—]\s{2,}|\s*[_~]\s*| +)(\d{3})(?![\d\-/])"),
}


def title_codes(title: str) -> list[str]:
    """Investigational codes in a ctgov title, as letters+digits keys (ABC-1234 -> ABC1234)."""
    if not isinstance(title, str):
        return []
    return sorted({(a + b).upper() for a, b in CODE.findall(title)
                   if a.upper() not in {"NCT", "COVID", "SARS", "HIV", "CD"}})  # CD123 etc. are antigens, not drugs


def protocol_key(protocol_number) -> int | None:
    """OnCore '17-123' -> 17123, the key PROTOCOL matches are compared on; None unless the form is NN-NNN."""
    m = re.fullmatch(r"\s*(\d{2})-(\d{3})\s*", str(protocol_number)) if isinstance(protocol_number, str) else None
    return int(m[1] + m[2]) if m else None


def nct_key(nct_id) -> str | None:
    """'NCT01234567' -> '1234567' (digits without leading zeros), the key NCT matches are compared on."""
    m = re.fullmatch(r"\s*NCT(\d{8})\s*", nct_id, re.I) if isinstance(nct_id, str) else None
    return m[1].lstrip("0") if m else None


def _nct_match_key(text: str) -> str:
    return re.sub(r"\D", "", text).lstrip("0")


@dataclass(frozen=True)
class TrialIdentifiers:
    """The identifiers one index event masks: those of every trial among its denominator rows."""
    protocol_keys: frozenset = frozenset()
    nct_keys: frozenset = frozenset()
    codes: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.protocol_keys or self.nct_keys or self.codes)


def event_trial_identifiers(events: pd.DataFrame, titles: Mapping[str, str]) -> tuple[dict, dict]:
    """pseudo_mrn -> TrialIdentifiers over all rows sharing it, and counts (no identifiers) for the run summary.

    ``events`` needs pseudo_mrn, protocol_number and nct_id; ``titles`` maps NCT ID -> ctgov title.
    """
    frame = events[["pseudo_mrn", "protocol_number", "nct_id"]].copy()
    frame["pkey"] = frame.protocol_number.map(protocol_key)
    frame["nkey"] = frame.nct_id.map(nct_key)
    has_title = frame.nct_id.map(lambda n: isinstance(titles.get(n), str) if isinstance(n, str) else False)
    frame["codes"] = frame.nct_id.map(lambda n: title_codes(titles.get(n)) if isinstance(n, str) else [])
    out = {}
    for pseudo_mrn, group in frame.groupby("pseudo_mrn", sort=False):
        out[int(pseudo_mrn)] = TrialIdentifiers(
            frozenset(int(k) for k in group.pkey.dropna()), frozenset(group.nkey.dropna()),
            tuple(sorted({c for codes in group.codes for c in codes})))
    counts = {
        "rows": int(len(frame)),
        "index_events": int(len(out)),
        "index_events_with_several_protocols": int(sum(len(v.protocol_keys) > 1 for v in out.values())),
        "rows_protocol_number_not_nn_nnn": int(frame.pkey.isna().sum()),
        "rows_nct_id_malformed": int(frame.nkey.isna().sum()),
        "rows_without_ctgov_title": int((~has_title).sum()),
        "index_events_with_title_codes": int(sum(bool(v.codes) for v in out.values())),
        "distinct_title_codes": int(len({c for v in out.values() for c in v.codes})),
    }
    return out, counts


@lru_cache(maxsize=8192)
def code_pattern(codes: tuple) -> re.Pattern | None:
    """One case-insensitive pattern for a tuple of title-code keys: letters, an optional dash or space, digits."""
    parts = []
    for code in sorted(set(codes), key=lambda c: (-len(c), c)):
        m = _CODE_KEY.fullmatch(code)
        if m:
            parts.append(re.escape(m[1]) + f"[{DASHES} ]?" + re.escape(m[2]))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I) if parts else None


def _protocol_form(match: re.Match) -> str:
    """Spelling category of a masked protocol mention (counted, never stored with text)."""
    prefix = match["prefix"].lower()
    body = match[0][len(match["prefix"]):]
    kinds = [w for w in ("dfhcc", "dfci", "protocol") if w in prefix] + (["#"] if "#" in prefix else [])
    dash = next(_DASH_NAMES[c] for c in body if c in _DASH_NAMES)
    spaced = "yes" if any(c.isspace() for c in body) else "no"
    return f"prefix={'+'.join(kinds) or 'none'};dash={dash};spaced={spaced}"


def mask_trial_identifiers(text: str, ids: TrialIdentifiers, forms: Counter | None = None,
                           codes_masked: Counter | None = None) -> tuple[str, int, int, int]:
    """Mask one event's own identifiers in ``text``. Returns (text, n_protocol, n_nct, n_code).

    ``forms`` (optional) counts the spelling of each masked protocol mention, ``codes_masked`` each masked code key.
    """
    if not isinstance(text, str) or not ids:
        return text, 0, 0, 0
    n = [0, 0, 0]
    if ids.protocol_keys and any(f"{k:05d}"[2:] in text for k in ids.protocol_keys):
        def protocol(m):
            if int(m[2] + m[3]) not in ids.protocol_keys:
                return m[0]
            n[0] += 1
            if forms is not None:
                forms[_protocol_form(m)] += 1
            return PROTOCOL_TOKEN
        text = PROTOCOL_MASK.sub(protocol, text)
        for name, pattern in LOOSE_PROTOCOL_MASKS.items():
            def loose(m, name=name):
                if int(m[1] + m[2]) not in ids.protocol_keys:
                    return m[0]
                n[0] += 1
                if forms is not None:
                    forms[f"loose={name}"] += 1
                return PROTOCOL_TOKEN
            text = pattern.sub(loose, text)
    if ids.nct_keys and _NCT_HINT.search(text):
        def nct(m):
            if _nct_match_key(m[0]) not in ids.nct_keys:
                return m[0]
            n[1] += 1
            return NCT_TOKEN
        text = NCT.sub(nct, text)
    pattern = code_pattern(ids.codes) if ids.codes and any(re.match(r"[A-Z]+(\d+)", c)[1] in text for c in ids.codes) else None
    if pattern is not None:
        def code(m):
            n[2] += 1
            if codes_masked is not None:
                codes_masked[re.sub(f"[{DASHES} ]", "", m[0]).upper()] += 1
            return CODE_TOKEN
        text = pattern.sub(code, text)
    return text, n[0], n[1], n[2]


RESIDUAL_CHECKS = ("own_protocol", "own_nct", "own_code", "own_code_any_case", *_LOOSE_PROTOCOL,
                   "nct_digits_without_prefix", "code_other_separator")


def residual_flags(text: str, ids: TrialIdentifiers) -> dict[str, bool]:
    """Does ``text`` still contain one of the event's own identifiers?

    The first three checks are measure_note_leakage.py's (``own_protocol``, ``own_nct``, ``own_code``, the last
    case-sensitive as there); after masking they must be False. ``own_code_any_case`` is the same with the
    case-insensitive matching the mask uses. The rest look for own identifiers in spellings the leakage regexes do
    not recognize (no dash, an amendment suffix, other separators, NCT digits without "NCT"): reported, not masked.
    """
    flags = dict.fromkeys(RESIDUAL_CHECKS, False)
    if not isinstance(text, str) or not ids:
        return flags
    if ids.protocol_keys:
        flags["own_protocol"] = bool({int(a + b) for a, b in PROTOCOL.findall(text)} & ids.protocol_keys)
        for name, pattern in _LOOSE_PROTOCOL.items():
            flags[name] = bool({int(a + b) for a, b in pattern.findall(text)} & ids.protocol_keys)
    if ids.nct_keys:
        flags["own_nct"] = bool({_nct_match_key(m) for m in NCT.findall(text)} & ids.nct_keys)
        flags["nct_digits_without_prefix"] = any(
            re.search(r"(?<![\dA-Za-z])0*" + re.escape(k) + r"(?!\d)", text) for k in ids.nct_keys if len(k) >= 6)
    if ids.codes:
        own = set(ids.codes)
        flags["own_code"] = bool({(a + b).upper() for a, b in CODE.findall(text)} & own)
        flags["own_code_any_case"] = bool({(a + b).upper() for a, b in CODE_ANY_CASE.findall(text)} & own)
        flags["code_other_separator"] = bool(_loose_code_pattern(ids.codes).search(text))
    return flags


@lru_cache(maxsize=8192)
def _loose_code_pattern(codes: tuple) -> re.Pattern:
    parts = []
    for code in sorted(set(codes), key=lambda c: (-len(c), c)):
        m = _CODE_KEY.fullmatch(code)
        if m:
            parts.append(re.escape(m[1]) + rf"(?:\s{{2,}}|\s*[_./]\s*|\s*[{DASHES}]\s+|\s+[{DASHES}]\s*)" + re.escape(m[2]))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I) if parts else re.compile(r"(?!x)x")
