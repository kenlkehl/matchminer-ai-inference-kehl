"""What the SQL workup agent is told about an exploration database before it starts.

Two code-made aids save the agent the exploration every requirement used to
repeat:

- :func:`store_profile` counts, once per exploration, what the store holds: rows
  and patients per item table and level, how many items are planned or dated,
  patients per code of every code column, and patients per cancer site group.
  :func:`render_profile` writes it into the agent's system prompt.
- :func:`search_values` is the agent's ``search`` action: for a few words it
  returns every matching code (with its patients) and every distinct text value
  that contains a word, rare ones included, with item and patient counts.

Both run trusted, code-written SELECT statements in the agent's isolated worker,
over the same tables the agent can already query. They show aggregate counts
and distinct extracted values, never patient ids, note text or quotes. Like
exploration results, they reach the agent's LLM endpoint.

The profile is a function of the database and of :data:`PROFILE_VERSION`, so the
cache key (database fingerprint plus the agent contract, which carries the
version) covers it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from matchminer_ai.patients.workup_queries.language import item_types

from .database import COMMON_COLUMNS, table_columns
from .sandbox import SQLSandbox, SQLSandboxError

PROFILE_VERSION = "mmai-workup-sql-profile-v1"

#: Cancer site groups for the profile's breakdown, matched (case-insensitively, RE2)
#: against each diagnosis's primary_site and histology text. Approximate on purpose:
#: a patient is counted in every group one of their diagnoses matches.
SITE_GROUPS: tuple[tuple[str, str], ...] = (
    ("lung", r"lung|lobe|bronch|pulmonary|\bnsclc\b|\bsclc\b"),
    ("breast", r"breast|ductal carcinoma|lobular carcinoma"),
    ("colon", r"colon|cecum|caecum|sigmoid|flexure|colorectal"),
    ("rectum", r"rect"),
    ("anus", r"\banal\b|\banus\b"),
    ("esophagus or stomach", r"esophag|oesophag|gastr|stomach|cardia"),
    ("pancreas", r"pancrea"),
    ("liver or biliary", r"liver|hepat|biliar|bile duct|cholangio|gallbladder|ampull"),
    ("prostate", r"prostat"),
    ("kidney", r"kidney|renal"),
    ("bladder or urothelial", r"bladder|urothel|ureter|urethra"),
    ("ovary, fallopian tube or peritoneum", r"ovar|fallopian|peritone"),
    ("uterus or cervix", r"uter|endometri|cervix"),
    ("head and neck", r"oral|tongue|tonsil|laryn|pharyn|parotid|salivary|nasal|sinus|"
     r"\blip\b|gingiv|mouth|palate|head and neck"),
    ("thyroid", r"thyroid"),
    ("brain or CNS", r"brain|glio|astrocyt|oligodendro|mening|cerebr|spinal cord"),
    ("melanoma", r"melanoma"),
    ("skin (not melanoma)", r"skin|cutaneous|merkel|basal cell"),
    ("sarcoma", r"sarcoma|gist\b|gastrointestinal stromal"),
    ("testis or germ cell", r"testi|germ cell|seminoma"),
    ("lymphoma", r"lymphoma|hodgkin"),
    ("leukemia", r"leuk"),
    ("myeloma or plasma cell", r"myeloma|plasma cell|plasmacytoma"),
    ("myelodysplastic or myeloproliferative", r"myelodysplas|myeloprolif|myelofibrosis|"
     r"polycythemia|thrombocythemia|\bmds\b"),
    ("neuroendocrine", r"neuroendocrine|carcinoid"),
    ("mesothelioma", r"mesothel"),
    ("unknown primary", r"unknown primary|occult primary"),
)
#: Stage IV, M1 or distant extent documented by any of the patient's diagnoses.
ADVANCED_SQL = ("(regexp_matches(coalesce(stage_group, ''), '(^|[^A-Za-z])IV([ABCabc]|[^A-Za-z]|$)')"
                " OR regexp_matches(coalesce(m_category, ''), '(?i)(^|[^a-z])[cpy]?m1')"
                " OR extent_at_diagnosis = 'distant')")
_SHARED = {name for name, _, _ in COMMON_COLUMNS}


class SearchFailed(RuntimeError):
    """A search statement failed in the worker (for example it timed out)."""
_NOT_TEXT = {"cancer_ids", "agents_ncit_codes", "procedure_group"}


def _literal_pattern(word: str) -> str:
    """``word`` as a literal RE2 pattern (punctuation escaped, letters and spaces as is)."""
    return "".join(c if c.isalnum() or c == " " else "\\" + c for c in word)


def _quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def code_columns(types) -> list[tuple[str, str, bool]]:
    """(table, column, is_list) of every code column, plus derived procedure_group."""
    found = []
    for item_type in types:
        for column in table_columns(item_type):
            if column.vocabulary is not None or column.name == "procedure_group":
                found.append((item_type.key, column.name, column.sql_type.endswith("[]")))
    return found


def text_columns(types) -> list[tuple[str, str, bool]]:
    """(table, column, is_list) of every free-text column the search reads."""
    found = []
    for item_type in types:
        for column in table_columns(item_type):
            if (column.vocabulary is None and column.source and column.part is None
                    and column.name not in _SHARED and column.name not in _NOT_TEXT
                    and column.sql_type in {"VARCHAR", "VARCHAR[]"}):
                found.append((item_type.key, column.name, column.sql_type.endswith("[]")))
    return found


def _values(table: str, column: str, is_list: bool) -> str:
    head = f"SELECT {_quote(table)} AS table_name, {_quote(column)} AS column_name, "
    if is_list:
        return (head + f"value, patient_key FROM (SELECT patient_key, unnest({column}) AS value "
                f"FROM {table})")
    return head + f"{column} AS value, patient_key FROM {table}"


def _run(box: SQLSandbox, sql: str, max_rows: int = 5000, timeout: float = 120.0) -> list[list]:
    reply = box.query(sql, max_rows=max_rows, max_cell_chars=None, count_limit=max_rows + 1,
                      timeout=timeout)
    if reply["error"]:
        raise SQLSandboxError(f"A profile query failed ({reply['error']}).")
    return reply["rows"]


def store_profile(box: SQLSandbox, extensions: Iterable[str] = ()) -> dict:
    """Counts of what the exploration database holds (code-written queries only)."""
    types = item_types(extensions)
    tables = [t.key for t in types]
    [[patients, incomplete]] = _run(
        box, "SELECT count(*), count(*) FILTER (WHERE extraction_incomplete) FROM patients")
    [[documents]] = _run(box, "SELECT count(*) FROM documents")
    per_table = " UNION ALL ".join(
        f"SELECT {_quote(t)} AS t, count(*) FILTER (WHERE level = 'document') AS doc, "
        "count(*) FILTER (WHERE level = 'patient') AS pat, count(DISTINCT patient_key) AS pts, "
        "count(*) FILTER (WHERE occurrence = 'planned') AS planned, "
        "count(*) FILTER (WHERE date_precision = 'day' AND date_source <> 'note_date') AS exact, "
        f"count(*) FILTER (WHERE date_bound IS NOT NULL) AS undated FROM {t}" for t in tables)
    table_rows = [dict(zip(("table", "document", "patient", "patients", "planned", "exact_date",
                            "undated"), row)) for row in _run(box, per_table)]
    codes_sql = " UNION ALL ".join(
        f"SELECT * FROM ({_values(t, c, is_list)}) WHERE value IS NOT NULL"
        for t, c, is_list in code_columns(types))
    codes_sql = (f"WITH v AS ({codes_sql} UNION ALL SELECT 'documents', 'document_type', "
                 "document_type, patient_key FROM documents WHERE document_type IS NOT NULL) "
                 "SELECT table_name, column_name, value, count(DISTINCT patient_key) AS n FROM v "
                 "GROUP BY ALL ORDER BY table_name, column_name, n DESC, value")
    codes: dict[str, list[tuple[str, int]]] = {}
    for table, column, value, count in _run(box, codes_sql, max_rows=20000):
        codes.setdefault(f"{table}.{column}", []).append((str(value), int(count)))
    groups = " UNION ALL ".join(f"SELECT {_quote(name)} AS site, {_quote('(?i)' + pattern)} "
                                "AS pattern" for name, pattern in SITE_GROUPS)
    sites_sql = (
        f"WITH g AS ({groups}), d AS (SELECT patient_key, coalesce(primary_site, '') || ' ' || "
        f"coalesce(histology, '') AS text, {ADVANCED_SQL} AS advanced FROM diagnoses), "
        "m AS (SELECT g.site, d.patient_key, bool_or(d.advanced) AS advanced FROM d JOIN g ON "
        "regexp_matches(d.text, g.pattern) GROUP BY ALL) "
        "SELECT site, count(*), count(*) FILTER (WHERE advanced) FROM m GROUP BY site "
        "UNION ALL SELECT '(no site group matched)', count(*), 0 FROM patients p WHERE NOT "
        "EXISTS (SELECT 1 FROM m WHERE m.patient_key = p.patient_key)")
    order = {name: n for n, (name, _) in enumerate(SITE_GROUPS)}
    sites = sorted(((str(s), int(n), int(a)) for s, n, a in _run(box, sites_sql)),
                   key=lambda row: order.get(row[0], len(order)))
    return {"version": PROFILE_VERSION, "patients": int(patients), "documents": int(documents),
            "incomplete": int(incomplete), "tables": table_rows, "codes": codes,
            "sites": sites}


def render_profile(profile: dict) -> str:
    """The profile as prompt text."""
    lines = [
        f"{profile['patients']} patients and {profile['documents']} notes with document-level "
        f"records; {profile['incomplete']} patients have an incomplete extraction.",
        "",
        "Item tables: rows by level (document / patient), patients with any item, planned "
        "items, items with an exact date stated in a note, undated items (date_bound set):",
    ]
    for row in profile["tables"]:
        lines.append(f"- {row['table']}: {row['document']} / {row['patient']} rows, "
                     f"{row['patients']} patients, {row['planned']} planned, "
                     f"{row['exact_date']} exactly dated, {row['undated']} undated")
    lines += ["", "Patients per code (any level, performed or planned; a code not listed has "
              "no items):"]
    for column, values in profile["codes"].items():
        lines.append(f"- {column}: " + ", ".join(f"{code} {n}" for code, n in values))
    lines += ["", "Patients per cancer site group (code matched primary_site and histology text "
              "to these groups; approximate, and a patient with several cancers is in each "
              "group): patients, of whom with stage IV, M1 or distant extent documented. Use it "
              "to judge whether counts are plausible, never to filter the final query:"]
    for site, n, advanced in profile["sites"]:
        lines.append(f"- {site}: {n}" + ("" if site.startswith("(") else f", {advanced}"))
    return "\n".join(lines)


def _code_index(profile: dict) -> dict[str, dict[str, int]]:
    return {column: dict(values) for column, values in profile["codes"].items()}


def search_values(box: SQLSandbox, terms: Sequence[str], profile: dict, *,
                  extensions: Iterable[str] = (), max_rows: int = 60,
                  max_cell_chars: int = 160, timeout: float | None = None) -> dict:
    """Every code and distinct text value containing one of ``terms`` (case-insensitive).

    A word of 3 or fewer characters (an abbreviation such as CEA, CT or EGD) matches
    whole words only; a longer one matches anywhere (a stem such as colonoscop).
    Underscores in codes count as spaces.

    Returns ``codes`` (table.column, code, meaning, patients) from the vocabulary,
    ``values`` (table.column, value, items, patients; values differing only in case
    or surrounding spaces are one value; most patients first, at most ``max_rows``)
    and ``columns`` (per table.column: matching distinct values and patients, so a
    cut list is visible). Rare values are included.
    """
    words = [w for w in dict.fromkeys(str(t).strip().lower() for t in terms) if len(w) >= 2]
    if not words:
        raise ValueError("search needs at least one word of 2 or more characters.")
    types = item_types(extensions)
    literal = ", ".join(
        f"({_quote(w)}, {'true' if len(w) <= 3 else 'false'}, "
        f"{_quote(r'(^|[^a-z0-9])' + _literal_pattern(w) + r'($|[^a-z0-9])')})" for w in words)
    match = ("EXISTS (SELECT 1 FROM w WHERE CASE WHEN w.whole THEN regexp_matches(lower({0}), "
             "w.pattern) ELSE strpos(lower({0}), w.word) > 0 END)")
    found = box.query(
        f"WITH w(word, whole, pattern) AS (VALUES {literal}) SELECT table_name, column_name, "
        "code, meaning FROM vocabulary WHERE " + match.format("replace(code, '_', ' ')")
        + " OR " + match.format("meaning")
        + " ORDER BY table_name, column_name, code", max_rows=500, max_cell_chars=None,
        timeout=timeout)
    if found["error"]:
        raise SearchFailed(str(found["error"]))
    vocabulary = found["rows"]
    counts = _code_index(profile)
    codes = [{"column": f"{table}.{column}", "code": code, "meaning": meaning,
              "patients": counts.get(f"{table}.{column}", {}).get(code, 0)}
             for table, column, code, meaning in vocabulary]
    union = " UNION ALL ".join(_values(t, c, is_list) for t, c, is_list in text_columns(types))
    base = (f"WITH w(word, whole, pattern) AS (VALUES {literal}), v AS ({union}), "
            "hits AS (SELECT table_name, "
            "column_name, min(trim(value)) AS value, count(*) AS items, "
            "count(DISTINCT patient_key) AS patients FROM v WHERE value IS NOT NULL AND "
            + match.format("value") + " GROUP BY table_name, column_name, lower(trim(value)))")
    reply = box.query(base + " SELECT table_name || '.' || column_name, value, items, patients "
                      "FROM hits ORDER BY patients DESC, items DESC, table_name, column_name, "
                      "value", max_rows=max_rows, max_cell_chars=max_cell_chars,
                      timeout=timeout)
    totals = box.query(base + " SELECT table_name || '.' || column_name, count(*), "
                       "sum(patients) FROM hits GROUP BY 1 ORDER BY 3 DESC", max_rows=200,
                       timeout=timeout)
    for result in (reply, totals):
        if result["error"]:
            raise SearchFailed(str(result["error"]))
    return {
        "words": words,
        "codes": codes,
        "values": [{"column": c, "value": v, "items": int(i), "patients": int(p)}
                   for c, v, i, p in reply["rows"]],
        "more_values": max(0, reply["row_count"] - len(reply["rows"])),
        "columns": [{"column": c, "values": int(n), "patient_sum": int(p)}
                    for c, n, p in totals["rows"]],
    }


def render_search(result: dict, max_chars: int) -> str:
    lines = ["Search for: " + ", ".join(repr(w) for w in result["words"]),
             "Codes whose code or meaning contains a word (patients with the code):"]
    lines += [f"- {c['column']} = {c['code']} ({c['meaning']}): {c['patients']}"
              for c in result["codes"]] or ["- none"]
    lines.append("Text columns with matching values: distinct values (patients summed over "
                 "values): " + ("; ".join(f"{c['column']} {c['values']} ({c['patient_sum']})"
                                          for c in result["columns"]) or "none"))
    if result["values"]:
        lines.append("Matching text values (items, patients), most patients first; values "
                     "differing only in case are merged:")
        lines += [f"- {v['column']}: {v['value']!r} {v['items']}, {v['patients']}"
                  for v in result["values"]]
    if result["more_values"] > 0:
        lines.append(f"- ... {result['more_values']} more values not shown: search narrower "
                     "words to see them")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[output cut]"
    return text


__all__ = [
    "PROFILE_VERSION",
    "SITE_GROUPS",
    "SearchFailed",
    "code_columns",
    "render_profile",
    "render_search",
    "search_values",
    "store_profile",
    "text_columns",
]
