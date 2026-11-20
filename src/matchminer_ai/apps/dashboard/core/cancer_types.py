"""Cancer types of a trial from regular expressions over its CT.gov cohorts (no LLM step; user, 2026-11-09).

Each CT.gov corpus cohort (``<NCT>-ctgov-cohort-<n>``: merged, then the v4 boilerplate step) is tagged from its
key inclusion lines (``items`` with ``criterion_type == "inclusion"`` and not boilerplate) and its name. Exclusion
and boilerplate lines are not read: they name the cancers a trial keeps out ("no other malignancy except basal
cell carcinoma ..."). A match is dropped when its clause (up to ". ; : )") says "except", "excluding", "other
than" or the like before it. A trial's types are the union of its CT.gov cohorts' types; protocol cohorts are not tagged.

These are keyword screens for browsing, not a classification: a cohort can mention a type it does not take
(e.g. "prior breast cancer allowed") or take one it names only by histology the patterns miss.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import re

NOUN = r"(?:cancers?|carcinomas?|adenocarcinomas?|neoplasms?|malignanc(?:y|ies)|tumou?rs?)"
# Gene and protein names that contain a cancer type: "anaplastic lymphoma kinase", "Wilms tumor 1", "breast cancer
# resistance protein", "melanoma-associated antigen".
GENE = r"(?![- ](?:genes?|susceptibility|resistance|antigens?|associated|kinases?|specific|protein|1\b|2\b))"


def organ(words: str, gap: int = 3) -> str:
    """``words`` followed within ``gap`` words by a cancer noun ("lung squamous cell carcinoma", "breast or
    ovarian cancer"), or a cancer noun "of the" ``words`` ("carcinoma of the uterine cervix"). Gap words that
    name a metastasis, an organ function or a lesion elsewhere break the phrase ("liver metastases")."""
    between = r"(?:[\s,/()-]+(?!metasta|function|involve|lesion|transplant|by\b|from\b|to\b|with\b|in\b)[\w+-]+)"
    return (rf"\b(?:{words})\w*{between}{{0,{gap}}}?[\s,/()-]+{NOUN}\b{GENE}"
            rf"|\b{NOUN}\s+(?:(?:located|arising|originating)\s+)?(?:of|in|from)\s+the\s+(?:[\w-]+\s+){{0,2}}?(?:{words})")


# (key, label, group, words, abbreviations): words match case-insensitively, abbreviations case-sensitively.
TYPES = [
    ("solid_any", "Solid tumors (any / basket)", "Basket",
     r"(?<!criteria in )\bso[il]{2}d\s+(?:tumou?rs?|malignanc|neoplasm|cancers?)|tumou?r[- ]agnostic|histology[- ]agnostic|"
     r"\b(?:MSI-H|dMMR|TMB-H)\s+(?:or\s+\S+\s+)?(?:cancers?|tumou?rs?|solid)|"
     r"\b(?:advanced|metastatic)\s+(?:or\s+metastatic\s+)?malignant\s+disease|"
     r"\b(?:any|all)\s+(?:solid\s+)?(?:tumou?r|cancer)\s+types?|"
     # generic wording with no site: "advanced malignant tumors", "metastatic cancer", "advanced malignancy"
     r"\b(?:advanced|metastatic|unresectable|refractory)\s+(?:or\s+(?:metastatic|recurrent|unresectable)\s+)?"
     r"(?:malignant\s+)?(?:tumou?rs?|cancers?|malignanc(?:y|ies)|neoplasms?)\b(?!\s+of\b)", None),
    ("heme_any", "Hematologic malignancy (any)", "Basket",
     r"ha?ematolog\w*\s+(?:malignanc|cancer|neoplas|tumou?r)|blood\s+cancers?|(?:lymphoid|myeloid)\s+malignanc", None),
    ("breast", "Breast", "Solid",
     organ(r"breast|mammary") + r"|triple[- ]negative\s+breast|invasive\s+(?:ductal|lobular)\s+(?:adeno)?carcinoma",
     r"\b(?:TNBC|DCIS)\b"),
    ("lung_any", "Lung (any)", "Solid",
     organ(r"lung|pulmonary|bronch") + r"|non[- ]?small[- ]?cell|small[- ]cell\s+lung|(?:extensive|limited)[- ]stage\s+small[- ]cell",
     r"\bN?SCLC\b"),
    ("lung_nsclc", "Lung: non-small cell", "Solid", r"non[- ]?small[- ]?cell", r"\bNSCLC\b"),
    ("lung_sclc", "Lung: small cell", "Solid",
     r"(?<!non-)(?<!non )(?<!non)\bsmall[- ]cell\s+lung|(?:extensive|limited)[- ]stage\s+small[- ]cell|"
     r"(?<!non-)(?<!non )(?<!non)\bsmall[- ]cell\s+(?:carcinoma|cancer)\s+of\s+the\s+lung", r"\bSCLC\b"),
    ("mesothelioma", "Mesothelioma", "Solid", r"mesothelioma", None),
    ("thymic", "Thymic (thymoma / thymic carcinoma)", "Solid", r"thymoma|thymic\s+(?:carcinoma|epithelial|tumou?r|cancer)", None),
    ("head_neck", "Head and neck", "Solid",
     r"head\s+(?:and|&)\s+neck|oro-?pharyn|naso-?pharyn|hypo-?pharyn|oral\s+cavity|sino-?nasal|paranasal|"
     + organ(r"laryn|oral|tongue|tonsil|salivary|parotid|submandibular|sublingual|lip|buccal|gingiv|palat|mouth"),
     r"\b(?:HNSCC|SCCHN|NPC)\b"),
    ("thyroid", "Thyroid", "Solid",
     organ(r"thyroid") + r"|(?:papillary|anaplastic|medullary|follicular|differentiated)\s+thyroid", None),
    ("esophageal", "Esophageal", "Solid", organ(r"o?esophag"), r"\bESCC\b"),
    ("gastric", "Gastric / GEJ", "Solid",
     organ(r"gastric|stomach") + r"|gastro-?o?esophageal\s+(?:junction|adeno|cancer|carcinoma)|o?esophagogastric",
     r"\b(?:GEJ|GEA|OGA)\b"),
    ("colorectal", "Colorectal", "Solid", organ(r"colon|colorectal|rectal|rectum|colonic|rectosigmoid|sigmoid|cec(?:al|um)") + r"|colorectal",
     r"\b(?:m?CRC|CRLM)\b"),
    ("small_bowel", "Small bowel / appendiceal", "Solid", organ(r"small\s+(?:intestin|bowel)|duoden|jejun|ile(?:al|um)|appendi"),
     None),
    ("anal", "Anal", "Solid", organ(r"anal|anus"), None),
    ("pancreatic", "Pancreatic", "Solid", organ(r"pancrea"), r"\bPDAC\b"),
    ("liver", "Liver (hepatocellular)", "Solid", r"hepatocellular|hepatoma|" + organ(r"liver|hepatic", gap=1),
     r"\bHCC\b"),
    ("biliary", "Biliary / gallbladder", "Solid",
     r"cholangiocarcinoma|" + organ(r"biliary|bile\s+duct|gallbladder|gall\s+bladder|ampulla|ampullary"), r"\bBTC\b"),
    ("neuroendocrine", "Neuroendocrine", "Solid",
     r"neuroendocrine|carcinoid|pheochromocytoma|paraganglioma|merkel", r"\b(?:p?NETs?|NEC|MCC)\b"),
    ("gist", "GIST", "Solid", r"gastrointestinal\s+stromal", r"\bGIST\b"),
    ("kidney", "Kidney (renal cell)", "Solid", r"renal\s+cell|wilms(?!\s+tumou?r\s*(?:1|gene|protein|antigen))|" + organ(r"kidney|renal(?!\s+pelvi)", gap=1),
     r"\b(?:cc)?RCC\b"),
    ("bladder", "Bladder / urothelial", "Solid",
     r"urothelial|transitional\s+cell\s+carcinoma|" + organ(r"bladder|ureter|urethra", gap=1),
     r"\b(?:NMIBC|MIBC|UTUC|m?UC)\b|\bCIS\s+(?:of\s+the\s+)?bladder"),
    ("prostate", "Prostate", "Solid", organ(r"prostat") + r"|castrat\w*[- ](?:resistant|sensitive)",
     r"\bm?(?:CRPC|HSPC|CSPC)\b|\bPCa\b"),
    ("testicular", "Testicular / germ cell", "Solid", r"germ[- ]cell\s+tumou?r|seminoma|" + organ(r"testic|testis"),
     None),
    ("penile", "Penile", "Solid", organ(r"penile|penis"), r"\bPSqCC\b"),
    ("ovarian", "Ovarian / fallopian / peritoneal", "Solid",
     organ(r"ovar", gap=6) + r"|fallopian\s+tube|primary\s+peritoneal|high[- ]grade\s+serous", None),
    ("endometrial", "Endometrial / uterine", "Solid",
     organ(r"endometri(?!oid)|uter(?!\w*\s+cervi)") + r"|uterine\s+(?:serous|sarcoma|carcinosarcoma|leiomyosarcoma)", None),
    ("cervical", "Cervical", "Solid", organ(r"cervical(?!\s+(?:spine|spinal|vertebra|lymph|node|esophag))|cervix"), None),
    ("vulvar_vaginal", "Vulvar / vaginal", "Solid", organ(r"vulva|vagina"), None),
    ("melanoma", "Melanoma", "Solid", r"melanoma\b" + GENE, None),
    ("skin_other", "Skin, non-melanoma", "Solid",
     r"cutaneous\s+squamous|basal[- ]cell\s+carcinoma|merkel[- ]cell|" + organ(r"skin", gap=1), r"\b(?:cSCC|MCC)\b"),
    ("sarcoma", "Sarcoma", "Solid",
     r"(?<!carcino)(?<!glio)sarcoma|gastrointestinal\s+stromal|desmoid|chordoma|ewing|rhabdoid", r"\b(?:GIST|STS)\b"),
    ("brain_cns", "Brain / CNS", "Solid",
     r"gliobl?astoma|glioma|astrocytoma|oligodendroglioma|ependymoma|medulloblastoma|meningioma|craniopharyngioma|"
     r"(?:brain|cns|central\s+nervous\s+system|intracranial)\s+(?:primary\s+)?(?:tumou?rs?|cancers?|neoplasms?|malignanc|lymphoma)|"
     r"primary\s+(?:brain|cns)\b", r"\b(?:GBM|DIPG|DMG|HGG|LGG)\b"),
    ("unknown_primary", "Unknown primary (CUP)", "Solid", r"unknown\s+primary|occult\s+primary", r"\bCUP\b"),
    ("pediatric_solid", "Pediatric solid (neuroblastoma, Wilms ...)", "Solid",
     r"neuroblastoma|wilms(?!\s+tumou?r\s*(?:1|gene|protein|antigen))|nephroblastoma|retinoblastoma|hepatoblastoma|rhabdomyosarcoma|osteosarcoma|ewing", None),
    ("lymphoma", "Lymphoma", "Hematologic",
     r"lymphoma\b" + GENE + r"|hodgkin|mycosis\s+fungoides|s[eé]zary|waldenstr|macroglobulin|lymphoproliferative",
     r"\b(?:B-)?NHL\b|\b(?:DLBCL|LBCL|PMBCL|PTCL|CTCL|ALCL|ATLL?|MCL|MZL|FL|SLL|ENKTC?L|NKTCL|PCNSL|cHL)\b|\bMF\s+(?:or|and|/)\s+SS\b"),
    ("leukemia", "Leukemia", "Hematologic", r"leuka?emi",
     r"\b(?:AML|CLL|CML|CMML|[BT]-ALL|APL)\b|\((?:[BT]-)?ALL\)|\b(?:MRD|Ph|R/R)[+-]?\s+ALL\b|"
     r"(?:positive|negative|[Rr]elapsed|[Rr]efractory)\s+(?:B-cell\s+)?ALL\b"),
    ("myeloma", "Myeloma / plasma cell", "Hematologic",
     r"myeloma|plasma[- ]cell\s+(?:neoplas|leuka?emi|dyscrasia|disorder)|plasmacytoma|light[- ]chain\s+amyloidosis",
     r"\bAL\b[^.;]{0,25}amyloid"),
    ("mds_mpn", "MDS / MPN", "Hematologic",
     r"myelodysplas|myeloproliferative|myelofibrosis|polycyth\w*\s+vera|essential\s+thrombocyth",
     r"\b(?:MDS|MPN|CMML|PV)\b"),
]
TYPE_LABELS = {key: label for key, label, *_ in TYPES}
_COMPILED = [(key, re.compile(words, re.I), re.compile(abbreviations) if abbreviations else None)
             for key, _, _, words, abbreviations in TYPES]
NEGATION = re.compile(r"\b(?:except|excluding|exclusive\s+of|other\s+than|with\s+the\s+exception\s+of|apart\s+from|rather\s+than|"
                      r"excepting)\b[^.;:)]*$", re.I)
VERSION = hashlib.sha256(json.dumps([TYPES, NEGATION.pattern]).encode()).hexdigest()[:16]


@lru_cache(maxsize=None)  # patients repeat their header lines (87,890 patients, ~9,000 distinct cancer types)
def _tag(text: str) -> frozenset:
    return frozenset(_tag_uncached(text))


def tag_text(text: str) -> set[str]:
    return set(_tag(text))


def _tag_uncached(text: str) -> set[str]:
    """The cancer types one text names, dropping matches whose clause says "except ..." (or the like) first."""
    found = set()
    for key, insensitive, sensitive in _COMPILED:
        for pattern in (insensitive, sensitive):
            if pattern is None or key in found:
                continue
            for match in pattern.finditer(text):
                if not NEGATION.search(text[max(0, match.start() - 120):match.start()]):
                    found.add(key)
                    break
    if found & {"lung_nsclc", "lung_sclc"}:
        found.add("lung_any")
    return found


def cohort_texts(row: dict) -> list[str]:
    """What a cohort is tagged from: its name and its key inclusion lines (else its key criteria text)."""
    items = row.get("items")
    if isinstance(items, str):
        items = json.loads(items)
    lines = [i.get("text") or "" for i in (items or [])
             if i.get("criterion_type") == "inclusion" and not i.get("boilerplate")]
    if not items:
        lines = [row.get("text") or ""]
    return [row.get("cohort_name") or ""] + lines


def cohort_types(row: dict) -> list[str]:
    found = set()
    for text in cohort_texts(row):
        found |= tag_text(text)
    return [key for key, *_ in TYPES if key in found]


BASKET = {key for key, _, group, *_ in TYPES if group == "Basket"}
SMALL_CELL = re.compile(r"(?<!non-)(?<!non )(?<!non)\bsmall[- ]cell", re.I)
NON_SMALL_CELL = re.compile(r"non[- ]?small|adenocarcinoma|squamous|large[- ]cell|adenosquamous", re.I)


def patient_types(header: dict) -> list[str]:
    """A synthetic patient's cancer types: the same patterns over the "Cancer type:" and "Histology:" lines at the
    top of its key summary (no basket types). A lung cancer whose histology is small cell is small cell lung
    cancer, one whose histology is adenocarcinoma, squamous or large cell is non-small cell."""
    cancer, histology = header.get("cancer_type") or "", header.get("histology") or ""
    found = (tag_text(cancer) | tag_text(histology)) - BASKET
    if "lung_any" in found and not found & {"lung_nsclc", "lung_sclc"}:
        if SMALL_CELL.search(histology):
            found.add("lung_sclc")
        elif NON_SMALL_CELL.search(histology):
            found.add("lung_nsclc")
    return [key for key, *_ in TYPES if key in found]


def describe() -> list[dict]:
    return [{"key": key, "label": label, "group": group} for key, label, group, *_ in TYPES]
