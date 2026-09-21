# How the Good Options pipeline works

An end-to-end walkthrough, in the order things happen. Two halves: building the
evidence catalog, which is done once, offline, with no patient involved; then
scoring one patient against one trial.

For the API surface see [Good Option Research and Scoring](../api/good-options.md).
For the settings named below see [Configuration](../reference/configuration.md).

## Part A — Building the catalog

### What feeds the research step

You supply a list of NCT numbers. The build downloads each trial's full
ClinicalTrials.gov record, then uses an LLM to read every `DRUG` and
`BIOLOGICAL` intervention in its arm context and decide which are actual
anticancer agents under test — dropping placebos, premedications, imaging
tracers, and unnamed standard-of-care placeholders. Each survivor gets a role:
investigational, control, background, supportive, or uncertain. Only
investigational and uncertain drugs are ever scored.

Surviving agent names are matched against the bundled NCI Thesaurus snapshot for
a canonical name, synonyms, and definition, then deduplicated across the whole
run. A drug with no NCIt match gets a name-hash identity instead. The result is
the unique drug set plus a table of which drugs appear in which trials in which
role.

### Step 1 — Research each drug on the open internet

For every unique drug the system searches five facets: mechanism and targets,
efficacy by tumor type, biomarker prevalence, biomarker-directed efficacy, and
safety.

It queries seven sources: ClinicalTrials.gov, PubMed, Europe PMC, CIViC,
DailyMed, cancer.gov, and a general web search that also downloads and extracts
the text of the pages it finds. Each source gets a query written in its own
syntax — boolean for the two literature indexes, prose for the web ones. The
registry, label, and curation sources search by exact agent name and ignore the
query.

Three adaptive rounds run: the first searches everything, later rounds re-search
only facets that came back nearly empty, with broadened phrasing. Failures retry
with `Retry-After` or exponential backoff, and a successful empty search stays
distinct from an exhausted technical failure. If every source fails on a facet
across all rounds the drug is marked blocked and never synthesized.

Each result becomes a *passage*: up to 5,000 characters of text plus its source,
URL, license, the query that found it, and a content hash. The drug's NCIt
definition is materialized as a passage too, so the model can cite it like
anything else.

A second pass searches the drug paired with a disease. The diseases come from
the `conditionsModule.conditions` of the trials the drug appears in — up to
`indication_terms_per_drug`, ranked by how many of those trials name each one —
and re-runs the two efficacy facets against the four query-driven sources with
the disease as a required clause. It is capped as a whole by
`max_indication_passages_per_drug` and interleaved across diseases and sources.
Every passage records the axis that found it in `query_scope`, so nothing
downstream confuses drug-name evidence with disease-conditioned evidence.

### Step 2 — Work out each drug's pharmacologic class

An LLM reads the drug's own mechanism passages, plus its name, aliases, and NCIt
definition as a hint, and names up to `class_max_per_drug` classes: each with a
basis (target, mechanism, or modality), a target, alternative names, a
confidence, and citations to the passages that establish it.

NCIt is deliberately not the source. It has nothing usable for the
first-in-human agents that most need class evidence, which is why DCBY02
resolves to "CD93 monoclonal antibody" and FHD-609 to "BRD9 degrader" from their
retrieved mechanism text rather than from an ontology entry.

Class identity is a hash of the normalized class name, so "PD-L1 inhibitor",
"PD-L1 Inhibitors", and "anti-PD-L1 inhibitor" collapse to one thing and two
drugs landing on the same class share everything downstream.

### Step 3 — Research each class

The same machinery as step 1, with the class name and aliases as the subject,
run once per class rather than once per drug. Three facets, four query-driven
sources, one round.

### Step 4 — Synthesize the drug evidence

A drug's passages go into one LLM call, numbered `P1`..`Pn`, under
`synthesis_evidence_max_tokens` of passage text. Where the ledger exceeds the
budget, passages are taken round-robin across `(query_scope, source)` so no
retrieval axis is starved.

The model returns JSON with six arrays — mechanism and targets, efficacy by
tumor, biomarker prevalence, biomarker-directed efficacy, safety, limitations.
Every item must be one self-contained sentence carrying its numbers, plus
`support_ids` citing passage numbers, plus labels for tumor type, histology,
regimen, biomarker, and prevalence denominator where they apply.

Code then checks the work: citations resolve back to real evidence IDs, uncited
items are dropped, and each surviving fact is stamped **`scope: agent`**.
Failures retry with the exact validation error quoted back. A drug that never
validates is left blocked rather than stored.

### Step 5 — Synthesize the class evidence

The same prompt, once per class, on its own `class_evidence_max_tokens` budget.
Facts are stamped **`scope: class`**.

The separation is the point rather than an implementation detail. A pooled
budget would let a flood of same-class passages evict an agent's own data, and
the agent's own data is what the rubric asks the scorer to prefer.

### Step 6 — Render readable summaries

Structured facts become plain text. Each bullet carries its scope and
attribution:

```
- [scope: agent | tumor type: Colon Cancer | histology: Stage III dMMR
   | regimen: Atezolizumab and mFOLFOX6 | evidence level: mature trial results]
  In a phase 3 trial of resected stage III dMMR colon cancer, adjuvant
  atezolizumab plus mFOLFOX6 improved 3-year DFS to 86.3% versus 76.2%
  (HR 0.50, p<0.001).
```

Two versions per drug — one without safety for scoring, one with safety for Help
Me Choose — plus one per class. The character budget is shared across sections
by max-min fair share rather than split evenly, so a section with one short line
releases its remainder to the section holding the disease-specific results.

These summaries are the only thing a patient-bearing prompt ever sees. URLs,
source names, queries, and registry metadata stay in the internal tables.

### Step 7 — Write and validate

Everything is written to Parquet: trial records, screening decisions, the
trial-drug index, drug summaries, drug passages, class assignments, class
passages, class summaries, and a full audit log of every search attempt. A
manifest records schema and prompt versions, a compatibility fingerprint, file
hashes, and counts.

Validation re-checks the bundle: hashes match, versions match the running code,
every citation points at a passage that exists for that drug or class, drug
facts carry `scope: agent` and class facts `scope: class`, no duplicate keys,
roles are legal. `load_good_option_catalog` runs this by default.

Every stage checkpoints to disk, so a build that dies at hour six resumes rather
than restarting.

## Part B — Scoring one patient against one trial

1. **Look up the trial.** If it is missing, its registry fetch failed, it has no
   scoreable drug, or any scoreable drug failed research or synthesis, the pair
   comes back explicitly unscored with a reason. There is no fallback to live
   search.
2. **Collect the Good Options summary for each scoreable drug.**
3. **Collect one class block per distinct class** across those drugs, each
   labelled with which drugs it speaks for. Two drugs sharing a class produce
   one block, not two.
4. **Build the prompt:** the patient's cancer history first, then the drug
   summaries and class blocks, then the rubric, then the exact list of drug
   names to score.
5. **The model answers four yes-or-no questions per drug:**
   1. Has this drug, or its class, helped people with what this patient has?
   2. Is the feature it targets common in this disease (20% or more)?
   3. Does this patient's tumor actually have that feature?
   4. Has targeting that feature helped people?

   Each needs a rationale naming the specific evidence or the specific gap, and
   saying whether it relied on the agent's own evidence or its class's.
6. **Code validates strictly** — drug names must match exactly, points must be
   the integers 0 or 1, rationales must be non-empty — and computes the score as
   total points over four times the drug count. Malformed answers retry
   individually with the exact parser error, and optionally once more with model
   thinking disabled.

A second scoring path exists: a four-logit classifier reading one patient and
one drug summary. It is wired up but ships with no trained model configured.

## Known limits

Two soft spots, both the same shape. The disease-conditioned search picks
diseases by what the *drug* is usually studied in rather than what the *trial
being scored* is about, and the class search has no disease attached at all.
Together they leave roughly a third of drug-trial pairs with no evidence
retrieved for the patient's own disease.
