# How the Good Options pipeline works

An end-to-end walkthrough, in the order things happen. There are two halves: building the evidence catalog, which
is done once, offline, with no patient involved; then scoring one patient against one trial.

For the API surface see [Good Option Research and Scoring](../api/good-options.md). For the settings named below
see [Configuration](../reference/configuration.md).

## Part A — Building the catalog

### What feeds the research step

You supply a list of NCT numbers. The build downloads each trial's full ClinicalTrials.gov record, then uses an LLM
to read every `DRUG`, `BIOLOGICAL`, and `COMBINATION_PRODUCT` intervention in its arm context and decide which are
actual anticancer agents under test. It drops placebos, premedications, imaging tracers, devices, unnamed
standard-of-care placeholders, and drugs that protect against toxicity or transplant complications (sodium
thiosulfate for ototoxicity, tacrolimus or mycophenolate as GVHD prophylaxis) unless the trial uses them against the
cancer. Each kept agent gets a role: investigational, control, background, supportive, or uncertain. Background
means given the same way to every participant; an agent given only in a comparator arm is control, even when it is
standard of care. Only investigational and uncertain drugs are ever scored.

Each agent's name must come from its entry's own text: the ctgov name, aliases, description, or the descriptions of
its arms, so an entry named "Phase I Arm A" contributes the agent its description names. Every agent is one drug:

- **Regimens** are listed as their components ("Pemetrexed + Cisplatin /Carboplatin" is three drugs), and a standard
  acronym such as FOLFIRI is expanded into its agents from knowledge, with the acronym recorded in `regimen_names`.
- **Names** are generic names (INN/USAN) or development codes, so "LV" is leucovorin and "5-Flurouracil"
  fluorouracil, with the entry's own spelling also recorded in `regimen_names`. A treatment category such as
  "adjuvant chemotherapy" is never an agent.
- **Grounding** is checked in code. A name the entry's text does not contain is rejected unless it expands a regimen
  or spelling that text names, and a name spelled as a combination gets one more attempt. A two-character name such
  as "SG" or "LV" counts as contained only as whole words of the text, never as part of a longer word. A listed
  regimen name the text does not contain is dropped, not failed, so the entry's other grounded agents are kept.

Kept agent names are matched against the bundled NCI Thesaurus snapshot for a canonical name, synonyms, and
definition, then deduplicated across the whole run. An exact NCIt name or synonym wins: the name as written, without
its parentheticals, or a parenthetical on its own ("[177Lu]Lu-PSMA-617 (AAA617)" is found by AAA617). Otherwise a
concept is taken only when it is the single one a rule finds, the strictest rule first:

1. The same name up to case, spaces, hyphens, separators ("/", "+", "and"), salt abbreviations, isotope notation
   ("Ac-225 X" is "Actinium Ac 225 X"), and "X liposome" for "liposomal X".
2. The same name without salts, taking the parent molecule when salts of it tie.
3. The name with route, dosage-form, and biosimilar words moved first or dropped ("Trastuzumab SC").
4. The name with one misspelled word of eight or more letters corrected ("5-Flurouracil").

A wrong match is worse than none, so nothing else is tried: a name's first word alone ("Polatuzumab"), an
abbreviation NCIt lacks ("LV"), and an agent NCIt does not file as a drug ("GM-CSF") stay unmatched. A drug with no
NCIt match gets an identity hashed from the same normalized name, so "Interleukin 2" and "Interleukin-2" stay one
drug. The result is the unique drug set plus a table of which drugs appear in which trials in which role.

### Step 1 — Research each drug on the open internet

For every unique drug the system searches five facets: mechanism and targets, efficacy by tumor type, biomarker
prevalence, biomarker-directed efficacy, and safety.

It queries seven sources: ClinicalTrials.gov, PubMed, Europe PMC, CIViC, DailyMed, cancer.gov, and a general web
search that also downloads and extracts the text of the pages it finds (the built-in provider is an anonymous DDGS
search). Each source gets a query written in its own syntax: boolean for the two literature indexes, prose for the
web ones. The ctgov, label, and curation sources search by exact agent name and ignore the query, so each of their
requests is sent once per drug and its response read by every facet and round.

Three adaptive rounds run. The first searches every facet; later rounds re-search only facets with fewer than two
passages, with the general-web query rephrased. Failures retry with `Retry-After` or exponential backoff, and a
successful empty search stays distinct from an exhausted technical failure. If every source's attempts at a facet
failed, the drug is marked blocked and never synthesized.

Many drugs are researched at once (`subject_concurrency`, `--subject-concurrency` on the command line), so requests
to ClinicalTrials.gov, DailyMed, and the NCBI E-utilities behind PubMed are paced per host in one shared budget,
retries included: by default 0.6, 1.0, and 2.0 requests a second, about 30% under each service's per-IP limit. An
NCBI API key raises E-utilities to 7 a second; give it as a file with `--ncbi-api-key-file` (or
`good_option_catalog.ncbi_api_key_file`), or in the `NCBI_API_KEY` environment variable. Rates and key only schedule
requests, so a checkpointed build resumes after either changes. See
[`good_option_catalog`](../reference/configuration.md#good_option_catalog).

Each result becomes a *passage*: up to 5,000 characters of text plus its source, URL, license, the query that found
it, and a content hash. The drug's NCIt definition is stored as a passage too, so the model can cite it like
anything else.

A second pass searches the drug paired with a disease. The diseases come from the `conditionsModule.conditions` of
the trials the drug appears in (their MeSH condition terms when a trial lists none). Generic conditions such as
"Solid Tumors" are dropped, and spellings of one disease ("Mesothelioma", "Pleural Mesothelioma") count once. Up to
`indication_terms_per_drug` (25) diseases are taken round-robin across the drug's trials, so every trial's disease
is searched before any trial gets a second. The pass re-runs efficacy by tumor, biomarker prevalence, and
biomarker-directed efficacy against the four query-driven sources (PubMed, Europe PMC, cancer.gov, general web)
with the disease as a required clause. It is capped as a whole by `max_indication_passages_per_drug` and
interleaved across diseases, sources, and facets so efficacy results cannot crowd out prevalence findings. Every
passage records the axis that found it in `query_scope` (`drug`, `drug_indication`, `class`, `class_indication`),
so nothing downstream confuses drug-name evidence with disease-conditioned evidence.

### Step 2 — Work out each drug's pharmacologic class

Classes come from the NCI Thesaurus hierarchy in the bundled snapshot. A class is an NCIt concept above some drug
concept, in the Pharmacologic Substance branch, other than a fixed list of umbrellas too broad to group agents that
act alike (`NCIT_UMBRELLA_CLASSES`: Drug, Food, Chemical or Biomedical Material; Pharmacologic Substance;
Antineoplastic Agent; Immunotherapeutic Agent; Biological Agent; Enzyme Inhibitor; Signal Transduction Inhibitor;
Targeted Therapy Agent; and the Antineoplastic Protein Inhibitor, Enzyme Inhibitor, Biological Agent, Antibody, and
Immunomodulating Agent concepts). Everything below them is kept, from Protein Kinase Inhibitor and Monoclonal
Antibody down to PD1 Inhibitor. The chemical-structure branch (Organic Chemical, Nitrogen Compound, ...) is not used.

A drug matched to NCIt gets every class above its concept, in code, with no LLM call: pembrolizumab gets Anti-PD1
Monoclonal Antibody, Monoclonal Antibody, PD1 Inhibitor, Immune Checkpoint Inhibitor, and Immune Checkpoint
Modulator. NCIt can leave a class out, so a local search of NCIt class names and synonyms against the drug's name,
definition, and mechanism passages finds up to `class_candidate_limit` (25) candidate classes, and an LLM may add,
by code, those the passages show the drug belongs to: an FLT3 inhibitor class for emavusertib, which NCIt files
only as an IRAK4 inhibitor. An added class brings its NCIt ancestors with it. The LLM never removes a hierarchy
class, codes outside the candidates are rejected, and an NCIt-classified drug with no candidates is never sent to
it.

A drug NCIt does not classify (no match, or only umbrellas above it, which is common for first-in-human agents such
as golcadomide or the BCL6 degrader BMS-986458) gets its classes from the LLM: candidates first, and a new class,
marked `class_source = "llm"`, only where no candidate fits. A new class is written in canonical form ("CD93
monoclonal antibody", "BRD9 degrader") and identified by a hash of its normalized name, so "PD-L1 inhibitor" and
"PD-L1 Inhibitors" collapse to one class; a new name that spells an NCIt class is stored as that class.
`class_max_per_drug` (3) caps the classes the LLM adds or names per drug, each with a basis (target, mechanism, or
modality), a target, a confidence, and citations to the passages that establish it.

An NCIt class's id is `NCIT:<code>`, its name NCIt's display or preferred name, and its aliases NCIt synonyms. Each
`drug_classes` row records `class_source` (`ncit` or `llm`), `ncit_code`, and `assigned_by`: `ncit_hierarchy`,
`llm_added_ncit`, `llm_added_ncit_ancestor`, or `llm_new`. Hierarchy rows carry no basis, target, confidence, or
citations; the hierarchy is their evidence. Two drugs landing on the same class share everything downstream.

### Step 3 — Research each class

The same machinery as step 1, with the class name and aliases as the subject, run once per class rather than once
per drug: three facets (efficacy by tumor, biomarker prevalence, biomarker-directed efficacy), the four query-driven
sources, one round. The class is also paired with the diseases named by its member drugs' trials, chosen the same
way as a drug's (up to 25, round-robin across trials), and searched on the same three facets. An NCIt class's
definition joins its corpus as a citable passage, as a drug's does.

### Step 4 — Synthesize the evidence

Each class, then each drug, is synthesized from its own corpus in its own chain of LLM calls. Its passages are
ordered round-robin across `(query_scope, source)` so no retrieval axis is starved, then split into chunks of at most
`synthesis_chunk_max_tokens` (60,000) of passage text for drugs and `class_evidence_max_tokens` (60,000) for classes.
Nothing is discarded: the chunks run in order, one call each, with the chunk's passages numbered `P1`..`Pn`. Bigger
chunks mean fewer calls but fewer facts, because the model condenses more of a large chunk.

Each call sees the facts built so far, numbered `F1`..`Fn`, and returns JSON with six arrays (mechanism and targets,
efficacy by tumor, biomarker prevalence, biomarker-directed efficacy, safety, limitations) holding only what this
chunk adds. Every item must be one self-contained sentence carrying its numbers, plus `support_ids` citing passage
numbers, plus labels for evidence level, tumor type, histology, regimen, biomarker, and prevalence denominator where
they apply. An item that corrects or supersedes an existing fact names it in `replaces`. Synthesis keeps supported
evidence at every maturity level, from ontology and preclinical results to mature trials, and labels its level.

Code then checks the work. Citations must resolve to real evidence IDs, uncited items are dropped, and each
surviving fact is stamped **`scope: agent`** (drug corpus) or **`scope: class`** (class corpus) in code, not by the
model. The new items are merged in code: every existing fact is kept unless a validated item replaces it, and an item
repeating an existing fact's text is dropped. Because existing facts never pass back through the model's output, a
broad class that has accumulated hundreds of them cannot silently lose any. Failures retry with the exact
validation error quoted back; if every ordinary attempt hit the output-token limit, the last one disables thinking.
A chunk that never validates leaves the facts so far unchanged.

The separation of agent and class corpora is the point rather than an implementation detail. A pooled budget would
let a flood of same-class passages evict an agent's own data, and the agent's own data is what the rubric asks the
scorer to prefer.

### Step 5 — Render readable summaries

Structured facts become plain text. Each bullet carries its scope and attribution:

```
- [scope: agent | evidence level: mature trial results | tumor type: Colon Cancer
   | histology: Stage III dMMR | regimen: Atezolizumab and mFOLFOX6]
  In a phase 3 trial of resected stage III dMMR colon cancer, adjuvant
  atezolizumab plus mFOLFOX6 improved 3-year DFS to 86.3% versus 76.2%
  (HR 0.50, p<0.001).
```

The catalog stores two renders per drug and one per class:

| Column | Content | Size (`good_option_catalog`) |
| --- | --- | --- |
| `good_option_summary` | Drug facts without safety | `good_option_summary_max_tokens` (6,000) |
| `help_me_choose_summary` | The same render with safety | `help_me_choose_summary_max_tokens` (9,000) |
| `class_option_summary` | Class facts without safety | `class_option_summary_max_tokens` (4,000) |

Sizes are converted at 4 characters a token. The character budget is shared across sections by weighted max-min
fair share: a section with one short line releases its remainder, and when every section overflows, efficacy by
tumor and biomarker-directed efficacy get three and two shares to each other section's one.

Large classes synthesize hundreds of facts, so what the budget keeps matters:

- **Ranking:** efficacy and safety facts are ranked by their evidence-level label (phase 3, regulatory or guideline
  first; then randomized, phase 2, or meta-analysis; then other human evidence; then case reports), taking turns
  across tumor types within a rank so one disease cannot crowd out another.
- **Preclinical results:** efficacy findings labelled preclinical (in vitro, animal, xenograft, and nothing marking
  them as human) move out of the human efficacy sections into "Preclinical efficacy (not human evidence)", which
  only receives budget the other sections leave unused.
- **Truncation:** a truncated section ends with a count of omitted findings, passage handles such as `(P3)` are
  removed, and repeated statements are printed once.

Ranking reads free-text labels with patterns, so it is a heuristic ordering, not a validated evidence grade. It
changes only the rendered text; structured facts are stored unchanged.

The stored renders are fixed-size. At scoring time the same renderer runs again from the stored facts with a budget
sized to the trial (see Part B), so the ranking above decides what survives there too.

Rendered text is the only catalog content a patient-bearing prompt ever sees. URLs, source names, queries, and
ctgov metadata stay in the internal tables.

### Step 6 — Write and validate

Everything is written to Parquet: trial records, screening decisions, the trial-drug index, drug summaries, drug
passages, class assignments, class passages, class summaries, and a full audit log of every search attempt. A
manifest records file hashes, a `compatibility_id` fingerprinting those hashes, a fingerprint of the catalog prompt
text, the NCIt version, and counts.

Validation re-checks the bundle: hashes match the files and the `compatibility_id`, every citation points at a
passage that exists for that drug or class, drug facts carry `scope: agent` and class facts `scope: class`, there
are no duplicate keys, and roles are legal. `load_good_option_catalog` runs this by default.

Every stage checkpoints to disk, so a build that dies at hour six resumes rather than restarting.

## Part B — Scoring one patient against one trial

1. **Look up the trial.** If it is missing, its ctgov fetch failed, it has no scoreable drug, or any scoreable drug
   failed research or synthesis, the pair comes back explicitly unscored with a reason in `good_option_status`.
   There is no fallback to live search.
2. **Collect the stored facts for each scoreable drug.**
3. **Collect one class block per distinct class** across those drugs, each labelled with which drugs it covers. Two
   drugs sharing a class produce one block, not two.
4. **Fit the evidence to the context.** Drugs and classes are rendered from their facts to fit what the scoring
   model's context leaves, not the fixed-size renders stored in the catalog. They share that room by weighted fair
   share, drugs ahead of classes. A subject short of room first shortens each line to keep every fact, and drops the
   weakest facts only when that is not enough.
5. **Build the prompt:** the patient's cancer history first, then the drug sections and class blocks, then the
   rubric, then the exact list of drug names to score.
6. **The model answers four yes-or-no questions per drug:**
    1. Has this drug, or its class, helped people with what this patient has? (A drug the patient has already
       received, or is receiving, scores 0 here.)
    2. Is the feature it targets common in this disease (20% or more)?
    3. Does this patient's tumor actually have that feature?
    4. Has targeting that feature helped people?

    Each needs a rationale naming the specific evidence or the specific gap, and saying whether it relied on the
    agent's own evidence or its class's.

7. **Code validates strictly.** The answer must assess exactly the supplied drugs (names compared without case),
   points must be the integers 0 or 1, and rationales must be non-empty. The score is total points over four times
   the drug count. Malformed answers retry individually with the exact parser error, and optionally once more with
   model thinking disabled.

`check_good_options` runs these steps on demand with the `good_option_check` preset settings. It loads the catalog
once and reuses it until its files change, caps the model's answer at 50,000 tokens, and fits the evidence to the
context length the serving endpoint reports. A malformed answer gets three tries, then one more with thinking off.

The score ranks trials for human review. It is not an eligibility decision, a response probability, or a treatment
recommendation.

A second scoring path, a four-logit classifier reading one patient and one drug summary, is deprecated: it never
saw class evidence and no trained model was published.

## Known limits

- **Patient-free retrieval.** Disease-conditioned searches use the diseases the drug's and class's trials name, not
  the patient's, and at most 25 per drug or class. A patient whose cancer is not among them, or whose trial's disease
  was cut by that cap, can be scored against evidence that never names the patient's disease.
- **Evidence as of the build.** Trial records and literature are fetched when the catalog is built. Rebuild it to
  pick up new results or changed protocols.
- **Heuristic ranking.** What survives a tight budget depends on free-text evidence-level labels, as described in
  step 5.
