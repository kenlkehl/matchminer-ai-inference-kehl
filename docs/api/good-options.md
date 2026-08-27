# Good Option Catalog and Scoring

Good Option scoring is a patient-trial ranking signal, not an eligibility
determination, response probability, or treatment recommendation. Research and
patient scoring are separate, versioned stages.

## Build the patient-free catalog

`build_good_option_catalog` accepts a list of NCT IDs. It fetches each current
ClinicalTrials.gov record, resolves `DRUG` and `BIOLOGICAL` active entities and
their investigational, uncertain, control, background, or supportive roles,
normalizes drug identities with the bundled NCIt snapshot, and deduplicates the
drugs across the complete run.

Each unique drug is researched once across mechanism and targets, human efficacy
by tumor type and histology, biomarker prevalence, biomarker-directed human
efficacy, and safety. The retriever uses bounded authoritative-source adapters
plus full-text general-web documents. Technical failures are retried with
`Retry-After` or exponential backoff; successful empty results remain distinct
from exhausted technical failures. Searches accept drug identity only and can
never receive patient text.

The configured LLM synthesizes the evidence ledger into structured facts with
validated passage support IDs. The bundle stores two clean projections: a
GoodOption summary without safety and a Help Me Choose summary with safety.
URLs, queries, source labels, registry metadata, and failure notices remain in
the internal Parquet ledger and do not appear in either patient-bearing prompt.

Catalog writes are atomic. `validate_good_option_catalog` verifies hashes,
component versions, NCIt compatibility, trial-drug references, terminal research
states, and structured-fact evidence support. `load_good_option_catalog`
validates by default.

## Score a patient-trial candidate

`evaluate_good_options` and the two scorer-specific APIs require a loaded
catalog. The prompt contains the patient cancer-history summary first, followed
only by clean summaries for scoreable investigational or unresolved drugs in
the trial. Control, background, and supportive drugs are indexed but are never
scored.

The unchanged rubric awards each scoreable drug up to four binary points:

1. evidence of benefit in the patient's disease type;
2. the targeted biomarker is commonly expressed in that disease type;
3. the patient's tumor is documented to have the targeted biomarker; and
4. human evidence of benefit from targeting that biomarker.

The code-derived LLM score is `total_points / (4 * scoreable_drug_count)`. The
four-logit GoodOptionChecker consumes one patient summary and one clean drug
summary per example and aggregates all per-drug, per-criterion probabilities.
Trials with missing catalog data, exhausted research, or missing synthesis are
returned as explicitly unscored; there is no live-search or legacy-snippet
fallback.

The default preset leaves `good_option_checker.model_name` empty because no
versioned public v2 checker artifact is bundled yet. Configure a compatible
trained model to use classifier mode; LLM mode is available through
`llm_good_option`.

::: matchminer_ai.good_options
    options:
      members:
        - build_good_option_catalog
        - validate_good_option_catalog
        - load_good_option_catalog
        - build_role_resolution_messages
        - build_synthesis_messages
        - build_good_option_messages
        - build_good_option_checker_text
        - score_good_options_with_llm
        - score_good_options
        - evaluate_good_options
