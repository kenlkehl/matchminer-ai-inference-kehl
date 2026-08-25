# Good Option Research and Scoring

Good Option scoring is a patient-trial ranking signal, not an eligibility
determination, response probability, or treatment recommendation. It has a
structural privacy boundary:

1. `research_good_options` accepts NCT IDs, fetches ClinicalTrials.gov arm and
   intervention metadata, and selects experimental drugs. Standard-of-care,
   placebo, and control-only interventions are excluded.
2. Web queries contain drug names and generic oncology/evidence terms only.
   They collect mechanism, efficacy, safety, and target or biomarker prevalence
   evidence across cancer types. Patient text is never supplied to web search.
3. Patient context is introduced only after research. It is sent either to the
   configured LLM endpoint or to the local GoodOptionChecker classifier.

The LLM scorer awards each experimental drug up to four binary evidence points:

1. evidence of benefit in the patient's disease type;
2. the targeted biomarker is commonly expressed in that disease type;
3. the patient's tumor is documented to have the targeted biomarker; and
4. there is human evidence of benefit from targeting that biomarker.

The code-derived score is `total_points / (4 * experimental_drug_count)`. A
multi-drug regimen therefore requires evidence for each experimental drug. The
classifier predicts the same 0-1 target from the patient summary, the saved web
research extract, and registry investigational-drug context. It does not infer
the research evidence from trial-space metadata alone.

The default preset leaves `good_option_checker.model_name` empty because no
versioned public GoodOptionChecker artifact is bundled yet. Configure a trained
model to use classifier mode; LLM mode is available through
`llm_good_option`.

::: matchminer_ai.good_options
    options:
      members:
        - research_good_options
        - build_experimental_drug_selection_messages
        - build_experimental_drug_search_queries
        - build_biomarker_expression_search_queries
        - build_good_option_messages
        - build_good_option_checker_text
        - score_good_options_with_llm
        - score_good_options
        - evaluate_good_options
