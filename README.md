# matchminer-ai

## Overview

`matchminer-ai` is a Python package for running the clinical trial matching inference workflow described in [Altreuter et al., MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching](https://doi.org/10.48550/arXiv.2412.17228). The package provides modular functions for the core MatchMiner-AI workflow: summarizing trials and patient histories, generating embeddings of each, retrieving candidate matches, scoring match quality, and assessing exclusion criteria.

For a specific TrialChecker or BoilerplateChecker prediction,
`interpret_match_quality` and `interpret_exclusion_criteria` provide on-demand
gradient-times-input token attribution mapped back to the original patient and
trial fields. These local sensitivity scores are intended for model debugging
and human review. They are not a clinical rationale, eligibility evidence, or
proof that a highlighted token caused the prediction.

Optional source-grounded extensions can also:

- extract text from local PDFs with page-aware embedded-text preservation and
  OCR fallback, producing a UTF-8 text file without sending document content to
  an external service;
- accept one or more patient-record PDFs, combine their locally extracted text
  in caller-supplied order, and pass that long note through the existing serial
  patient summarization workflow;
- extract complete, source-grounded inclusion and exclusion criteria relevant
  to one trial space from an OCR eligibility-checklist text file using the
  configured LLM backend;
- answer a focused question over one patient's raw notes through local
  embedding retrieval and a bounded, evidence-citing LLM agent, with one
  code-derived source date per chunk when note-level dated input is available;
- screen complete trial eligibility criteria by decomposing them into grounded
  raw-note questions, embedding the note index once on CUDA, reusing its CPU
  vectors across concurrent question processes against an authorized endpoint,
  and synthesizing a human-reviewable JSON result;
- transform one or a batch of free-text cancer histories into JSON with one
  record per active cancer, dependency-ready LLM batching, hierarchical
  OncoTree coding, and locally searched NCIt drug normalization;
- transform a clinical-space summary into JSON while preserving age, sex,
  disease burden, treatment, response, and biomarker requirements;
- research matched trials using ClinicalTrials.gov drug/biological intervention
  names and compare them after that patient-free web step; and
- contextualize a trial space using heading-aware NCI PDQ, FDA
  companion-diagnostic and DailyMed material, accepted CIViC evidence, focused
  PubMed searches, and permissively licensed Europe PMC guideline/consensus
  full text.

Trial-space retrieval rejects patient-bearing columns. Patient personalization
is a separate API that sends patient context only to the configured LLM
backend. These extensions produce research considerations, not treatment
recommendations, guideline compliance, or eligibility determinations.

For detailed instructions, please see the
[documentation website](https://dfci.github.io/matchminer-ai-inference/).

## Ontology attribution

The optional structured patient-summary and trial-space workflows bundle and
use the following ontology snapshots locally:

- **OncoTree**, developed at Memorial Sloan Kettering Cancer Center, stable
  hierarchy snapshot downloaded 2026-07-31. OncoTree is licensed under the
  [Creative Commons Attribution 4.0 International License](https://github.com/cBioPortal/oncotree/blob/master/LICENSE.md).
  Project and source: [cBioPortal/oncotree](https://github.com/cBioPortal/oncotree).
- **NCI Thesaurus (NCIt) 26.07d**, produced by the National Cancer Institute
  Enterprise Vocabulary Services group, Center for Biomedical Informatics and
  Information Technology, National Cancer Institute, Maryland, USA. NCIt is
  licensed under the
  [Creative Commons Attribution 4.0 International License](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ThesaurusTermsofUse.pdf).
  Source and terms of use:
  [NCI Enterprise Vocabulary Services](https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/).

OncoTree codes and NCIt records remain attributable to their respective
creators. The bundled snapshots are unmodified; MatchMiner-AI adds local search,
LLM-guided selection, and output formatting around them. NCI Thesaurus is a
trademark of the National Cancer Institute. MatchMiner-AI does not imply
endorsement by MSK or NCI.

> [!WARNING]
> This package is currently pre-v1 and under active development.

## Compute requirements

A GPU is recommended for the clinical trial matching inference workflow. Please see the
[requirements documentation](https://dfci.github.io/matchminer-ai-inference/getting-started/requirements/)
for more information on compute expectations and GPU recommendations.

## Installation

This package requires Python 3.12+.

The package has been tested in Linux environments.

We recommend using [`uv`](https://docs.astral.sh/uv/) to create the Python
environment and install the package:

```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install matchminer-ai
```

## Quickstart
See the example notebook for a full walkthrough using sample input data:
[example notebook](https://github.com/dfci/matchminer-ai-inference/blob/main/examples/run_examples.ipynb)

## Citation

If you use `matchminer-ai`, please cite:
>Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA, Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M, Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026. doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)

## Contributing

Contributions are welcome! Please follow our
[contribution instructions][contributing] if you are interested in contributing
to this project.

[contributing]: https://dfci.github.io/matchminer-ai-inference/development/contributing/
