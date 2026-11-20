# MatchMiner-AI

MatchMiner-AI is an open-source pipeline for matching patients with cancer to clinical trials. It reads trial
documents and longitudinal patient notes and writes comparable summaries of each. From those summaries it retrieves
candidate patient-trial pairs and scores whether each pair deserves a closer look. It works in both directions:
trials for a patient, and patients for a trial. The method is described in
[Altreuter et al., *MatchMiner-AI: An Open-Source Solution for Cancer Clinical Trial Matching*](https://doi.org/10.48550/arXiv.2412.17228).

!!! warning "Research use"
    MatchMiner-AI is a research tool for prioritizing human review. It is not an eligibility engine, a medical
    device or a source of treatment recommendations. Its scores rank and screen candidates. Every suggested match
    needs review by a clinician and the trial team against the complete, current protocol.

!!! warning "Pre-v1 package"
    The package is under active development. APIs, configuration options and outputs may change.

## What the package contains

The `matchminer_ai` package holds the whole pipeline. Training renders its examples with the same prompts, text
formats and schemas the runtime uses.

- **Matching.** Seven steps: trials to cohorts, notes to patient summaries, TrialSpace embedding, retrieval, and
  three checks of each candidate pair by the decision model (match quality, boilerplate criteria, criteria line by
  line). The [quickstart](getting-started/quickstart.md) runs them end to end, and
  [How matching works](user-guide/patient-centric-matching.md) explains the concepts behind them.
- **Related review tools.**
    - [GoodOption drug evidence](user-guide/good-options-pipeline.md);
    - guideline catalogs from [NCCN](user-guide/guideline-extraction.md) and [PDQ](user-guide/pdq-catalogs.md);
    - [structured clinical extraction](api/extraction.md);
    - workup checks ([extraction-based](api/workup-queries.md), [SQL agent](api/workup-sql.md));
    - [population workup review](user-guide/population-workup.md);
    - [note-search question answering](user-guide/note-search-qa.md).
- **[Training](training/index.md).** `mmai-train` produces TrialSpace and the decision model from synthetic and
  public data, and optionally distills [OncoReasoning](training/oncoreasoning.md).
- **[Evaluation](evaluation/index.md).** Evaluations on real patients, run only where that data is authorized.
- **Apps.** The [dashboard](apps/dashboard.md) (`mmai-dashboard`) and the
  [guideline fetcher](apps/guideline-fetcher.md) (`mmai-guideline-fetcher`).

## Where to start

1. [Requirements](getting-started/requirements.md) and [GPU recommendations](getting-started/gpu-recommendations.md):
   compute and input data.
2. [Installation](getting-started/installation.md).
3. [Choosing an inference setup](user-guide/local-vs-remote.md): in-process models or an OpenAI-compatible server.
4. [Quickstart](getting-started/quickstart.md) and the [example notebook](getting-started/example-notebook.md).
5. References:
    - [Configuration](reference/configuration.md): the preset's sections;
    - [Site configuration](reference/site-config.md): installation-specific values and the data root;
    - [LLM runtime](reference/llm-runtime.md): model profiles, the endpoint pool and tracing.

Contributions are welcome; see [contributing](development/contributing.md).

## Citation

If you use MatchMiner-AI, please cite:

> Altreuter J, Trukhanov P, Paul MA, Hassett MJ, Riaz IB, Afzal MU, Mohammed AA,
> Sammons S, Lindsay J, Mallaber E, Klein HR, Gungor G, Galvin M, Deletto M,
> Van Nostrand SC, Provencher J, Yu J, Tahir N, Wischhusen J, Kozyreva O, Ortiz
> T, Tuncer H, Masri JE, Malcolm A, Mazor T, Cerami E, Kehl KL. MatchMiner-AI:
> An Open-Source Solution for Cancer Clinical Trial Matching. *arXiv*. 2026.
> doi: [10.48550/arXiv.2412.17228](https://doi.org/10.48550/arXiv.2412.17228)
