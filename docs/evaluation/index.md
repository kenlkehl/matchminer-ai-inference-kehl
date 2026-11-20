# Evaluations

`matchminer_ai.evaluation` holds the evaluations of the pipeline on real patients. Their inputs and outputs are
clinical data (PHI): they read and write only under the data root's `phi/` tree (or a PHI directory the site config
names, `paths.phi_roots`), create files 600 and directories 700, and log counts only. Run them only where that data is
authorized, with LLM endpoints approved for it (the site config's `endpoints.*`).

Install with the evaluation extra: `pip install -e '.[training,evaluation]'`.

| Module | What it does |
| --- | --- |
| `evaluation.retro_cohorts` | Retrospective denominators from an institution's structured EHR extract (its location is the site config's `paths.retro_source_root`): trial enrollments and standard-of-care (SOC) treatment starts, index dates, note-level datasets cut before the index date with the event's own trial identifiers masked, and their serial summaries. |
| `evaluation.retro_fidelity` | Distillation fidelity on those patients, with the teacher's labels as the reference: P@20 of three rankers (base EmbeddingGemma 2; TrialSpace; TrialSpace reranked by the decision model) and the decision model's AUROC, AUPRC, Brier score and calibration for every question family. |
| `evaluation.dual_judge` | Two frontier LLM judges (Claude on Vertex AI, GPT on Azure) grade cohort extraction, patient summaries and the decision questions; the user adjudicates disagreements in a loopback-only app. |
| `evaluation.cohort_lineitems` | Cohorts for a table of ctgov-style listings with training's ctgov extractor. |

## Retrospective fidelity

Each stage is a module that writes a path-free `stage_receipt.json` last and answers `--check` (0 complete, 10 not
yet, 20 begun under other settings); the orchestrator runs them in dependency order and resumes:

```bash
python -m matchminer_ai.evaluation.retro_fidelity.run --root <data root>/phi/<evaluation> \
    --trials <denominators>/soc_trial_sample.csv --summaries <denominators>/summaries/soc_validation/summaries \
    --denominator <denominators>/soc_validation/processed_soc_treatments.csv
```

Teacher servers come from the site config (`endpoints.teacher.url`, and optionally `endpoints.secondary_teacher.url`,
reached through `retro_fidelity.teacher_proxy`) or `--teacher-url` / `--secondary-teacher-url`. With
`--extraction-source SNAPSHOT` the trial cohorts are extracted by a code snapshot (`git archive` with a
`commits.json`) rather than the training run's frozen programs.

## Dual judge

```bash
python -m matchminer_ai.evaluation.dual_judge.build_items --run-dir RUN ...    # where the data is
python -m matchminer_ai.evaluation.dual_judge.judge_worker --run-dir RUN --judge vertex
python -m matchminer_ai.evaluation.dual_judge.judge_worker --run-dir RUN --judge azure
python -m matchminer_ai.evaluation.dual_judge.compare --run-dir RUN
python -m matchminer_ai.evaluation.dual_judge.adjudicate --run-dir RUN         # http://127.0.0.1:8791
python -m matchminer_ai.evaluation.dual_judge.metrics --run-dir RUN
```

The judge, comparison, adjudication and metrics modules need only the standard library (plus `anthropic[vertex]` for
the Vertex judge), so they can run on another machine that can read and write the run directory. The Vertex project
is `--project`, `ANTHROPIC_VERTEX_PROJECT_ID` or the site config's `vertex.project`; Azure uses
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT`. The judge prompts are `prompts/judge.*.txt`.

## Caveats

The teacher is the reference, not clinician review: these numbers measure agreement with the teacher on real
summaries, not clinical accuracy. MatchMiner-AI is a research prioritization tool; its scores are screening signals
for human review.
