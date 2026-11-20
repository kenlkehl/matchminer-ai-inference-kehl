"""Retrospective distillation-fidelity evaluation of the current pipeline on real patients (PHI).

User, 2026-11-17: run retrospective distillation-fidelity evaluations of the patients-for-a-trial and
trials-for-a-patient workflows, with Nemotron's teacher calls as the gold standard, on the validation SOC cohort.
This replaces the v22-era real-data evaluations (eval_phi_soc/, eval_phi_enrollments/: ModernBERT TrialChecker and
BoilerplateChecker, OncoReasoning, GPT checks) for that purpose; those files are kept unchanged.

What it measures, with the teacher (training's labeler and label contracts) as the reference:
- retrieval: P@20 and the mean teacher 0-6 score of three rankers' top-20 lists, (a) base EmbeddingGemma 2,
  (b) TrialSpace round 2 and (c) TrialSpace's top K reranked by the fine-tuned decision model's reasonableness;
- the decision model: AUROC, AUPRC, Brier score and calibration of its P(yes) for the five match questions, the
  boilerplate question and the per-criterion question, on the merged top-20 pairs.

Stages (each a script that writes stage_receipt.json last and answers --check, trialspace/stage.py):

  cohorts.py   extraction         ctgov cohorts of the trial sample, with training's frozen programs (teacher server)
  prepare.py   patients / cohorts / corpus   the labeler-format corpus and the seeded query samples
  embed.py     embed              key texts of one side under one embedder (CPU or GPU)
  retrieve.py  retrieve           exact cosine top K per query and embedder
  lists.py     lists-ab / lists-c rankers (a)-(c), the merged union, the stratified per-criterion sample
  label.py     label              Nemotron teacher labels (match, boilerplate, criteria) (teacher servers)
  decide.py    decide             the decision model's P(yes) (GPU)
  metrics.py   metrics            metrics.json and report.md
  run.py       orchestrator       runs them in order, resumably, and writes oa_pause/release after the last
                                  teacher request (a waiter can then resume other teacher work)

Rerun (every finished stage is skipped; teacher servers from the site config or --teacher-url):

  E=<data root>/phi/<evaluation directory>
  R=<data root>/phi/<denominators directory>
  python -m matchminer_ai.evaluation.retro_fidelity.run --root $E \
      --trials $R/soc_trial_sample.csv --summaries $R/summaries/soc_validation/summaries \
      --denominator $R/soc_validation/processed_soc_treatments.csv

All outputs are PHI: directories 700, files 600, counts only in logs.
"""
