# Example Notebook

[`examples/run_examples.ipynb`](https://github.com/kenlkehl/matchminer-ai-kehl/blob/main/examples/run_examples.ipynb)
runs the matching workflow on the sample data in `examples/data/`: ClinicalTrials.gov listings and the notes of
twelve synthetic patients. It runs:

1. `process_trials`, from trials to cohorts;
2. `summarize_patients`, including an update with newer notes;
3. TrialSpace embedding and retrieval in both directions;
4. the decision model's three checks;
5. the optional LLM checks;
6. a review table with one row per patient-cohort pair.

Run it from the `examples/` directory of a checkout:

```shell
cd examples
jupyter lab run_examples.ipynb
```

Before running, set the TrialSpace checkpoint (`embedding.model_path`) and choose where the LLM steps run:

- in-process vLLM, the default;
- an OpenAI-compatible server (`USE_SERVER = True`).

The notebook writes its results to `examples/output/`, which git ignores. The [quickstart](quickstart.md) explains
the same steps.
