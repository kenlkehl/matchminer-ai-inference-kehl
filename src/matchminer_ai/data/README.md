# Bundled ontology resources

- `Thesaurus_26.07d.FLAT.zip` is the unmodified NCI Thesaurus 26.07d flat-file
  release. Editing was completed on 2026-07-27. NCIt is produced by the NCI
  Enterprise Vocabulary Services group and distributed under CC BY 4.0.
  SHA-256:
  `589f385e0b463221e909a11c0dc3f444108f8986fa2191826c4937bd21f99e67`.

License and source links are recorded in the repository's main `README.md`.

The GoodOption catalog loads the NCIt file locally to normalize drug names and
to assign drug classes from the concept hierarchy above each drug. Only matched
drugs' records, and for each drug a short list (by default at most 25) of
candidate classes found by local search, reach its LLM prompts; the complete
ontology never enters model context.
