# Bundled ontology resources

- `oncotree_stable_7-31-26.json` is a stable OncoTree hierarchy snapshot
  downloaded on 2026-07-31. OncoTree is produced by Memorial Sloan Kettering
  Cancer Center and distributed under CC BY 4.0. The bundled snapshot is
  unmodified. SHA-256:
  `0e2b7b424e189bb8527489e6b719f0f729039b6fc49f9806e40041bebb9e58db`.
- `Thesaurus_26.07d.FLAT.zip` is the unmodified NCI Thesaurus 26.07d flat-file
  release. Editing was completed on 2026-07-27. NCIt is produced by the NCI
  Enterprise Vocabulary Services group and distributed under CC BY 4.0.
  SHA-256:
  `589f385e0b463221e909a11c0dc3f444108f8986fa2191826c4937bd21f99e67`.

License and source links are recorded in the repository's main `README.md`.

The structuring workflow loads these files locally. It sends only one OncoTree
level or a bounded NCIt candidate page to the configured LLM; it never places a
complete ontology in model context.
