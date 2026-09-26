# Local guideline conversion

`convert_guidelines.py` derives from the user-supplied converter v1.0.3 preserved
in the standalone `space_paradigms_nccn` prototype (code commit `4ffc944`).
Its original SHA-256 was
`34fab68b017c5e7548b917e57d8492ac775c5a47babfbf599a4f99c50c12986d`.
The integration requires explicit paths outside this code repository; conversion
semantics and manifest format remain unchanged. No guideline text is bundled.

Install the `guideline-conversion` extra and Poppler's `pdftotext`.

```bash
python scripts/convert_guidelines.py \
  --input-dir /path/to/local/pdfs \
  --output-dir /path/to/local/pdfs/markdown
```

Use `--validate-only` with the same paths to check an existing library. The
extractor expects `markdown` alongside the original PDF collection. The script
is local-only text/layout conversion, without OCR or LLM calls. Missing graphical
relationships require review of the original PDF. Keep both the source library
and subsequent extraction outputs outside this repository and public packages.
