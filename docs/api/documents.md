# PDF document OCR

`ocr_pdf` is a local, page-aware text extractor for trial documents, patient records and other PDFs. It keeps a
page's embedded text when the page has a usable text layer, because that is generally more accurate than recognizing
a rendered page, and falls back to RapidOCR for pages that are image-only or nearly empty. It writes one UTF-8 `.txt`
file with form-feed characters between pages and does not overwrite an existing file unless asked.

The package uses it wherever it reads PDFs: `summarize_patients` for patient-record PDFs, and
`extract_protocol_cohorts` (or `process_trials(..., source="protocol")`) for protocol PDFs.

```python
from matchminer_ai.documents import ocr_pdf

text_path = ocr_pdf("eligibility-checklist.pdf")    # writes eligibility-checklist.txt
```

For PDFs with damaged embedded text, or form values that appear only when the page is rendered, force OCR for every
page:

```python
text_path = ocr_pdf(
    "completed-checklist.pdf",
    "completed-checklist.ocr.txt",
    force_ocr=True,
    overwrite=True,
)
```

- **Local processing:** `ocr_pdf` sends no document text or page images to an LLM, web search or hosted OCR service.
  The text it produces reaches the configured LLM only when a caller passes it on, as `summarize_patients` and the
  protocol extraction do, so use an endpoint authorized for the source documents.
- **Sensitivity:** the output inherits the source document's sensitivity and must be stored and handled under the
  same controls.
- **Accuracy:** OCR can omit or misread characters, especially in tables, handwriting, low-resolution scans and
  complex layouts. Compare important text with the source PDF. OCR output does not establish trial eligibility or
  replace review of the complete, current protocol.

::: matchminer_ai.documents
    options:
      members:
        - ocr_pdf
        - PDFOCRError
        - PDFOCRNoTextError
