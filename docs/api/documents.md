# PDF document OCR

::: matchminer_ai.documents
    options:
      members:
        - ocr_pdf

`ocr_pdf` is a local, page-aware extractor for trial documents and other PDFs.
It retains usable embedded text because it is generally more accurate than
rendering a digitally produced page, then falls back to RapidOCR for pages that
are image-only or nearly empty. It writes a UTF-8 `.txt` file with form-feed
characters between pages and does not overwrite an existing file unless asked.

```python
from matchminer_ai.documents import ocr_pdf

text_path = ocr_pdf("eligibility-checklist.pdf")
```

For PDFs with damaged embedded text, or form values that appear only when the
page is rendered, force OCR for every page:

```python
text_path = ocr_pdf(
    "completed-checklist.pdf",
    "completed-checklist.ocr.txt",
    force_ocr=True,
    overwrite=True,
)
```

OCR runs locally. The function does not send document text or page images to an
LLM, web search, or hosted OCR service. This local processing boundary is
important if the API is later used for authorized patient documents. The output
still inherits the source document's sensitivity and must be stored and handled
under the same controls.

OCR is probabilistic and can omit or misread characters, especially in tables,
handwriting, low-resolution scans, and complex layouts. Compare important text
with the source PDF before using it in research review. OCR output does not
establish trial eligibility or replace review of the complete, current protocol.
