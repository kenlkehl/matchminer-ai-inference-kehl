"""PDQ ingestion, complete catalog round trips and tampering: invented text only."""

import copy
import io
import json

import httpx
import pandas as pd
import pytest

from matchminer_ai import load_default_preset
from matchminer_ai._storage import digest, read_json
from matchminer_ai.trials import (
    audit_guideline_catalog,
    get_guideline_considerations,
    list_guidelines,
    load_guideline_catalog,
    prepare_pdq_library,
    summarize_guidelines,
    summarize_pdq,
)
from matchminer_ai.trials._guideline_sources import InputNotReady, load_guideline
from matchminer_ai.trials import _guideline_prompts as prompts
from matchminer_ai.trials._pdq_sources import parse_pdq_html
from test_guideline_extraction import TEXT, catalog, extraction, quoted_state
from test_guideline_api import fake_endpoint, guideline_run  # noqa: F401
from test_workup_review import harness  # noqa: F401

URL = "https://www.cancer.gov/types/fictional/hp/fictional-treatment-pdq"
PDQ_TEXT = (
    TEXT
    + " Synthetic therapy B is an alternative after finding C. PDQ level of evidence A1."
)


def html_source(clinical=None):
    return (
        f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="dcterms.type" content="pdqCancerInfoSummary">
<meta name="dcterms.issued" content="2025-02-01">
<link rel="canonical" href="{URL}"></head><body>
<nav>Website navigation must not be evidence</nav>
<h1>Fictional Disease Treatment (PDQ®)–Health Professional Version</h1>
<div id="cgvBody"><div class="summary-sections">
<section id="about"><h2>About This PDQ Summary</h2><p>Fictional editorial text.</p></section>
<section id="care"><h2>Fictional disease care</h2>
{clinical if clinical is not None else '<p>' + PDQ_TEXT + '</p>'}
</section></div></div><time datetime="2025-03-02">March 2, 2025</time>
<footer>Website footer must not be evidence</footer></body></html>"""
    ).encode()


@pytest.fixture
def downloads(monkeypatch):
    calls = []

    def get(self, url, **kwargs):
        calls.append(url)
        assert kwargs == {"follow_redirects": False}
        return httpx.Response(
            200,
            content=html_source(),
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    return calls


def test_complete_html_hierarchy_tables_references_and_no_truncation():
    clinical = (
        """<p>Introduction with ER<span>+</span> and HER2-negative.</p>
<h3 id="first">First-line disease</h3><p>Consider option A.</p>
<ul><li>Alternative B<ul><li>If finding C</li><li>Or finding D</li></ul></li><li>Option E</li></ul>
<table><caption>Fictional treatment table</caption><thead><tr><th colspan="2">Setting</th><th>Options</th></tr></thead>
<tbody><tr><td rowspan="2">Later-line</td><td>Finding F</td><td><p>A</p><p>OR</p><p>B</p></td></tr>
<tr><td>Finding G</td><td>Option Z<sup>1</sup></td></tr></tbody>
<tfoot><tr><td colspan="3">Third-party permission notice retained.</td></tr></tfoot></table>
<p>Evidence <a href="#ref1">[1]</a> <a href="/publications/pdq/levels-evidence/treatment">Level of evidence A1</a>.</p>
<p><img alt="Unavailable diagram" src="/never-download.png">Image caption.</p>
<h6>References</h6><ol><li id="ref1">Fictional citation <a href="https://pubmed.ncbi.nlm.nih.gov/999999999/">Study</a></li></ol>
<h3 id="late">Second-line disease</h3><p>"""
        + "Long fictional prose. " * 5000
        + "Final sentinel.</p>"
    )
    sections, metadata = parse_pdq_html(html_source(clinical), URL)
    combined = "\n".join(s["text"] for s in sections)
    assert len(combined) > 100000 and "Final sentinel." in combined
    assert "Website navigation" not in combined and "Website footer" not in combined
    assert "ER+" in combined and "HER2-negative" in combined
    first = next(s for s in sections if s["source_url"] == URL + "#first")
    assert first["section_path"] == ["Fictional disease care", "First-line disease"]
    assert (
        "- Alternative B\n  - If finding C\n  - Or finding D\n- Option E"
        in first["text"]
    )
    assert '<td rowspan="2">Later-line</td>' in combined
    assert '<th colspan="2">Setting</th>' in combined
    assert "<p>A</p><p>OR</p><p>B</p>" in combined
    assert "Third-party permission notice retained." in combined
    assert any(x["url"] == URL + "#ref1" for x in first["links"])
    assert any("ref1" in s["anchors"] for s in sections)
    assert first["warnings"] == [
        "Non-text content not interpreted: Unavailable diagram"
    ]
    assert metadata["updated_at"] == "2025-03-02"
    assert metadata["published_at"] == "2025-02-01"
    assert not sections[0]["primary"]


def test_minified_block_boundaries_and_lists_within_table_cells_are_preserved():
    sections, _ = parse_pdq_html(
        html_source("""<ul><li><p>First sentence.</p><p>Second sentence.</p></li></ul>
<table><tr><td><ol start="3"><li>Drug A<ul><li>When X</li><li>When Y</li></ul></li><li>Drug B</li></ol></td></tr></table>
<p>ER<span>+</span> <b>remains</b> explicit.</p>"""),
        URL,
    )
    text = sections[1]["text"]
    assert "First sentence. Second sentence." in text
    assert (
        '<ol start="3"><li>Drug A<ul><li>When X</li><li>When Y</li></ul></li><li>Drug B</li></ol>'
        in text
    )
    assert "ER+ remains explicit." in text


def test_heading_and_unwrapped_links_and_continued_reference_numbers():
    sections, _ = parse_pdq_html(
        html_source("""
<h3 id="context">Context from <a href="/types/related/hp/related-pdq">Related</a></h3>
<a href="#last-reference">Source detail</a>
<ol start="7"><li>Seventh reference</li><li value="9" id="last-reference">Ninth reference</li></ol>
"""),
        URL,
    )
    last = sections[-1]
    assert "7. Seventh reference\n9. Ninth reference" in last["text"]
    assert {x["url"] for x in last["links"]} == {
        "https://www.cancer.gov/types/related/hp/related-pdq",
        URL + "#last-reference",
    }


def test_snapshot_offline_reuse_and_source_integrity(tmp_path, downloads, monkeypatch):
    source = tmp_path / "source"
    rows = prepare_pdq_library({"fictional": URL}, output_dir=source)
    assert rows.disease.tolist() == ["fictional"]
    assert len(downloads) == 1
    guideline = load_guideline(source, "fictional")
    assert len(guideline.pages) == 2 and [p.id for p in guideline.primary] == ["p0002"]
    assert guideline.pages["p0002"].source_url == URL + "#care"
    assert (
        source / "markdown/fictional/html/document-001.html"
    ).read_bytes() == html_source()
    monkeypatch.setattr(
        httpx.Client,
        "get",
        lambda *a, **k: pytest.fail("Snapshot resume must be offline"),
    )
    pd.testing.assert_frame_equal(
        prepare_pdq_library({"fictional": URL}, output_dir=source), rows
    )
    assert load_guideline(source, "fictional").fingerprint == guideline.fingerprint
    with pytest.raises(ValueError, match="selection changed"):
        prepare_pdq_library({"another": URL}, output_dir=source)
    section = source / "markdown/fictional/sections/s0002.md"
    section.write_text(section.read_text().replace("test A", "test Z"))
    with pytest.raises(InputNotReady, match="differs"):
        load_guideline(source, "fictional")


@pytest.mark.parametrize(
    "tamper", ["html", "section_hash", "url", "heading", "primary", "drop_section"]
)
def test_snapshot_tampering_fails_even_with_updated_render_hash(
    tmp_path, downloads, tamper
):
    from matchminer_ai._storage import digest

    source = tmp_path / "source"
    prepare_pdq_library({"fictional": URL}, output_dir=source)
    directory = source / "markdown/fictional"
    p = directory / "manifest.json"
    m = read_json(p)
    if tamper == "html":
        (directory / m["documents"][0]["path"]).write_bytes(b"Changed original")
    elif tamper == "section_hash":
        section = directory / m["pages"][1]["path"]
        section.write_text("Forged source text")
        m["generated_files"][m["pages"][1]["path"]] = digest(section.read_bytes())
    elif tamper == "url":
        m["pages"][1]["source_url"] = URL + "#another-population"
    elif tamper == "heading":
        m["pages"][1]["section_path"] = ["Different treatment line"]
    elif tamper == "primary":
        m["pages"][1]["primary"] = False
    else:
        m["pages"].pop()
        m["page_count"] -= 1
    p.write_text(json.dumps(m))
    with pytest.raises(InputNotReady, match="changed|differs"):
        load_guideline(source, "fictional")


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.invalid/types/a/hp/b-pdq",
        "http://www.cancer.gov/types/a/hp/b-pdq",
        "https://www.cancer.gov/types/a/patient/b-pdq",
        URL + "?patient=private",
        URL + "#care",
    ],
)
def test_no_arbitrary_sources(url, tmp_path, downloads):
    with pytest.raises(ValueError, match="health-professional"):
        prepare_pdq_library({"fictional": url}, output_dir=tmp_path / "source")
    assert not downloads


def test_redirect_is_validated_before_fetch(tmp_path, monkeypatch):
    calls = []

    def get(self, url, **kwargs):
        calls.append(url)
        return httpx.Response(
            302,
            headers={"location": "http://127.0.0.1/private"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    with pytest.raises(ValueError, match="health-professional"):
        prepare_pdq_library({"fictional": URL}, output_dir=tmp_path / "source")
    assert calls == [URL]
    assert not (tmp_path / "source/markdown/manifest.json").exists()


def test_partial_download_resumes_completed_diseases_and_captures_multiple_documents(
    tmp_path, monkeypatch
):
    second = URL.replace("fictional-treatment", "fictional-secondary")
    calls, failing = [], [True]

    def get(self, url, **kwargs):
        calls.append(url)
        if url == second and failing[0]:
            raise httpx.ConnectError("Synthetic download interruption")
        return httpx.Response(
            200,
            content=html_source().replace(URL.encode(), url.encode()),
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.Client, "get", get)
    source = tmp_path / "sources"
    mapping = {"first": URL, "combined": [URL, second]}
    with pytest.raises(httpx.ConnectError):
        prepare_pdq_library(mapping, output_dir=source)
    assert not (source / "markdown/manifest.json").exists()
    first = load_guideline(source, "first").fingerprint
    failing[0] = False
    calls.clear()
    inventory = prepare_pdq_library(mapping, output_dir=source)
    assert inventory.disease.tolist() == ["first", "combined"]
    assert calls == [URL, second]
    assert load_guideline(source, "first").fingerprint == first
    combined = load_guideline(source, "combined")
    assert len(combined.pages) == 4 and len(combined.primary) == 2
    assert combined.pages["p0002"].source_url == URL + "#care"
    assert combined.pages["p0004"].source_url == second + "#care"
    manifest = read_json(source / "markdown/manifest.json")
    assert (
        manifest["guidelines"][1]["source_sha256"] == combined.metadata["source_sha256"]
    )


@pytest.mark.parametrize(
    "change", ["patient", "missing_body", "missing_sections", "wrong_canonical"]
)
def test_changed_web_layout_does_not_silently_extract_partial_page(change):
    raw = html_source()
    if change == "patient":
        raw = raw.replace(b"Health Professional", b"Patient")
    elif change == "missing_body":
        raw = raw.replace(b'"cgvBody"', b'"changed"')
    elif change == "missing_sections":
        raw = raw.replace(b'"summary-sections"', b'"changed"')
    else:
        raw = raw.replace(
            URL.encode(), b"https://www.cancer.gov/types/other/hp/other-pdq"
        )
    with pytest.raises(ValueError):
        parse_pdq_html(raw, URL)


@pytest.fixture
def pdq_endpoint(monkeypatch):
    calls = []
    values = {"candidates": extraction(), "states": catalog(), "space": quoted_state()}
    for value in values.values():
        owners = value.get("candidates", value.get("states", [value]))
        for owner in owners:
            if "treatment_options" in owner:
                owner["treatment_options"] = [
                    {
                        "name": "Synthetic therapy B",
                        "conditions": "Alternative after finding C",
                        "category": "PDQ level of evidence A1",
                        "evidence": copy.deepcopy(owner["evidence"]),
                    }
                ]
            if "defining_branch" in owner:
                owner["defining_branch"]["line_ids"] = [2]
            for item in [
                owner,
                *owner.get("diagnostic_workup", []),
                *owner.get("treatment_options", []),
            ]:
                if "category" in item:
                    item["category"] = "PDQ level of evidence A1"
                for e in item["evidence"]:
                    if "line_ids" in e:
                        e["line_ids"] = [2]
                    if "source_text" in e:
                        e["source_text"] = PDQ_TEXT

    def respond(request, timeout):
        body = json.loads(request.data) if request.data else None
        calls.append((request.full_url, body))
        if request.full_url.endswith("/v1/models"):
            return io.BytesIO(
                json.dumps(
                    {"data": [{"id": "synthetic-Qwen3.8", "max_model_len": 262144}]}
                ).encode()
            )
        if request.full_url.endswith("/tokenize"):
            return io.BytesIO(
                json.dumps(
                    {
                        "count": sum(
                            len(m["content"].encode()) for m in body["messages"]
                        )
                        // 4
                        + 100
                    }
                ).encode()
            )
        assert request.full_url.endswith("/v1/chat/completions")
        assert body["max_tokens"] == 100000
        assert body["reasoning_effort"] == "xhigh"
        assert body["chat_template_kwargs"]["enable_thinking"] is True
        assert (
            body["temperature"] == 1 and body["top_p"] == 0.95 and body["top_k"] == 20
        )
        assert "PDQ is an evidence summary" in body["messages"][0]["content"]
        assert "pdftotext -layout" not in body["messages"][0]["content"]
        schema = body["response_format"]["json_schema"]["schema"]
        value = next(
            copy.deepcopy(v) for key, v in values.items() if key in schema["properties"]
        )
        event = {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": json.dumps(value)},
                    "finish_reason": "stop",
                }
            ]
        }
        return io.BytesIO(
            ("data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n").encode()
        )

    monkeypatch.setattr("matchminer_ai.llm.structured.urlopen", respond)
    return calls


def test_public_pdq_catalog_roundtrip_audit_lookup_and_offline_resume(
    tmp_path,
    downloads,
    pdq_endpoint,
    monkeypatch,
    harness,  # noqa: F811
):
    source, output = tmp_path / "source", tmp_path / "catalog"
    prepare_pdq_library({"fictional": URL}, output_dir=source)
    config = load_default_preset()
    config.remote.update(
        enabled=True,
        server_urls=["http://synthetic.invalid/v1"],
        max_concurrent_requests=2,
        max_retries=1,
    )
    result, metadata, qc = summarize_pdq(
        source,
        disease="fictional",
        output_dir=output,
        config=config,
        return_metadata=True,
        return_qc=True,
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row.space_trial_id.startswith("pdq-fictional-")
    assert row.trial_id.startswith("pdq:fictional:snapshot-")
    assert row.source["source_kind"] == "nci_pdq"
    assert row.source["is_clinical_practice_guideline"] is False
    assert "source_pdf" not in row.source
    assert row.evidence[0]["source_url"] == URL + "#care"
    assert row.evidence[0]["section_number"] == 2 and "pdf_page" not in row.evidence[0]
    assert row.evidence[0]["quote"] == PDQ_TEXT
    assert row.diagnostic_workup[0]["category"] == "PDQ level of evidence A1"
    assert row.treatment_options[0]["conditions"] == "Alternative after finding C"
    assert row.treatment_options[0]["evidence"][0]["source_url"] == URL + "#care"
    assert "NCI" in row.notice and "NCCN" not in row.notice
    assert metadata["validation"]["verified_raw_provider_responses"] == 3
    assert metadata["validation"]["source_sections"] == 2
    assert "source_pdf_pages" not in metadata["validation"]
    assert metadata["run_config"]["prompt_version"].startswith("pdq-")
    for filename in prompts.PDQ_PROMPT_FILES:
        assert metadata["run_config"]["prompt_resources_sha256"][filename] == digest(
            prompts.load_prompt_text(filename).encode()
        )
    assert list(qc.columns) == ["metric", "value", "denominator", "percent", "ids"]
    report = (output / "report.md").read_text()
    assert (
        URL + "#care" in report
        and "PDF page" not in report
        and "NCCN-endorsed" not in report
    )
    loaded = load_guideline_catalog(output)
    selected = get_guideline_considerations(output, space_trial_id=row.space_trial_id)
    pd.testing.assert_frame_equal(result, loaded)
    assert selected.iloc[0].diagnostic_workup == row.diagnostic_workup
    assert selected.iloc[0].treatment_options == row.treatment_options
    assert list_guidelines(source).disease.tolist() == ["fictional"]
    assert len([u for u, _ in pdq_endpoint if u.endswith("completions")]) == 3
    definitions = []
    for url, body in pdq_endpoint:
        if not url.endswith("completions"):
            continue
        # Source adaptation must retain the opening orientation in every generation stage.
        system = body["messages"][0]
        assert system["role"] == "system"
        assert system["content"] == prompts.PDQ_SYSTEM
        assert system["content"].startswith("TASK CONTEXT AND PURPOSE\n")
        # The complete current trial prompts must reach the endpoint, not a copied
        # paraphrase, followed by the adaptation that preserves PDQ/JSON semantics.
        for filename in prompts.PDQ_TRIAL_PROMPT_FILES:
            reference = prompts.load_prompt_text(filename)
            assert system["content"].count(reference) == 1
            assert system["content"].index(reference) < system["content"].index(
                "APPLY THAT POPULATION FORMAT TO THE CURRENT PDQ TASK"
            )
        assert "Return the requested JSON schema for the active stage" in system["content"]
        assert "obey the\n  fixed-definition rules" in system["content"]
        assert (
            "searchable research library from NCI PDQ health-professional cancer"
            in system["content"]
        )
        assert (
            "downstream matching software converts ONLY the nine named population fields"
            in system["content"]
        )
        assert (
            "do not attempt the entire workflow or rewrite fields "
            "that the active task declares fixed" in system["content"]
        )
        schema = body["response_format"]["json_schema"]["schema"]
        if key := next((k for k in ("candidates", "states") if k in schema["properties"]), None):
            definitions.append(key)
            assert prompts.FIELD_PLACEMENT_RULES in body["messages"][-1]["content"]
            assert prompts.SPECIFICITY_RULES in body["messages"][-1]["content"]
            fields = schema["properties"][key]["items"]["properties"]["space"]["properties"]
            assert "Not a catch-all" in fields["cancer_burden_allowed"]["description"]
            assert "Treatment ineligibility is NOT absence of prior exposure" in fields["prior_treatment_excluded"]["description"]
    assert definitions == ["candidates", "states"]
    monkeypatch.setattr(
        "matchminer_ai.llm.structured.urlopen",
        lambda *a, **k: pytest.fail("Completed resume is offline"),
    )
    # Both public entrypoints share the same accepted requests and offline audit.
    resumed = summarize_guidelines(
        source, disease="fictional", output_dir=output, config=config
    )
    pd.testing.assert_frame_equal(result, resumed)
    assert (
        audit_guideline_catalog(source, disease="fictional", output_dir=output)[
            "status"
        ]
        == "passed"
    )
    # The public retrieval and workup consumers use these actual generated
    # records. Only the local embedding/checker models and patient LLM are fake.
    from matchminer_ai.matching import (
        retrieve_guideline_considerations,
        write_guideline_considerations_report,
    )
    from matchminer_ai.patients import review_patient_workup

    def embed(frame, *, entity_type, **kwargs):
        key = "patient_id" if entity_type == "patient" else "space_trial_id"
        return frame[[key]].assign(embedding=[[1.0, 0.0]] * len(frame)), {
            "model_metadata": {"embedding_model": {"model": "synthetic-same-encoder"}}
        }

    def score(frame, **kwargs):
        assert kwargs["filter_low_quality"] is False
        return frame[["patient_id", "space_trial_id"]].assign(
            match_quality_score=0.9,
            match_quality_pass=True,
        ), {"model_metadata": {"match_quality_checker": {"model": "synthetic"}}}

    monkeypatch.setattr("matchminer_ai.embedding.embed_for_matching", embed)
    monkeypatch.setattr("matchminer_ai.matching.score_match_quality", score)
    patient = pd.DataFrame(
        [
            {
                "patient_id": "fictional",
                "cancer_history_summary": "Invented patient in state alpha.",
            }
        ]
    )
    ranked = retrieve_guideline_considerations(patient, output, top_n=1, candidate_k=5)
    assert ranked.iloc[0].diagnostic_workup == row.diagnostic_workup
    report_path = write_guideline_considerations_report(
        ranked, patient, tmp_path / "review.md"
    )
    report_text = report_path.read_text()
    assert URL + "#care" in report_text and "PDF page" not in report_text
    assert "NCI PDQ health-professional evidence summary" in report_text
    assert "PDQ level of evidence A1" in report_text
    patient_config, workup_calls, responder, _ = harness
    responder[0] = lambda p: [
        {
            "name": r["name"],
            "status": "completed",
            "applicability": "applies",
            "bottom_line": "Documented in invented notes.",
        }
        for r in p["recommendations"]
    ]
    reviewed = review_patient_workup(
        "Test A done.",
        ranked.iloc[0].diagnostic_workup,
        population_context=ranked.iloc[0].clinical_space_summary,
        config=patient_config,
    )
    finding = reviewed["assessments"][0]
    assert finding["status"] == "completed"
    assert finding["recommendation"] == row.diagnostic_workup[0]
    assert finding["evidence"][0]["quote"] == "Test A done."
    assert (
        workup_calls[0]["recommendations"][0]["conditions"]
        == row.diagnostic_workup[0]["conditions"]
    )
    assert reviewed["metadata"]["patient_summary_used"] is False
    # Editing a displayed source address cannot survive the final audit.
    payload = json.loads((output / "paradigms.jsonl").read_text())
    payload["evidence"][0]["source_url"] = URL + "#wrong"
    (output / "paradigms.jsonl").write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="provenance differs"):
        audit_guideline_catalog(source, disease="fictional", output_dir=output)


def test_pdq_entrypoint_rejects_nccn_sources(tmp_path):
    from test_guideline_extraction import make_library

    with pytest.raises(ValueError, match="PDQ source library"):
        summarize_pdq(
            make_library(tmp_path), disease="fictional", output_dir=tmp_path / "catalog"
        )


def test_pdq_prompt_changes_do_not_invalidate_nccn_identity(
    guideline_run,  # noqa: F811
    fake_endpoint,  # noqa: F811
):
    # Fixtures from the NCCN public API suite are imported below. An unused PDQ
    # prompt must never enter an NCCN resume identity or its sent instructions.
    root, output, config = guideline_run
    summarize_guidelines(root, disease="fictional", output_dir=output, config=config)
    run = read_json(output / "run_config.json")
    assert not set(prompts.PDQ_PROMPT_FILES) & run["prompt_resources_sha256"].keys()
    assert run["prompt_version"] == "nccn-trialspace-v14-field-contract"
    for _, body, _ in fake_endpoint:
        if body and "messages" in body:
            assert "PDQ is an evidence summary" not in body["messages"][0]["content"]
            assert "CURRENT MATCHMINER-AI TRIAL-SPACE PROMPT" not in body["messages"][0]["content"]


@pytest.mark.parametrize("filename", prompts.PDQ_PROMPT_FILES)
def test_pdq_resume_rejects_changed_trial_reference_dependencies(
    tmp_path, downloads, pdq_endpoint, filename,
):
    source, output = tmp_path / "source", tmp_path / "catalog"
    prepare_pdq_library({"fictional": URL}, output_dir=source)
    config = load_default_preset()
    config.remote.update(enabled=True, server_urls=["http://synthetic.invalid/v1"])
    summarize_pdq(source, disease="fictional", output_dir=output, config=config)
    path = output / "run_config.json"
    saved = read_json(path)
    # Represent an internally consistent older run whose reference was different.
    saved["prompt_resources_sha256"][filename] = digest(b"Previous reference text")
    identity = {
        k: copy.deepcopy(v) for k, v in saved.items()
        if k not in {"config_sha256", "runtime", "stage_versions"}
    }
    for key in ("timeout", "attempts", "api_key_env", "stream", "max_concurrent_requests"):
        identity["llm"].pop(key)
    identity["llm"].pop("transport_base_url", None)
    saved["config_sha256"] = digest(identity)
    path.write_text(json.dumps(saved))
    pdq_endpoint.clear()
    with pytest.raises(ValueError, match="choose a new output_dir"):
        summarize_pdq(source, disease="fictional", output_dir=output, config=config)
    assert not any(url.endswith("completions") for url, _ in pdq_endpoint)
