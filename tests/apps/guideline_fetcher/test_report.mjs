import assert from "node:assert/strict";
import test from "node:test";
import { buildExportReport, patientEvidenceCountLabel, reviewMethodLabel, workupReviewDetail, workupStatusLabel } from "../../../src/matchminer_ai/apps/guideline_fetcher/static/report.mjs";

function result() {
  return {
    report: "# Synthetic guideline report\n\nOriginal considerations and citations.\n",
    spaces: ["Alpha", "Beta", "Gamma"].map((name, index) => ({
      rank: index + 1, name, space_trial_id: `space${index}`,
      source: { title: "Fabricated guideline", version: "Test edition" },
    })),
  };
}

function review(assessments) {
  return {
    notice: "Not documented does not mean not done. This does not establish overall guideline concordance.",
    metadata: { note_count: 2, model: "testModel" },
    assessments,
  };
}

function item(name, status = "completed", applicability = "applies") {
  return { name, status, applicability, bottom_line: "Synthetic finding", evidence: [] };
}

test("unreviewed and invalidated results preserve the original export exactly", () => {
  const data = result();
  assert.equal(buildExportReport(data), data.report);
  assert.equal(buildExportReport(data, {}), data.report);
  assert.equal(buildExportReport(data, { 0: null, 1: review([]) }), data.report);
  assert.equal(buildExportReport(data, { 9: review([item("Old space")]) }), data.report);
});

test("all completed reviews are attached to their own spaces in display order", () => {
  const data = result();
  const reviews = { 2: review([item("Gamma workup")]), 0: review([item("Alpha workup")]) };
  const original = structuredClone({ data, reviews });
  const report = buildExportReport(data, reviews);
  assert.ok(report.startsWith(data.report.trimEnd()));
  assert.match(report, /### 1\. Alpha[\s\S]*\*\*Space ID:\*\* space0[\s\S]*Alpha workup/);
  assert.match(report, /### 3\. Gamma[\s\S]*\*\*Space ID:\*\* space2[\s\S]*Gamma workup/);
  assert.ok(report.indexOf("Alpha workup") < report.indexOf("Gamma workup"));
  assert.doesNotMatch(report, /### 2\. Beta/);
  assert.match(report, /\*\*Raw notes reviewed:\*\* 2/);
  assert.match(report, /\*\*Review model:\*\* testModel/);
  assert.match(report, /Not documented does not mean not done/);
  assert.match(report, /does not establish overall guideline concordance/);
  assert.deepEqual({ data, reviews }, original);
  assert.equal(buildExportReport(data, reviews), report, "Repeated exports must not duplicate findings");
});

test("export preserves status and applicability distinctions", () => {
  const report = buildExportReport(result(), { 0: review([
    item("A", "completed"), item("B", "partially_completed"), item("C", "planned"),
    item("D", "not_done", "not_applicable"), item("E", "not_documented", "uncertain"),
    item("F", "unclear"),
  ]) });
  for (const label of ["Completed", "Partially completed", "Planned / pending",
    "Explicitly not done", "Not documented", "Unclear / conflicting"]) {
    assert.ok(report.includes(`**Status:** ${label}`));
  }
  for (const label of ["Applies", "Not applicable", "Uncertain"]) {
    assert.ok(report.includes(`**Applicability:** ${label}`));
  }
  assert.match(report, /Synthetic finding/);
  assert.match(report, /No supporting evidence located in the supplied notes/);
});

test("evidence retains source note numbers, dates, and multiline quoted text", () => {
  const assessment = item("Fabricated workup");
  assessment.evidence = [
    { note_number: 2, note_date: "2026-01-02", note_type: "Radiology", quote: "First quoted line\nSecond quoted line" },
    { note_number: 1, note_date: null, quote: "Undated evidence" },
  ];
  const report = buildExportReport(result(), { 0: review([assessment]) });
  assert.ok(report.includes("Note 2 · 2026\\-01\\-02"));
  assert.ok(report.includes("2026\\-01\\-02 · Radiology"));
  assert.ok(report.includes("> First quoted line  \n> Second quoted line  \n"));
  assert.ok(report.includes("Note 1 · Date unavailable"));
  assert.ok(report.includes("> Undated evidence"));
});

test("source text is escaped rather than interpreted as Markdown or HTML", () => {
  const assessment = item("[Link](https://example.invalid)");
  assessment.bottom_line = "<script>example</script> & **literal**";
  assessment.evidence = [{ note_number: 1, quote: "# Literal heading\n> Comparison < 5", note_date: null }];
  const report = buildExportReport(result(), { 0: review([assessment]) });
  assert.ok(report.includes("\\[Link\\]"));
  assert.ok(report.includes("&lt;script&gt;example&lt;/script&gt; &amp; \\*\\*literal\\*\\*"));
  assert.ok(report.includes("> \\# Literal heading  \n> &gt; Comparison &lt; 5"));
  assert.doesNotMatch(report, /<script>|\[Link\]/);
});

test("automatic excerpts are labeled as review context rather than selected support", () => {
  const assessment = item("Fabricated assay");
  assessment.evidence = [{ note_number: 2, note_date: null, quote: 'é: Ordered.\nCompleted.' }];
  const data = review([assessment, item("Missing", "not_documented", "uncertain")]);
  data.metadata.evidence_selection = "automatic_reviewed_excerpts";
  const report = buildExportReport(result(), { 0: data });
  assert.match(report, /Reviewed patient-note excerpts/);
  assert.match(report, /Automatically retained review context/);
  assert.match(report, /No patient-note excerpts retained for review/);
  assert.doesNotMatch(report, /\*\*Patient-note evidence\*\*/);
  assert.ok(report.includes('> é: Ordered\\.  \n> Completed\\.'));
  assert.equal(patientEvidenceCountLabel(assessment, data), "1 fragment");
  assessment.evidence.push({note_number: 3, quote: "Fabricated second fragment."});
  assert.equal(patientEvidenceCountLabel(assessment, data), "2 fragments");
  delete data.metadata.evidence_selection;
  assert.equal(patientEvidenceCountLabel(assessment, data), "2 quotes");
});

test("agentic reports distinguish selected excerpts, missing evidence and failed searches", () => {
  const agentic = review([
    { ...item("Missing", "not_documented", "uncertain"), limitations: ["Search may miss synonyms"] },
    { ...item("Failure", "error", "uncertain"), limitations: ["Worker timed out"] },
  ]);
  agentic.metadata.method = "agentic";
  agentic.metadata.scope = "searched_excerpts";
  const report = buildExportReport(result(), { 0: agentic });
  assert.match(report, /Review method:\*\* Agentic note search · selected excerpts/);
  assert.match(report, /Raw notes available to search:\*\* 2/);
  assert.doesNotMatch(report, /Raw notes reviewed/);
  assert.match(report, /Not located in searched excerpts/);
  assert.match(report, /Review failed/);
  assert.match(report, /No documentation assessment available/);
  assert.match(report, /Search may miss synonyms/);
  assert.match(report, /Worker timed out/);
});

test("compressed review labels evidence and missing documentation as transformed text", () => {
  const scan = item("Fabricated scan");
  scan.evidence = [{ note_number: 1, note_date: "2026-01-01", quote: "CT done." }];
  const compressed = review([scan, item("Missing", "not_documented", "uncertain")]);
  compressed.metadata.method = "full_notes";
  compressed.metadata.compression = { input_characters: 1000, output_characters: 500 };
  const before = structuredClone(compressed);
  const report = buildExportReport(result(), { 0: compressed });
  assert.equal(reviewMethodLabel(compressed), "Full-note review · compressed notes");
  assert.match(report, /Compressed notes reviewed:\*\* 2/);
  assert.match(report, /Compression:\*\* 1000 → 500 characters; losslessness unverified/);
  assert.match(report, /\*\*Compressed-note excerpts\*\*/);
  assert.match(report, /LLM-compressed text; confirm findings against the original notes/);
  assert.match(report, /Not documented in compressed notes/);
  assert.match(report, /No supporting evidence located in the compressed notes/);
  assert.doesNotMatch(report, /Raw notes reviewed|\*\*Patient-note evidence\*\*/);
  assert.deepEqual(compressed, before);
});

test("compressed agentic findings preserve coverage limits and automatic excerpt labels", () => {
  const missing = item("Missing", "not_documented", "uncertain");
  const full = { ...item("Full fallback", "not_documented", "uncertain"), metadata: {
    followup_review: { status: "complete", full_record: { complete: true } },
  } };
  const data = review([missing, full]);
  data.metadata.method = "agentic";
  data.metadata.evidence_selection = "automatic_reviewed_excerpts";
  data.metadata.compression = { input_characters: 100, output_characters: 50 };
  missing.evidence = [{ note_number: 1, note_date: null, quote: "CT planned." }];
  const report = buildExportReport(result(), { 0: data });
  assert.equal(reviewMethodLabel(data), "Agentic note search · selected excerpts · compressed notes");
  assert.equal(workupStatusLabel(missing, data), "Not located in searched compressed excerpts");
  assert.equal(workupStatusLabel(full, data), "Not documented in compressed notes");
  assert.match(workupReviewDetail(full, data), /reviewed all compressed notes/);
  assert.match(report, /Compressed notes available to search/);
  assert.match(report, /Reviewed compressed-note excerpts/);
  assert.match(report, /Automatically retained review context/);
  assert.doesNotMatch(report, /Raw notes reviewed|reviewed all supplied notes/);
});

test("enhanced reviews preserve per-item coverage and full-record fallback in exports", () => {
  const completed = { ...item("Targeted"), metadata: { followup_review: {
    status: "complete", full_record: null,
  } } };
  const full = { ...item("Missing after full review", "not_documented"), metadata: {
    followup_review: { status: "complete", full_record: { chunks: 2, complete: true } },
  } };
  const partial = { ...item("Partial fallback", "not_documented"), metadata: {
    followup_review: { status: "incomplete", full_record: { chunks: 1, complete: false } },
  }, limitations: ["Follow-up request budget exhausted"] };
  const failed = { ...item("Retained plan", "planned"), metadata: {
    followup_review: { status: "incomplete", full_record: null },
  }, limitations: ["The last validated assessment is retained"] };
  const agentic = review([completed, full, partial, failed]);
  agentic.metadata.method = "agentic";
  agentic.metadata.followup_review = { max_review_passes: 2 };
  const before = structuredClone(agentic);
  const report = buildExportReport(result(), { 0: agentic });
  assert.match(report, /Agentic note search · coverage review/);
  assert.equal(workupReviewDetail(completed), "Search coverage review finished.");
  assert.match(workupReviewDetail(full), /reviewed all supplied notes/);
  assert.equal(workupStatusLabel(full, agentic), "Not documented in supplied notes");
  assert.equal(workupStatusLabel(partial, agentic), "Not located in searched excerpts");
  assert.match(workupReviewDetail(partial), /stopped before all supplied notes were reviewed/);
  assert.match(workupReviewDetail(failed), /Follow-up review incomplete/);
  assert.match(report, /Review coverage:\*\* Full\\-record fallback reviewed all supplied notes/);
  assert.match(report, /Follow\\-up review incomplete; see limitations/);
  assert.match(report, /last validated assessment is retained/);
  assert.deepEqual(agentic, before);
  assert.equal(workupReviewDetail(item("Legacy")), "");
  agentic.metadata.followup_review.max_review_passes = 0;
  assert.equal(reviewMethodLabel(agentic), "Agentic note search · selected excerpts");
});

test("backup provenance is retained per item and full review in exports", () => {
  const finding = item("Fabricated assay");
  finding.metadata = { backup_used: true, backup_events: [{
    primary_model: "smallModel", backup_model: "largeModel",
    reason: "consecutive_python_errors", failure_threshold: 3,
  }] };
  const data = review([finding]);
  data.metadata.backup_used = true;
  data.metadata.backup_items = 1;
  const report = buildExportReport(result(), { 0: data });
  assert.match(report, /Backup model used for 1 workup item/);
  assert.match(report, /Backup model largeModel used after 3 consecutive Python errors with smallModel/);
  data.metadata.backup_events = [{...finding.metadata.backup_events[0],
    reason: "request_or_validation_failures"}];
  assert.match(buildExportReport(result(), {0: data}), /3 failed request or validation attempts/);
});
