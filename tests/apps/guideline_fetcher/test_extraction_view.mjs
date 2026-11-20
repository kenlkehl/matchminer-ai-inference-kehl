import assert from "node:assert/strict";
import test from "node:test";
import { costLine, extractionAssessment, highlightedNote } from "../../../src/matchminer_ai/apps/guideline_fetcher/static/extraction_view.mjs";
import { distributionTable, patientDetail, patientRows } from "../../../src/matchminer_ai/apps/guideline_fetcher/static/population_view.mjs";
import { buildExportReport, reviewMethodLabel, workupStatusLabel } from "../../../src/matchminer_ai/apps/guideline_fetcher/static/report.mjs";

// Fabricated extraction-query result.
const item = {
  name: "CT chest", recommendation: { name: "CT chest", conditions: "" }, recommendation_index: 0,
  metadata: {}, status: "met", applicability: "applies", reason: "documented",
  bottom_line: "Met: CT chest: CT (chest) <b>", query_summary: "A CT of the chest.", limitations: ["Approximation."],
  criteria: [{ label: "CT", item_type: "imaging_study", status: "performed", state: "yes", definite_count: 1, possible_count: 0, min_count: 1, anchor: null }],
  evidence: [{ note_id: "note-a", note_date: "2026-01-05", quote: "CT CHEST", start: 3, end: 11, state: "yes", criterion: "CT", item: "CT (chest)", date_text: "2026-01-05", note_number: 1 }],
};
const review = { notice: "Review aid.", metadata: { method: "extraction_queries", note_count: 1, model: "m", coverage: { documents: 1, items: 1 } }, assessments: [item] };

test("assessments escape text, name extraction statuses and link quotes to notes", () => {
  const html = extractionAssessment(item, { noteRef: (v) => `data-show-note="${v.note_number}"` });
  assert.match(html, /data-show-note="1"/);
  assert.match(html, /&lt;b&gt;/);
  assert.match(html, /Met · documented in extraction/);
  assert.equal(highlightedNote("ab CT CHEST z", 3, 11), "ab <mark>CT CHEST</mark> z");
  assert.equal(highlightedNote("<x>", 5, 9), "&lt;x&gt;");
});

test("population tables and drilldowns use the method's statuses", () => {
  const rows = [{ recommendation_index: 0, recommendation: { name: "CT" }, statuses: { met: 1, not_met: 0, unknown: 1, not_applicable: 0 }, applicability: { applies: 2, not_applicable: 0, uncertain: 0 }, applicable_assessed: 1, completed_applicable: 1 }];
  assert.match(distributionTable(rows, "extraction_queries"), /Met \/ applicable assessed/);
  assert.match(patientRows([{ patient_id: "p", status: "not_met", applicability: "applies", bottom_line: "b" }], "", "extraction_queries"), /Not documented in extraction/);
  const detail = patientDetail({ patient: { cancer_history_summary: "s" }, notes: [{ note_id: "note-a", note_text: "ab CT CHEST z", note_date: "2026-01-05" }], review }, 0, [{ query: { summary: "q" } }]);
  assert.match(detail, /data-pop-note="0"/);
  assert.match(detail, /id="pop-note-0"/);
});

test("exports and labels identify the extraction method", () => {
  assert.equal(reviewMethodLabel(review), "Structured extraction queries");
  assert.equal(workupStatusLabel({ status: "unknown" }, review), "Unknown");
  const report = buildExportReport({ report: "# R", spaces: [{ rank: 1, name: "S", source: { title: "T", version: "1" }, space_trial_id: "s" }] }, { 0: review });
  assert.match(report, /Notes extracted/);
  assert.match(costLine({ reused_extraction: true, compile_requests: 0, workup_items: 2, requests: 0 }), /already extracted/);
});

// Fabricated SQL-agent result: the same statuses, with the item's SQL instead of criteria.
const sqlEntry = { status: "compiled", sql: "SELECT p.patient_key FROM patients p -- <x>", summary: "A CT.", accepted_by: "agent", action_count: 9, submit_count: 2, model: "m", distribution: { patients: 1000, status_reason: { "met/documented": 700, "not_met/not_documented": 300 } } };
const sqlItem = { ...item, criteria: [], sql: sqlEntry.sql };
const sqlReview = { ...review, metadata: { ...review.metadata, method: "extraction_sql" }, assessments: [sqlItem] };

test("SQL-agent results render as extraction results with their SQL", () => {
  const html = extractionAssessment(sqlItem, { sql: sqlEntry });
  assert.match(html, /SQL query/);
  assert.match(html, /-- &lt;x&gt;/);
  assert.match(html, /Accepted by the agent after 9 actions \(2 submissions\)/);
  assert.match(html, /met \/ documented 700/);
  assert.doesNotMatch(html, /Query criteria/);
  assert.equal(reviewMethodLabel(sqlReview), "SQL agent over extraction records");
  assert.equal(workupStatusLabel({ status: "met" }, sqlReview), "Met · documented in extraction");
  assert.match(distributionTable([{ recommendation_index: 0, recommendation: { name: "CT" }, statuses: { met: 1, not_met: 0, unknown: 0, not_applicable: 0 }, applicability: { applies: 1, not_applicable: 0, uncertain: 0 }, applicable_assessed: 1, completed_applicable: 1 }], "extraction_sql"), /SQL-agent results/);
  const detail = patientDetail({ patient: { cancer_history_summary: "s" }, notes: [{ note_id: "note-a", note_text: "ab CT CHEST z", note_date: "2026-01-05" }], review: sqlReview }, 0, [sqlEntry]);
  assert.match(detail, /SQL agent over extraction records · 1 document records/);
  assert.match(detail, /data-pop-note="0"/);
  assert.match(detail, /an LLM agent wrote each workup item/);
});

test("SQL-agent cost lines name the store and the request ceiling", () => {
  const cost = { reused_extraction: true, compile_requests: 40, cached_queries: 1, workup_items: 3, requests: 40, sql_agent: { store: "synthetic", patients: 1000, database_built: false, items_to_explore: 2, max_actions_per_item: 20 } };
  const line = costLine(cost);
  assert.match(line, /SQL agent: 2 of 3 workup items to explore \(1 cached\), up to 20 requests each/);
  assert.match(line, /synthetic \(1,000 patients; its database is built first\)/);
  assert.match(line, /without patient ids, note text or quotes/);
  assert.match(line, /At most 40 requests/);
  assert.match(costLine({ ...cost, sql_agent: { ...cost.sql_agent, items_to_explore: 0 } }), /All 3 workup SQL queries are cached for synthetic/);
});

test("steps the guideline advises against have their own status and no completion meter", () => {
  const rows = [{ recommendation_index: 0, recommendation: { name: "PET/CT not routinely indicated" }, recommends_against: true, statuses: { met: 0, not_met: 3, unknown: 0, not_applicable: 0, done_not_recommended: 2 }, applicability: { applies: 5, not_applicable: 0, uncertain: 0 }, applicable_assessed: 0, completed_applicable: 0 }];
  const html = distributionTable(rows, "extraction_sql");
  assert.match(html, /Advised against · 2 done anyway/);
  assert.match(html, /Done though not recommended/);
  assert.doesNotMatch(html, /<meter/);
  const item = { ...sqlItem, status: "done_not_recommended", reason: "documented_not_recommended" };
  const detail = extractionAssessment(item, { sql: { ...sqlEntry, recommends_against: true } });
  assert.match(detail, /Done though not recommended/);
  assert.match(detail, /Documented although the guideline advises against it/);
  assert.match(detail, /advises against this step/);
});
