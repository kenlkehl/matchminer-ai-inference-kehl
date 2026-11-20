import assert from "node:assert/strict";
import test from "node:test";
import { clinicalNoteTable, savedCompressionMatches } from "../../../src/matchminer_ai/apps/guideline_fetcher/static/note_view.mjs";

const text = "Fabricated source record";
const notes = [
  {note_text: "A long fabricated note\n".repeat(80) + "Final detail retained.", note_date: "2026-01-01", note_type: "Visit"},
  {note_text: "<script>alert('source')</script> & tail", note_date: null},
];
const saved = {
  notes_input: {text, notes},
  notes: notes.map(row => ({...row, compressed_note_text: "<img src=x> & compressed tail"})),
};

test("table shows full text, dates, note types and optional compressed column safely", () => {
  const original = structuredClone({text, notes, saved});
  const view = clinicalNoteTable(text, notes, saved);
  assert.equal(view.count, 2);
  assert.equal(view.hasCompression, true);
  assert.match(view.html, /2026-01-01/);
  assert.match(view.html, /Visit/);
  assert.match(view.html, /Date unavailable/);
  assert.match(view.html, /Final detail retained\./);
  assert.equal(view.html.split("A long fabricated note\n").length - 1, 80);
  assert.match(view.html, /Compressed note/);
  assert.match(view.html, /Show whole note/);
  assert.match(view.html, /&lt;script&gt;/);
  assert.match(view.html, /&lt;img src=x&gt; &amp; compressed tail/);
  assert.doesNotMatch(view.html, /<script>|<img/);
  assert.deepEqual({text, notes, saved}, original);
});

test("changed text, dates, types, note order or note count hides stale compression", () => {
  for (const changed of [
    [text + " changed", notes],
    [text, [{...notes[0], note_text: "New patient"}, notes[1]]],
    [text, [{...notes[0], note_date: "2026-02-01"}, notes[1]]],
    [text, [{...notes[0], note_type: "Different type"}, notes[1]]],
    [text, [...notes].reverse()],
    [text, [notes[0]]],
  ]) {
    assert.equal(savedCompressionMatches(saved, ...changed), false);
    const view = clinicalNoteTable(...changed, saved);
    assert.equal(view.hasCompression, false);
    assert.doesNotMatch(view.html, /<th scope="col">Compressed note<\/th>/);
  }
});

test("source matching ignores key order and null optional metadata", () => {
  const reordered = notes.map(row => ({note_type: row.note_type ?? null,
    note_date: row.note_date, note_text: row.note_text}));
  assert.equal(savedCompressionMatches(saved, text, reordered), true);
});

test("free text remains one undated complete note; empty input shows no rows", () => {
  const view = clinicalNoteTable("Pasted\ncomplete text", null, null);
  assert.equal(view.count, 1);
  assert.equal(view.hasCompression, false);
  assert.match(view.html, /Date unavailable/);
  assert.match(view.html, /Pasted\ncomplete text/);
  assert.equal(clinicalNoteTable("  ", null, null).count, 0);
});
