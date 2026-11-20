const escape = (value) => String(value ?? "").replace(/[&<>"']/g, ch => (
  {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[ch]
));

const dateLabel = value => value
  ? `<time datetime="${escape(value)}">${escape(value.replace("T", "\n"))}</time>`
  : "Date unavailable";

export function savedCompressionMatches(saved, text, notes) {
  if (!saved || saved.notes_input.text !== text) return false;
  const normalize = rows => rows?.length ? rows.map(row => ({
    note_text: row.note_text, note_date: row.note_date ?? null, note_type: row.note_type ?? null,
  })) : null;
  return JSON.stringify(normalize(saved.notes_input.notes)) === JSON.stringify(normalize(notes));
}

export function clinicalNoteTable(text, notes, saved) {
  const rows = notes?.length ? notes : text.trim() ? [{note_text: text, note_date: null}] : [];
  const compressed = savedCompressionMatches(saved, text, notes) ? saved.notes : null;
  return {
    count: rows.length,
    hasCompression: Boolean(compressed),
    html: `<thead><tr><th scope="col">Date / note</th><th scope="col">Original note</th>${compressed ? '<th scope="col">Compressed note</th>' : ""}</tr></thead><tbody>${rows.map((row, index) => `<tr data-note-row="${index}"><th scope="row"><strong>Note ${index + 1}</strong><span class="note-date">${dateLabel(row.note_date)}</span>${row.note_type ? `<span class="note-type">${escape(row.note_type)}</span>` : ""}<button class="note-expand" data-expand-note="${index}" aria-expanded="false">Show whole note</button></th><td><div class="note-text" tabindex="0" aria-label="Original note ${index + 1}">${escape(row.note_text)}</div></td>${compressed ? `<td><div class="note-text" tabindex="0" aria-label="Compressed note ${index + 1}">${escape(compressed[index].compressed_note_text)}</div></td>` : ""}</tr>`).join("")}</tbody>`,
  };
}
