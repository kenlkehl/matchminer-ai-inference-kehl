import { applicabilityLabels, buildExportReport, isAgentic, isCompressed, reviewMethodLabel, workupStatusLabel, workupReviewDetail, workupBackupDetail, patientEvidenceLabel, patientEvidenceCountLabel, missingEvidenceLabel } from "./report.mjs";
import { clinicalNoteTable, savedCompressionMatches } from "./note_view.mjs";
import { costLine, extractionAssessment, highlightedNote, isExtractionReview, reviewAid } from "./extraction_view.mjs";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
// Workup methods that check structured extraction records of the original notes.
const usesExtraction = (method) => method === "extractions" || method === "extraction_sql";
const state = {
  catalogs: [],
  config: {},
  busy: false,
  ready: false,
  imported: null,
  syntheticCase: "",
  result: null,
  retrievalJob: null,
  workup: {},
  workupMethod: "full_notes",
  extractionLevels: "document",
  noteSource: "original",
  noteCompression: null,
  input: "summary",
  started: 0,
  timer: null,
  activeJob: null,
  stopping: false,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...options,
  });
  // PHI mode only: a restarted server with a new token, or a signed-out browser, needs sign-in.
  if (response.status === 401) location.assign("/login");
  if (!response.ok) {
    let message = "The request could not be completed. Please try again.";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (response.status === 422)
        message = "Check the input values and settings, then try again.";
    } catch {
      /* use safe fallback */
    }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}
function post(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
function error(message) {
  $("#error").textContent = message;
  $("#error").hidden = !message;
}
let toastTimer;
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("#toast").hidden = true), 5000);
}
function updateInput() {
  const savedCompression = Boolean(state.noteCompression);
  const words = $("#summary").value.trim().split(/\s+/).filter(Boolean).length;
  $("#summary-count").textContent = `${words.toLocaleString()} words`;
  $("#retrieve").disabled =
    !state.ready || state.busy || !words || !state.catalogs.length || state.input !== "summary";
  $("#summarize").disabled = !state.ready || state.busy || !$("#notes").value.trim();
  $$("[data-synthetic-case], #synthetic-notes, #example").forEach((el) => {
    el.disabled = !state.ready || state.busy || !state.syntheticCase;
  });
  $$("[data-review-workup]").forEach((el) => {
    el.disabled = !state.ready || state.busy || !state.retrievalJob || !$("#notes").value.trim();
    el.textContent = state.workupMethod === "extraction_sql"
      ? "Check workup with the SQL agent · original notes"
      : state.workupMethod === "extractions"
      ? "Check workup with extraction queries · original notes"
      : state.noteSource === "compressed"
        ? "Review workup in saved compressed notes" : "Review workup in original notes";
  });
  $$("[data-workup-method]").forEach((el) => {
    el.disabled = state.busy;
    el.checked = el.value === state.workupMethod;
  });
  $$("[data-note-source]").forEach((el) => {
    el.value = state.noteSource;
    el.disabled = state.busy;
    const option = el.querySelector('option[value="compressed"]');
    option.disabled = !savedCompression;
    option.textContent = savedCompression ? "Saved compressed notes" : "Saved compressed notes · unavailable";
  });
  $("#compress-notes").disabled = !state.ready || state.busy || !$("#notes").value.trim() || savedCompression;
  $("#compress-notes").textContent = savedCompression ? "Compression saved" : "Compress notes";
  $("#compression-status").textContent = savedCompression
    ? `${state.noteCompression.metadata.note_count} notes saved · ${state.noteCompression.metadata.input_characters.toLocaleString()} → ${state.noteCompression.metadata.output_characters.toLocaleString()} characters. Choose original or compressed notes below. Losslessness is unverified.`
    : "Optional standalone action. Compression may lose information; original notes stay intact.";
  $("#notes-table-button").disabled = !$("#notes").value.trim();
}
let compressionReset = Promise.resolve();
function discardSavedCompression(clearServer = true) {
  const hadCompression = Boolean(state.noteCompression);
  state.noteCompression = null;
  state.noteSource = "original";
  renderNotesView();
  if (clearServer && hadCompression) {
    compressionReset = compressionReset.catch(() => {}).then(() =>
      api("/api/note-compression", {method:"DELETE"}),
    );
    compressionReset.catch(e => error(e.message));
  }
  return compressionReset;
}
function restoreSavedCompression(saved, restoreNotes = false) {
  if (saved && restoreNotes && !$("#notes").value) {
    $("#notes").value = saved.notes_input.text;
    state.imported = saved.notes_input.notes;
  }
  state.noteCompression = savedCompressionMatches(saved, $("#notes").value, state.imported)
    ? saved : null;
  if (!state.noteCompression) state.noteSource = "original";
  renderNotesView();
  updateInput();
}
function renderNotesView() {
  if (!$("#notes-dialog").open) return;
  const view = clinicalNoteTable($("#notes").value, state.imported, state.noteCompression);
  $("#clinical-notes-table").innerHTML = view.html;
  $("#clinical-notes-table").classList.toggle("has-compression", view.hasCompression);
  $("#notes-view-status").textContent = `${view.count} ${view.count === 1 ? "note" : "notes"} · original text and clinical dates`;
  $("#notes-view-compression").textContent = view.hasCompression
    ? `Saved compression from ${state.noteCompression.model}. Available for optional reuse in summaries and workup checks. Compression may lose information; compare with the original notes.`
    : "Run Compress notes to add compressed text alongside the originals. Each note has a scrollable reading area; Show whole note expands it.";
  $("#expand-all-notes").textContent = "Show whole notes";
}
$("#notes-table-button").addEventListener("click", () => {
  $("#notes-dialog").showModal();
  renderNotesView();
});
function expandNote(row, expanded) {
  row.classList.toggle("expanded", expanded);
  const button = row.querySelector("[data-expand-note]");
  button.setAttribute("aria-expanded", String(expanded));
  button.textContent = expanded ? "Collapse note" : "Show whole note";
}
$("#clinical-notes-table").addEventListener("click", event => {
  const button = event.target.closest("[data-expand-note]");
  if (button) expandNote(button.closest("tr"), button.getAttribute("aria-expanded") !== "true");
});
$("#expand-all-notes").addEventListener("click", () => {
  const rows = $$("#clinical-notes-table tbody tr");
  const expanded = !rows.every(row => row.classList.contains("expanded"));
  rows.forEach(row => expandNote(row, expanded));
  $("#expand-all-notes").textContent = expanded ? "Collapse notes" : "Show whole notes";
});
document.addEventListener("change", (event) => {
  if (event.target.matches("[data-note-source]") && !state.busy) {
    state.noteSource = event.target.value;
    updateInput();
  }
});
function clearWorkup() {
  state.workup = {};
  if (state.result) renderResults();
}
function invalidate() {
  state.result = null;
  state.retrievalJob = null;
  state.workup = {};
  $("#results").hidden = true;
  $("#download").hidden = true;
  $("#empty").hidden = false;
  $("#space-list").replaceChildren();
}
function setBusy(value, population = false) {
  if (!population) document.dispatchEvent(new CustomEvent("patient-busy", {detail: value}));
  state.busy = value;
  $$(
    "#summary, #notes, #upload, #clear, #catalog, #top-n, #candidate-k, #refresh-library, [data-use-catalog]",
  ).forEach((el) => (el.disabled = value));
  $$("[data-input]").forEach((el) => (el.disabled = value));
  updateInput();
}
function inputTab(name) {
  state.input = name;
  $$("[data-input]").forEach((el) => {
    const active = el.dataset.input === name;
    el.setAttribute("aria-selected", String(active));
    el.tabIndex = active ? 0 : -1;
  });
  $("#summary-pane").hidden = name !== "summary";
  $("#notes-pane").hidden = name !== "notes";
  $("#summary-actions").hidden = name !== "notes";
  updateInput();
}
$$("[data-input]").forEach((button) =>
  button.addEventListener("click", () => inputTab(button.dataset.input)),
);
$(".input-tabs").addEventListener("keydown", (event) => {
  if (
    ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key) &&
    !state.busy
  ) {
    event.preventDefault();
    inputTab(state.input === "summary" ? "notes" : "summary");
    $(`#${state.input}-tab`).focus();
  }
});
$("#summary").addEventListener("input", () => {
  invalidate();
  $("#exclusion-section").hidden = true;
  $("#exclusion-evidence").textContent = "";
  $("#summary-compression-notice").hidden = true;
  $("#summary-label").textContent = "Your summary stays editable";
  updateInput();
});
$("#notes").addEventListener("input", () => {
  state.imported = null;
  discardSavedCompression();
  clearWorkup();
  $("#upload-notice").textContent =
    "Edited free text will be summarized. Dates are preserved only when stated in the text.";
  updateInput();
});
$("#catalog").addEventListener("change", invalidate);
$("#top-n").addEventListener("change", () => {
  if (+$("#candidate-k").value < +$("#top-n").value)
    $("#candidate-k").value = $("#top-n").value;
  invalidate();
});
$("#candidate-k").addEventListener("change", invalidate);

async function loadCatalogs(refresh = false) {
  const data = await api(`/api/catalogs?refresh=${refresh}`);
  state.catalogs = data.catalogs;
  const selected = $("#catalog").value;
  $("#catalog").innerHTML =
    `<option value="">All completed guidelines (${data.catalogs.length})</option>` +
    data.catalogs
      .map(
        (row) =>
          `<option value="${escape(row.id)}">${escape(row.title)} · ${row.spaces} spaces</option>`,
      )
      .join("");
  if (data.catalogs.some((r) => r.id === selected))
    $("#catalog").value = selected;
  $("#nav-count").textContent = data.catalogs.length;
  $("#library-total").textContent =
    `${data.catalogs.length} completed guidelines · ${data.spaces.toLocaleString()} catalog spaces`;
  renderLibrary();
  updateInput();
}
function renderLibrary() {
  const query = $("#library-search").value.toLowerCase();
  const rows = state.catalogs.filter((r) =>
    `${r.title} ${r.disease}`.toLowerCase().includes(query),
  );
  $("#library-list").innerHTML = rows.length
    ? rows
        .map(
          (r) =>
            `<div class="library-row"><div><h3>${escape(r.title)}</h3><p>${escape(r.version)} · ${r.spaces} spaces · Stored audit passed</p></div><button class="button secondary" data-use-catalog="${escape(r.id)}" ${state.busy ? "disabled" : ""}>Select for search →</button></div>`,
        )
        .join("")
    : '<p class="empty-menu">No completed guidelines match this search.</p>';
}
$("#library-search").addEventListener("input", renderLibrary);
$("#library-nav").addEventListener("click", () =>
  $("#library-dialog").showModal(),
);
$$("[data-close]").forEach((el) =>
  el.addEventListener("click", () => $(`#${el.dataset.close}`).close()),
);
$$("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) {
      const r = dialog.getBoundingClientRect();
      if (
        e.clientX < r.left ||
        e.clientX > r.right ||
        e.clientY < r.top ||
        e.clientY > r.bottom
      )
        dialog.close();
    }
  }),
);
$("#library-list").addEventListener("click", (event) => {
  const button = event.target.closest("[data-use-catalog]");
  if (!button || state.busy) return;
  $("#catalog").value = button.dataset.useCatalog;
  invalidate();
  $("#library-dialog").close();
  $("#summary").focus();
  toast("Guideline selected for the next search.");
});
$("#refresh-library").addEventListener("click", async () => {
  try {
    $("#refresh-library").disabled = true;
    await loadCatalogs(true);
    toast("Completed catalog list refreshed.");
  } catch (e) {
    error(e.message);
  } finally {
    $("#refresh-library").disabled = false;
  }
});
function selectSyntheticCase(source) {
  state.syntheticCase = source;
  $$("[data-synthetic-case]").forEach((el) => { el.value = source; });
}
$$("[data-synthetic-case]").forEach((select) => {
  select.addEventListener("change", () => selectSyntheticCase(select.value));
});
async function useSyntheticCase(createSummary = false) {
  if (state.busy || !state.ready || !state.syntheticCase) return;
  setBusy(true);
  error("");
  try {
    const source = encodeURIComponent(state.syntheticCase);
    const data = await api(`/api/synthetic-notes?source=${source}`);
    if (!savedCompressionMatches(state.noteCompression, data.text, data.notes)) {
      await discardSavedCompression();
    }
    $("#notes").value = data.text;
    state.imported = data.notes;
    $("#upload-notice").textContent = data.notice;
    $("#summary").value = "";
    $("#exclusion-section").hidden = true;
    $("#exclusion-evidence").textContent = "";
    $("#summary-compression-notice").hidden = true;
    $("#summary-label").textContent = "Create a summary from the synthetic notes";
    invalidate();
    if (createSummary) {
      await runJob("summarize", { text: data.text, notes: data.notes, note_source: state.noteSource });
    } else {
      toast("Synthetic notes loaded. Create a patient summary, then find guideline spaces.");
    }
  } catch (e) {
    error(e.message);
  } finally {
    setBusy(false);
  }
}
$("#example").addEventListener("click", () => useSyntheticCase(true));
$("#synthetic-notes").addEventListener("click", () => useSyntheticCase());
$("#upload").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > (state.config.max_upload_mb || 20) * 1024 * 1024) {
    error("The file exceeds the 20 MB upload limit.");
    event.target.value = "";
    return;
  }
  setBusy(true);
  error("");
  $("#upload-notice").textContent = "Reading the uploaded record…";
  try {
    const form = new FormData();
    form.append("file", file);
    const data = await api("/api/upload", { method: "POST", body: form });
    await discardSavedCompression();
    $("#notes").value = data.text;
    state.imported = data.notes;
    clearWorkup();
    $("#upload-notice").textContent = data.notice;
    toast("Record imported. Review the text, then create a summary.");
  } catch (e) {
    error(e.message);
    $("#upload-notice").textContent =
      "No record was imported. Choose another file or paste notes.";
  } finally {
    setBusy(false);
    event.target.value = "";
  }
});

async function runJob(kind, body, existingJob = null) {
  error("");
  setBusy(true);
  state.stopping = false;
  $("#stop-job").disabled = true;
  $("#stop-job").textContent = "Stop";
  $("#progress").hidden = false;
  $("#empty").hidden = true;
  $("#progress-title").textContent =
    kind === "retrieve"
      ? "Finding relevant guideline spaces"
      : kind === "compress" ? "Compressing clinical notes"
      : kind === "workup" ? (state.workupMethod === "extraction_sql"
        ? "Checking workup with the SQL agent"
        : state.workupMethod === "extractions"
        ? "Checking workup with extraction queries"
        : state.noteSource === "compressed"
          ? "Reviewing workup in saved compressed notes" : "Reviewing workup in original notes")
      : state.noteSource === "compressed"
        ? "Creating the summary from saved compressed notes" : "Creating the patient summary from original notes";
  $("#progress-message").textContent = "Preparing your request";
  state.started = Date.now();
  const tick = () => {
    const seconds = Math.floor((Date.now() - state.started) / 1000);
    $("#elapsed").textContent =
      `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  };
  tick();
  state.timer = setInterval(tick, 1000);
  try {
    await compressionReset;
    const job_id = existingJob || (await post(`/api/${kind}`, body)).job_id;
    state.activeJob = job_id;
    $("#stop-job").disabled = false;
    let failures = 0;
    while (true) {
      let job;
      try {
        job = await api(`/api/jobs/${job_id}`);
        failures = 0;
      } catch (e) {
        if (++failures >= 5) throw e;
        $("#progress-message").textContent =
          "Connection interrupted. Reconnecting to the job…";
        await new Promise((r) => setTimeout(r, 3000));
        continue;
      }
      $("#progress-message").textContent = job.progress.replace(
        /^Embedding 1 patient summaries\b/,
        "Embedding 1 patient summary",
      );
      if (job.status === "failed") throw new Error(job.progress);
      if (job.status === "cancelling") {
        state.stopping = true;
        $("#stop-job").disabled = true;
        $("#stop-job").textContent = "Stopping…";
      }
      if (job.status === "cancelled") {
        toast(kind === "workup"
          ? "Workup review stopped. Your notes and guideline results are available to use again."
          : "Job stopped. You can start another job.");
        if (!state.result) $("#empty").hidden = false;
        break;
      }
      if (job.status === "complete") {
        if (kind === "compress") {
          toast("Notes compressed and saved. Choose original or compressed notes for your next summary or workup check.");
          if (!state.result) $("#empty").hidden = false;
        } else if (kind === "summarize") {
          restoreSummary(job.result);
          invalidate();
          inputTab("summary");
          toast(
            "Summary ready. Review or edit it, then find guideline spaces.",
          );
        } else if (kind === "workup") {
          restoreWorkup(job.result);
          toast("Workup documentation review ready. Inspect the findings and source excerpts.");
        } else {
          state.result = job.result;
          state.retrievalJob = job_id;
          $("#summary").value = job.result.summary;
          renderResults();
          toast("Guideline considerations ready for review.");
        }
        break;
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  } catch (e) {
    error(e.message);
    if (!state.result) $("#empty").hidden = false;
  } finally {
    clearInterval(state.timer);
    state.activeJob = null;
    state.stopping = false;
    $("#stop-job").disabled = true;
    $("#progress").hidden = true;
    try {
      const saved = await api("/api/note-compression");
      restoreSavedCompression(saved.note_compression);
    } catch { /* completed inference remains available if cache refresh fails */ }
    setBusy(false);
  }
}
$("#stop-job").addEventListener("click", async () => {
  if (!state.activeJob || state.stopping) return;
  const key = state.activeJob;
  state.stopping = true;
  $("#stop-job").disabled = true;
  $("#stop-job").textContent = "Stopping…";
  try {
    await post(`/api/jobs/${key}/stop`, {});
  } catch (e) {
    if (state.activeJob === key) {
      state.stopping = false;
      $("#stop-job").disabled = false;
      $("#stop-job").textContent = "Stop";
      error(e.message);
    }
  }
});
$("#retrieve").addEventListener("click", () => {
  const top_n = +$("#top-n").value,
    candidate_k = +$("#candidate-k").value;
  if (
    !Number.isInteger(candidate_k) ||
    candidate_k < top_n ||
    candidate_k > 100
  ) {
    error(
      "Candidate count must be an integer between the requested result count and 100.",
    );
    return;
  }
  invalidate();
  runJob("retrieve", {
    summary: $("#summary").value,
    catalogs: $("#catalog").value ? [$("#catalog").value] : [],
    top_n,
    candidate_k,
  });
});
$("#summarize").addEventListener("click", () =>
  runJob("summarize", { text: $("#notes").value, notes: state.imported, note_source: state.noteSource }),
);
$("#compress-notes").addEventListener("click", () =>
  runJob("compress", {
    text: $("#notes").value, notes: state.imported, summary: $("#summary").value,
    retrieval_job_id: state.retrievalJob, note_source: state.noteSource,
  }),
);

function evidence(items) {
  if (!items?.length) return "";
  return `<details class="evidence"><summary>View source evidence · ${items.length} ${items.length === 1 ? "excerpt" : "excerpts"}</summary>${items.map((item) => `<blockquote>${escape(item.quote)}<cite>${escape(item.printed_label || item.page_id)}${item.pdf_page ? ` · PDF page ${item.pdf_page}` : ""} · Lines ${escape((item.line_ids || []).join(", "))}</cite></blockquote>`).join("")}</details>`;
}

function citationReview(owner) {
  const review = owner.citation_review;
  if (!review?.issues?.length) return "";
  return `<div class="uncertainty"><strong>Source support needs review</strong><ul>${review.issues.map((issue) => `<li>${escape(issue)}</li>`).join("")}</ul><p>Any excerpts below support only part of this catalog item.</p></div>`;
}
function renderConditions(value) {
  const text = String(value ?? "").trim();
  const normalized = text.toLowerCase().replace(/\s+/g, " ").replace(/\.+$/, "");
  if (["", "na", "n/a", "not specified", "not specified in supplied evidence",
    "not specified in the supplied evidence"].includes(normalized)) return "";
  return `<p class="conditions">${escape(text)}</p>`;
}
function menu(items) {
  if (!items?.length)
    return '<p class="empty-menu">No considerations in this category were recorded for this catalog space.</p>';
  return items
    .map(
      (item, i) =>
        `<article class="consideration"><div class="option-top"><span class="option-number">${String(i + 1).padStart(2, "0")}</span><div class="option-content"><h4>${escape(item.name)}</h4><span class="category">${escape(item.category)}</span>${renderConditions(item.conditions)}${citationReview(item)}${evidence(item.evidence)}</div></div></article>`,
    )
    .join("");
}
function fields(values, labels = {}) {
  return `<dl class="field-grid">${Object.entries(values)
    .map(
      ([key, value]) =>
        `<div><dt>${escape(labels[key] || key)}</dt><dd>${escape(value)}</dd></div>`,
    )
    .join("")}</dl>`;
}
function content(row, type) {
  if (type === "workup") {
    const index = state.result.spaces.indexOf(row);
    const review = state.workup[index];
    const controls = row.diagnostic_workup.length
      ? `<section class="workup-review"><h4>Was this workup done?</h4><p>Choose how to review the listed items in the clinical notes. Each method uses your saved concordance model and its enabled backup. The patient summary is not used as evidence.</p><fieldset class="workup-method"><legend>Review method</legend><label><input type="radio" name="workup-method-${index}" data-workup-method value="full_notes" ${state.workupMethod === "full_notes" ? "checked" : ""}><span><strong>Full-note question answering</strong><small>Reviews every supplied note chunk for each item.</small></span></label><label><input type="radio" name="workup-method-${index}" data-workup-method value="agentic" ${state.workupMethod === "agentic" ? "checked" : ""}><span><strong>Agentic note search</strong><small>Searches each item in parallel, checks coverage, and reviews more notes when needed. May miss evidence.</small></span></label><label><input type="radio" name="workup-method-${index}" data-workup-method value="extractions" ${state.workupMethod === "extractions" ? "checked" : ""}><span><strong>Structured extraction queries</strong><small>Extracts each original note into structured items (one LLM request per note), compiles each workup item once into a cached query, and checks the queries in code. Each result cites its notes.</small></span></label><label><input type="radio" name="workup-method-${index}" data-workup-method value="extraction_sql" ${state.workupMethod === "extraction_sql" ? "checked" : ""}><span><strong>SQL agent over extractions</strong><small>Extracts each original note into structured items, then an LLM agent explores the extraction record store with read-only SQL and writes one cached query per workup item; code runs it over this patient's items. Each result cites its notes. The agent's query results (extracted values, without patient ids or note text) go to the concordance connection.</small></span></label></fieldset>${usesExtraction(state.workupMethod) ? `<div class="workup-extraction"><label><input type="checkbox" data-extraction-levels ${state.extractionLevels === "document_and_patient" ? "checked" : ""}> Also build a patient-level record from all notes (more requests)</label><p class="helper" data-extraction-cost="${index}">Estimating what this sends…</p></div>` : ""}<div class="workup-note-source"><label for="workup-note-source-${index}">Notes to use</label><select id="workup-note-source-${index}" data-note-source><option value="original">Original notes</option><option value="compressed">Saved compressed notes</option></select><p class="helper">Choose saved compressed notes after using Compress notes in The patient panel. This review does not run compression.</p></div><button class="button primary" data-review-workup="${index}">Review workup in original notes</button><p class="helper">Add notes under The patient → Clinical notes. Conditional items may not apply. Missing documentation does not mean “not done.”</p>${review ? renderWorkup(review, index) : ""}</section>`
      : "";
    return controls + menu(row.diagnostic_workup);
  }
  if (type === "treatment") return menu(row.treatment_options);
  if (type === "population")
    return (
      `<p class="population-text">${escape(row.cohort_key_criteria || row.clinical_space_summary)}</p>` +
      `<p class="helper">The population's key criteria, as TrialSpace and the decision model read them, written like a trial cohort's key criteria (protocol-style sentences in upgraded catalogs; the nine population fields, NA fields omitted, in older ones). A match does not establish that a patient meets the full guideline population.</p>${citationReview(row)}${evidence(row.evidence)}`
    );
  let html = fields({
    Guideline: row.source.title,
    Edition: row.source.version,
    "Space ID": row.space_trial_id,
    "Match quality": Number(row.match_quality_score).toFixed(4),
    "TrialSpace cosine": Number(row.similarity_score).toFixed(4),
    "Retrieval rank": row.retrieval_rank,
  });
  html +=
    '<p class="helper">Match-quality scores from the decision model are ranking signals, not probabilities. The cutoff flag is retained; low-scoring results are not silently removed.</p>';
  if (row.uncertainties?.length)
    html += `<div class="uncertainty"><strong>Catalog uncertainties</strong><ul>${row.uncertainties.map((t) => `<li>${escape(t)}</li>`).join("")}</ul></div>`;
  else
    html +=
      '<p class="helper">No explicit uncertainties recorded. This does not establish clinical correctness.</p>';
  return html;
}
function renderResults() {
  const search = state.result.search;
  if (search) {
    $("#catalog").value = search.catalogs.length === 1 ? search.catalogs[0] : "";
    $("#top-n").value = String(search.top_n);
    $("#candidate-k").value = String(search.candidate_k);
  }
  const rows = state.result.spaces;
  $("#empty").hidden = true;
  $("#results").hidden = false;
  $("#download").hidden = !rows.length;
  $("#results-summary").innerHTML =
    `<strong>${rows.length} ranked ${rows.length === 1 ? "space" : "spaces"}</strong> · ${escape(state.result.provenance?.catalog?.catalog_spaces ?? "Selected")} catalog spaces searched`;
  $("#space-list").innerHTML = rows.length
    ? rows
        .map(
          (row, i) =>
            `<details class="space-card" data-index="${i}" ${i === 0 ? "open" : ""}><summary><span class="rank">${row.rank}</span><div class="space-title"><h3>${escape(row.name)}</h3><p class="space-subtitle">${escape(row.source.title)} · ${escape(row.source.version)}</p><div class="score-row"><span class="score-pill ${row.match_quality_pass ? "" : "low"}">Match quality ${Number(row.match_quality_score).toFixed(3)}${row.match_quality_pass ? "" : " · Below cutoff"}</span><span class="counts">${row.diagnostic_workup.length} workup · ${row.treatment_options.length} treatment${row.uncertainties.length ? ` · ${row.uncertainties.length} caveats` : ""}</span></div></div><span class="chevron" aria-hidden="true">⌄</span></summary><div class="card-body"><div class="result-tabs" role="tablist" aria-label="Consideration category">${[
              ["workup", "Workup"],
              ["treatment", "Treatment"],
              ["population", "Population"],
              ["source", "Source & caveats"],
            ]
              .map(
                ([key, label], j) =>
                  `<button role="tab" id="tab-${i}-${key}" aria-controls="content-${i}" aria-selected="${j === 0}" tabindex="${j === 0 ? 0 : -1}" data-section="${key}">${label}</button>`,
              )
              .join(
                "",
              )}</div><div class="card-content" id="content-${i}" role="tabpanel" aria-labelledby="tab-${i}-workup">${content(row, "workup")}</div></div></details>`,
        )
        .join("")
    : '<p class="empty-menu">No spaces were returned. Try a broader guideline scope.</p>';
  $("#provenance").innerHTML =
    `<pre>${escape(JSON.stringify(state.result.provenance, null, 2))}</pre>`;
  $$("[data-extraction-cost]").forEach((el) => refreshExtractionCost(+el.dataset.extractionCost));
  updateInput();
}
function extractionNoteRef(evidence) {
  return evidence.note_number ? `data-show-note="${evidence.note_number}" data-start="${Number(evidence.start)}" data-end="${Number(evidence.end)}"` : "";
}
function renderExtractionWorkup(review, index) {
  const meta = review.metadata, x = meta.extraction || {}, q = meta.queries || {};
  const queries = q.queries || [];
  const sql = meta.method === "extraction_sql";
  const compiled = sql
    ? `SQL queries ${q.compiled ?? 0}/${q.distinct ?? 0} accepted, ${q.cached ?? 0} cached, ${q.requests ?? 0} agent requests over ${escape(q.exploration?.name || "the store")} (${(q.exploration?.patients ?? 0).toLocaleString()} patients)`
    : `queries ${q.compiled ?? 0}/${q.distinct ?? 0} compiled, ${q.cached ?? 0} cached, ${q.requests ?? 0} compile requests`;
  const options = (item) => sql ? {sql: queries[item.recommendation_index]} : {query: queries[item.recommendation_index]?.query};
  return `<div class="workup-findings"><div class="workup-heading"><h4>${sql ? "SQL-agent findings" : "Extraction-query findings"}</h4><button class="button secondary" data-download-workup="${index}">Download JSON</button></div><p><strong>${escape(reviewMethodLabel(review))}</strong></p><p class="review-note">${escape(reviewAid(meta.method))}</p><p class="helper">${escape(review.notice)}</p><p class="helper">${meta.note_count} notes · ${x.reused ? "session extraction reused" : `${x.requests ?? 0} extraction requests`} (${escape((x.levels || []).join(" + "))} level${x.failed ? `, ${x.failed} notes failed` : ""}${x.partial ? `, ${x.partial} partial` : ""}) · ${compiled} · ${escape(meta.model || "")}</p>${review.assessments.map((item) => extractionAssessment(item, {noteRef: extractionNoteRef, ...options(item)})).join("")}</div>`;
}
async function refreshExtractionCost(index) {
  const target = $(`[data-extraction-cost="${index}"]`);
  if (!target || !state.retrievalJob || !$("#notes").value.trim()) {
    if (target) target.textContent = "Add clinical notes to estimate what extraction sends.";
    return;
  }
  try {
    const cost = await post("/api/workup/extraction-cost", {
      text: $("#notes").value, notes: state.imported, retrieval_job_id: state.retrievalJob,
      space_index: index, extraction_levels: state.extractionLevels,
      method: state.workupMethod === "extraction_sql" ? "extraction_sql" : "extractions",
    });
    target.textContent = costLine(cost) + " Notes go to the saved concordance connection; use only an endpoint authorized for them.";
  } catch (e) {
    target.textContent = e.message;
  }
}
function renderWorkup(review, index) {
  if (isExtractionReview(review)) return renderExtractionWorkup(review, index);
  const notes = review.metadata.note_count === 1 ? "note" : "notes";
  const scope = isCompressed(review) ? (isAgentic(review)
    ? `compressed ${notes} available to search` : `compressed ${notes} reviewed`)
    : isAgentic(review) ? `${notes} available to search` : `raw ${notes} reviewed`;
  const compression = review.metadata.compression;
  const compressionDetail = compression
    ? `<p class="helper">${compression.reused ? "Saved compression reused" : "Compression"}: ${compression.input_characters.toLocaleString()} → ${compression.output_characters.toLocaleString()} characters · losslessness unverified. Confirm findings against original notes.</p>` : "";
  return `<div class="workup-findings"><div class="workup-heading"><h4>Documentation findings</h4><button class="button secondary" data-download-workup="${index}">Download JSON</button></div><p><strong>${escape(reviewMethodLabel(review))}</strong></p><p class="helper">${escape(review.notice)}</p>${compressionDetail}${workupBackupDetail(review.metadata) ? `<p class="helper">${escape(workupBackupDetail(review.metadata))}</p>` : ""}${review.assessments.map((item) => `<article class="workup-finding"><div class="workup-heading"><h4>${escape(item.name)}</h4><span class="workup-status ${item.status === "completed" ? "completed" : ""}">${escape(workupStatusLabel(item, review))}</span></div><p class="applicability">Applicability: ${escape(applicabilityLabels[item.applicability] || item.applicability)}</p><p>${escape(item.bottom_line)}</p>${workupReviewDetail(item, review) ? `<p class="helper">${escape(workupReviewDetail(item, review))}</p>` : ""}${workupBackupDetail(item.metadata) ? `<p class="helper">${escape(workupBackupDetail(item.metadata))}</p>` : ""}${item.limitations?.length ? `<p class="helper">Limitations: ${item.limitations.map(escape).join("; ")}</p>` : ""}${item.evidence.length ? `<details class="evidence"><summary>${escape(patientEvidenceLabel(item, review))} · ${escape(patientEvidenceCountLabel(item, review))}</summary>${item.evidence.map((e) => `<blockquote>${escape(e.quote)}<cite>Note ${e.note_number} · ${escape(e.note_date || "Date unavailable")}${e.note_type ? " · " + escape(e.note_type) : ""}</cite></blockquote>`).join("")}</details>` : `<p class="helper">${escape(missingEvidenceLabel(item, review))}</p>`}</article>`).join("")}<p class="helper">${review.metadata.note_count} ${scope} · ${review.metadata.requests} LLM ${isCompressed(review) ? "review " : ""}requests</p></div>`;
}
function restoreNoteInput(result) {
  if (result.notes_input) {
    $("#notes").value = result.notes_input.text;
    state.imported = result.notes_input.notes;
  }
  state.noteSource = result.note_source || "original";
}
function restoreCompression(result) {
  if (result.summary_result) restoreSummary(result.summary_result);
  if (result.workup_result) restoreWorkup(result.workup_result);
  else if (result.retrieval) {
    state.result = result.retrieval;
    state.retrievalJob = result.retrieval_job_id;
    renderResults();
  }
  restoreNoteInput(result);
  $("#summary").value = result.summary || "";
}
function restoreSummary(result) {
  restoreNoteInput(result);
  $("#summary").value = result.summary;
  $("#summary-label").textContent = result.compression
    ? "Generated from compressed notes · review before searching" : "Generated summary · review before searching";
  $("#summary-compression-notice").hidden = !result.compression;
  if (result.compression) {
    const stats = result.compression;
    $("#summary-compression-notice").textContent = `${stats.reused ? "Saved compression reused" : "Compression"}: ${stats.input_characters.toLocaleString()} → ${stats.output_characters.toLocaleString()} characters. Losslessness is unverified; confirm the summary against the original notes.`;
  }
  $("#exclusion-evidence").textContent = result.exclusion_evidence;
  $("#exclusion-section").hidden = !result.exclusion_evidence;
}
function restoreWorkup(result) {
  state.result = result.retrieval;
  state.retrievalJob = result.retrieval_job_id;
  if (result.review) state.workup[result.space_index] = result.review;
  state.workupMethod = result.method || (result.review?.metadata?.method === "extraction_queries"
    ? "extractions" : result.review?.metadata?.method) || "full_notes";
  if (!["full_notes", "agentic", "extractions", "extraction_sql"].includes(state.workupMethod)) state.workupMethod = "full_notes";
  state.extractionLevels = result.extraction_levels || state.extractionLevels;
  $("#summary").value = state.result.summary;
  restoreNoteInput(result);
  renderResults();
  $(`.space-card[data-index="${result.space_index}"]`).open = true;
}
function resultTab(button) {
  const card = button.closest(".space-card");
  const index = +card.dataset.index;
  card.querySelectorAll("[data-section]").forEach((el) => {
    const active = el === button;
    el.setAttribute("aria-selected", String(active));
    el.tabIndex = active ? 0 : -1;
  });
  const panel = card.querySelector(".card-content");
  panel.setAttribute("aria-labelledby", button.id);
  panel.innerHTML = content(state.result.spaces[index], button.dataset.section);
  if (button.dataset.section === "workup" && usesExtraction(state.workupMethod)) refreshExtractionCost(index);
  updateInput();
}
$("#space-list").addEventListener("change", (event) => {
  if (event.target.matches("[data-workup-method]") && !state.busy) {
    state.workupMethod = event.target.value;
    rerenderWorkupTab(event.target);
    updateInput();
  }
  if (event.target.matches("[data-extraction-levels]") && !state.busy) {
    state.extractionLevels = event.target.checked ? "document_and_patient" : "document";
    refreshExtractionCost(+event.target.closest(".space-card").dataset.index);
  }
});
function rerenderWorkupTab(element) {
  // Show or hide the extraction options (and their cost) in this space's Workup tab.
  const card = element.closest(".space-card");
  const index = +card.dataset.index;
  card.querySelector(".card-content").innerHTML = content(state.result.spaces[index], "workup");
  if (usesExtraction(state.workupMethod)) refreshExtractionCost(index);
}
function showEvidenceNote(button) {
  const number = Number(button.dataset.showNote);
  const note = state.imported ? state.imported[number - 1] : (number === 1 ? {note_text: $("#notes").value} : null);
  if (!note) return toast("That note is no longer in Clinical notes.");
  $("#evidence-note-title").textContent = `Note ${number} · ${note.note_date || "date unavailable"}${note.note_type ? " · " + note.note_type : ""}`;
  $("#evidence-note-text").innerHTML = highlightedNote(note.note_text, Number(button.dataset.start), Number(button.dataset.end));
  $("#evidence-note-dialog").showModal();
  $("#evidence-note-text mark")?.scrollIntoView({block: "center"});
}
$("#space-list").addEventListener("click", (event) => {
  const noteButton = event.target.closest("[data-show-note]");
  if (noteButton) return showEvidenceNote(noteButton);
  const button = event.target.closest("[data-section]");
  if (button) resultTab(button);
  const review = event.target.closest("[data-review-workup]");
  if (review && !state.busy) {
    const index = +review.dataset.reviewWorkup;
    delete state.workup[index];
    renderResults();
    $(`.space-card[data-index="${index}"]`).open = true;
    runJob("workup", {
      text: $("#notes").value, notes: state.imported,
      retrieval_job_id: state.retrievalJob, space_index: index,
      method: state.workupMethod,
      note_source: usesExtraction(state.workupMethod) ? "original" : state.noteSource,
      ...(usesExtraction(state.workupMethod) ? {extraction_levels: state.extractionLevels} : {}),
    });
  }
  const download = event.target.closest("[data-download-workup]");
  if (download) {
    const index = +download.dataset.downloadWorkup;
    const row = state.result.spaces[index];
    const report = {space_trial_id: row.space_trial_id, space_name: row.name,
      source: row.source, ...state.workup[index]};
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)],
      {type: "application/json"}));
    const link = document.createElement("a");
    link.href = url; link.download = "workup-documentation-review.json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
});
$("#space-list").addEventListener("keydown", (event) => {
  const button = event.target.closest("[data-section]");
  if (
    !button ||
    !["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
  )
    return;
  event.preventDefault();
  const buttons = [...button.parentElement.children],
    i = buttons.indexOf(button);
  const next =
    event.key === "Home"
      ? buttons[0]
      : event.key === "End"
        ? buttons.at(-1)
        : buttons[
            (i + (event.key === "ArrowRight" ? 1 : buttons.length - 1)) %
              buttons.length
          ];
  resultTab(next);
  next.focus();
});
$("#download").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(
    new Blob([buildExportReport(state.result, state.workup)], { type: "text/markdown;charset=utf-8" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "guideline-considerations.md";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$("#clear").addEventListener("click", async () => {
  try {
    await api("/api/session", { method: "DELETE" });
    $("#summary").value = "";
    $("#notes").value = "";
    $("#exclusion-evidence").textContent = "";
    $("#exclusion-section").hidden = true;
    $("#summary-compression-notice").hidden = true;
    state.noteSource = "original";
    state.imported = null;
    discardSavedCompression(false);
    $("#summary-label").textContent = "Your summary stays editable";
    $("#upload-notice").textContent =
      "CSV: note_text, optional note_date and patient_id. One patient per file. PDF uploads require readable text.";
    invalidate();
    inputTab("summary");
    error("");
    updateInput();
    toast("Patient workspace cleared.");
  } catch (e) {
    error(e.message);
  }
});
let localConnectionDrafts = {};
let connectionDrafts = {};
let connectionTask = "summarize";
let connectionProvider = null;
function connectionValues() {
  return {
    task: connectionTask,
    llm_backend: $("#llm-backend").value,
    endpoint: $("#llm-endpoint").value.trim(),
    model: $("#llm-model").value.trim(),
    tokenizer: $("#llm-tokenizer").value.trim(),
    reasoning: $("#llm-reasoning").checked,
    backup_enabled: connectionTask.endsWith("_backup") && $("#llm-backup-enabled").checked,
    failure_threshold: Number($("#llm-failure-threshold").value),
  };
}
// Claude through Databricks serves only the SQL agent task.
function claudeProfile(backend) {
  return (state.config.claude_profiles || {})[backend];
}
function renderProviderChoices() {
  $$("#llm-backend option").forEach((option) => {
    const claude = claudeProfile(option.value);
    option.disabled = Boolean(claude) && claude.task !== connectionTask;
  });
}
function renderProvider() {
  $("#backup-settings").hidden = !connectionTask.endsWith("_backup");
  const backend = $("#llm-backend").value;
  const claude = claudeProfile(backend);
  const hosted = Boolean(state.config.vertex_profiles[backend]) || Boolean(claude);
  const gemini = backend === "gemini4_vertex" || backend === "gemini_vertex";
  $("#llm-endpoint").readOnly = hosted;
  // Claude: the server fixes the gateway URL; enter the gateway's endpoint name.
  $("#llm-model").readOnly = hosted && !claude;
  $("#llm-model").placeholder = claude
    ? "ai_gateway_model_services_prod.<team>_<project>.<team>_<project>-claude-sonnet-5-5"
    : "Served model name";
  $("#tokenizer-settings").hidden = hosted;
  $("#endpoint-help").hidden = hosted;
  $("#set-all-tasks").disabled = Boolean(claude);
  $("#provider-help").textContent = claude
    ? `Uses Claude through the Databricks Unity AI Gateway with the dashboard server’s Databricks login (profile ${claude.databricks_profile}). Only the SQL agent can use it: its queries and their results (extracted values, without patient ids, note text or quotes) go to Databricks; credentials stay on the dashboard server. Enter the full endpoint name from your gateway onboarding email.`
    : hosted
    ? `Uses Google Cloud project ${state.config.vertex_profiles[backend].project} through Vertex / Enterprise Agent Platform. Raw notes are sent to Google Cloud; credentials stay on the dashboard server.`
    : `Uses your selected OpenAI-compatible endpoint for ${state.config.tasks[connectionTask].label.toLowerCase()}.`;
  $("#llm-reasoning").disabled = gemini || Boolean(claude);
  $("#thinking-help").textContent = claude
    ? "Claude always uses adaptive thinking; the SQL agent asks for high effort."
    : gemini
    ? "Gemini always uses reasoning; its API does not support switching it off."
    : "Off by default. Applied through the model’s API thinking switch where supported.";
  $("#check-endpoint").textContent = hosted ? "Check connection" : "Discover model";
  $("#connection-check-help").textContent = hosted
    ? "Sends a short connection test with no patient data."
    : "Reads the endpoint’s model list and fills in the model below.";
  $("#connection-details").innerHTML = fields({
    "Local model device": state.config.device,
    Task: state.config.tasks[connectionTask].label,
    Thinking: claude ? "Adaptive · effort high" : gemini ? "Always enabled · high" : $("#llm-reasoning").checked ? "On where supported" : "Off where supported",
    Sampling: claude ? "Managed by Claude" : gemini ? "Managed by Gemini" : /qwen[-_ ]?3[._]5/i.test($("#llm-model").value)
      ? `Qwen 3.5 general-task defaults · temperature ${$("#llm-reasoning").checked ? "1.0" : "0.7"}, top-p ${$("#llm-reasoning").checked ? "0.95" : "0.8"}, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1.0`
      : "Model vendor defaults from MatchMiner-AI",
  });
}
function renderConnection() {
  const values = connectionDrafts[connectionTask] || state.config.tasks[connectionTask];
  $("#llm-task").value = connectionTask;
  $("#llm-backend").value = values.llm_backend;
  connectionProvider = values.llm_backend;
  $("#llm-endpoint").value = values.endpoint;
  $("#llm-model").value = values.model;
  $("#llm-tokenizer").value = values.tokenizer !== values.model ? values.tokenizer : "";
  $("#llm-reasoning").checked = values.reasoning;
  $("#llm-backup-enabled").checked = Boolean(values.backup_enabled);
  $("#llm-failure-threshold").value = values.failure_threshold || 3;
  renderProviderChoices();
  renderProvider();
}
$("#llm-task").addEventListener("change", () => {
  connectionDrafts[connectionTask] = connectionValues();
  if (connectionProvider === "local") localConnectionDrafts[connectionTask] = connectionValues();
  connectionTask = $("#llm-task").value;
  $("#endpoint-models").replaceChildren();
  $("#endpoint-status").textContent = "Save settings to apply changes to this task.";
  renderConnection();
});
$("#llm-reasoning").addEventListener("change", renderProvider);
$("#llm-model").addEventListener("input", renderProvider);
$("#llm-backend").addEventListener("change", () => {
  const backend = $("#llm-backend").value;
  if (connectionProvider === "local") {
    localConnectionDrafts[connectionTask] = connectionValues();
  }
  const values = state.config.vertex_profiles[backend] || claudeProfile(backend)
    || localConnectionDrafts[connectionTask] || state.config.local_defaults;
  connectionProvider = backend;
  $("#llm-endpoint").value = values.endpoint;
  $("#llm-model").value = values.model;
  $("#llm-tokenizer").value = values.tokenizer;
  $("#endpoint-models").replaceChildren();
  $("#endpoint-status").textContent = "Save settings to use this provider for new jobs.";
  renderProvider();
});
async function openConnection() {
  connectionDrafts = {};
  localConnectionDrafts = {};
  $("#connection-dialog").showModal();
  $("#endpoint-status").textContent = "";
  $("#endpoint-models").replaceChildren();
  connectionPending(true);
  try {
    state.config = await api("/api/config");
    renderConnection();
  } catch (e) {
    $("#endpoint-status").textContent = e.message;
  } finally {
    connectionPending(false);
  }
}
function connectionPending(pending) {
  $$("#connection-form input, #connection-form select, #connection-form button").forEach((el) => {
    el.disabled = pending;
  });
  if (!pending && state.config) renderProvider();
}
$("#connection-nav").addEventListener("click", openConnection);
$("#llm-endpoint").addEventListener("input", () => {
  $("#endpoint-models").replaceChildren();
  $("#endpoint-status").textContent = "Endpoint changed. Discover its model, then save settings.";
});
$("#check-endpoint").addEventListener("click", async () => {
  if (!$("#llm-endpoint").reportValidity()) return;
  const endpoint = $("#llm-endpoint").value.trim();
  const model = $("#llm-model").value.trim();
  const llm_backend = $("#llm-backend").value;
  const profile = state.config.vertex_profiles[llm_backend];
  const claude = claudeProfile(llm_backend);
  connectionPending(true);
  $("#endpoint-status").textContent = claude
    ? "Sending a short test to Claude through the Databricks gateway…"
    : profile
    ? `Checking ${profile.label} access in project ${profile.project}…` : "Reading the endpoint’s model list…";
  try {
    const result = await post("/api/endpoint-check", {endpoint, model, llm_backend, task:connectionTask});
    if (!result.models.length) throw new Error(result.error);
    $("#llm-endpoint").value = result.endpoint;
    $("#endpoint-models").innerHTML = result.models
      .map((name) => `<option value="${escape(name)}"></option>`).join("");
    if (result.model !== model) $("#llm-tokenizer").value = "";
    $("#llm-model").value = result.model;
    $("#endpoint-status").textContent = claude
      ? "Claude answered through the Databricks gateway. Save settings to use it for the SQL agent."
      : profile
      ? `Connected to ${profile.label} in ${profile.project}. Save settings to use it.`
      : result.models.length > 1
      ? `Found ${result.models.length} models. Select a model from the field’s suggestions, then save settings.`
      : "Model discovered and filled in. Save settings to use it.";
  } catch (e) {
    $("#endpoint-status").textContent = e.message;
  } finally {
    connectionPending(false);
  }
});
async function saveConnection(allTasks = false) {
  if (!$("#connection-form").reportValidity()) return;
  const values = connectionValues();
  connectionPending(true);
  try {
    state.config = await post("/api/config", {...values, apply_to_all: allTasks});
    if (allTasks) {
      connectionDrafts = {};
      localConnectionDrafts = {};
    } else {
      delete connectionDrafts[connectionTask];
    }
    const saved = await api("/api/note-compression");
    restoreSavedCompression(saved.note_compression);
    renderConnection();
    $("#endpoint-status").textContent = allTasks
      ? "Saved this model connection for all tasks. Further task-specific saves apply only to the selected task."
      : `Saved for ${state.config.tasks[connectionTask].label.toLowerCase()}. Other tasks keep their settings.`;
    updateInput();
    toast(allTasks ? "All tasks now use this model connection." : "Workspace LLM settings saved.");
  } catch (e) {
    $("#endpoint-status").textContent = e.message;
  } finally {
    connectionPending(false);
  }
}
$("#connection-form").addEventListener("submit", (event) => {
  event.preventDefault();
  saveConnection();
});
$("#set-all-tasks").addEventListener("click", () => saveConnection(true));
document.addEventListener("population-busy", (event) => setBusy(event.detail, true));

async function init() {
  try {
    state.config = await api("/api/config");
    $$("[data-synthetic-case]").forEach((select) => {
      select.replaceChildren(
        ...state.config.synthetic_note_sources.map(({id, label}) => new Option(label, id)),
      );
    });
    selectSyntheticCase(state.config.synthetic_note_sources[0]?.id || "");
    renderConnection();
    await loadCatalogs();
    const saved = await api("/api/session");
    if (saved.job && (!$("#summary").value && !$("#notes").value)) {
      if (saved.job.restore) {
        if (saved.job.kind === "workup") restoreWorkup(saved.job.restore);
        else if (saved.job.kind === "compress") restoreCompression(saved.job.restore);
        else restoreNoteInput(saved.job.restore);
      }
      if (saved.job.status === "failed") {
        error(saved.job.progress);
      } else if (saved.job.status === "cancelled") {
        toast("The previous job was stopped. You can start another job.");
      } else if (saved.job.status === "complete") {
        if (saved.job.kind === "retrieve") {
          state.result = saved.job.result;
          state.retrievalJob = saved.job.id;
          $("#summary").value = saved.job.result.summary;
          renderResults();
        } else if (saved.job.kind === "workup") {
          restoreWorkup(saved.job.result);
        } else if (saved.job.kind === "compress") {
          restoreCompression(saved.job.result);
        } else {
          restoreSummary(saved.job.result);
        }
        if (saved.job.kind !== "compress") {
          $("#summary-label").textContent = saved.job.kind === "summarize" && saved.job.result.compression
            ? "Restored summary from compressed notes" : "Restored from this browser session";
        }
      } else {
        runJob(saved.job.kind, null, saved.job.id);
      }
    }
    restoreSavedCompression(saved.note_compression, true);
    if (saved.job?.kind === "compress" && !$("#summary").value) inputTab("notes");
    state.ready = true;
  } catch (e) {
    error("Could not load the workspace. " + e.message);
  }
  updateInput();
}
init();
