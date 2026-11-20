import {statusLabels, applicabilityLabels, workupStatusLabel} from './report.mjs';
import {escapeHTML, extractionAssessment, extractionStatusLabels, isExtractionMethod, isExtractionReview, methodNames, reviewAid} from './extraction_view.mjs';
export {escapeHTML};
export const labels = {...statusLabels, not_documented: 'Not documented in reviewed notes'};
// Extraction methods (compiled queries, SQL agent) report their own statuses; full-note reviews keep theirs.
export const methodLabels = method => isExtractionMethod(method) ? extractionStatusLabels : labels;
const e = escapeHTML;
export function startupError(error) {
  if (error.status===404 && error.path==='/api/population') {
    return 'Population management is unavailable on this running server. Restart the guideline-fetcher app to load the updated backend, then retry the connection.';
  }
  return `Could not connect to population management. ${error.message || 'Check the server connection and retry.'}`;
}
export function distributionTable(rows, method='full_notes') {
  if (!rows?.length) return '';
  const names = methodLabels(method), statuses = Object.keys(names);
  const extraction = isExtractionMethod(method);
  return `<table class="population-table"><caption>${extraction ? `${method === 'extraction_sql' ? 'SQL-agent' : 'Extraction-query'} results across the active cohort (met / met or not documented, among patients it applies to)` : 'Documentation statuses across the active cohort'}</caption><thead><tr><th scope="col">Workup guideline</th><th scope="col">${extraction ? 'Met / applicable assessed' : 'Completed / applicable assessed'}</th>${statuses.map(s=>`<th scope="col">${e(names[s])}</th>`).join('')}<th scope="col">Applies</th><th scope="col">Not applicable</th><th scope="col">Uncertain</th></tr></thead><tbody>${rows.map(r=>`<tr><th scope="row"><button class="population-link" data-pop-item="${r.recommendation_index}">${e(r.recommendation.name)}</button></th><td>${r.recommends_against ? `Advised against · ${r.statuses.done_not_recommended ?? 0} done anyway` : r.applicable_assessed ? `<meter min="0" max="${r.applicable_assessed}" value="${r.completed_applicable}" aria-label="Documented completion among applicable assessed patients"></meter> ${r.completed_applicable} / ${r.applicable_assessed}` : '— · no applicable assessments'}</td>${statuses.map(s=>`<td class="pop-status-${s}">${r.statuses[s]}</td>`).join('')}${['applies','not_applicable','uncertain'].map(s=>`<td>${r.applicability[s]}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
export function patientRows(patients, filter='', method='full_notes') {
  const names = methodLabels(method);
  const visible = patients.filter(p=>!filter || p.status === filter);
  return `<table class="population-table"><caption>${visible.length} patients shown</caption><thead><tr><th>Patient</th><th>Documentation</th><th>Applicability</th><th>Finding</th></tr></thead><tbody>${visible.map(p=>`<tr><th><button class="population-link" data-pop-patient="${e(p.patient_id)}">${e(p.patient_id)}</button></th><td>${e(names[p.status] || p.status)}</td><td>${e(applicabilityLabels[p.applicability])}</td><td>${e(p.bottom_line)}</td></tr>`).join('')}</tbody></table>`;
}
// Evidence names its note by extraction note id; link it to that note in this dialog.
function extractionDetail(data, review, selectedItem, queries) {
  const index = new Map(data.notes.map((n, i) => [n.note_id, i]));
  const noteRef = v => index.has(v.note_id) ? `data-pop-note="${index.get(v.note_id)}" data-start="${Number(v.start)}" data-end="${Number(v.end)}"` : '';
  const numbered = a => ({...a, evidence: (a.evidence || []).map(v => ({...v, note_number: index.has(v.note_id) ? index.get(v.note_id) + 1 : null}))});
  const coverage = review.metadata?.coverage || {};
  const method = review.metadata?.method;
  const options = a => method === 'extraction_sql'
    ? {sql: queries?.[a.recommendation_index] || (a.sql ? {sql: a.sql, status: 'compiled'} : null)}
    : {query: queries?.[a.recommendation_index]?.query};
  return `<p class="review-note">${e(reviewAid(method))}</p><p class="helper">${e(review.notice)}</p><p class="helper">${e(methodNames[method] || 'Extraction queries')} · ${coverage.documents ?? 0} document records${coverage.patient_records ? ` · ${coverage.patient_records} patient-level record` : ''}${coverage.notes_without_records ? ` · ${coverage.notes_without_records} notes without records` : ''}${coverage.failed_documents ? ` · ${coverage.failed_documents} failed` : ''} · ${coverage.items ?? 0} extracted items</p>${review.assessments.map(a => extractionAssessment(numbered(a), {noteRef, open: a.recommendation_index === selectedItem, ...options(a)})).join('')}`;
}
function notesList(data, extraction) {
  return `<h3>Source notes</h3>${data.notes.map((n,i)=>`<details class="population-assessment" id="pop-note-${i}"><summary>Note ${i+1} · ${e(n.note_date || 'date unavailable')} · ${e(n.note_type || '')}${extraction && n.note_id ? ` · ${e(n.note_id)}` : ''}</summary><h4>Original note</h4><pre data-note-text>${e(n.note_text)}</pre>${n.compressed_note_text?`<h4>Saved compressed note · losslessness unverified</h4><pre>${e(n.compressed_note_text)}</pre>`:''}</details>`).join('')}`;
}
export function patientDetail(data, selectedItem=null, queries=null) {
  const review = data.review;
  if (isExtractionReview(review)) {
    return `<h3>Cancer history summary</h3><p class="population-text">${e(data.patient.cancer_history_summary)}</p>${extractionDetail(data, review, selectedItem, queries)}${notesList(data, true)}`;
  }
  return `<h3>Cancer history summary</h3><p class="population-text">${e(data.patient.cancer_history_summary)}</p>
    ${review ? `<p class="review-note">${e(review.notice)}</p><p class="helper">${e(review.metadata?.note_source || 'original')} notes · Full-note review · ${e(review.metadata?.model || 'See individual findings for model provenance')}</p>${review.assessments.map(a=>`<details class="population-assessment" ${a.recommendation_index===selectedItem?'open':''}><summary>${e(a.name || a.recommendation.name)} · ${e(workupStatusLabel(a,review))}</summary><p><strong>${e(applicabilityLabels[a.applicability])}</strong> · ${e(a.bottom_line)}</p><p class="helper">${e(a.recommendation.conditions)}</p>${(a.limitations||[]).map(l=>`<p class="helper">${e(l)}</p>`).join('')}<h4>${review.metadata?.compression?'Reviewed compressed-note excerpts':'Reviewed patient-note excerpts'}</h4><p class="helper">Automatically retained review context, including potentially irrelevant passages.</p>${(a.evidence||[]).map(v=>`<blockquote><small>Note ${e(v.note_number)} · ${e(v.note_date || 'date unavailable')} · ${e(v.note_type || '')}</small><pre>${e(v.quote)}</pre></blockquote>`).join('')}<details><summary>Guideline recommendation & source provenance</summary><pre>${e(JSON.stringify(a.recommendation,null,2))}</pre></details></details>`).join('')}` : '<p>No workup review available for this patient yet.</p>'}
    <h3>Source notes</h3>${data.notes.map((n,i)=>`<details class="population-assessment"><summary>Note ${i+1} · ${e(n.note_date || 'date unavailable')} · ${e(n.note_type || '')}</summary><h4>Original note</h4><pre>${e(n.note_text)}</pre>${n.compressed_note_text?`<h4>Saved compressed note · losslessness unverified</h4><pre>${e(n.compressed_note_text)}</pre>`:''}</details>`).join('')}`;
}
