// Extraction-based workup checks: labels and rendering shared by the patient and population views.
export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const e = escapeHTML;

export const extractionStatusLabels = {
  met: 'Met · documented in extraction', not_met: 'Not documented in extraction',
  unknown: 'Unknown', not_applicable: 'Not applicable',
  done_not_recommended: 'Done though not recommended',
};
export const extractionReasonLabels = {
  documented: 'Documented by extracted items', not_documented: 'No extracted item matched',
  planned_only: 'Only planned or ordered', uncertain_dates: 'Dates too imprecise to tell',
  fields_not_stated: 'Needed details not stated', anchor_unavailable: 'Reference event undated or missing',
  incomplete_extraction: 'Some notes lack extraction records', no_extraction: 'No extraction records',
  not_expressible: 'Not checkable from extracted items', query_unavailable: 'No valid query compiled',
  documented_not_applicable: 'Documented facts make it not apply',
  documented_not_recommended: 'Documented although the guideline advises against it',
};
export const stateLabels = {yes: 'yes', unknown: 'unknown', no: 'no', planned: 'planned'};
export const REVIEW_AID = 'Review aid, not a determination: an LLM compiled each workup item into a query and extracted the items from the notes; code ran the query. Check each cited quote against its note. Not documented does not mean not done.';
export const SQL_REVIEW_AID = 'Review aid, not a determination: an LLM agent wrote each workup item\'s SQL query after exploring extracted items, an LLM extracted the items from the notes, and code ran each query once over the cohort. Check each cited quote against its note. Not documented does not mean not done.';
// The report methods that check extraction records (statuses met / not_met / unknown / not_applicable).
export const EXTRACTION_METHODS = ['extraction_queries', 'extraction_sql'];
export const isExtractionMethod = method => EXTRACTION_METHODS.includes(method);
export const methodNames = {extraction_queries: 'Extraction queries', extraction_sql: 'SQL agent over extraction records'};
export const reviewAid = method => method === 'extraction_sql' ? SQL_REVIEW_AID : REVIEW_AID;

export function isExtractionReview(review) {
  return isExtractionMethod(review?.metadata?.method);
}

// One evidence quote with a link to its note; noteRef(evidence) returns the link's data attributes or ''.
function quote(evidence, noteRef) {
  const ref = noteRef(evidence);
  const where = `${evidence.note_number ? `Note ${evidence.note_number}` : 'Note'} · ${e(evidence.note_date || 'date unavailable')}`;
  const link = ref ? `<button type="button" class="population-link" ${ref}>${where} · show in note</button>` : `<span>${where} · note not loaded</span>`;
  return `<blockquote class="extraction-quote"><span class="extraction-state state-${e(evidence.state)}">${e(stateLabels[evidence.state] || evidence.state)}</span> ${e(evidence.quote)}<cite>${e(evidence.criterion || '')} · ${e(evidence.item || '')} · ${e(evidence.date_text || '')}<br>${link}</cite></blockquote>`;
}

export function criteriaTable(criteria) {
  if (!criteria?.length) return '';
  return `<table class="population-table extraction-criteria"><caption>Query criteria (code-evaluated)</caption><thead><tr><th>Criterion</th><th>Item type</th><th>Holds</th><th>Certain / possible / needed</th><th>Anchor</th></tr></thead><tbody>${criteria.map(c => `<tr><th>${e(c.label)}</th><td>${e(c.item_type.replace(/_/g, ' '))}${c.status !== 'any' ? ` · ${e(c.status)}` : ''}</td><td>${e(stateLabels[c.state] || c.state)}</td><td>${c.definite_count} / ${c.possible_count} / ${c.min_count}</td><td>${c.anchor ? e(`${c.anchor.kind.replace(/_/g, ' ')}${c.anchor.date ? ` ${c.anchor.date}` : ''}${c.anchor.fallback ? ' (reference date)' : ''}`) : 'any time'}</td></tr>`).join('')}</tbody></table>`;
}

// The SQL agent's accepted query for one workup item, with how the agent reached it.
export function sqlQueryDetail(entry) {
  if (!entry) return '';
  const how = entry.expressible === false
    ? `Not expressible with extracted items: ${entry.not_expressible_reason || ''}`
    : `${entry.status === 'compiled' ? 'Accepted' : 'Not accepted'}${entry.accepted_by === 'budget' ? ' at the end of the action budget' : entry.accepted_by === 'agent' ? ' by the agent' : ''} after ${entry.action_count ?? 0} actions (${entry.submit_count ?? 0} submission${entry.submit_count === 1 ? '' : 's'})${entry.cached ? ' · cached' : ''}${entry.model ? ` · ${entry.model}` : ''}`;
  const spread = entry.distribution?.status_reason
    ? `<p class="helper">On the exploration population (${entry.distribution.patients} patients): ${e(Object.entries(entry.distribution.status_reason).map(([k, n]) => `${k.replace('/', ' / ').replace(/_/g, ' ')} ${n}`).join(' · '))}</p>` : '';
  const against = entry.recommends_against ? '<p class="helper">The guideline advises against this step: patients who had it are reported as done though not recommended.</p>' : '';
  return `<p class="helper">${e(how)}</p>${against}${entry.error ? `<p class="helper">${e(entry.error)}</p>` : ''}${spread}${entry.sql ? `<pre class="sql-query">${e(entry.sql)}</pre>` : ''}`;
}

export function extractionAssessment(item, {noteRef = () => '', open = false, query = null, sql = null} = {}) {
  const evidence = item.evidence || [];
  const sqlEntry = sql || (item.sql ? {sql: item.sql, status: 'compiled'} : null);
  return `<details class="population-assessment extraction-assessment" ${open ? 'open' : ''}><summary>${e(item.name || item.recommendation?.name)} · <strong>${e(extractionStatusLabels[item.status] || item.status)}</strong></summary>
    <p><strong>Applicability: ${e({applies: 'Applies', not_applicable: 'Not applicable', uncertain: 'Uncertain'}[item.applicability] || item.applicability)}</strong> · ${e(extractionReasonLabels[item.reason] || item.reason)}</p>
    <p>${e(item.bottom_line)}</p>
    ${item.query_summary ? `<p class="helper">Query: ${e(item.query_summary)}</p>` : ''}
    ${(item.limitations || []).map(l => `<p class="helper">Limitation: ${e(l)}</p>`).join('')}
    ${criteriaTable(item.criteria)}
    ${evidence.length ? `<h4>Extracted-item evidence · ${evidence.length} quote${evidence.length === 1 ? '' : 's'}</h4>${evidence.map(v => quote(v, noteRef)).join('')}` : '<p class="helper">No extracted item quotes for this result.</p>'}
    <details><summary>Guideline recommendation${sqlEntry ? ' & SQL query' : query ? ' & compiled query' : ''}</summary><pre>${e(JSON.stringify(item.recommendation, null, 2))}</pre>${sqlEntry ? sqlQueryDetail(sqlEntry) : query ? `<pre>${e(JSON.stringify(query, null, 2))}</pre>` : ''}</details>
  </details>`;
}

// A note's text with one quote's character span marked (offsets index the note's own text).
export function highlightedNote(text, start, end) {
  const value = String(text ?? '');
  if (!(Number.isInteger(start) && Number.isInteger(end) && 0 <= start && start < end && end <= value.length)) return e(value);
  return `${e(value.slice(0, start))}<mark>${e(value.slice(start, end))}</mark>${e(value.slice(end))}`;
}

export function costLine(cost) {
  if (!cost) return '';
  const extraction = cost.reused_extraction
    ? 'This session already extracted these notes; they will be reused.'
    : `On-demand extraction: ${cost.notes} note${cost.notes === 1 ? '' : 's'} (${cost.characters.toLocaleString()} characters) → ${cost.document_requests} LLM request${cost.document_requests === 1 ? '' : 's'}${cost.patient_requests ? ` + ${cost.patient_requests} patient-level chunk request${cost.patient_requests === 1 ? '' : 's'}` : ''} to ${cost.model}, each reasoning ${cost.typical_reasoning_tokens} with up to ${cost.output_reserve_per_request.toLocaleString()} output tokens.`;
  const agent = cost.sql_agent;
  const queries = agent
    ? (agent.items_to_explore
      ? ` SQL agent: ${agent.items_to_explore} of ${cost.workup_items} workup items to explore (${cost.cached_queries} cached), up to ${agent.max_actions_per_item} requests each, over the extraction records of ${agent.store} (${agent.patients.toLocaleString()} patients${agent.database_built ? '' : '; its database is built first'}). The agent's query results (extracted values, without patient ids, note text or quotes) go to the SQL agent's connection${agent.model ? ` (${agent.model})` : ''}.`
      : ` All ${cost.workup_items} workup SQL queries are cached for ${agent.store}.`)
    : cost.compile_requests
      ? ` Query compilation: ${cost.compile_requests} request${cost.compile_requests === 1 ? '' : 's'} (${cost.cached_queries} of ${cost.workup_items} workup items already cached).`
      : ` All ${cost.workup_items} workup queries are cached.`;
  return extraction + queries + ` ${agent ? 'At most' : 'About'} ${cost.requests} requests before retries.`;
}
