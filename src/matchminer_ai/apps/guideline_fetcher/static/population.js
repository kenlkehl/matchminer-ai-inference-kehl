import {escapeHTML as e, methodLabels, distributionTable, patientRows, patientDetail, startupError} from './population_view.mjs';
import {highlightedNote} from './extraction_view.mjs';
// Review methods that read extraction records instead of having an LLM read the notes.
const EXTRACTION_REVIEW=['extraction_queries','extraction_sql'];
const $ = s=>document.querySelector(s);
// The server renders index.html for its mode; PHI mode adds the DuckDB and cache controls.
const phi = document.documentElement.dataset.mode==='phi';
const state = {population:null, spaces:[], busy:false, patientBusy:false, job:null, patients:[], item:null, ready:false, initializing:false};
function showView(name) {
  history.replaceState(null, "", `#${name}`);
  document.body.classList.toggle('population-mode',name==='population');
  for (const view of ['patient','population']) {
    $(`#${view}-view`).hidden = name !== view;
    $(`#${view}-tab`).setAttribute('aria-selected', String(name===view));
    $(`#${view}-tab`).tabIndex = name === view ? 0 : -1;
  }
  if (name==='population' && !state.ready) init();
}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>showView(b.dataset.view)));
$('.view-tabs').addEventListener('keydown',event=>{
  if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
  event.preventDefault();
  const name = event.key==='Home'?'population':event.key==='End'?'patient':$('#population-view').hidden?'population':'patient';
  showView(name); $(`#${name}-tab`).focus();
});
$('#pop-connections').addEventListener('click',()=>$('#connection-nav').click());
function error(message='') {
  $('#pop-error-message').textContent=message;
  $('#pop-error').hidden=!message;
  $('#pop-retry').hidden=state.ready||!message;
}
async function api(path, options={}) {
  const r = await fetch(path,{credentials:'same-origin',...options});
  if (r.status===401) location.assign('/login');  // PHI mode only
  if (!r.ok) {
    let body; try {body=await r.json();} catch {}
    const failure=new Error(typeof body?.detail === 'string' ? body.detail : 'The request failed. Check inputs and model settings, then retry.');
    failure.status=r.status; failure.path=path;
    throw failure;
  }
  return r.status===204?null:r.json();
}
const post = (path,body)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function busy(value) {
  state.busy=value;
  $('#pop-progress .spinner').hidden=!value;
  document.dispatchEvent(new CustomEvent('population-busy',{detail:value}));
  controls();
}
document.addEventListener('patient-busy',event=>{state.patientBusy=event.detail; controls();});
function controls() {
  const locked=state.busy||state.patientBusy||!state.ready;
  $('#population-view').querySelectorAll('input,select,button').forEach(el=>{el.disabled=locked;});
  $('#pop-retry').disabled=state.initializing||state.busy||state.patientBusy;
  $('#pop-stop').disabled=!state.job || !state.busy;
  const active=state.population?.active;
  const count=active?.patient_count || 0;
  $('#pop-retrieve').disabled=locked||!state.population||!$('#pop-space').value;
  for (const id of ['pop-compress','pop-export']) $(`#${id}`).disabled=locked||!count;
  const method=$('#pop-method').value, extraction=EXTRACTION_REVIEW.includes(method);
  $('#pop-note-source option[value="compressed"]').disabled=!active?.compression||extraction;
  if (!active?.compression||extraction) $('#pop-note-source').value='original';
  $('#pop-extraction-controls').hidden=!extraction;
  $('#pop-extraction-help').hidden=method!=='extraction_queries';
  $('#pop-sql-help').hidden=method!=='extraction_sql';
  $('#pop-concurrency-controls').hidden=extraction;
  $('#pop-review').disabled=locked||!count||!active?.space?.diagnostic_workup?.length;
  for (const id of ['pop-json','pop-csv']) $(`#${id}`).disabled=locked||!active?.report;
}
function render() {
  const p=state.population, a=p?.active;
  const origin=p?.provenance?.cache?.name || p?.provenance?.database?.name;
  const notes=p?.notes_on_demand?'cached source notes, read for matched patients':'source notes loaded';
  $('#pop-source-status').textContent=p?`${p.source_count.toLocaleString()} patients · ${p.source_note_count.toLocaleString()} ${notes}${origin?` from ${origin}`:''}`:'No population loaded.';
  const metrics=[['Source population',p?.source_count||0],['Active patients',a?.patient_count||0],['Patients reviewed',a?.report?.patient_count||0],['Workup items',a?.space?.diagnostic_workup?.length||0]];
  $('#pop-metrics').innerHTML=metrics.map(([label,value])=>`<div class="panel population-metric"><span>${e(label)}</span><strong>${value.toLocaleString()}</strong></div>`).join('');
  $('#pop-active-title').textContent=a?.space?.name || 'Choose a guideline population';
  $('#pop-active-definition').textContent=a?.space?.cohort_key_criteria || a?.space?.clinical_space_summary || '';
  $('#pop-active-status').textContent=a?`${a.patient_count} active patients / ${a.retrieval.scored_count ?? a.matches.length} match-quality scored / ${a.retrieval.population_count ?? p.source_count} in source population. Score threshold: ${a.retrieval.min_match_score ?? 'none'}. New selection settings apply when you fetch again.`:'Load a population, then select a finalized trial space to find matching patients.';
  $('#pop-matches').hidden=!a?.matches?.length;
  $('#pop-match-table').innerHTML=a?.matches?.length?`<table class="population-table"><thead><tr><th>Patient</th><th>Match quality</th><th>TrialSpace similarity</th><th>Retrieval rank</th></tr></thead><tbody>${a.matches.map(r=>`<tr><th><button class="population-link" data-pop-patient="${e(r.patient_id)}">${e(r.patient_id)}</button></th><td>${Number(r.match_quality_score).toFixed(3)}</td><td>${Number(r.similarity_score).toFixed(3)}</td><td>${r.retrieval_rank}</td></tr>`).join('')}</tbody></table>`:'';
  $('#pop-compression-status').textContent=a?.compression?`${a.compression.note_count} notes compressed · ${a.compression.input_characters.toLocaleString()} → ${a.compression.output_characters.toLocaleString()} characters. Original notes retained; losslessness unverified.`:'Optional. Original notes are retained; compression may lose information.';
  const r=a?.report;
  $('#pop-report-status').textContent=!r?'Check every active patient against every workup guideline.'
    :r.method==='extraction_sql'?`${r.patient_count} patients checked with the SQL agent · records ${r.extraction_store?.name || ''} (${(r.extraction_store?.levels||[]).join(' + ')} level) · ${r.extraction?.documents_used ?? 0} document records used${r.extraction?.documents_ignored ? `, ${r.extraction.documents_ignored} for other notes ignored` : ''}${r.extraction?.notes_without_records ? `, ${r.extraction.notes_without_records} notes without records` : ''} · SQL queries ${r.compile?.compiled ?? 0}/${r.compile?.distinct ?? 0} accepted (${r.compile?.cached ?? 0} cached, ${r.compile?.requests ?? 0} agent requests over ${(r.compile?.exploration?.patients ?? 0).toLocaleString()} store patients) by ${r.compile?.model || r.connection?.model || ''}; each ran once over the cohort. Review aids, not determinations.`
    :r.method==='extraction_queries'?`${r.patient_count} patients checked with extraction queries · records ${r.extraction_store?.name || ''} (${(r.extraction_store?.levels||[]).join(' + ')} level) · ${r.extraction?.documents_used ?? 0} document records used${r.extraction?.documents_ignored ? `, ${r.extraction.documents_ignored} for other notes ignored` : ''}${r.extraction?.notes_without_records ? `, ${r.extraction.notes_without_records} notes without records` : ''} · queries ${r.compile?.compiled ?? 0}/${r.compile?.distinct ?? 0} compiled (${r.compile?.cached ?? 0} cached, ${r.compile?.requests ?? 0} requests) by ${r.compile?.model || r.connection?.model || ''}. Review aids, not determinations.`
    :`${r.patient_count} patients reviewed · full ${r.note_source} notes · ${r.max_concurrent_patients || 1} at once. ${r.connection?.model || ''}`;
  $('#pop-distribution').innerHTML=distributionTable(r?.aggregate, r?.method);
  controls();
}
function applyPopulation(value) {
  state.population=value;
  // A new cohort/run invalidates open patient drilldowns.
  for (const id of ['pop-item-dialog','pop-patient-dialog']) if ($(`#${id}`).open) $(`#${id}`).close();
  state.patients=[]; state.item=null;
  if (value?.active?.report?.note_source) $('#pop-note-source').value=value.active.report.note_source;
  if (value?.active?.report?.method) $('#pop-method').value=value.active.report.method;
  render();
}
async function run(path,body,existing=null,upload=null) {
  error(); busy(true); $('#pop-progress').hidden=false;
  $('#pop-progress-message').textContent='Preparing population workflow…';
  try {
    const submitted=existing?{job_id:existing}:upload?await api(path,{method:'POST',body:upload}):await post(path,body);
    state.job=submitted.job_id; controls();
    while (true) {
      const job=await api(`/api/jobs/${encodeURIComponent(state.job)}`);
      $('#pop-progress-message').textContent=job.progress;
      if (job.status==='complete') {applyPopulation(job.result); break;}
      if (job.status==='failed') throw new Error(job.progress);
      if (job.status==='cancelled') {$('#pop-progress-message').textContent='Stopped. The last completed cohort stage is preserved.'; break;}
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
  } catch(err) {error(err.message);}
  finally {state.job=null;busy(false);}
}
function version() {return {version:state.population?.version};}
function valid(ids) {return ids.every(id=>$(`#${id}`).reportValidity());}
$('#pop-load').addEventListener('click',()=>{
  if (valid(['pop-size','pop-seed'])) run('/api/population/sample',{size:Number($('#pop-size').value),seed:Number($('#pop-seed').value),...(phi?{database:$('#pop-database').value.trim()||null}:{})});
});
const fileSize=n=>n>=1e9?`${(n/1e9).toFixed(1)} GB`:n>=1e6?`${(n/1e6).toFixed(1)} MB`:`${Math.ceil(n/1e3)} kB`;
const fileDate=seconds=>new Date(seconds*1000).toLocaleDateString();
let databaseSequence=0;
async function checkDatabase(){
  const sequence=++databaseSequence, path=$('#pop-database').value.trim();
  $('#pop-database-status').textContent='Checking database tables…';
  try{
    const db=await api(`/api/population/database?path=${encodeURIComponent(path)}`);
    if(sequence===databaseSequence)$('#pop-database-status').textContent=`${db.name} · ${fileSize(db.size)} · modified ${fileDate(db.mtime_ns/1e9)} · patient tables found`;
  }catch(err){if(sequence===databaseSequence)$('#pop-database-status').textContent=err.message;}
}
let cacheSequence=0;
async function checkCache(){
  const sequence=++cacheSequence, path=$('#pop-cache').value.trim();
  $('#pop-cache-status').textContent='Checking cache manifest…';
  try{
    const c=await api(`/api/population/cache?path=${encodeURIComponent(path)}`);
    if(sequence===cacheSequence)$('#pop-cache-status').textContent=`${c.patients.toLocaleString()} patients · ${c.notes.toLocaleString()} notes · built ${new Date(c.created_utc).toLocaleString()} from ${c.database}${c.sample_limit?` (trial sample of ${c.sample_limit.toLocaleString()})`:''} · ${c.embedding_model || 'TrialSpace'}`;
  }catch(err){if(sequence===cacheSequence)$('#pop-cache-status').textContent=err.message;}
}
// The dialog chooses DuckDB files for sampling or cache folders for loading.
let browseMode='database';
const browseCopy={
  database:['Choose a DuckDB database','Folders and .duckdb files on this server, within the configured database folders. Choosing a file checks its tables without reading patient rows.'],
  cache:['Choose a population cache','Folders and preprocessed population caches on this server, within the configured database folders. Choosing a cache reads only its manifest.'],
};
async function browse(path=''){
  let listing;
  try{listing=await api(`/api/population/databases?path=${encodeURIComponent(path)}`);}
  catch(err){
    // A typed path outside the browsable folders falls back to the default folder.
    if(!path)throw err;
    listing=await api('/api/population/databases');
  }
  $('#pop-database-folder').textContent=listing.path;
  const choices=browseMode==='cache'
    ?listing.caches.map(c=>`<tr><th><button class="population-link" data-pop-pick="${e(c.path)}">${e(c.name)}</button></th><td>Population cache</td><td></td></tr>`)
    :listing.files.map(f=>`<tr><th><button class="population-link" data-pop-pick="${e(f.path)}">${e(f.name)}</button></th><td>${fileSize(f.size)}</td><td>${fileDate(f.modified)}</td></tr>`);
  const rows=[
    ...(listing.parent?[`<tr><th><button class="population-link" data-pop-dir="${e(listing.parent)}">↑ Parent folder</button></th><td></td><td></td></tr>`]:[]),
    ...choices,
    ...listing.directories.map(name=>`<tr><th><button class="population-link" data-pop-dir="${e(`${listing.path.replace(/\/$/,'')}/${name}`)}">${e(name)}/</button></th><td>Folder</td><td></td></tr>`),
  ];
  const kind=browseMode==='cache'?`${listing.caches.length} population caches`:`${listing.files.length} DuckDB files`;
  $('#pop-database-entries').innerHTML=`<table class="population-table"><caption>${kind} · ${listing.directories.length} folders${listing.truncated?' · listing truncated':''}</caption><thead><tr><th>Name</th><th>Size</th><th>Modified</th></tr></thead><tbody>${rows.join('')}</tbody></table>`;
}
async function openBrowser(mode){
  browseMode=mode; error();
  [$('#pop-database-title').textContent,$('#pop-database-help').textContent]=browseCopy[mode];
  const input=mode==='cache'?'#pop-cache':'#pop-database';
  // Start beside the current cache rather than inside it.
  const start=mode==='cache'?$(input).value.trim().replace(/\/[^/]*\/?$/,''):$(input).value.trim();
  try{await browse(start);$('#pop-database-dialog').showModal();}
  catch(err){error(err.message);}
}
if(phi){
  $('#pop-database').addEventListener('change',checkDatabase);
  $('#pop-cache').addEventListener('change',checkCache);
  $('#pop-cache-load').addEventListener('click',()=>run('/api/population/cache',{path:$('#pop-cache').value.trim()||null}));
  $('#pop-browse').addEventListener('click',()=>openBrowser('database'));
  $('#pop-cache-browse').addEventListener('click',()=>openBrowser('cache'));
}
$('#pop-tables').addEventListener('click',()=>{
  const summaries=$('#pop-summary-file').files[0], notes=$('#pop-notes-file').files[0];
  if (!summaries||!notes) return error('Choose both a summary table and a notes table.');
  const form=new FormData();form.append('summaries',summaries);form.append('notes',notes);
  run('/api/population/tables',null,null,form);
});
$('#pop-import').addEventListener('click',async()=>{
  const file=$('#pop-archive').files[0]; if (!file) return error('Choose an exported cohort ZIP.');
  busy(true);error();
  try {const form=new FormData();form.append('file',file);applyPopulation((await api('/api/population/import',{method:'POST',body:form})).population);}
  catch(err){error(err.message);}finally{busy(false);}
});
$('#pop-retrieve').addEventListener('click',()=>{
  if(valid(['pop-candidates','pop-threshold']))run('/api/population/retrieve',{...version(),catalog_id:$('#pop-catalog').value,space_trial_id:$('#pop-space').value,candidate_k:Number($('#pop-candidates').value),min_match_score:Number($('#pop-threshold').value)});
});
$('#pop-compress').addEventListener('click',()=>run('/api/population/compress',version()));
$('#pop-review').addEventListener('click',()=>{
  const method=$('#pop-method').value;
  if(EXTRACTION_REVIEW.includes(method)){
    run('/api/population/review',{...version(),method,extraction_records:$('#pop-extractions').value.trim()||null});
  } else if(valid(['pop-concurrency']))run('/api/population/review',{...version(),note_source:$('#pop-note-source').value,max_concurrent_patients:Number($('#pop-concurrency').value)});
});
let extractionSequence=0;
async function checkExtractions(){
  const sequence=++extractionSequence, path=$('#pop-extractions').value.trim();
  $('#pop-extractions-status').textContent='Checking extraction records…';
  try{
    const x=await api(`/api/population/extractions?path=${encodeURIComponent(path)}`);
    const docs=x.counts.documents, pts=x.counts.patient_records;
    if(sequence===extractionSequence)$('#pop-extractions-status').textContent=`${x.counts.patients.toLocaleString()} patients · ${x.levels.join(' + ')} records${docs?` · ${(docs.complete+docs.partial).toLocaleString()} notes extracted, ${docs.failed.toLocaleString()} failed`:''}${pts?` · ${pts.complete+pts.partial} patient records`:''} · built ${new Date(x.created_utc).toLocaleString()} by ${x.model || 'an unrecorded model'} from ${x.source.name}`;
  }catch(err){if(sequence===extractionSequence)$('#pop-extractions-status').textContent=err.message;}
}
$('#pop-extractions').addEventListener('change',checkExtractions);
$('#pop-method').addEventListener('change',()=>{controls();if(EXTRACTION_REVIEW.includes($('#pop-method').value))checkExtractions();});
$('#pop-stop').addEventListener('click',async()=>{
  if(!state.job)return;
  $('#pop-stop').disabled=true;
  try{await post(`/api/jobs/${encodeURIComponent(state.job)}/stop`,{});}catch(err){error(err.message);}
});
$('#pop-note-source').addEventListener('change',controls);
let catalogSequence=0;
async function catalogs(refresh=false){
  const selected=$('#pop-catalog').value;
  const result=await api(`/api/catalogs?refresh=${refresh}`);
  $('#pop-catalog').replaceChildren(new Option('Choose a catalog',''),...result.catalogs.map(c=>new Option(`${c.title} · ${c.version} · ${c.spaces} spaces`,c.id)));
  if(result.catalogs.some(c=>c.id===selected))$('#pop-catalog').value=selected;
}
function filterSpaces(){
  const selected=$('#pop-space').value, query=$('#pop-filter').value.toLocaleLowerCase();
  const visible=state.spaces.filter(s=>`${s.name || ''} ${s.space_trial_id} ${s.cohort_key_criteria || s.clinical_space_summary}`.toLocaleLowerCase().includes(query));
  $('#pop-space').replaceChildren(new Option(`${visible.length} spaces · choose one`,''),...visible.map(s=>new Option(s.name || s.clinical_space_summary.slice(0,180),s.space_trial_id)));
  if(visible.some(s=>s.space_trial_id===selected))$('#pop-space').value=selected;
  definition();
}
function definition(){
  const space=state.spaces.find(s=>s.space_trial_id===$('#pop-space').value);
  $('#pop-definition').textContent=space?.cohort_key_criteria||space?.clinical_space_summary||'No space selected.';controls();
}
$('#pop-catalog').addEventListener('change',async()=>{
  const sequence=++catalogSequence;state.spaces=[];filterSpaces();
  if(!$('#pop-catalog').value)return;
  try{const result=await api(`/api/population/spaces?catalog_id=${encodeURIComponent($('#pop-catalog').value)}`);if(sequence===catalogSequence){state.spaces=result.spaces;filterSpaces();}}
  catch(err){error(err.message);}
});
$('#pop-filter').addEventListener('input',filterSpaces);$('#pop-space').addEventListener('change',definition);
$('#pop-refresh').addEventListener('click',async()=>{try{await catalogs(true);$('#pop-catalog').dispatchEvent(new Event('change'));}catch(err){error(err.message);}});
async function download(path,name){
  busy(true);error();
  try{
    const r=await fetch(path,{credentials:'same-origin'});
    if(!r.ok){const body=await r.json();throw new Error(body.detail||'Export failed.');}
    const url=URL.createObjectURL(await r.blob()), link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(err){error(err.message);}finally{busy(false);}
}
$('#pop-export').addEventListener('click',()=>{
  // Let the browser stream large archives directly to its download destination.
  const link=document.createElement('a');
  link.href=`/api/population/export?version=${encodeURIComponent(state.population.version)}`;
  link.download='guideline-cohort.zip'; link.click();
  $('#pop-active-status').textContent='Preparing cohort archive on the server. The browser download begins when it is ready.';
});
for(const format of ['json','csv'])$(`#pop-${format}`).addEventListener('click',()=>download(`/api/population/results?format=${format}&version=${encodeURIComponent(state.population.version)}`,`population-results.${format}`));
$('#pop-clear').addEventListener('click',async()=>{try{await api('/api/population',{method:'DELETE'});applyPopulation(null);$('#pop-progress').hidden=true;error();}catch(err){error(err.message);}});
async function openPatient(pid){
  const selectedVersion=state.population.version;
  try{
    const data=await api(`/api/population/patient?version=${encodeURIComponent(selectedVersion)}&patient_id=${encodeURIComponent(pid)}`);
    if(state.population?.version!==selectedVersion)return;
    $('#pop-patient-title').textContent=`Patient ${pid}`;
    $('#pop-patient-detail').innerHTML=patientDetail(data,state.item,state.population.active?.report?.queries);
    state.patientNotes=data.notes;
    $('#pop-patient-dialog').showModal();
  }catch(err){error(err.message);}
}
document.addEventListener('click',async event=>{
  const close=event.target.closest('[data-pop-close]');if(close)$(`#${close.dataset.popClose}`).close();
  const folder=event.target.closest('[data-pop-dir]');
  if(folder){try{await browse(folder.dataset.popDir);}catch(err){$('#pop-database-folder').textContent=err.message;}return;}
  const pick=event.target.closest('[data-pop-pick]');
  if(pick){
    $(browseMode==='cache'?'#pop-cache':'#pop-database').value=pick.dataset.popPick;
    $('#pop-database-dialog').close();
    (browseMode==='cache'?checkCache:checkDatabase)();
    return;
  }
  const patient=event.target.closest('[data-pop-patient]');if(patient)openPatient(patient.dataset.popPatient);
  const note=event.target.closest('[data-pop-note]');
  if(note){
    // Open the cited note and mark the quoted span (offsets index the original note text).
    const index=Number(note.dataset.popNote), target=$(`#pop-note-${index}`);
    if(target&&state.patientNotes?.[index]){
      target.open=true;
      target.querySelector('[data-note-text]').innerHTML=highlightedNote(state.patientNotes[index].note_text,Number(note.dataset.start),Number(note.dataset.end));
      target.scrollIntoView({block:'start'});target.querySelector('mark')?.scrollIntoView({block:'center'});
    }
    return;
  }
  const item=event.target.closest('[data-pop-item]');if(!item)return;
  const number=Number(item.dataset.popItem), selectedVersion=state.population.version;
  try{
    const data=await api(`/api/population/items/${number}?version=${encodeURIComponent(selectedVersion)}`);
    if(state.population?.version!==selectedVersion)return;
    state.patients=data.patients;state.item=number;
    const report=state.population.active.report, recommendation=report.aggregate[number].recommendation;
    $('#pop-item-title').textContent=recommendation.name;
    $('#pop-item-conditions').textContent=recommendation.conditions||'';
    statusFilter(report.method);
    $('#pop-status-filter').value='';$('#pop-item-patients').innerHTML=patientRows(state.patients,'',report.method);
    $('#pop-item-dialog').showModal();
  }catch(err){error(err.message);}
});
function statusFilter(method){
  $('#pop-status-filter').replaceChildren(new Option('All patients',''),...Object.entries(methodLabels(method)).map(([key,label])=>new Option(label,key)));
}
statusFilter('full_notes');
$('#pop-status-filter').addEventListener('change',()=>{$('#pop-item-patients').innerHTML=patientRows(state.patients,$('#pop-status-filter').value,state.population?.active?.report?.method);});
async function init(){
  if(state.initializing)return;state.initializing=true;
  error();controls();
  try{
    await catalogs();const saved=await api('/api/population');
    $('#pop-size').value=saved.defaults.size;$('#pop-seed').value=saved.defaults.seed;
    if(!$('#pop-extractions').value)$('#pop-extractions').value=saved.defaults.extraction_records||'';
    if(phi){
      if(!$('#pop-database').value)$('#pop-database').value=saved.defaults.database;
      if(!$('#pop-cache').value)$('#pop-cache').value=saved.defaults.cache;
      checkDatabase();checkCache();
    }
    state.ready=true;applyPopulation(saved.population);
    if(saved.job && ['queued','running','cancelling'].includes(saved.job.status))run(null,null,saved.job.id);
    else if(saved.job?.status==='failed')error(saved.job.progress);
  }catch(err){error(startupError(err));}finally{state.initializing=false;controls();}
}
$('#pop-retry').addEventListener('click',init);
render();

showView(location.hash === '#patient' ? 'patient' : 'population');
window.addEventListener('hashchange',()=>{
  if (['#patient','#population'].includes(location.hash)) showView(location.hash.slice(1));
});
