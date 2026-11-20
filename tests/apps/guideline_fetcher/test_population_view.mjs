import test from 'node:test';
import assert from 'node:assert/strict';
import {distributionTable, patientRows, patientDetail, labels, startupError} from '../../../src/matchminer_ai/apps/guideline_fetcher/static/population_view.mjs';
test('missing population API explains the backend restart and recovery',()=>{
  const message=startupError({status:404,path:'/api/population',message:'Not Found'});
  assert.match(message,/Restart the guideline-fetcher app/);
  assert.match(message,/retry the connection/);
  assert.doesNotMatch(message,/Not Found/);
});
test('other startup errors retain their cause without misdiagnosing an old backend',()=>{
  for (const failure of [
    {status:503,path:'/api/population',message:'Service unavailable'},
    {status:404,path:'/api/catalogs?refresh=false',message:'Catalog service unavailable'},
    new Error('Failed to fetch'),
  ]) {
    const message=startupError(failure);
    assert.ok(message.includes(failure.message));
    assert.doesNotMatch(message,/Restart the guideline-fetcher app/);
  }
});
test('distribution retains statuses and workup drilldown',()=>{
  const row={recommendation_index:0,recommendation:{name:'A <script>test</script>'},applicable_assessed:2,completed_applicable:1,statuses:Object.fromEntries(Object.keys(labels).map(s=>[s,1])),applicability:{applies:2,not_applicable:2,uncertain:3}};
  const html=distributionTable([row]);
  assert.match(html,/data-pop-item="0"/);assert.match(html,/1 \/ 2/);assert.doesNotMatch(html,/<script>/);
  assert.match(html,/Review failed/);assert.match(html,/Not documented in reviewed notes/);
});
test('filtering and untrusted patient identifiers stay escaped',()=>{
  const html=patientRows([{patient_id:'"><img src=x>',status:'error',applicability:'uncertain',bottom_line:'<b>x</b>'},{patient_id:'b',status:'completed',applicability:'applies',bottom_line:'done'}],'error');
  assert.match(html,/1 patients shown/);assert.doesNotMatch(html,/<img/);assert.doesNotMatch(html,/>b<\/button>/);
});
test('patient drilldown includes original and compressed notes',()=>{
  const html=patientDetail({patient:{cancer_history_summary:'Fabricated'},notes:[{note_text:'Original <script>',compressed_note_text:'Compact',note_date:null}],review:null});
  assert.match(html,/Original &lt;script&gt;/);assert.match(html,/Compact/);assert.match(html,/date unavailable/);
});
