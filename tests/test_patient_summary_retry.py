from types import SimpleNamespace
from dataclasses import replace

from matchminer_ai.llm.backends import LLMGenerationResult
from matchminer_ai.llm.prompt_rendering import Prompt
from matchminer_ai.patients.summary_retry import generate_checked_summaries


def result(answers):
    return LLMGenerationResult(final_outputs=answers, reasoning_outputs=['reasoning']*len(answers),
        raw_outputs=['raw']*len(answers), finish_reasons=['stop']*len(answers), model_metadata={})


def test_retries_only_failed_rows_with_original_prompt_and_archives_reasoning(tmp_path):
    calls=[]
    outputs=iter([result(['Good clinical history.', 'Wait, I should rewrite this.']), result(['Repaired clinical history.'])])
    def generate(**kwargs):
        calls.append(kwargs['prompt_list'])
        return next(outputs)
    backend=SimpleNamespace(generate_llm_outputs=generate)
    prompts=[Prompt(i,f'original notes {i}',100) for i in range(2)]
    final,attempts=generate_checked_summaries(backend,prompts,llm_config={},metadata_cache_dir=None,trace_dir=tmp_path)
    assert final.final_outputs==['Good clinical history.','Repaired clinical history.']
    assert [p.row_idx for p in calls[1]]==[0]
    assert calls[1][0].prompt_text=='original notes 1'
    assert len(attempts[1])==2 and attempts[1][0]['reasoning']=='reasoning'
    assert len(list(tmp_path.glob('*.json')))==3


def test_exhaustion_marks_row_failed_and_preserves_failed_attempts():
    outputs=iter([result(['Good clinical history.', 'Final Polish: malformed']), result(['Final Polish: malformed'])])
    backend=SimpleNamespace(generate_llm_outputs=lambda **kw:next(outputs))
    prompts=[Prompt(i,f'notes {i}',100) for i in range(2)]
    final,attempts=generate_checked_summaries(backend,prompts,llm_config={},metadata_cache_dir=None,max_attempts=2)
    assert final.final_outputs==['Good clinical history.','Final Polish: malformed']
    assert final.finish_reasons==['stop','error']
    assert len(attempts[0])==1 and len(attempts[1])==2
    assert [a['finish_reason'] for a in attempts[1]]==['stop','stop']


def test_remote_raw_absence_and_token_limit_are_handled():
    outputs=iter([replace(result(['Partial history']),raw_outputs=[],finish_reasons=['length']),
                  replace(result(['Complete history']),raw_outputs=[])])
    backend=SimpleNamespace(generate_llm_outputs=lambda **kw:next(outputs))
    final,attempts=generate_checked_summaries(backend,[Prompt(0,'notes',100)],llm_config={},metadata_cache_dir=None)
    assert final.final_outputs==['Complete history'] and final.raw_outputs==[]
    assert attempts[0][0]['retry_codes']==['incomplete_generation']
    assert attempts[0][0]['raw_output'] is None


def test_shared_primary_fields_without_duplicate_histories_do_not_force_retry():
    block='Age: 60\nCancer type: Lung\nHistology: Adenocarcinoma\nBiomarkers: KRAS G12C\n'
    text=block+'Current extent: Left lung\nTreatment history: 2020 lobectomy\n'+block+'Current extent: Right lung\nTreatment history: 2024 radiation'
    backend=SimpleNamespace(generate_llm_outputs=lambda **kw:result([text]))
    final,attempts=generate_checked_summaries(backend,[Prompt(0,'notes',100)],llm_config={},metadata_cache_dir=None)
    assert final.final_outputs==[text] and len(attempts[0])==1
