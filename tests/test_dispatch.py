from matchminer_ai.llm.dispatch import least_loaded_endpoint


def test_sparse_work_prefers_unused_capacity_over_equal_endpoint_shares():
    assert least_loaded_endpoint([('cloud', 70, 96, 0), ('gb300', 10, 320, 1)]) == 'gb300'
    assert least_loaded_endpoint([('small', 0, 2, 2), ('large', 0, 8, 1)]) == 'large'
    assert least_loaded_endpoint([('full', 2, 2, 0), ('disabled', 0, 0, 1)]) is None


def test_cap_reduction_does_not_admit_more_work_until_existing_calls_drain():
    assert least_loaded_endpoint([('draining', 8, 4, 0), ('healthy', 2, 4, 1)]) == 'healthy'
