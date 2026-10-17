import asyncio
import json
from pathlib import Path

import pytest

from matchminer_ai.trials.hierarchy import (
    FIELDS,
    Completion,
    ReadyHierarchy,
    load_prompts,
    run_ready_hierarchies,
)

PROMPTS = load_prompts(Path(__file__).parents[1] / "src/matchminer_ai/prompts")


def answer(request):
    if request.field_name == "boilerplate":
        return Completion('{"exclusions": []}')
    values = ["fast", "slow"] if request.field_name == "age" else ["value"]
    return Completion(
        json.dumps(
            {
                "branches": [
                    {
                        "value": value,
                        "scope": value
                        if request.field_name == "age"
                        else request.scope,
                    }
                    for value in values
                ],
                "prune_reason": "",
            }
        )
    )


def test_fast_branch_crosses_field_barrier_and_new_trial_replaces_finished_trial():
    async def scenario():
        advanced, third_started = asyncio.Event(), asyncio.Event()
        events, completed = [], []
        in_flight = peak = 0
        states = [ReadyHierarchy(i, f"synthetic {i}", PROMPTS) for i in range(3)]

        async def complete(request):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            events.append((request.trial_index, request.field_name, request.scope))
            assert list(request.parent) == [
                key for key, _ in FIELDS[: len(request.parent)]
            ]
            try:
                if request.trial_index == 1 and request.field_name == "age":
                    await third_started.wait()
                if request.trial_index == 2 and request.field_name == "age":
                    assert 0 in completed and 1 not in completed
                    third_started.set()
                if request.trial_index == 0:
                    if request.field_name == "boilerplate" or (
                        request.field_name == "sex" and request.scope == "slow"
                    ):
                        await advanced.wait()
                    if request.field_name == "histology" and request.scope == "fast":
                        advanced.set()
                await asyncio.sleep(0)
                return answer(request)
            finally:
                in_flight -= 1

        await asyncio.wait_for(
            run_ready_hierarchies(
                iter(states),
                complete,
                concurrency=4,
                active_trials=2,
                on_complete=lambda state: completed.append(state.trial_index),
            ),
            timeout=3,
        )
        assert peak <= 4
        assert advanced.is_set() and third_started.is_set()
        assert set(completed) == {0, 1, 2}
        assert all(
            len(s.output().split("\nBoilerplate")[0].splitlines()) == 2 for s in states
        )
        assert sum(field == "boilerplate" for _, field, _ in events) == 3

    asyncio.run(scenario())


def test_numbering_is_independent_of_completion_order_and_scopes_survive():
    async def extract(slow_scope):
        state = ReadyHierarchy(0, "synthetic", PROMPTS)

        async def complete(request):
            if request.scope == slow_scope:
                await asyncio.sleep(0.001)
            return answer(request)

        await run_ready_hierarchies([state], complete, concurrency=4, active_trials=1)
        return state.output()

    assert asyncio.run(extract("fast")) == asyncio.run(extract("slow"))


@pytest.mark.parametrize("limit", ["max_nodes", "max_calls"])
def test_graph_limits_fail_without_exporting_partial_spaces(limit):
    async def scenario():
        state = ReadyHierarchy(0, "synthetic", PROMPTS, **{limit: 1})

        async def complete(request):
            return answer(request)

        await run_ready_hierarchies([state], complete, concurrency=2, active_trials=1)
        with pytest.raises(ValueError, match=limit):
            state.output()

    asyncio.run(scenario())


def test_failed_branch_never_exports_completed_siblings():
    async def scenario():
        state = ReadyHierarchy(0, "synthetic", PROMPTS)

        async def complete(request):
            if request.field_name == "biomarkers_excluded" and request.scope == "slow":
                return Completion("invalid")
            return answer(request)

        await run_ready_hierarchies([state], complete, concurrency=4, active_trials=1)
        assert state.error
        with pytest.raises(ValueError):
            state.output()

    asyncio.run(scenario())


def test_cancellation_drains_running_requests():
    async def scenario():
        started = asyncio.Event()
        running = 0

        async def complete(request):
            nonlocal running
            running += 1
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                running -= 1

        task = asyncio.create_task(
            run_ready_hierarchies(
                [ReadyHierarchy(0, "synthetic", PROMPTS)],
                complete,
                concurrency=2,
                active_trials=1,
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert running == 0

    asyncio.run(scenario())


def test_added_server_capacity_is_used_while_old_requests_are_still_waiting():
    async def scenario():
        capacity = 1
        root_started, boilerplate_started = asyncio.Event(), asyncio.Event()
        state = ReadyHierarchy(0, "synthetic", PROMPTS)

        async def complete(request):
            if request.field_name == "age":
                root_started.set()
                await boilerplate_started.wait()
            if request.field_name == "boilerplate":
                boilerplate_started.set()
            return answer(request)

        async def add_capacity():
            nonlocal capacity
            await root_started.wait()
            capacity = 2

        grow = asyncio.create_task(add_capacity())
        await asyncio.wait_for(
            run_ready_hierarchies(
                [state],
                complete,
                concurrency=lambda: capacity,
                active_trials=1,
            ),
            timeout=3,
        )
        await grow
        assert boilerplate_started.is_set() and state.done

    asyncio.run(scenario())
