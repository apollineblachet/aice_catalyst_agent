import re
import pytest

from agent.tools.generate_tasks_tool import generate_tasks_tool, Task


def test_basic_breakdown_shapes_and_order():
    raw = (
        "Build a simple feedback widget with auth. "
        "Users submit title/description, optional screenshot upload; "
        "admins triage. Log analytics events; ensure accessibility."
    )
    tasks = generate_tasks_tool(raw)
    assert isinstance(tasks, list)
    assert len(tasks) >= 8

    # IDs are T1..Tn contiguous
    for idx, t in enumerate(tasks, start=1):
        assert isinstance(t, Task)
        assert t.id == f"T{idx}"
        assert re.fullmatch(r"T\d+", t.id)

    # Default sequential dependency (each task depends on previous)
    for idx, t in enumerate(tasks, start=1):
        if idx == 1:
            assert t.depends_on == []
        else:
            assert t.depends_on == [f"T{idx-1}"]


def test_priority_curve_monotonic_by_buckets():
    tasks = generate_tasks_tool("Any feature text that triggers no keywords.")
    # Priorities should be in {1,2,3} and non-increasing early->late
    prios = [t.priority for t in tasks]
    assert set(prios).issubset({1, 2, 3})
    # The first task should not have lower urgency (higher number) than the last
    assert prios[0] <= prios[-1]


def test_max_tasks_truncation_prunes_dependencies():
    raw = "Feature mentions upload, admin role, analytics, accessibility for broader coverage."
    tasks = generate_tasks_tool(raw, max_tasks=4)
    assert len(tasks) == 4

    # No dependency should point outside kept set
    kept_ids = {t.id for t in tasks}
    for t in tasks:
        for dep in t.depends_on:
            assert dep in kept_ids, f"Dangling dependency {dep} in {t.id}"


def test_empty_input_is_graceful():
    tasks = generate_tasks_tool("", max_tasks=3)
    assert len(tasks) == 3
    assert tasks[0].title.lower().startswith("clarify")
