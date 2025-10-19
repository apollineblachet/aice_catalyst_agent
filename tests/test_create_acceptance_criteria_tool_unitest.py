from __future__ import annotations
import re
import pytest

from agent.tools.create_acceptance_criteria_tool_unittest import (
    create_acceptance_criteria_tool,
    TaskInput,
    _create_acceptance_criteria_impl,  
    AcceptanceSpec,
)

def _has_gwt(s: str) -> bool:
    # Each scenario should contain Given/When/Then on separate lines (case-sensitive as generated)
    return all(kw in s for kw in ("Given ", "When ", "Then "))


def test_basic_happy_path_and_formatting():
    tasks = [
        TaskInput(
            id="T-01",
            title="Add Feedback widget form",
            description="React UI form; logged-in users submit to Postgres; log analytics events"
        ),
        TaskInput(
            id="T-02",
            title="Create /api/feedback POST endpoint",
            description="FastAPI endpoint that persists to database and enforces authz"
        ),
    ]

    # Use the tool as LangChain would
    result = create_acceptance_criteria_tool.invoke({"tasks": tasks})
    assert "items" in result and isinstance(result["items"], list)
    assert len(result["items"]) == 2

    # Validate GWT presence and test suites
    for item in result["items"]:
        assert {"id", "title", "acceptance_criteria", "test_suite"} <= set(item.keys())
        criteria = item["acceptance_criteria"]
        assert isinstance(criteria, list) and len(criteria) >= 2  # happy + validation at minimum
        for scenario in criteria:
            assert _has_gwt(scenario), f"Scenario missing GWT: {scenario}"

        suite = item["test_suite"]
        assert "unit_tests" in suite and "integration_tests" in suite
        assert len(suite["unit_tests"]) >= 2
        assert len(suite["integration_tests"]) >= 1
        # Ensure names are non-empty strings
        assert all(isinstance(t["name"], str) and t["name"] for t in suite["unit_tests"])
        assert all(isinstance(t["name"], str) and t["name"] for t in suite["integration_tests"])


def test_schema_objects_and_keyword_driven_scenarios():
    # Internal implementation returns Pydantic objects; easier to assert richer conditions
    tasks = [
        TaskInput(
            id="T-03",
            title="Update status in triage view",
            description="UI allows admin to change status; status transitions must follow rules"
        ),
        TaskInput(
            id="T-04",
            title="Persist feedback to Postgres",
            description="Repository saves and reads back records"
        ),
        TaskInput(
            id="T-05",
            title="Emit analytics event on submission",
            description="Log a single event with schema and no PII"
        ),
    ]
    specs = _create_acceptance_criteria_impl(tasks)
    # Strong typing
    assert all(isinstance(s, AcceptanceSpec) for s in specs)

    # Keyword-driven branches should add targeted scenarios
    s_status = next(s for s in specs if s.id == "T-03")
    assert any("status" in sc.lower() for sc in s_status.acceptance_criteria), "Expected status/workflow scenario"
    assert any(
        t.name == "status_transition_rules" for t in s_status.test_suite.unit_tests
    ), "Expected unit test for status transitions"

    s_db = next(s for s in specs if s.id == "T-04")
    assert any("database" in sc.lower() or "committed" in sc.lower() for sc in s_db.acceptance_criteria)
    assert any(
        t.name == "repository_persists_and_reads_back" for t in s_db.test_suite.unit_tests
    )

    s_analytics = next(s for s in specs if s.id == "T-05")
    assert any("analytics" in sc.lower() for sc in s_analytics.acceptance_criteria)
    assert any(
        t.name == "analytics_event_emitted_once" for t in s_analytics.test_suite.integration_tests
    )


def test_minimum_scenarios_always_present():
    # Even with a very generic task, we still want at least happy + validation
    tasks = [TaskInput(id="GEN-1", title="Do the thing")]
    result = create_acceptance_criteria_tool.invoke({"tasks": tasks})
    item = result["items"][0]
    assert len(item["acceptance_criteria"]) >= 2
    assert all(_has_gwt(s) for s in item["acceptance_criteria"][:2])
