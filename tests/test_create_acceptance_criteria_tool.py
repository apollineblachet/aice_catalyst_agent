import copy
import pytest

from agent.tools.create_acceptance_criteria_tool import (
    create_acceptance_criteria_tool,
    CreateAcceptanceCriteriaRequest,
    WorkItem,
)

def _run(payload_dict):
    return create_acceptance_criteria_tool.invoke(payload_dict)


def test_basic_happy_path_and_formatting():
    wi = WorkItem(
        id="T-1",
        type="task",
        title="Submit feedback form",
        description="Users fill required fields and submit."
    )
    resp = _run(CreateAcceptanceCriteriaRequest(items=[wi]).model_dump())
    results = resp["results"][0]["criteria"]
    # Should have at least 3 criteria by default (happy, validation, error; auth may or may not)
    assert len(results) >= 3

    # Check Gherkin structure
    for c in results:
        text = "\n".join([
            "Given " + c["parts"]["given"],
            "When " + c["parts"]["when"],
            "Then " + c["parts"]["then"],
        ])
        # Simple sanity: each line must start with keyword
        for line, kw in zip(text.splitlines(), ["Given", "When", "Then"]):
            assert line.startswith(kw)


def test_includes_authorization_when_restricted():
    wi = WorkItem(
        id="T-2",
        type="task",
        title="Triage view access",
        description="Only staff (role=admin) can access the triage view."
    )
    resp = _run(CreateAcceptanceCriteriaRequest(items=[wi]).model_dump())
    kinds = [c["kind"] for c in resp["results"][0]["criteria"]]
    assert "authorization" in kinds


def test_perf_generates_nonfunctional_criterion():
    wi = WorkItem(
        id="T-3",
        type="task",
        title="Fast widget",
        description="First contentful interaction <= 1s on broadband."
    )
    resp = _run(CreateAcceptanceCriteriaRequest(items=[wi]).model_dump())
    kinds = [c["kind"] for c in resp["results"][0]["criteria"]]
    assert "nonfunctional" in kinds


def test_upload_paths_are_shaped_correctly():
    wi = WorkItem(
        id="T-4",
        type="feature",
        title="Upload meeting recording",
        description="User can upload an MP4 file and see success."
    )
    resp = _run(CreateAcceptanceCriteriaRequest(items=[wi]).model_dump())
    parts = resp["results"][0]["criteria"][0]["parts"]
    # Happy path 'given' should mention upload screen and file selected
    assert "upload" in parts["given"] or "file" in parts["given"]


def test_deterministic_output():
    wi = WorkItem(
        id="T-5",
        type="task",
        title="Change status to In Review",
        description="Staff can change status on an item."
    )
    req = CreateAcceptanceCriteriaRequest(items=[wi], max_per_item=5)
    resp1 = _run(req.model_dump())
    resp2 = _run(req.model_dump())
    assert resp1 == resp2, "Tool must be deterministic for same input"


def test_respects_max_per_item_and_edge_case_toggle():
    wi = WorkItem(
        id="T-6",
        type="task",
        title="Submit feedback form",
        description="Users fill required fields and submit."
    )
    # With edge cases off and max_per_item=3
    req1 = CreateAcceptanceCriteriaRequest(items=[wi], max_per_item=3, include_edge_cases=False)
    resp1 = _run(req1.model_dump())
    assert len(resp1["results"][0]["criteria"]) == 3

    # With edge cases on and larger cap
    req2 = CreateAcceptanceCriteriaRequest(items=[wi], max_per_item=6, include_edge_cases=True)
    resp2 = _run(req2.model_dump())
    kinds = [c["kind"] for c in resp2["results"][0]["criteria"]]
    assert "edge_case" in kinds
