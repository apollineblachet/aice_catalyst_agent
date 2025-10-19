# tests/test_generate_prompt_for_copilot_tool.py
import re
from agent.tools.generate_prompt_for_copilot_tool import (
    generate_prompt_for_copilot_tool,
    GeneratePromptInput,
)

def _mk_input(**overrides):
    base = dict(
        task_title="Add triage filter for status",
        task_description="Implement status filter (New, In Review, Closed) in the staff triage view.",
        acceptance_criteria=[
            "Given items with mixed statuses, when staff selects 'In Review', then only 'In Review' items are shown.",
            "Given a selected filter, when page reloads, then the selection persists via query string.",
            "Given no results for a filter, show an empty state with a reset link.",
        ],
        constraints=[
            "Do not break mobile layout.",
            "Filter must not trigger full page reloads.",
            "Follow existing design tokens."
        ],
        tech_stack="React + Node/Express + Postgres",
        repo_context={"frontend": "apps/web/src", "api": "services/api"},
        files_to_edit=["apps/web/src/pages/admin/triage.tsx", "apps/web/src/components/TriageFilters.tsx"],
        done_definition="E2E happy-path passes; unit tests added for the filter util.",
    )
    base.update(overrides)
    return GeneratePromptInput(**base)

def test_basic_happy_path_and_formatting():
    raw = _mk_input()
    out = generate_prompt_for_copilot_tool(raw)

    # Has required sections
    for section in [
        "### Goal", "### Context", "### Constraints",
        "### Acceptance criteria", "### Codebase pointers",
        "### Focus files", "### Done when", "### Assistant instructions"
    ]:
        assert section in out.prompt, f"Missing section: {section}"

    # Contains all criteria bullets
    for crit in raw.acceptance_criteria:
        assert crit in out.prompt

    # Reasonable size (not bloated)
    assert len(out.prompt) < 3000
    # Lines wrapped sensibly (no ultra-long lines)
    longest = max(len(line) for line in out.prompt.splitlines())
    assert longest <= 120, f"Line too long: {longest}"

def test_deterministic_output():
    raw = _mk_input()
    out1 = generate_prompt_for_copilot_tool(raw)
    out2 = generate_prompt_for_copilot_tool(raw)
    assert out1.prompt == out2.prompt
    assert out1.meta == out2.meta

def test_handles_minimal_required_fields():
    raw = GeneratePromptInput(
        task_title="Add healthcheck endpoint",
        task_description="Create a simple /health endpoint that returns 200 and app version.",
        acceptance_criteria=[
            "GET /health returns 200 with JSON {status:'ok', version:'<semver>'}."
        ],
    )
    out = generate_prompt_for_copilot_tool(raw)

    # Still includes core sections
    assert "### Goal" in out.prompt
    assert "### Acceptance criteria" in out.prompt
    # Optional sections may be omitted
    assert "### Context" not in out.prompt or "Tech:" not in out.prompt

def test_truncates_excessive_criteria_to_10_items():
    crits = [f"Criterion {i}" for i in range(1, 30)]
    raw = _mk_input(acceptance_criteria=crits)
    out = generate_prompt_for_copilot_tool(raw)
    # Only first 10 included, deterministically
    for i in range(1, 11):
        assert f"Criterion {i}" in out.prompt
    assert "Criterion 11" not in out.prompt
