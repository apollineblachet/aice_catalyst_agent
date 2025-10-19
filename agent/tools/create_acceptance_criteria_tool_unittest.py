from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool



class TaskInput(BaseModel):
    """Minimal description of a task or feature the tool will cover."""
    id: str = Field(..., description="Stable task identifier")
    title: str = Field(..., description="Short, action-oriented task title")
    description: Optional[str] = Field("", description="1–3 lines describing the task context")


class TestCase(BaseModel):
    name: str
    description: str


class TestSuite(BaseModel):
    unit_tests: List[TestCase]
    integration_tests: List[TestCase]


class AcceptanceSpec(BaseModel):
    id: str
    title: str
    acceptance_criteria: List[str]  
    test_suite: TestSuite


class CreateACArgs(BaseModel):
    tasks: List[TaskInput]


KEYWORDS_AUTH = {"auth", "login", "logged-in", "admin", "role", "permission", "staff"}
KEYWORDS_PERSIST = {"db", "database", "persist", "save", "postgres", "store"}
KEYWORDS_API = {"api", "endpoint", "http", "rest", "graphql"}
KEYWORDS_UI = {"ui", "page", "form", "widget", "button", "react", "screen", "view"}
KEYWORDS_STATUS = {"status", "state", "workflow"}
KEYWORDS_PERF = {"perf", "performance", "latency", "throughput", "concurrent", "1s", "5 minutes", "30 seconds"}
KEYWORDS_ANALYTICS = {"analytics", "event", "log", "telemetry", "tracking"}

def _has(words: set[str], text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in words)

def _title_as_action(title: str) -> str:
    # Make a human-friendly action phrase from a task title
    t = title.strip().rstrip(".")
    return t[0].lower() + t[1:] if t else "complete the task"

def _happy_path_scenario(task: TaskInput) -> str:
    actor = "a logged-in user" if _has(KEYWORDS_AUTH, task.description + " " + task.title) else "a user"
    action = _title_as_action(task.title)
    outcome = "the expected result is shown"  # generic fallback
    if _has(KEYWORDS_UI, task.title + " " + task.description):
        outcome = "the UI reflects the change with a success indicator"
    elif _has(KEYWORDS_API, task.title + " " + task.description):
        outcome = "the API responds with 2xx and a correct payload"
    elif _has(KEYWORDS_PERSIST, task.title + " " + task.description):
        outcome = "the data is persisted and can be retrieved"
    return (
        f"Given {actor} is on the relevant screen or context\n"
        f"When they {_title_as_action(task.title)} with valid inputs\n"
        f"Then {outcome}"
    )

def _validation_scenario(task: TaskInput) -> str:
    invalid = "missing required field or invalid format"
    if _has(KEYWORDS_UI, task.title + " " + task.description):
        then = "an accessible inline error is displayed and no changes are saved"
    else:
        then = "the operation is rejected with a clear error and nothing is persisted"
    return (
        f"Given a user attempts to {_title_as_action(task.title)}\n"
        f"When the input is {invalid}\n"
        f"Then {then}"
    )

def _auth_or_perm_scenario(task: TaskInput) -> Optional[str]:
    if not _has(KEYWORDS_AUTH, task.title + " " + task.description):
        return None
    return (
        f"Given a user without the required role/permissions\n"
        f"When they attempt to {_title_as_action(task.title)}\n"
        f"Then access is denied and an appropriate message is shown"
    )

def _persistence_scenario(task: TaskInput) -> Optional[str]:
    if not _has(KEYWORDS_PERSIST, task.title + " " + task.description):
        return None
    return (
        f"Given the database is available\n"
        f"When a user {_title_as_action(task.title)}\n"
        f"Then the change is committed and can be read back consistently"
    )

def _status_or_workflow_scenario(task: TaskInput) -> Optional[str]:
    if not _has(KEYWORDS_STATUS, task.title + " " + task.description):
        return None
    return (
        f"Given an item in a known initial status\n"
        f"When a user updates the status following allowed transitions\n"
        f"Then the new status is persisted and visible to authorized users"
    )

def _performance_scenario(task: TaskInput) -> Optional[str]:
    if not _has(KEYWORDS_PERF, task.title + " " + task.description):
        return None
    return (
        f"Given a typical load and network conditions\n"
        f"When a user {_title_as_action(task.title)}\n"
        f"Then the action completes within the documented latency budget"
    )

def _analytics_scenario(task: TaskInput) -> Optional[str]:
    if not _has(KEYWORDS_ANALYTICS, task.title + " " + task.description):
        return None
    return (
        f"Given analytics is enabled\n"
        f"When a user {_title_as_action(task.title)}\n"
        f"Then a single analytics event is emitted with correct schema and no PII"
    )


def _derive_test_suite(task: TaskInput) -> TestSuite:
    unit: List[TestCase] = [
        TestCase(
            name="success_path_renders_or_returns_expected",
            description="Valid inputs produce the correct view or API payload."
        ),
        TestCase(
            name="validation_errors_block_commit",
            description="Invalid inputs surface clear errors and do not persist changes."
        ),
    ]
    if _has(KEYWORDS_STATUS, task.title + " " + task.description):
        unit.append(TestCase(
            name="status_transition_rules",
            description="Only allowed status transitions pass; illegal transitions fail."
        ))
    if _has(KEYWORDS_PERSIST, task.title + " " + task.description):
        unit.append(TestCase(
            name="repository_persists_and_reads_back",
            description="Repository/DAO writes and reads back consistent state."
        ))

    integration: List[TestCase] = []
    if _has(KEYWORDS_UI, task.title + " " + task.description):
        integration.append(TestCase(
            name="ui_end_to_end_happy_path",
            description="From user action to persisted state and refreshed UI."
        ))
    if _has(KEYWORDS_API, task.title + " " + task.description):
        integration.append(TestCase(
            name="api_contract_and_status_codes",
            description="Endpoint returns expected 2xx/4xx and OpenAPI contract fields."
        ))
    if _has(KEYWORDS_AUTH, task.title + " " + task.description):
        integration.append(TestCase(
            name="authz_enforced",
            description="Unauthorized users cannot perform the action; authorized can."
        ))
    if _has(KEYWORDS_ANALYTICS, task.title + " " + task.description):
        integration.append(TestCase(
            name="analytics_event_emitted_once",
            description="Back-end or client emits a single well-formed event without PII."
        ))

    # Always include a basic end-to-end when relevant signals are present; if none, still have at least one integration
    if not integration:
        integration.append(TestCase(
            name="end_to_end_smoke",
            description="Primary flow works across boundaries (API/DB/UI) if applicable."
        ))

    return TestSuite(unit_tests=unit, integration_tests=integration)


def _build_acceptance_spec(task: TaskInput) -> AcceptanceSpec:
    scenarios: List[str] = [
        _happy_path_scenario(task),
        _validation_scenario(task),
    ]

    # Optional scenarios based on keywords
    for opt in (
        _auth_or_perm_scenario(task),
        _persistence_scenario(task),
        _status_or_workflow_scenario(task),
        _performance_scenario(task),
        _analytics_scenario(task),
    ):
        if opt:
            scenarios.append(opt)

    # Guarantee each scenario has Given/When/Then (already ensured), and keep concise
    test_suite = _derive_test_suite(task)
    return AcceptanceSpec(
        id=task.id,
        title=task.title,
        acceptance_criteria=scenarios,
        test_suite=test_suite,
    )


def _create_acceptance_criteria_impl(tasks: List[TaskInput]) -> List[AcceptanceSpec]:
    return [_build_acceptance_spec(t) for t in tasks]


def _create_acceptance_criteria_tool(tasks: List[TaskInput]) -> Dict[str, Any]:
    """
    Tool wrapper: returns a JSON-serializable dict so the agent can display/pipe it.
    """
    specs = _create_acceptance_criteria_impl(tasks)
    # Convert Pydantic models to plain dicts
    return {"items": [s.model_dump() for s in specs]}


create_acceptance_criteria_tool = StructuredTool.from_function(
    name="create_acceptance_criteria_tool",
    description=(
        "Produce clear, testable acceptance criteria in Given/When/Then format for each task. "
        "Also returns a proposed unit and integration test suite per task."
    ),
    func=_create_acceptance_criteria_tool,
    args_schema=CreateACArgs,
)
