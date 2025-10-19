from __future__ import annotations

from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, Field
from dataclasses import dataclass
import re
from langchain_core.tools import tool


ItemType = Literal["feature", "task"]


class WorkItem(BaseModel):
    """
    A single feature or task to turn into acceptance criteria.
    """
    id: str = Field(..., description="Stable identifier, e.g. T-12 or FEAT-1")
    type: ItemType = Field(..., description='"feature" or "task"')
    title: str
    description: Optional[str] = Field(
        default="",
        description="Short, concrete description of the behavior to validate",
    )
    notes: Optional[str] = Field(
        default="",
        description="Any extra constraints like perf, a11y, roles, states, limits"
    )


class CriteriaParts(BaseModel):
    given: str
    when: str
    then: str

    def as_text(self) -> str:
        return f"Given {self.given}\nWhen {self.when}\nThen {self.then}"


class Criteria(BaseModel):
    """
    One Gherkin-style acceptance criterion.
    """
    kind: Literal["happy_path", "validation", "authorization", "error_state", "nonfunctional", "edge_case"]
    parts: CriteriaParts


class AcceptanceCriteria(BaseModel):
    item_id: str
    item_title: str
    criteria: List[Criteria]


class CreateAcceptanceCriteriaRequest(BaseModel):
    items: List[WorkItem]
    language: Literal["en"] = "en"
    max_per_item: int = Field(5, ge=1, le=12)
    include_edge_cases: bool = True


class CreateAcceptanceCriteriaResponse(BaseModel):
    results: List[AcceptanceCriteria]
    notes: Optional[str] = ""



@dataclass(frozen=True)
class Heuristics:
    needs_auth: bool
    has_perf: Optional[str]  # e.g. "< 1s", "≤ 30s"
    mentions_accessibility: bool
    has_upload: bool
    has_form: bool
    has_status_change: bool


_AUTH_PATTERNS = [
    r"\bonly\s+(staff|admins?|admin role|role\s*=\s*admin)\b",
    r"\bauthenticated\b|\blogged[-\s]?in\b",
]
_PERF_PATTERNS = [
    r"\b(<|≤|<=)\s*\d+(\.\d+)?\s*(ms|s|sec|seconds?)\b",
    r"\b(performance|latency)\b.*?\b(target|budget|limit)\b",
]
_A11Y_PATTERNS = [r"\ba11y\b", r"\baccessibilit(y|ies)\b", r"\baria[-\w]*\b", r"\bkeyboard\b"]
_UPLOAD_PATTERNS = [r"\bupload\b|\bfile(s)?\b|\bscreenshot\b"]
_FORM_PATTERNS = [r"\bform\b|\bsubmit\b|\bfield(s)?\b|\blabel(s)?\b"]
_STATUS_PATTERNS = [r"\bstatus\b|\b(approve|close|reopen|change)\b"]


def _match_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def _extract_perf_target(text: str) -> Optional[str]:
    t = text.lower()
    # capture things like "< 1s", "≤ 30 seconds"
    m = re.search(r"(<|≤|<=)\s*\d+(\.\d+)?\s*(ms|s|sec|seconds?)", t)
    if m:
        return m.group(0).replace("sec", "s").replace("seconds", "s")
    return None


def _heuristics(wi: WorkItem) -> Heuristics:
    blob = " ".join(filter(None, [wi.title, wi.description, wi.notes]))
    return Heuristics(
        needs_auth=_match_any(blob, _AUTH_PATTERNS),
        has_perf=_extract_perf_target(blob),
        mentions_accessibility=_match_any(blob, _A11Y_PATTERNS),
        has_upload=_match_any(blob, _UPLOAD_PATTERNS),
        has_form=_match_any(blob, _FORM_PATTERNS),
        has_status_change=_match_any(blob, _STATUS_PATTERNS),
    )


def _happy_path(wi: WorkItem, h: Heuristics) -> Criteria:
    if h.has_upload:
        parts = CriteriaParts(
            given="the user is on the upload screen with a valid file selected",
            when="they start the upload",
            then="the system stores the file, returns a success confirmation, and shows the new entry in the list"
        )
    elif h.has_form:
        parts = CriteriaParts(
            given="the user is on the form with all required fields filled correctly",
            when="they click Submit",
            then="the system persists the data and shows a visible success state without page reload"
        )
    elif h.has_status_change:
        parts = CriteriaParts(
            given="a staff user is viewing the item in the triage view",
            when="they change the status using the UI control",
            then="the new status is saved and immediately reflected in filters and counts"
        )
    else:
        parts = CriteriaParts(
            given="the user is on the relevant screen and prerequisites are met",
            when="they perform the primary action",
            then="the system completes the action and presents a clear confirmation"
        )
    return Criteria(kind="happy_path", parts=parts)


def _validation(wi: WorkItem, h: Heuristics) -> Criteria:
    if h.has_form:
        parts = CriteriaParts(
            given="a required field is empty or exceeds its max length",
            when="the user attempts to submit",
            then="the system blocks submission and shows inline, accessible error messages describing what to fix"
        )
    else:
        parts = CriteriaParts(
            given="an input parameter is missing or malformed",
            when="the user triggers the action",
            then="the system returns a validation error without side effects"
        )
    return Criteria(kind="validation", parts=parts)


def _authorization(wi: WorkItem, h: Heuristics) -> Optional[Criteria]:
    if not h.needs_auth:
        return None
    parts = CriteriaParts(
        given="a non-privileged user tries to access a restricted capability",
        when="they navigate directly or call the endpoint",
        then="the system denies with 403/redirect and no restricted data is leaked"
    )
    return Criteria(kind="authorization", parts=parts)


def _error_state(wi: WorkItem, h: Heuristics) -> Criteria:
    if h.has_upload:
        parts = CriteriaParts(
            given="the selected file is unsupported or the network fails mid-upload",
            when="the user starts the upload",
            then="the system shows a recoverable error and allows retry without losing prior inputs"
        )
    else:
        parts = CriteriaParts(
            given="a dependent service is unavailable",
            when="the user initiates the action",
            then="the system shows a non-destructive error state and logs the incident"
        )
    return Criteria(kind="error_state", parts=parts)


def _nonfunctional(wi: WorkItem, h: Heuristics) -> Optional[Criteria]:
    if h.has_perf:
        parts = CriteriaParts(
            given="the happy path scenario",
            when="the user performs the primary action under normal broadband conditions",
            then=f"the visible response completes {h.has_perf}"
        )
        return Criteria(kind="nonfunctional", parts=parts)

    if h.mentions_accessibility or h.has_form:
        parts = CriteriaParts(
            given="the UI is navigated using only a keyboard and a screen reader",
            when="the user traverses fields and triggers the primary action",
            then="focus order is logical, labels are announced, and success/error states are conveyed"
        )
        return Criteria(kind="nonfunctional", parts=parts)

    return None


def _edge_case(wi: WorkItem, h: Heuristics) -> Criteria:
    parts = CriteriaParts(
        given="two users modify the same item concurrently",
        when="the second save occurs",
        then="the system prevents silent overwrites and prompts the user to reconcile changes"
    )
    return Criteria(kind="edge_case", parts=parts)


def _build_for_item(wi: WorkItem, req: CreateAcceptanceCriteriaRequest) -> AcceptanceCriteria:
    h = _heuristics(wi)

    buckets: List[Criteria] = []
    buckets.append(_happy_path(wi, h))
    buckets.append(_validation(wi, h))
    auth = _authorization(wi, h)
    if auth:
        buckets.append(auth)
    buckets.append(_error_state(wi, h))
    nonfunc = _nonfunctional(wi, h)
    if nonfunc:
        buckets.append(nonfunc)
    if req.include_edge_cases:
        buckets.append(_edge_case(wi, h))

    # Respect max_per_item, but always keep at least 1
    selected = buckets[: max(1, req.max_per_item)]

    return AcceptanceCriteria(
        item_id=wi.id,
        item_title=wi.title,
        criteria=selected,
    )


@tool(
    "create_acceptance_criteria_tool",
    args_schema=CreateAcceptanceCriteriaRequest,  # <-- key change
    return_direct=False,
)
def create_acceptance_criteria_tool(
    items: List[WorkItem],
    language: Literal["en"] = "en",
    max_per_item: int = 5,
    include_edge_cases: bool = True,
) -> Dict:
    """
    Produce clear, testable Given/When/Then acceptance criteria for each item.
    Deterministic and template-driven; no LLM call required.
    """
    req = CreateAcceptanceCriteriaRequest(
        items=items,
        language=language,
        max_per_item=max_per_item,
        include_edge_cases=include_edge_cases,
    )
    results = [_build_for_item(wi, req) for wi in req.items]
    resp = CreateAcceptanceCriteriaResponse(
        results=results,
        notes="Deterministic criteria generated via keyword heuristics. Tune by updating patterns or titles/descriptions."
    )
    return resp.model_dump()
