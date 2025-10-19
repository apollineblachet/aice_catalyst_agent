from __future__ import annotations

from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str = Field(..., description="Stable task identifier like 'T1'")
    title: str
    description: str
    priority: int = Field(..., ge=1, le=5, description="1=highest urgency")
    depends_on: List[str] = Field(default_factory=list, description="List of task ids this task depends on")


def _base_pipeline() -> List[Dict]:
    """
    A pragmatic default pipeline that fits most 'build a small feature' asks.
    Ordered for sensible default dependencies (each depends on previous).
    """
    return [
        {"title": "Clarify scope & constraints",
         "description": "Confirm must-haves, non-goals, and edge cases; note unknowns & risks."},
        {"title": "Data model / schema sketch",
         "description": "Identify entities, fields, and minimal schema changes needed."},
        {"title": "Backend API surface",
         "description": "Define endpoints, request/response contracts, and validation rules."},
        {"title": "Frontend UI skeleton",
         "description": "Create minimal UI flow and states (loading, success, error)."},
        {"title": "Integrations & external services",
         "description": "Wire any external APIs, keys, and error handling paths."},
        {"title": "Persistence layer",
         "description": "Implement DB access patterns and migrations if any."},
        {"title": "Functional tests (happy paths)",
         "description": "Automate the main success scenarios end-to-end or via API."},
        {"title": "Error & edge-case tests",
         "description": "Cover invalid inputs, timeouts, and permission errors."},
        {"title": "Docs & handoff",
         "description": "Write a concise README with setup, env vars, run & test commands."},
    ]


def _keyword_inserts(raw_text: str) -> List[Dict]:
    """
    Drop in a few context-aware tasks when the feature mentions certain capabilities.
    Keeps this tool 'simple yet functional' without depending on an LLM.
    """
    text = raw_text.lower()
    inserts: List[Dict] = []

    def add(title: str, description: str):
        inserts.append({"title": title, "description": description})

    # File uploads / media handling
    if any(k in text for k in ["upload", "mp3", "mp4", "wav", "file", "attachment"]):
        add("File handling & validation",
            "Accept and validate allowed file types, size limits, and safe temp storage.")

    # Auth / roles
    if any(k in text for k in ["auth", "login", "role", "permission", "admin"]):
        add("Authorization & roles",
            "Enforce who can access/modify; add guards and permission checks.")

    # Performance / SLAs
    if any(k in text for k in ["performance", "sla", "concurrent", "latency", "throughput"]):
        add("Performance checks",
            "Add basic timing, concurrency tests, and guardrails for target SLAs.")

    # Accessibility
    if any(k in text for k in ["a11y", "accessibility", "aria", "keyboard"]):
        add("Accessibility pass",
            "Ensure keyboard navigation, aria labels, and color contrast basics.")

    # Analytics / logging
    if any(k in text for k in ["analytics", "telemetry", "logging", "metrics"]):
        add("Analytics & logging events",
            "Emit events for key user actions and important failures.")

    return inserts


def _assign_priorities(n: int) -> List[int]:
    """
    Simple priority curve: earliest tasks are highest urgency.
    1 for first third, 2 for second third, 3 for final third.
    """
    if n == 0:
        return []
    one = max(1, n // 3)
    two = max(1, (n - one) // 2)
    out = []
    for i in range(n):
        if i < one:
            out.append(1)
        elif i < one + two:
            out.append(2)
        else:
            out.append(3)
    return out


def _reindex_and_prune(tasks: List[Task], keep_count: Optional[int]) -> List[Task]:
    """
    If max_tasks is set, truncate while keeping IDs contiguous (T1..Tk)
    and pruning dangling dependencies.
    """
    if keep_count is None or keep_count >= len(tasks):
        return tasks

    kept = tasks[:keep_count]
    kept_ids: Set[str] = {t.id for t in kept}

    # Remove deps not in kept_ids
    for t in kept:
        t.depends_on = [d for d in t.depends_on if d in kept_ids]
    return kept


def generate_tasks_tool(raw_text: str, max_tasks: Optional[int] = None) -> List[Task]:
    """
    Break a feature into granular tasks with short descriptions and a first guess
    at dependencies and priority.

    Args:
        raw_text: Free-form requirement text.
        max_tasks: Optional hard cap; keeps earliest tasks and prunes dangling deps.

    Returns:
        List[Task]: ordered from start to finish.
    """
    if not raw_text or not raw_text.strip():
        # Minimal, still useful output
        base = [{"title": "Clarify scope & constraints",
                 "description": "Confirm must-haves and success criteria with the requester."},
                {"title": "Skeleton implementation",
                 "description": "Create minimal end-to-end path with placeholders."},
                {"title": "Basic tests & README",
                 "description": "Cover the happy path and document the run steps."}]
    else:
        base = _base_pipeline()
        inserts = _keyword_inserts(raw_text)
        # Insert contextual tasks just after UI skeleton (index 3) to keep flow logical
        inject_at = min(4, len(base))
        base = base[:inject_at] + inserts + base[inject_at:]

    # Materialize as Task objects with sequential dependencies
    priorities = _assign_priorities(len(base))
    tasks: List[Task] = []
    for i, item in enumerate(base, start=1):
        tid = f"T{i}"
        depends = [f"T{i-1}"] if i > 1 else []
        tasks.append(Task(
            id=tid,
            title=item["title"],
            description=item["description"],
            priority=priorities[i - 1] if i - 1 < len(priorities) else 3,
            depends_on=depends
        ))

    # Truncate & prune if requested
    tasks = _reindex_and_prune(tasks, max_tasks)

    return tasks


__all__ = ["Task", "generate_tasks_tool"]
