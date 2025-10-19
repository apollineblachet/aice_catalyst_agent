from __future__ import annotations

from typing import List, Dict, Set, Tuple, Optional
from pydantic import BaseModel, Field, field_validator
import re
import itertools
from collections import defaultdict, deque


class TaskIn(BaseModel):
    id: Optional[str] = Field(
        default=None,
        description="Stable id. If missing, a slug is generated from title."
    )
    title: str
    description: Optional[str] = None
    depends_on: List[str] = Field(
        default_factory=list,
        description="Pre-known dependencies (ids)."
    )

    @field_validator("title")
    def title_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("title must be non-empty")
        return v.strip()


class TaskOut(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    depends_on: List[str] = Field(default_factory=list)


class DependencyEdge(BaseModel):
    source: str  # blocks
    target: str  # is blocked by source


class DetectDependenciesResult(BaseModel):
    tasks: List[TaskOut]
    edges: List[DependencyEdge]
    topo_order: List[str] = Field(
        default_factory=list,
        description="Topologically sorted ids when acyclic; empty if cycles exist."
    )
    cycles: List[List[str]] = Field(
        default_factory=list,
        description="List of simple cycles (ids) if any; empty otherwise."
    )
    notes: str = ""



STAGE_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("foundation", ("init", "setup", "scaffold", "configure", "env", "schema", "database", "db", "migrations")),
    ("backend", ("backend", "service", "api", "endpoint", "controller", "fastapi", "express", "auth")),
    ("integration", ("integrate", "wire", "hook", "webhook", "connect")),
    ("frontend", ("ui", "ux", "page", "widget", "component", "react", "view")),
    ("testing", ("test", "unit", "e2e", "integration test", "cypress", "pytest")),
    ("docs", ("readme", "docs", "documentation")),
    ("deploy", ("deploy", "docker", "infra", "pipeline", "ci", "cd")),
    ("monitoring", ("monitor", "metrics", "logging", "alerting", "observability")),
]

STAGE_INDEX = {name: idx for idx, (name, _) in enumerate(STAGE_KEYWORDS)}


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "task"


def stage_for(title: str) -> int:
    t = title.lower()
    for name, kws in STAGE_KEYWORDS:
        if any(kw in t for kw in kws):
            return STAGE_INDEX[name]
    return 1  


def soft_ref_ids(title: str, id_by_token: Dict[str, str]) -> Set[str]:
    """
    If a title mentions a token from another task id or key noun, infer a soft dep.
    Very conservative: only exact token matches ≥ 4 chars.
    """
    tokens = {tok for tok in re.findall(r"[a-z0-9]{4,}", title.lower())}
    refs = set()
    for tok in tokens:
        if tok in id_by_token:
            refs.add(id_by_token[tok])
    return refs


def detect_dependencies_tool(tasks: List[TaskIn]) -> DetectDependenciesResult:
    """
    From a list of tasks, infer blocking relationships and suggest a sensible execution order.
    Deterministic: outputs are stable across runs for the same input.
    """
    if not tasks:
        return DetectDependenciesResult(tasks=[], edges=[], notes="No tasks provided.")

    # Normalize ids and keep original order to break ties deterministically
    normed: List[TaskOut] = []
    seen_ids: Set[str] = set()
    for t in tasks:
        tid = t.id or slugify(t.title)
        # ensure unique
        base = tid
        k = 2
        while tid in seen_ids:
            tid = f"{base}-{k}"
            k += 1
        seen_ids.add(tid)
        normed.append(TaskOut(id=tid, title=t.title, description=t.description, depends_on=sorted(set(t.depends_on))))

    # Build quick lookups
    by_id: Dict[str, TaskOut] = {t.id: t for t in normed}
    id_by_token: Dict[str, str] = {}
    for t in normed:
        for tok in set(re.findall(r"[a-z0-9]{4,}", t.title.lower())):
            # map token -> first id that contains it to remain deterministic
            id_by_token.setdefault(tok, t.id)

    # Stage ranking and deterministic tie-breakers by original order
    stage_idx: Dict[str, int] = {t.id: stage_for(t.title) for t in normed}
    order_key = {t.id: i for i, t in enumerate(normed)}

    # Infer dependencies:
    # - A task at stage S should depend on at least one task from any strictly earlier stage that is relevant.
    # - Soft text references add dependencies.
    for t in normed:
        deps: Set[str] = set(t.depends_on)

        # Soft references
        deps |= soft_ref_ids(t.title, id_by_token)

        # Stage-based single hop: depend on the *closest* earlier stage tasks, if any exist.
        s = stage_idx[t.id]
        if s > 0:
            # candidate providers: tasks with stage < s
            providers = [u for u in normed if stage_idx[u.id] < s and u.id != t.id]
            if providers:
                # Pick the provider with the largest stage < s (closest), then stable by original order
                best_stage = max(stage_idx[u.id] for u in providers)
                closest = [u for u in providers if stage_idx[u.id] == best_stage]
                closest_sorted = sorted(closest, key=lambda x: (order_key[x.id], x.id))
                # To avoid over-constraining, only depend on the first provider within the closest stage
                deps.add(closest_sorted[0].id)

        # Remove self-deps and normalize
        deps.discard(t.id)
        t.depends_on = sorted(deps)

    # Build edge list
    edges = sorted(
        [DependencyEdge(source=d, target=t.id) for t in normed for d in t.depends_on],
        key=lambda e: (e.source, e.target),
    )

    # Topological sort + cycle detection
    topo, cycles = _toposort_with_cycles([t.id for t in normed], edges)

    notes = "Cycles detected; topo_order omitted." if cycles else "Acyclic dependency graph."
    return DetectDependenciesResult(
        tasks=normed,
        edges=edges,
        topo_order=topo if not cycles else [],
        cycles=cycles,
        notes=notes,
    )


def _toposort_with_cycles(nodes: List[str], edges: List[DependencyEdge]) -> Tuple[List[str], List[List[str]]]:
    # Kahn's algorithm for topo; then simple cycle enumeration of remaining nodes
    incoming = defaultdict(int)
    outgoing = defaultdict(list)
    for e in edges:
        outgoing[e.source].append(e.target)
        incoming[e.target] += 1
        incoming.setdefault(e.source, incoming.get(e.source, 0))

    q = deque(sorted([n for n in nodes if incoming.get(n, 0) == 0]))
    result: List[str] = []

    while q:
        n = q.popleft()
        result.append(n)
        for m in sorted(outgoing.get(n, [])):
            incoming[m] -= 1
            if incoming[m] == 0:
                q.append(m)

    if len(result) == len(nodes):
        return result, []

    # Cycles exist: attempt to list simple cycles (bounded, deterministic)
    remaining = [n for n in nodes if n not in result]
    graph = {n: sorted(outgoing.get(n, [])) for n in remaining}

    cycles: Set[Tuple[str, ...]] = set()

    def dfs(start: str, current: str, path: List[str], seen: Set[str]):
        for nxt in graph.get(current, []):
            if nxt == start:
                cyc = tuple(sorted(_rotate_to_smallest(path + [start])))
                cycles.add(cyc)
            elif nxt not in seen and nxt in graph:
                dfs(start, nxt, path + [nxt], seen | {nxt})

    for n in sorted(graph.keys()):
        dfs(n, n, [n], {n})

    # Convert to list of lists
    cycles_list = [list(c) for c in sorted(cycles)]
    return result, cycles_list


def _rotate_to_smallest(path: List[str]) -> List[str]:
    """Rotate cyclic list so that lexicographically smallest id comes first; ensures deterministic cycle signatures."""
    if not path:
        return path
    k = min(range(len(path)), key=lambda i: path[i])
    return path[k:] + path[:k]
