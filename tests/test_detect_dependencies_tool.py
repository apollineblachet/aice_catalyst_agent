import copy
import pytest

from agent.tools.detect_dependencies_tool import (
    TaskIn,
    detect_dependencies_tool,
)


def _titles(tasks_out):
    return [t.title for t in tasks_out]


def _ids(tasks_out):
    return [t.id for t in tasks_out]


def _edges_to_tuples(edges):
    return sorted([(e.source, e.target) for e in edges])


def test_basic_happy_path_and_formatting():
    # Scrambled input order to verify deterministic topo & edges
    raw = [
        TaskIn(title="Build UI widget in React"),
        TaskIn(title="Create database schema & migrations"),
        TaskIn(title="Expose API endpoints (FastAPI)"),
        TaskIn(title="Write unit tests (pytest)"),
        TaskIn(title="Update README and usage docs"),
    ]

    res = detect_dependencies_tool(raw)

    # Each task receives a generated id, stable slugified
    assert all(t.id for t in res.tasks)
    assert len(set(_ids(res.tasks))) == len(res.tasks)

    # Expected coarse order: schema -> api -> ui -> tests -> docs
    # The tool only adds minimal necessary edges (closest earlier stage).
    edges = _edges_to_tuples(res.edges)
    assert ("create-database-schema-migrations", "expose-api-endpoints-fastapi") in edges
    assert ("expose-api-endpoints-fastapi", "build-ui-widget-in-react") in edges
    assert ("build-ui-widget-in-react", "write-unit-tests-pytest") in edges or \
           ("expose-api-endpoints-fastapi", "write-unit-tests-pytest") in edges
    assert any(topo in res.topo_order for topo in ["create-database-schema-migrations"])

    # Graph must be acyclic
    assert res.cycles == []
    assert len(res.topo_order) == len(res.tasks)


def test_respects_existing_dependencies_and_does_not_self_depend():
    raw = [
        TaskIn(id="schema", title="DB schema", depends_on=[]),
        TaskIn(id="api", title="API endpoints", depends_on=["schema"]),
        TaskIn(id="ui", title="UI page", depends_on=["api"]),
    ]
    res = detect_dependencies_tool(raw)

    # Should keep the chain; not add self deps
    for t in res.tasks:
        assert t.id not in t.depends_on

    # Edges should at least contain the provided ones
    edges = _edges_to_tuples(res.edges)
    assert ("schema", "api") in edges
    assert ("api", "ui") in edges

    # Deterministic topo equals the chain
    assert res.topo_order.index("schema") < res.topo_order.index("api") < res.topo_order.index("ui")


def test_parallel_tasks_not_overconstrained():
    raw = [
        TaskIn(title="Setup project & env"),
        TaskIn(title="Add logging/metrics"),
        TaskIn(title="Add CI pipeline"),
    ]
    res = detect_dependencies_tool(raw)

    # foundation vs monitoring/deploy buckets: each later-stage task
    # should depend on at most one closest earlier-stage task
    for t in res.tasks:
        if t.title.lower().startswith("add "):
            assert len(t.depends_on) <= 1


def test_cycle_detection():
    raw = [
        TaskIn(id="a", title="A", depends_on=["c"]),
        TaskIn(id="b", title="B", depends_on=["a"]),
        TaskIn(id="c", title="C", depends_on=["b"]),
    ]
    res = detect_dependencies_tool(raw)

    # Cycles must be reported; topo_order empty
    assert res.topo_order == []
    assert res.cycles != []
    # Cycle should include a,b,c in some canonical rotation
    cyc_ids = set(sum(res.cycles, []))
    assert {"a", "b", "c"}.issubset(cyc_ids)


def test_deterministic_output():
    raw = [
        TaskIn(title="Create database schema"),
        TaskIn(title="Implement API"),
        TaskIn(title="Build UI"),
        TaskIn(title="Write tests"),
    ]
    res1 = detect_dependencies_tool(copy.deepcopy(raw))
    res2 = detect_dependencies_tool(copy.deepcopy(raw))

    # identical edges & topo across runs
    assert _edges_to_tuples(res1.edges) == _edges_to_tuples(res2.edges)
    assert res1.topo_order == res2.topo_order
    # ids stable
    assert _ids(res1.tasks) == _ids(res2.tasks)
