import json
import re
import types

import pytest

from agent.tools.parse_requirements_tool import (
    parse_requirements_tool,   
    RequirementParse,
    Feature,
    Constraint,
    Stakeholder,
    SuccessCriterion,
)

def _invoke_tool(sample: str):
    """
    Works whether the object is a LangChain StructuredTool (has .invoke)
    or a plain callable (in case you refactor the decorator away).
    """
    if hasattr(parse_requirements_tool, "invoke"):
        return parse_requirements_tool.invoke({"raw_text": sample})
    # fallback: plain callable
    return parse_requirements_tool(sample)

def _as_model(obj) -> RequirementParse:
    """
    Ensure we get a RequirementParse model object.
    If the tool was modified to return a dict, reconstruct the model.
    """
    if isinstance(obj, RequirementParse):
        return obj
    if isinstance(obj, dict):
        return RequirementParse(**obj)
    # Some LangChain wrappers might return a Pydantic-like object
    if hasattr(obj, "model_dump"):
        return RequirementParse(**obj.model_dump())
    raise TypeError(f"Unexpected tool return type: {type(obj)}")

def test_basic_extraction_happy_path():
    sample = """
Stakeholders:
- Admin: manages access and reviews flagged items
- End users: submit feedback via widget

Constraints:
- Must run on Azure; use Postgres
- GDPR compliant; PII encrypted at rest
- MVP within 2 weeks

Success Criteria:
- Collect 80% of feedback within the widget
- Average triage under 2h

Features:
- Build an embeddable feedback widget
- Implement triage dashboard with filter & search
"""
    raw = _invoke_tool(sample)
    model = _as_model(raw)

    # Features
    titles = [f.title.lower() for f in model.features]
    assert any("feedback widget" in t for t in titles)
    assert any("triage dashboard" in t for t in titles)

    # Constraints typed
    kinds = {c.kind for c in model.constraints}
    assert "tech" in kinds       # Azure/Postgres
    assert "compliance" in kinds # GDPR
    assert "time" in kinds       # within 2 weeks

    # Stakeholders
    roles = {s.role.lower() for s in model.stakeholders}
    assert "admin" in roles
    assert "end users" in roles or "user" in " ".join(roles)

    # Success metrics (should capture at least the % or duration token)
    metrics = [sc.metric for sc in model.success_criteria if sc.metric]
    assert any("%" in m for m in metrics) or any(re.search(r"\b\d+\s?(ms|s|sec|min|h)\b", m or "") for m in metrics)

def test_empty_input_returns_unknown_and_no_items():
    raw = _invoke_tool("   ")
    model = _as_model(raw)

    assert model.features == []
    assert model.constraints == []
    assert model.stakeholders == []
    assert model.success_criteria == []
    assert "Empty requirement text." in model.unknowns

def test_unknowns_detection_when_missing_key_topics():
    sample = """
We need a small portal for partners to upload CSVs and view processed results.
No mention of SLAs or KPIs. Keep it simple for MVP.
"""
    model = _as_model(_invoke_tool(sample))
    unknowns = " | ".join(model.unknowns).lower()
    # Should flag at least these gaps
    assert "timeline" in unknowns or "deadline" in unknowns
    assert "auth/authorization" in unknowns
    assert "storage" in unknowns or "db" in unknowns
    assert "success criteria" in unknowns

def test_constraint_kind_classification_matrix():
    sample = """
Constraints:
- Deadline: deliver by Oct 31
- Budget capped at $5k
- Must use React + Node on Azure
- JWT-based auth; PII encrypted at rest
- Must comply with GDPR and SOC2
- Depends on CRM integration with HubSpot
- Out of scope: mobile app
"""
    model = _as_model(_invoke_tool(sample))
    by_kind = {}
    for c in model.constraints:
        by_kind.setdefault(c.kind, []).append(c.text.lower())

    assert any("oct" in t or "deadline" in t for t in by_kind.get("time", []))
    assert any("$" in t or "budget" in t for t in by_kind.get("budget", []))
    assert any("react" in t or "azure" in t for t in by_kind.get("tech", []))
    assert any("jwt" in t or "encrypted" in t for t in by_kind.get("security", []))
    assert any("gdpr" in t or "soc" in t for t in by_kind.get("compliance", []))
    assert any("depends on" in t or "integration" in t for t in by_kind.get("dependency", []))
    assert any("out of scope" in t for t in by_kind.get("scope", []))

def test_no_explicit_sections_still_finds_feature_and_roles():
    sample = """
Admins need to approve vendor access requests.
Users submit forms; security reviews sensitive fields.
Build a simple dashboard and notify approvers by email.
"""
    model = _as_model(_invoke_tool(sample))

    # Should infer at least one feature from imperative lines
    assert any(isinstance(f, Feature) and f.title for f in model.features)

    # Should infer stakeholders from role words in free text
    roles = {s.role.lower() for s in model.stakeholders}
    assert "admin" in roles or "security" in roles or "user" in roles

def test_success_metrics_extraction_simple_tokens():
    sample = """
Success:
- 80% of approvals handled in-app
- Median response under 30min
- Handle 200 rps during spikes
"""
    model = _as_model(_invoke_tool(sample))
    metrics = [m.metric for m in model.success_criteria if m.metric]
    # Expect tokens like "80%", "30min", "200 rps"
    assert any(m == "80%" for m in metrics)
    assert any(m.lower() == "30min" for m in metrics)
    assert any(m.lower() == "200 rps" for m in metrics)

def test_json_serializable_and_deterministic():
    sample = """
Features:
- Allow SSO login
Constraints:
- Must support SAML and OAuth2
"""
    m1 = _as_model(_invoke_tool(sample))
    m2 = _as_model(_invoke_tool(sample))

    assert m1.model_dump() == m2.model_dump()

    dumped = m1.model_dump_json()
    loaded = json.loads(dumped)
    assert isinstance(loaded, dict)
    assert "features" in loaded and isinstance(loaded["features"], list)

