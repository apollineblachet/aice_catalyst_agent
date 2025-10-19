import copy
import pytest

from agent.tools.estimate_complexity_tool import (
    estimate_complexity_tool,
    ComplexityInput,
    ComplexityLabel,
)

from pydantic import ValidationError


def _invoke(payload: dict) -> dict:
    """
    The tool is decorated; call .invoke if available (LangChain),
    else call it like a normal function. This keeps tests portable.
    """
    fn = getattr(estimate_complexity_tool, "invoke", None)
    return fn(payload) if callable(fn) else estimate_complexity_tool(payload)


def test_minimal_input_runs_and_shapes_output():
    payload = {
        "text": "Add a simple feedback widget and store entries."
    }
    out = _invoke(payload)

    # Required top-level keys
    for k in ["label", "estimated_days", "confidence", "drivers", "notable_risks", "notes"]:
        assert k in out, f"missing key {k} in output"

    # Label sanity
    assert out["label"] in {e.value for e in ComplexityLabel}

    # Estimated days sanity
    days = out["estimated_days"]
    assert isinstance(days, (list, tuple)) and len(days) == 2
    assert days[0] > 0 and days[1] >= days[0]

    # Drivers and risks are lists
    assert isinstance(out["drivers"], list)
    assert isinstance(out["notable_risks"], list)


def test_structured_input_affects_estimate_and_label():
    base_text = (
        "Add a Feedback widget (React) for logged-in users with "
        "category/title/description/upload. Create an internal triage view "
        "with filters by category/status and ability to change status. "
        "Store in Postgres (simple schema). Only admins access triage. "
        "First contentful interaction < 1s. Keyboard navigable with aria labels."
    )

    # Unstructured (let the tool infer)
    unstructured = {"text": base_text}
    out_unstructured = _invoke(unstructured)

    # Structured hints (should generally increase points -> widen days and/or label)
    structured = {
        "feature_name": "Feedback Inbox MVP",
        "text": base_text,
        "num_screens": 2,
        "backend_endpoints": 2,
        "touches_datamodel": True,
        "roles_or_auth": True,
        "test_rigor": "moderate",
        "perf_sla_seconds": 1.0,
    }
    out_structured = _invoke(structured)

    # Both should be valid
    for out in (out_unstructured, out_structured):
        assert out["label"] in {e.value for e in ComplexityLabel}
        assert out["estimated_days"][0] > 0

    # Structured should not be *smaller* than unstructured in most realistic cases
    assert out_structured["estimated_days"][1] >= out_unstructured["estimated_days"][1]


def test_confidence_drops_with_unknowns_and_integrations():
    payload_low_conf = {
        "text": "Integrate payment provider and webhooks; several unknowns remain.",
        "external_integrations": ["Stripe"],
        "unknowns_count": 3,
    }
    out = _invoke(payload_low_conf)
    assert out["confidence"] == "low"


def test_external_integrations_add_risk_entry():
    payload = {
        "text": "Call external API and process responses.",
        "external_integrations": ["SomeAPI"],
    }
    out = _invoke(payload)
    risks_texts = " | ".join(r["risk"] for r in out["notable_risks"])
    assert "External API" in risks_texts or "rate limit" in risks_texts


def test_validation_negative_values_raise():
    with pytest.raises(ValidationError):
        _invoke({"text": "anything", "num_screens": -1})
    with pytest.raises(ValidationError):
        _invoke({"text": "anything", "backend_endpoints": -2})
    with pytest.raises(ValidationError):
        _invoke({"text": "anything", "concurrency_target": -10})


def test_deterministic_results_for_same_input():
    payload = {
        "text": "Simple dashboard view with one form; save to DB.",
        "num_screens": 1,
        "backend_endpoints": 1,
        "touches_datamodel": True,
        "test_rigor": "basic",
    }
    out1 = _invoke(copy.deepcopy(payload))
    out2 = _invoke(copy.deepcopy(payload))
    assert out1 == out2, "Tool should be deterministic for identical inputs"


@pytest.mark.parametrize(
    "points_case,expected_label_max",
    [
        (
            # Tiny/small-ish: 1 screen, no DB, no auth, basic tests
            {"text": "One small UI widget.", "num_screens": 1, "test_rigor": "basic"},
            # hi should be <= 2.0 → Small or Tiny
            ComplexityLabel.SMALL.value,
        ),
        (
            # Medium/Large: 2 screens, 2 endpoints, db, auth, moderate tests
            {
                "text": "Two UI screens with API and DB schema; admin access only.",
                "num_screens": 2,
                "backend_endpoints": 2,
                "touches_datamodel": True,
                "roles_or_auth": True,
                "test_rigor": "moderate",
            },
            # hi likely between >2 and <=10 → Medium or Large
            ComplexityLabel.LARGE.value,
        ),
    ],
)
def test_label_monotonicity(points_case, expected_label_max):
    out = _invoke(points_case)
    # sanity: the tool should not exceed XL for these scenarios
    order = ["Tiny", "Small", "Medium", "Large", "XLarge"]
    assert order.index(out["label"]) <= order.index(expected_label_max)
