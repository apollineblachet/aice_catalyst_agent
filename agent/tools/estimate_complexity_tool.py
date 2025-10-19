from __future__ import annotations
from typing import List, Optional, Literal, Tuple
from enum import Enum
from dataclasses import dataclass
import math
import re

from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool


# ---------- Schemas ----------

class ComplexityLabel(str, Enum):
    TINY = "Tiny"        # spike / trivial change
    SMALL = "Small"      # 1–2 days
    MEDIUM = "Medium"    # 3–5 days
    LARGE = "Large"      # 6–10 days
    XLARGE = "XLarge"    # 10–15 days

class RiskItem(BaseModel):
    risk: str
    why_it_matters: Optional[str] = None
    mitigation: Optional[str] = None

class ComplexityInput(BaseModel):
    """Primary input. Pass raw text or structured fields (or both)."""
    feature_name: Optional[str] = Field(default=None, description="Short feature title.")
    text: str = Field(..., description="Raw requirement text for the feature (100–300 words ideal).")
    num_screens: Optional[int] = Field(default=None, description="How many distinct UI screens/forms/views?")
    backend_endpoints: Optional[int] = Field(default=None, description="Approximate count of new/changed API endpoints.")
    touches_datamodel: Optional[bool] = Field(default=None, description="Does this require DB schema changes?")
    external_integrations: Optional[List[str]] = Field(default=None, description="Names of external services/APIs used.")
    roles_or_auth: Optional[bool] = Field(default=None, description="Does RBAC or auth gating matter?")
    perf_sla_seconds: Optional[float] = Field(default=None, description="Explicit performance SLA in seconds, if any.")
    concurrency_target: Optional[int] = Field(default=None, description="Expected concurrent users/jobs.")
    test_rigor: Optional[Literal["basic","moderate","high"]] = Field(default=None, description="Testing expectations.")
    unknowns_count: Optional[int] = Field(default=None, description="Number of open questions/unknowns called out.")

    @field_validator("num_screens", "backend_endpoints", "concurrency_target", "unknowns_count")
    def non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("Value must be non-negative")
        return v

class ComplexityOutput(BaseModel):
    label: ComplexityLabel
    estimated_days: Tuple[float, float]  # (min_days, max_days) for 1 dev
    confidence: Literal["low","medium","high"]
    drivers: List[str]                   # top factors that pushed the estimate
    notable_risks: List[RiskItem]
    notes: Optional[str] = None          # any extra commentary

@dataclass
class Score:
    points: float = 0.0
    drivers: List[str] = None

    def add(self, pts: float, why: str):
        self.points += pts
        if self.drivers is None:
            self.drivers = []
        self.drivers.append(f"+{pts:g} • {why}")

def _kw_present(text: str, *patterns: str) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)

def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))

def _suggest_risks(text: str, inp: ComplexityInput) -> List[RiskItem]:
    risks: List[RiskItem] = []

    if inp.external_integrations or _kw_present(text, r"\b(api|webhook|oauth|stripe|slack|whisper|assemblyai)\b"):
        risks.append(RiskItem(
            risk="External API instability or rate limits",
            why_it_matters="3rd-party latency/errors can block critical paths",
            mitigation="Add retries/backoff, circuit breakers, and sandbox keys; mock in tests"
        ))

    if inp.roles_or_auth or _kw_present(text, r"\b(role|rbac|permission|admin only|auth|sso|oauth)\b"):
        risks.append(RiskItem(
            risk="Authorization gaps",
            why_it_matters="Data leakage or privilege escalation",
            mitigation="Centralize checks; add negative tests for non-privileged users"
        ))

    if inp.touches_datamodel or _kw_present(text, r"\b(schema|migration|postgres|index|ddl)\b"):
        risks.append(RiskItem(
            risk="Data migration or schema mismatch",
            why_it_matters="Downtime or broken queries in production",
            mitigation="Write reversible migrations; migrate in phases; add DB constraints"
        ))

    if (inp.perf_sla_seconds and inp.perf_sla_seconds <= 1.0) or _kw_present(text, r"\b(<\s*1\s*s|p\d+ latency|realtime|low latency)\b"):
        risks.append(RiskItem(
            risk="Tight performance SLA",
            why_it_matters="May require caching/queueing and careful IO",
            mitigation="Profile early; add async/queue; cache hot paths"
        ))

    if (inp.concurrency_target and inp.concurrency_target >= 10) or _kw_present(text, r"\bconcurrent|simultaneous|scale\b"):
        risks.append(RiskItem(
            risk="Concurrency/hotspot contention",
            why_it_matters="Race conditions and throughput bottlenecks",
            mitigation="Idempotency, locks or queues, load tests"
        ))

    if _kw_present(text, r"\bupload|pdf|file\b"):
        risks.append(RiskItem(
            risk="Large file handling / storage costs",
            why_it_matters="Time-outs and storage bloat",
            mitigation="Set size limits, stream uploads, virus scan, lifecycle policies"
        ))

    if _kw_present(text, r"\baccessibility|aria|wcag|a11y\b"):
        risks.append(RiskItem(
            risk="Accessibility non-compliance",
            why_it_matters="Usability + potential legal risk",
            mitigation="Keyboard nav, semantic labels, axe checks in CI"
        ))

    if (inp.unknowns_count or 0) >= 2 or _kw_present(text, r"\bunknown|tbd|assume|clarify\b"):
        risks.append(RiskItem(
            risk="Ambiguities/assumptions",
            why_it_matters="Scope creep and rework",
            mitigation="Time-box discovery; document assumptions; confirm with stakeholders"
        ))

    return risks[:6]  # keep it concise

def _score_feature(inp: ComplexityInput) -> Score:
    text = inp.text
    s = Score(0.0, [])

    # Base
    s.add(1.0, "Non-trivial baseline")

    # UI surfaces
    nscreens = inp.num_screens if inp.num_screens is not None else max(
        _count(r"\b(form|screen|page|view|dashboard|widget)\b", text), 1 if _kw_present(text, r"\bui|frontend|react|form\b") else 0
    )
    s.add(0.7 * nscreens, f"{nscreens} UI screen(s)")

    # Backend endpoints
    nendp = inp.backend_endpoints if inp.backend_endpoints is not None else (
        _count(r"\bendpoint|api\b", text)
    )
    if nendp:
        s.add(0.6 * nendp, f"{nendp} backend endpoint(s)")

    # Data model / DB
    if bool(inp.touches_datamodel) or _kw_present(text, r"\b(schema|table|migration|postgres|sql)\b"):
        s.add(1.2, "DB schema changes")

    # Auth / roles
    if bool(inp.roles_or_auth) or _kw_present(text, r"\b(role|rbac|admin only|permission|auth)\b"):
        s.add(0.8, "Authorization complexity")

    # External integrations
    ext = inp.external_integrations or []
    if ext:
        s.add(0.8 + 0.4 * len(ext), f"External integrations: {', '.join(ext)}")
    elif _kw_present(text, r"\bwhisper|assemblyai|stripe|slack|oauth|webhook|s3|gcs\b"):
        s.add(0.8, "External integration (keyword detected)")

    # Performance SLA
    if inp.perf_sla_seconds is not None:
        if inp.perf_sla_seconds <= 1.0:
            s.add(1.0, "Tight perf SLA ≤ 1s")
        elif inp.perf_sla_seconds <= 5.0:
            s.add(0.5, "Moderate perf SLA ≤ 5s")
    elif _kw_present(text, r"\b<\s*1\s*s|fast|realtime|snappy\b"):
        s.add(0.6, "Implied performance expectation")

    # Concurrency / throughput
    if inp.concurrency_target is not None:
        if inp.concurrency_target >= 50:
            s.add(1.0, "High concurrency target")
        elif inp.concurrency_target >= 10:
            s.add(0.6, "Moderate concurrency target")
    elif _kw_present(text, r"\bconcurrent|simultaneous|scale\b"):
        s.add(0.4, "Concurrency considerations")

    # Test rigor
    tr = inp.test_rigor or ("high" if _kw_present(text, r"\bunit test|coverage|acceptance criteria\b") else "moderate")
    s.add({"basic": 0.2, "moderate": 0.4, "high": 0.7}[tr], f"Test rigor: {tr}")

    # Unknowns
    unk = inp.unknowns_count if inp.unknowns_count is not None else _count(r"\bunknown|tbd|clarify|assumption\b", text)
    if unk:
        s.add(min(0.3 * unk, 1.2), f"{unk} open question(s)")

    return s

def _points_to_days(points: float) -> Tuple[float, float]:
    """
    Map total points to a range of developer-days (1 developer).
    Tuned for small MVP-scale features.
    """
    # Smooth curve: base 0.5 day + 0.6 * points, ±20%
    center = 0.5 + 0.6 * points
    lo = max(0.25, round(center * 0.8, 1))
    hi = round(center * 1.2, 1)
    return (lo, hi)

def _label_from_days(days: Tuple[float, float]) -> ComplexityLabel:
    _, hi = days
    if hi <= 1.0:
        return ComplexityLabel.TINY
    if hi <= 2.0:
        return ComplexityLabel.SMALL
    if hi <= 5.0:
        return ComplexityLabel.MEDIUM
    if hi <= 10.0:
        return ComplexityLabel.LARGE
    return ComplexityLabel.XLARGE

def _confidence(inp: ComplexityInput) -> Literal["low","medium","high"]:
    # Lower confidence when unknowns are high or external APIs are present
    unknowns = inp.unknowns_count or 0
    has_ext = bool(inp.external_integrations)
    if unknowns >= 3 or (unknowns >= 2 and has_ext):
        return "low"
    if unknowns == 0 and not has_ext:
        return "high"
    return "medium"

@tool("estimate_complexity_tool", args_schema=ComplexityInput, return_direct=False)
def estimate_complexity_tool(**payload) -> dict:
    """
    For a feature, return a complexity label, estimated days, and notable risks to guide planning.
    Accepts top-level fields matching ComplexityInput (no 'payload' wrapper).
    """
    inp = ComplexityInput(**payload)
    sc = _score_feature(inp)
    days = _points_to_days(sc.points)
    label = _label_from_days(days)
    risks = _suggest_risks(inp.text, inp)
    conf = _confidence(inp)

    out = ComplexityOutput(
        label=label,
        estimated_days=days,
        confidence=conf,
        drivers=sc.drivers or [],
        notable_risks=risks,
        notes="Estimate assumes 1 developer, typical web stack, and normal review cadence."
    )
    # Return plain dict for easy JSON serialization in agents
    return out.model_dump()

