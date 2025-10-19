

import re
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict
from langchain_core.tools import tool


class FieldRule(BaseModel):
    name: str
    max_length: Optional[int] = None
    enum: Optional[List[str]] = None
    optional: bool = False

class Feature(BaseModel):
    title: str
    description: Optional[str] = None
    fields: List[FieldRule] = []
    endpoints: List[str] = []
    roles_required: List[str] = []

class NonFunctional(BaseModel):
    type: Literal["performance","accessibility","observability"]
    details: Dict

class ParsedOut(BaseModel):
    features: List[Feature]
    nonfunctional: List[NonFunctional]
    constraints: List[str]
    success_criteria: List[str]
    stakeholders: List[str]
    unknowns: List[str]


class Stakeholder(BaseModel):
    role: str = Field(..., description="Role or group, e.g., 'End user', 'Admin', 'Legal'")
    notes: Optional[str] = Field(None, description="Context about responsibilities, pain points, or goals")

class Constraint(BaseModel):
    kind: str = Field(..., description="Type: time, budget, tech, security, compliance, dependency, scope, other")
    text: str = Field(..., description="The raw constraint text")

class SuccessCriterion(BaseModel):
    text: str = Field(..., description="Crisp, ideally measurable outcome")
    metric: Optional[str] = Field(None, description="Extracted metric/threshold if present")

class Feature(BaseModel):
    title: str = Field(..., description="Short feature name")
    description: Optional[str] = Field(None, description="One-line expansion if available")
    notes: Optional[Dict[str, str]] = Field(default_factory=dict, description="Misc parsed hints, e.g., priority")

class RequirementParse(BaseModel):
    features: List[Feature]
    constraints: List[Constraint]
    stakeholders: List[Stakeholder]
    success_criteria: List[SuccessCriterion]
    unknowns: List[str] = Field(default_factory=list, description="Ambiguities the agent should resolve next")

_SECTION_HINTS = {
    "stakeholders": re.compile(r"\b(stakeholders?|audience|users?|personas?)\b", re.I),
    "constraints": re.compile(r"\b(constraints?|limitations?|must|should(?!\s*we)|non[- ]functional|compliance|security)\b", re.I),
    "success": re.compile(r"\b(success|acceptance|criteria|done|definition of done|metrics?)\b", re.I),
    "features": re.compile(r"\b(features?|scope|deliverables|requirements?)\b", re.I),
}
_GENERIC_HEADINGS = {"project", "context", "overview", "background", "goal", "objective"}

_BULLET = re.compile(
    r"^\s*(?:[-*•–—]\s+|\(?\d+\)?[.)]\s+)(.*)$"
)
_METRIC = re.compile(r"(\b\d{1,3}\s?%|\b\d+\s?(ms|s|sec|min|h|req/s|rps|users?)\b|\b\d{1,3}\s?/\s?\d{1,3}\b|\b[A-Z]{2,}\d+\b)", re.I)
_IMPERATIVE = re.compile(r"^(build|create|add|implement|support|allow|enable|integrate|collect|store|display|send|export|import|track|log|notify|authenticate|authorize)\b", re.I)
_ROLE_WORDS = re.compile(
    r"\b("
    r"admins?|administrators?|"
    r"end[- ]?users?|users?|customers?|clients?|"
    r"ops|support|analysts?|pm|product|qa|security|legal|sales|marketing|finance|"
    r"data|engineers?|devops"
    r")\b",
    re.I,
)
_CONSTRAINT_KEYWORDS = {
    "time": re.compile(r"\b(deadline|by\s+\w+ \d{1,2}|today|tomorrow|this (week|month|quarter)|\b\d+\s?(days?|weeks?)\b)\b", re.I),
    "budget": re.compile(r"\b(budget|cost|capex|opex|\$\d)", re.I),
    "tech": re.compile(r"\b(stack|tech|language|framework|database|cloud|on[- ]prem|azure|aws|gcp|postgres|mysql|react|node|python|java|kotlin|swift)\b", re.I),
    "security": re.compile(r"\b(security|oauth|sso|saml|jwt|encryption|encrypted|secret|pii|rbac|least privilege)\b", re.I),
    "compliance": re.compile(r"\b(gdpr|ccpa|hipaa|soc ?2|iso\s?27001|pci[- ]?dss|accessibility|wcag)\b", re.I),
    "dependency": re.compile(r"\b(depends on|blocked by|requires|integration with)\b", re.I),
    "scope": re.compile(r"\b(out of scope|not required|exclude|mvp only)\b", re.I),
}

def _split_lines(text: str) -> List[str]:
    # Normalize bullets and keep non-empty lines
    text = text.replace("\u2022", "•")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]

def _sectionize(lines: List[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"features": [], "constraints": [], "success": [], "stakeholders": [], "other": []}
    current = "other"
    for ln in lines:
        header_hit = None
        for name, pat in _SECTION_HINTS.items():
            if pat.search(ln) and (ln.isupper() or ln.lower().endswith(":") or len(ln.split()) <= 5):
                header_hit = name
                break
        if header_hit:
            current = header_hit if header_hit in sections else "other"
            continue
        sections[current].append(ln)
    return sections

def _collect_bullets(candidates: List[str]) -> List[str]:
    out = []
    for ln in candidates:
        m = _BULLET.match(ln)
        out.append(m.group(1).strip() if m else ln)
    return out

def _extract_features(lines: List[str]) -> List[Feature]:
    feats: List[Feature] = []
    for ln in _collect_bullets(lines):
        lower = ln.lower().strip(" :")
        if lower in _GENERIC_HEADINGS:
            continue
        if _IMPERATIVE.search(ln) or len(ln.split()) <= 12 or "feature" in lower:
            title = re.sub(r"[:.]\s*$", "", ln)
            feats.append(Feature(title=title, description=None))
    if not feats and lines:
        condensed = " ".join(l for l in lines if l.lower().strip(" :") not in _GENERIC_HEADINGS)
        if condensed:
            feats.append(Feature(title=condensed[:80] + "…" if len(condensed) > 80 else condensed))
    return feats

def _extract_constraints(lines: List[str]) -> List[Constraint]:
    out: List[Constraint] = []
    PRIORITY = ["compliance", "security", "time", "tech", "dependency", "scope", "budget", "other"]
    for ln in _collect_bullets(lines):
        kinds = []
        for kind, pat in _CONSTRAINT_KEYWORDS.items():
            if pat.search(ln):
                kinds.append(kind)
        if kinds:
            # choose by explicit priority (compliance outranks security, etc.)
            kind = next((k for k in PRIORITY if k in kinds), kinds[0])
        else:
            kind = "constraints" if re.search(r"\bmust|should\b", ln, re.I) else "other"
        out.append(Constraint(kind=kind, text=ln))
    return out

def _extract_stakeholders(lines: List[str]) -> List[Stakeholder]:
    out: List[Stakeholder] = []
    for ln in _collect_bullets(lines):
        if _ROLE_WORDS.search(ln):
            parts = re.split(r"[:\-–]\s+", ln, maxsplit=1)
            if len(parts) > 1:
                left = parts[0].strip()
                # accept "Role: notes" only if LEFT matches a role word
                if _ROLE_WORDS.search(left):
                    role = left.capitalize()
                    notes = parts[1].strip() or None
                    out.append(Stakeholder(role=role, notes=notes))
                    continue
            # Otherwise extract each role word present (Users, Admin, Security…)
            roles = sorted(set(w.capitalize() for w in _ROLE_WORDS.findall(ln)))
            for role in roles:
                out.append(Stakeholder(role=role, notes=None))
    if not out and lines:
        joined = " ".join(lines)
        candidates = set(w.capitalize() for w in _ROLE_WORDS.findall(joined))
        for role in sorted(candidates):
            out.append(Stakeholder(role=role, notes=None))
    return out



def _extract_success(lines: List[str]) -> List[SuccessCriterion]:
    out: List[SuccessCriterion] = []
    for ln in _collect_bullets(lines):
        m = _METRIC.search(ln)
        metric = m.group(0) if m else None
        out.append(SuccessCriterion(text=ln, metric=metric))
    return out


def _guess_unknowns(text: str) -> List[str]:
    unknowns = []

    # Broad timeline signals
    time_pattern = re.compile(
        r"(deadline|due date|deliver by|by\s+\w+\s+\d{1,2}|by\s+(?:eod|eom|eoy)|"
        r"within\s+\d+(?:[–-]| to )?\d*\s?(?:days?|weeks?|months?|hours?|h|w|d)|"
        r"\bin\s+\d+(?:[–-]| to )?\d*\s?(?:days?|weeks?|months?)\b|"
        r"\bmvp\s+within\b|\bq[1-4]\b)",
        re.I,
    )
    if not time_pattern.search(text):
        unknowns.append("Missing timeline or deadline.")

    if not re.search(r"\b(auth|login|sso|oauth|jwt|saml|rbac)\b", text, re.I):
        unknowns.append("Unclear auth/authorization requirements.")

    if not re.search(r"\b(db|database|postgres|mysql|sqlite|dynamo|mongo|storage)\b", text, re.I):
        unknowns.append("Data storage/DB choice unspecified.")

    # Treat NEGATED mentions as absence: "no KPIs", "no mention of KPIs", "without metrics", "lack of success criteria"
    negated_success = re.search(
        r"\b(?:no(?:\s+mention\s+of)?|without|lack(?:ing)?(?:\s+of)?)\b[^.:\n]*\b"
        r"(?:kpis?|metrics?|acceptance|success|criteria|definition of done|done)\b",
        text,
        re.I,
    )
    has_success_terms = re.search(
        r"\b(acceptance|success|done|criteria|kpis?|metrics?)\b",
        text,
        re.I,
    )
    if negated_success or not has_success_terms:
        unknowns.append("Success criteria not explicitly stated.")

    return unknowns


class ParseRequirementsInput(BaseModel):
    raw_text: str = Field(..., description="The raw requirement text to parse")

@tool("parse_requirements_tool", args_schema=ParseRequirementsInput, return_direct=False)
def parse_requirements_tool(raw_text: str) -> RequirementParse:  # type: ignore[override]
    """
    Extract features, constraints, stakeholders, and success criteria from raw text.
    Returns a RequirementParse Pydantic object (JSON-serializable).
    """
    if not raw_text or not raw_text.strip():
        return RequirementParse(features=[], constraints=[], stakeholders=[], success_criteria=[], unknowns=["Empty requirement text."])
    lines = _split_lines(raw_text)
    sections = _sectionize(lines)

    features = _extract_features(sections.get("features", []) or sections.get("other", []))
    constraints = _extract_constraints(sections.get("constraints", []))
    stakeholders = _extract_stakeholders(sections.get("stakeholders", []))
    success = _extract_success(sections.get("success", []))

    # If stakeholders missing, try to infer from "other"
    if not stakeholders:
        stakeholders = _extract_stakeholders(sections.get("other", []))

    unknowns = _guess_unknowns(raw_text)

    return RequirementParse(
        features=features,
        constraints=constraints,
        stakeholders=stakeholders,
        success_criteria=success,
        unknowns=unknowns,
    )


