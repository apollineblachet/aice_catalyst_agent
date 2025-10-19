from __future__ import annotations
from typing import List, Literal, Tuple
from pydantic import BaseModel, Field

class RiskItem(BaseModel):
    risk: str
    mitigation: str

class AcceptanceCriterion(BaseModel):
    given: str
    when: str
    then: str

class Task(BaseModel):
    id: str = Field(..., description="Stable id for dependency references, e.g. T-001")
    title: str
    description: str
    estimate_days: float = Field(..., ge=0.0)
    complexity: Literal["XS", "S", "M", "L", "XL"]
    dependencies: List[str] = Field(default_factory=list)
    priority: Literal["P0", "P1", "P2"] = "P1"
    acceptance_criteria: List[AcceptanceCriterion]
    prompt_for_code_assistant: str
    deliverables: List[str] = Field(default_factory=list)
    risks: List[RiskItem] = Field(default_factory=list)

class Phase(BaseModel):
    name: str
    goal: str
    tasks: List[Task]
    estimate_days: float
    risks: List[RiskItem] = Field(default_factory=list)

class DevelopmentPlan(BaseModel):
    objective: str
    assumptions: List[str] = Field(default_factory=list)
    unknowns_or_questions: List[str] = Field(default_factory=list)
    phases: List[Phase]
    total_estimate_days: float
    risks_and_mitigations: List[RiskItem] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)
