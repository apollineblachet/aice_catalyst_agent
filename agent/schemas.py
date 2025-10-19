from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field

class RiskItem(BaseModel):
    risk: str
    mitigation: str

class AcceptanceCriterion(BaseModel):
    given: str
    when: str
    then: str

class TestCase(BaseModel):
    name: str
    description: str

class TestSuite(BaseModel):
    unit_tests: List[TestCase] = Field(default_factory=list)
    integration_tests: List[TestCase] = Field(default_factory=list)

class Task(BaseModel):
    id: str = Field(..., description="Stable id for dependency references, e.g. T-001")
    title: str
    description: str
    estimate_days: float = Field(..., ge=0.0)
    complexity: Literal["XS", "S", "M", "L", "XL"]
    dependencies: List[str] = Field(default_factory=list)
    priority: Literal["P0", "P1", "P2"] = "P1"
    acceptance_criteria: List[AcceptanceCriterion] = Field(default_factory=list)
    test_suite: TestSuite = Field(default_factory=TestSuite)        # ← NEW
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
