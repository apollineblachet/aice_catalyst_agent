from __future__ import annotations
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from .llm import build_llm
from .schemas import DevelopmentPlan

from .tools.parse_requirements_tool import parse_requirements_tool
from .tools.estimate_complexity_tool import estimate_complexity_tool
from .tools.generate_tasks_tool import generate_tasks_tool
from .tools.create_acceptance_criteria_tool import create_acceptance_criteria_tool
from .tools.generate_prompt_for_copilot_tool import generate_prompt_for_copilot_tool
from .tools.detect_dependencies_tool import detect_dependencies_tool

SYSTEM_PROMPT = (
    "You are a planning agent that transforms raw business requirements into a structured, "
    "phased development plan. Extract structure, estimate complexity per feature, propose tasks, "
    "compute dependencies, add acceptance criteria, and generate concise prompts for code assistants. "
    "Finally, emit a clean JSON plan with phases, task estimates, and totals."
)

TOOLS = [
    parse_requirements_tool,
    estimate_complexity_tool,
    generate_tasks_tool,
    create_acceptance_criteria_tool,
    generate_prompt_for_copilot_tool,
    detect_dependencies_tool,
]


def build_agent():
    llm = build_llm()
    return create_react_agent(
        llm,
        tools=TOOLS,
        response_format=DevelopmentPlan,
    )

def system_message() -> SystemMessage:
    return SystemMessage(content=SYSTEM_PROMPT)
