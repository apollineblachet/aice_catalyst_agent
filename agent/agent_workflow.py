from __future__ import annotations
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from .llm import build_llm
from .schemas import DevelopmentPlan, Phase, Task, RiskItem, AcceptanceCriterion
from .utils import recompute_plan_estimate
from .io_utils import as_dict, call_tool, as_list_of_dicts, build_requirement_brief

# --- Tools
from .tools.parse_requirements_tool import parse_requirements_tool
from .tools.estimate_complexity_tool import estimate_complexity_tool
from .tools.generate_tasks_tool import generate_tasks_tool
from .tools.detect_dependencies_tool import detect_dependencies_tool
from .tools.create_acceptance_criteria_tool import create_acceptance_criteria_tool
from .tools.generate_prompt_for_copilot_tool import generate_prompt_for_copilot_tool

SYSTEM_PROMPT = (
    "You are a planning agent that transforms raw business requirements into a structured, "
    "phased development plan. Extract structure, estimate complexity per feature, propose tasks, "
    "compute dependencies, add acceptance criteria, and generate concise prompts for code assistants. "
    "Finally, emit a clean JSON plan with phases, task estimates, and totals."
)

def system_message() -> SystemMessage:
    return SystemMessage(content=SYSTEM_PROMPT)

# ---------------------------
# Deterministic Agent State
# ---------------------------
class AgentState(TypedDict, total=False):
    raw_text: str                             # incoming requirement
    # Parsed
    objective: str
    assumptions: List[str]
    unknowns_or_questions: List[str]
    features: List[str]                       # from parse_requirements_tool
    constraints: List[str]
    # Tools’ outputs (intermediate)
    task_suggestions: List[dict]
    tasks: List[Task]
    phases: List[Phase]
    risks: List[RiskItem]
    deliverables: List[str]
    # Final
    plan: DevelopmentPlan
    messages: list                            # for transparency/streaming

# -------------
# Node helpers
# -------------
def _append_msg(state: AgentState, role: str, text: str):
    msgs = state.get("messages", [])
    msgs.append({"role": role, "content": text})
    state["messages"] = msgs

# Node: parse the requirement (tool)
def node_parse(state: AgentState) -> AgentState:
    parsed_obj = call_tool(parse_requirements_tool, {"raw_text": state["raw_text"]})
    parsed = as_dict(parsed_obj)

    state["objective"] = parsed.get("objective", "") or ""
    state["assumptions"] = parsed.get("assumptions") or []
    state["unknowns_or_questions"] = parsed.get("unknowns_or_questions") or []
    state["features"] = parsed.get("features") or []
    state["constraints"] = parsed.get("constraints") or []
    _append_msg(state, "tool", f"parse_requirements_tool: features={len(state['features'])}")
    return state


# Node: identify gaps (LLM-only, deterministic prompt)
def node_identify_gaps(state: AgentState) -> AgentState:
    llm = build_llm()
    prompt = (
        "Given the objective, assumptions, features and constraints below, list any critical missing information "
        "the team would need to start. Keep it concise, bullet points only.\n\n"
        f"Objective:\n{state.get('objective','')}\n\n"
        f"Assumptions:\n{state.get('assumptions',[])}\n\n"
        f"Features:\n{state.get('features',[])}\n\n"
        f"Constraints:\n{state.get('constraints',[])}\n"
    )
    resp = llm.invoke([system_message(), HumanMessage(content=prompt)])
    gaps = [line.strip("-• ").strip() for line in resp.content.splitlines() if line.strip()]
    # Merge with existing unknowns
    merged = list(dict.fromkeys((state.get("unknowns_or_questions") or []) + gaps))
    state["unknowns_or_questions"] = merged
    _append_msg(state, "assistant", f"identified_gaps: {len(merged)}")
    return state

# Node: fill gaps with tool calls (tasks, estimates, deps, AC, prompts)
def node_fill_gaps(state: AgentState) -> AgentState:
    # 1) generate tasks
    brief = build_requirement_brief(state)
    gen = call_tool(generate_tasks_tool, {"raw_text": brief})
    gen = as_dict(gen)
    task_suggestions = as_list_of_dicts(gen.get("tasks", []))

    # 2) estimates
    est = call_tool(estimate_complexity_tool, {"tasks": task_suggestions})
    est = as_dict(est)
    est_tasks = as_list_of_dicts(est.get("tasks", task_suggestions))

    # 3) dependencies
    deps = call_tool(detect_dependencies_tool, {"tasks": est_tasks})
    deps = as_dict(deps)
    dep_tasks = as_list_of_dicts(deps.get("tasks", est_tasks))

    # 4) acceptance criteria
    ac = call_tool(create_acceptance_criteria_tool, {"tasks": dep_tasks})
    ac = as_dict(ac)
    ac_tasks = as_list_of_dicts(ac.get("tasks", dep_tasks))

    # 5) prompts
    prompts = call_tool(generate_prompt_for_copilot_tool, {"tasks": ac_tasks})
    prompts = as_dict(prompts)
    final_tasks = as_list_of_dicts(prompts.get("tasks", ac_tasks))


    # Convert to Pydantic Task models if your tools return dicts
    from .schemas import Task as TaskModel, Phase as PhaseModel, RiskItem as RiskModel, AcceptanceCriterion as AcModel
    tasks: List[TaskModel] = []
    for t in final_tasks:
        ac_list = [AcModel(**ac) if isinstance(ac, dict) else ac for ac in t.get("acceptance_criteria", [])]
        risks = [RiskModel(**r) if isinstance(r, dict) else r for r in t.get("risks", [])]
        tasks.append(TaskModel(
            id=t["id"],
            title=t["title"],
            description=t["description"],
            estimate_days=float(t["estimate_days"]),
            complexity=t["complexity"],
            dependencies=t.get("dependencies", []),
            priority=t.get("priority","P1"),
            acceptance_criteria=ac_list,
            prompt_for_code_assistant=t["prompt_for_code_assistant"],
            deliverables=t.get("deliverables", []),
            risks=risks,
        ))
    # For simplicity group into a single phase “Core”
    phase = PhaseModel(
        name="Core",
        goal="Deliver MVP scope derived from features",
        tasks=tasks,
        estimate_days=0.0,
        risks=[],
    )
    state["phases"] = [phase]
    state["tasks"] = tasks
    state["deliverables"] = sorted({d for t in tasks for d in t.deliverables})
    _append_msg(state, "tool", f"tools: tasks={len(tasks)}")
    return state

# Node: assemble final DevelopmentPlan
def node_assemble_plan(state: AgentState) -> AgentState:
    plan = DevelopmentPlan(
        objective=state["objective"],
        assumptions=state.get("assumptions", []),
        unknowns_or_questions=state.get("unknowns_or_questions", []),
        phases=state["phases"],
        total_estimate_days=0.0,
        risks_and_mitigations=[],   # could be filled by a risk tool later
        deliverables=state.get("deliverables", []),
    )
    # recompute totals from tasks
    recompute_plan_estimate(plan)
    state["plan"] = plan
    _append_msg(state, "assistant", "assembled_plan")
    return state

# -----------
# Build graph
# -----------
def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse", node_parse)
    graph.add_node("identify_gaps", node_identify_gaps)
    graph.add_node("fill_gaps", node_fill_gaps)
    graph.add_node("assemble_plan", node_assemble_plan)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "identify_gaps")
    graph.add_edge("identify_gaps", "fill_gaps")
    graph.add_edge("fill_gaps", "assemble_plan")
    graph.add_edge("assemble_plan", END)

    return graph.compile()  # returns a Runnable

# Backwards-compatible helpers used elsewhere
def build_agent_workflow():
    """Return a runnable that accepts {'messages': [...]} *or* {'raw_text': '...'} and emits the same contract you used before."""
    runnable_graph = build_agent_graph()

    def _shim(inputs):
        # accept your previous calling pattern
        if "messages" in inputs:
            # last HumanMessage is the raw text
            msgs = inputs["messages"]
            last_human = next((m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), "")
            state_in = {"raw_text": last_human}
        else:
            state_in = {"raw_text": inputs["raw_text"]}
        state_out = runnable_graph.invoke(state_in)
        # fabricate the LangChain-like result object you used:
        plan = state_out["plan"]
        return {
            "messages": [system_message(), HumanMessage(content="Structured plan ready.")],
            "structured_response": plan,
        }

    return RunnableLambda(_shim)
