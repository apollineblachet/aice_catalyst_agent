from __future__ import annotations
from typing import Tuple, Dict, Any
from langchain_core.messages import HumanMessage
from .agent import build_agent, system_message
from .agent_workflow import build_agent_workflow
from .io_utils import read_text
from .utils import recompute_plan_estimate

def run_agent(input_text_or_path: str, stream: bool = False) -> Tuple[str, Any]:
    """Returns (raw_text, plan_obj)."""
    agent = build_agent()
    raw_req = read_text(input_text_or_path)
    messages = [system_message(), HumanMessage(content=raw_req)]

    if stream:
        for ev in agent.stream({"messages": messages}):
            print(ev)

    result = agent.invoke({"messages": messages})
    raw_text = result["messages"][-1].content
    plan = result["structured_response"]
    recompute_plan_estimate(plan)
    return raw_text, plan
