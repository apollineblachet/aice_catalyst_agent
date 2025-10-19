from __future__ import annotations
import os
from pathlib import Path

def read_text(maybe_path: str) -> str:
    if os.path.exists(maybe_path) and os.path.isfile(maybe_path):
        return Path(maybe_path).read_text(encoding="utf-8")
    return maybe_path

def write_text(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return str(p)

def save_plan_json(plan, out_path: str) -> str:
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if hasattr(plan, "model_dump_json"):
        payload = plan.model_dump_json(indent=2, ensure_ascii=False)
    else:
        payload = plan.json(indent=2, ensure_ascii=False)
    Path(out_path).write_text(payload, encoding="utf-8")
    return out_path

def as_dict(obj):
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    # Already a dict
    if isinstance(obj, dict):
        return obj
    # Fallback: try to read __dict__ 
    return getattr(obj, "__dict__", {})


def call_tool(tool, payload: dict):
    """
    Call a tool that may be:
    - LangChain Runnable/Tool: .invoke(payload)
    - Plain function with keyword args: tool(**payload)
    - Plain function expecting a single dict: tool(payload)
    - (Optional) legacy BaseTool: .run(payload)
    """
    if hasattr(tool, "invoke"):
        return tool.invoke(payload)

    if callable(tool):
        try:
            return tool(**payload)     
        except TypeError:
            try:
                return tool(payload)    
            except TypeError:
                pass  

    if hasattr(tool, "run"):
        return tool.run(payload)

    raise TypeError(f"Unsupported tool type: {type(tool)}")



def as_list_of_dicts(objs):
    if objs is None:
        return []
    out = []
    for x in objs:
        out.append(as_dict(x) if not isinstance(x, dict) else x)
    return out


def build_requirement_brief(state) -> str:
    """Compose a compact, deterministic requirement string for tools that expect raw_text."""
    lines = []
    obj = state.get("objective") or ""
    if obj:
        lines.append(f"Objective: {obj}")

    feats = state.get("features") or []
    if feats:
        lines.append("Features:")
        for f in feats:
            lines.append(f"- {f}")

    ass = state.get("assumptions") or []
    if ass:
        lines.append("Assumptions:")
        for a in ass:
            lines.append(f"- {a}")

    cons = state.get("constraints") or []
    if cons:
        lines.append("Constraints:")
        for c in cons:
            lines.append(f"- {c}")

    return "\n".join(lines).strip()



