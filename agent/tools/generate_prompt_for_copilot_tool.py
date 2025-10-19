# tools/generate_prompt_for_copilot_tool.py
from __future__ import annotations
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
import textwrap


class GeneratePromptInput(BaseModel):
    task_title: str = Field(..., description="Short, action-oriented task title.")
    task_description: str = Field(..., description="1–3 sentences describing what to build/change.")
    acceptance_criteria: List[str] = Field(
        ..., description="List of testable criteria. Prefer Given/When/Then, but plain bullets are fine."
    )
    constraints: Optional[List[str]] = Field(
        default=None, description="Non-functional constraints or guardrails (perf, security, style, etc.)."
    )
    tech_stack: Optional[str] = Field(
        default=None, description="Key tech/context (e.g., 'React + FastAPI', 'Node/Express + Postgres')."
    )
    repo_context: Optional[Dict[str, str]] = Field(
        default=None,
        description="Key repo pointers like paths/modules, e.g. {'frontend':'/app/web','api':'/services/api'}",
    )
    files_to_edit: Optional[List[str]] = Field(
        default=None,
        description="If we already know likely targets (paths/globs), list them to focus the assistant."
    )
    done_definition: Optional[str] = Field(
        default=None,
        description="Optional 'Definition of Done' (e.g., 'tests pass, lint clean, docs updated')."
    )

    # ---- v2-style validators ----

    @field_validator("task_title", "task_description", mode="after")
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("acceptance_criteria", mode="after")
    def _non_empty_criteria(cls, v: List[str]) -> List[str]:
        cleaned = [c.strip() for c in v if c and str(c).strip()]
        if not cleaned:
            raise ValueError("acceptance_criteria must include at least one non-empty item.")
        return cleaned[:10]  # deterministic cap

    @field_validator("constraints", "files_to_edit", mode="before")
    def _norm_list(cls, v):
        if v is None:
            return None
        return [str(x).strip() for x in v if str(x).strip()]

class GeneratePromptOutput(BaseModel):
    prompt: str
    meta: Dict[str, str]


SECTION_ORDER = (
    "Goal",
    "Context",
    "Constraints",
    "Acceptance criteria",
    "Codebase pointers",
    "Focus files",
    "Done when",
    "Assistant instructions",
)


def _bullet(lines: List[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _wrap(s: str) -> str:
    return "\n".join(textwrap.fill(line, width=96) if "```" not in line else line for line in s.splitlines())


def generate_prompt_for_copilot_tool(raw: GeneratePromptInput) -> GeneratePromptOutput:
    """
    Turn a task + acceptance criteria into a concise, high-signal prompt for GitHub Copilot or Claude.
    Deterministic, sectioned, and guardrailed to keep signal high.
    """
    sections: Dict[str, str] = {}

    sections["Goal"] = _wrap(f"{raw.task_title}. {raw.task_description}")

    ctx_bits: List[str] = []
    if raw.tech_stack:
        ctx_bits.append(f"Tech: {raw.tech_stack}")
    if raw.repo_context:
        # Stable ordering by key
        kv = ", ".join(f"{k}={v}" for k, v in sorted(raw.repo_context.items()))
        ctx_bits.append(f"Repo context: {kv}")
    if ctx_bits:
        sections["Context"] = _wrap(" · ".join(ctx_bits))

    if raw.constraints:
        sections["Constraints"] = _bullet(raw.constraints)

    # Ensure criteria are crisp; leave as bullets to avoid extra tokens
    sections["Acceptance criteria"] = _bullet(raw.acceptance_criteria)

    if raw.repo_context:
        sections["Codebase pointers"] = _wrap("Prefer working within the existing modules/paths above. "
                                              "Reuse utilities and follow local patterns.")

    if raw.files_to_edit:
        sections["Focus files"] = _bullet(raw.files_to_edit)

    done_when = raw.done_definition or "All acceptance criteria satisfied; tests & lints pass."
    sections["Done when"] = done_when

    sections["Assistant instructions"] = _bullet([
        "Make minimal, focused changes; follow existing conventions.",
        "Explain risky changes briefly in comments near the diff.",
        "Output: patch/diff and updated files; include any new tests.",
    ])

    # Assemble in fixed, human-friendly order
    ordered = []
    for name in SECTION_ORDER:
        if name in sections and sections[name].strip():
            ordered.append(f"### {name}\n{sections[name].strip()}")

    prompt = "\n\n".join(ordered).strip()

    meta = {
        "sections": ", ".join([name for name in SECTION_ORDER if name in sections]),
        "chars": str(len(prompt)),
        "criteria_count": str(len(raw.acceptance_criteria)),
    }

    return GeneratePromptOutput(prompt=prompt, meta=meta)
