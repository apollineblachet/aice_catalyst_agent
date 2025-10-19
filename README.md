# AICE Catalyst Agent Challenge

This project implements an **AI planning agent** that transforms unstructured business requirements into a **structured, phased development plan**.  
Built for the *AICE Catalyst Agent Challenge*, it demonstrates how an autonomous LangGraph workflow can extract structure, estimate effort, define acceptance criteria, and generate concise prompts for code assistants like GitHub Copilot or Claude.

---

## Overview

The agent reads a short project requirement and outputs:
- Phases, tasks, and dependencies  
- Complexity estimates and risks  
- Acceptance criteria (Given/When/Then format)  
- Developer prompts for AI code assistants  

It uses LangGraph’s `create_react_agent` to coordinate reasoning and tool use through a ReAct-style workflow.
---

## Implemented Tools

| Tool | Purpose |
|------|----------|
| `parse_requirements_tool` | Extracts project features, constraints, and goals. |
| `estimate_complexity_tool` | Labels complexity, estimates effort, and lists risks. |
| `generate_tasks_tool` | Breaks features into granular tasks. |
| `create_acceptance_criteria_tool` | Generates testable acceptance criteria. |
| `generate_prompt_for_copilot_tool` | Creates concise developer prompts. |
| `detect_dependencies_tool` | Identifies task dependencies. |

---

## Setup

### Requirements
Python 3.12+  
Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Agent

Run the agent on an example prompt:

```bash
python -m scripts.run_agent \
  --input examples/inputs/prompt.md \
  --name prompt \
  --stream
```

Use `--stream` to see intermediate reasoning steps.

## MLflow Integration

Log the final agent as a LangChain model in MLflow:
```bash
python -m scripts.mlflow_langchain_flavor --mode log
```

Reload and use the logged model for inference:
```bash
python -m scripts.mlflow_langchain_flavor --mode infer \
  --model-uri runs:/<ID>/model \
  --text "Project: Minimal Customer Feedback Inbox ..."
```

## Run Tests

```bash
pytest -q
```

## Project Structure

```
aice_catalyst_agent/
├── agent/
│   ├── tools/
│   ├── schema.py
│   ├── runner.py
│   └── ...
├── scripts/
│   ├── run_agent.py
│   ├── mlflow_langchain_flavor.py
│   └── ...
├── examples/inputs/
│   ├── low_specificity_prompt.md
│   ├── medium_specificity_prompt.md
│   ├── high_specificity_prompt.md
│   └── prompt1.md
└── tests/
```