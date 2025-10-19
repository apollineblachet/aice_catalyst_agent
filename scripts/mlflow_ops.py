from __future__ import annotations
import argparse
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel

from agent.agent import build_agent, system_message
from langchain_core.messages import HumanMessage


class AgentPyFunc(PythonModel):
    def load_context(self, context):
        self._agent = build_agent()

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        for text in model_input["raw_text"].tolist():
            result = self._agent.invoke({
                "messages": [system_message(), HumanMessage(content=text)]
            })
            raw_text_out = result["messages"][-1].content
            plan = result["structured_response"]
            plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
            outputs.append({"raw_text": raw_text_out, "plan": plan_dict})
        return pd.DataFrame(outputs)


def log_model(run_name: str = "agent_run", name: str = "model"):
    model = AgentPyFunc()
    input_ex = pd.DataFrame({"raw_text": ["Example requirement text"]})
    output_ex = pd.DataFrame([{"raw_text": "...", "plan": {"objective": "...", "phases": []}}])

    with mlflow.start_run(run_name=run_name) as run:
        try:
            mlflow.pyfunc.log_model(
                name=name,
                python_model=model,
                input_example=input_ex,
                signature=mlflow.models.infer_signature(input_ex, output_ex),
            )
        except TypeError:
            mlflow.pyfunc.log_model(
                artifact_path=name,
                python_model=model,
                input_example=input_ex,
                signature=mlflow.models.infer_signature(input_ex, output_ex),
            )
        print("Model logged to run:", run.info.run_id)


def load_and_infer(model_uri: str, raw_text: str):
    model = mlflow.pyfunc.load_model(model_uri)
    df = pd.DataFrame({"raw_text": [raw_text]})
    return model.predict(df).to_dict(orient="records")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["log", "infer"], required=True)
    ap.add_argument("--model-uri", help="runs:/<run_id>/model")
    ap.add_argument("--text", help="Raw requirement text (for infer)")
    args = ap.parse_args()

    if args.mode == "log":
        log_model()
    else:
        if not args.model_uri or not args.text:
            raise SystemExit("--model-uri and --text are required for infer mode")
        out = load_and_infer(args.model_uri, args.text)
        print(out)


if __name__ == "__main__":
    main()
