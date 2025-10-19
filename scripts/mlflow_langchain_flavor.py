from __future__ import annotations
import argparse
import mlflow
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from agent.agent import build_agent, system_message

def make_runnable():
    def _predict(inputs: dict):
        text = inputs["raw_text"]
        agent = build_agent()  
        result = agent.invoke({"messages": [system_message(), HumanMessage(content=text)]})
        raw_text_out = result["messages"][-1].content
        plan = result["structured_response"]
        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
        return {"raw_text": raw_text_out, "plan": plan_dict}
    return RunnableLambda(_predict)

def log_model(name: str = "model"):
    with mlflow.start_run(run_name="agent_langchain_flavor") as run:
        # Minimal examples for UI + schema
        input_ex = {"raw_text": "Example requirement text"}
        output_ex = {"raw_text": "...", "plan": {"objective": "...", "phases": []}}

        try:
            mlflow.langchain.log_model(
                make_runnable(),
                name=name,
                input_example=input_ex,
                signature=mlflow.models.infer_signature(input_ex, output_ex),
            )
        except TypeError:
            mlflow.langchain.log_model(
                make_runnable(),
                artifact_path=name,
                input_example=input_ex,
            )
        print("LangChain-flavor model run:", run.info.run_id)


def infer(model_uri: str, text: str):
    chain = mlflow.langchain.load_model(model_uri)
    return chain.invoke({"raw_text": text})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["log","infer"], required=True)
    ap.add_argument("--model-uri")
    ap.add_argument("--text")
    args = ap.parse_args()

    if args.mode == "log":
        log_model()
    else:
        if not args.model_uri or not args.text:
            raise SystemExit("--model-uri and --text are required for infer")
        print(infer(args.model_uri, args.text))

if __name__ == "__main__":
    main()
