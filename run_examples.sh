python -m scripts.run_agent --input examples/inputs/prompt1.md --name prompt1 --stream
python -m scripts.run_agent --input examples/inputs/high_specificity_prompt.md --name high_specificity_prompt --stream
python -m scripts.run_agent --input examples/inputs/medium_specificity_prompt.md --name medium_specificity_prompt --stream
python -m scripts.run_agent --input examples/inputs/low_specificity_prompt.md --name low_specificity_prompt --stream

python -m scripts.mlflow_langchain_flavor --mode log

python -m scripts.mlflow_langchain_flavor --mode infer \
  --model-uri runs:/66704930d6e84e99a826f731d83695e2/model \
  --text "Project: Minimal Customer Feedback Inbox ..."