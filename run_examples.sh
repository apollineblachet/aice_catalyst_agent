python -m scripts.run_agent --input examples/inputs/prompt1.md --name prompt1 --stream
python -m scripts.run_agent --input examples/inputs/high_specificity_prompt.md --name high_specificity_prompt --stream
python -m scripts.run_agent --input examples/inputs/medium_specificity_prompt.md --name medium_specificity_prompt --stream
python -m scripts.run_agent --input examples/inputs/low_specificity_prompt.md --name low_specificity_prompt --stream

python -m scripts.mlflow_langchain_flavor --mode log

python -m scripts.mlflow_langchain_flavor --mode infer \
  --model-uri runs:/a3a64609986a42b5b7590fb7b2b8e3ce/model \
  --text "Project: Minimal Customer Feedback Inbox ..."