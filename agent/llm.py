from __future__ import annotations
import os
from langchain_openai import AzureChatOpenAI

def build_llm() -> AzureChatOpenAI:
    endpoint  = os.getenv("AZURE_OPENAI_ENDPOINT",  "https://catalyst-agent.openai.azure.com/")
    deployment= os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-4o-mini")
    api_key   = os.getenv("AZURE_OPENAI_API_KEY",   "")
    api_ver   = os.getenv("AZURE_OPENAI_API_VERSION","2025-01-01-preview")
    if not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_API_KEY.")
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_ver,
        api_key=api_key,
        temperature=0.0,
    )
