import os
import re
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from llama_guard_filter import LlamaGuardClassifier

load_dotenv()

app = FastAPI(title="LLM Financial Advisor Guarded Proxy")

# 🔒 Load finance-specific regex policies
with open("policies/finance_policy.yaml", "r") as f:
    finance_policies = yaml.safe_load(f)

# 🛡️ Initialize Llama Guard classifier
llama_guard = LlamaGuardClassifier()

# 🤖 Initialize GPT-4.1 backend (GitHub Inference API)
gpt_client = OpenAI(
    base_url="https://models.github.ai/inference",  # GitHub Inference endpoint
    api_key=os.getenv("GITHUB_TOKEN")                 # GitHub PAT required
)
GPT_MODEL = "openai/gpt-4.1"

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    print("[Server] Llama Guard loaded and ready in memory ✅")

@app.post("/query")
async def process_query(request: QueryRequest):
    text = request.query

    # 1️⃣ Static regex policy check
    for rule in finance_policies.get("blocked_patterns", []):
        if re.search(rule["regex"], text, re.IGNORECASE):
            return {
                "allowed": False,
                "reason": f"Blocked by Finance Policy: {rule['reason']}"
            }

    # 2️⃣ Run Llama Guard classifier
    guard_result = llama_guard.classify(text)
    print(f"Llama Guard results => {guard_result}")    
    if not guard_result.get("safe", False):
        return {
            "allowed": False,
            "reason": "Blocked by Llama Guard",
            "details": guard_result
        }

    # 3️⃣ Forward safe queries to GPT-4.1 (GitHub Inference)
    try:
        response = gpt_client.chat.completions.create(
            model=GPT_MODEL,  # or gpt-4.1
            messages=[
                {"role": "system", "content": "You are a financial advisor restricted by safety guardrails."},
                {"role": "user", "content": text}
            ],
        )

        answer = response.choices[0].message.content

        return {"allowed": True, "response": answer}

    except Exception as e:
        return {"allowed": False, "reason": "Error calling GPT backend", "error": str(e)}

