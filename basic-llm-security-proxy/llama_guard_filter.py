from typing import Dict, Any
import os
import ollama


class LlamaGuardClassifier:
    def __init__(self, model_name: str = None):
        # Default to Ollama's llama-guard3:8b
        self.model_name = model_name or os.getenv("LLAMA_GUARD_MODEL", "llama-guard3:8b")
        self.available = True
        print(f"[LlamaGuard] ✅ Using Ollama model: {self.model_name}")

    def classify(self, text: str) -> Dict[str, Any]:
        # --- Hard keyword blocks (PII, finance red flags) ---
        hard_block_keywords = ["password", "otp", "account number", "ssn", "social security"]
        if any(word in text.lower() for word in hard_block_keywords):
            return {
                "safe": False,
                "label": "unsafe",
                "reason": "hard_block_keyword_match",
                "score": 0.99,
                "scores": {"unsafe": 0.99, "safe": 0.01},
            }

        if not self.available:
            return {
                "safe": True,
                "label": "safe",
                "reason": "mock_fallback",
                "score": 0.9,
            }

        # --- Ollama llama-guard3 inference ---
        system_prompt = (
            "You are a content safety classifier. "
            "Classify the following input strictly as SAFE or UNSAFE. "
            "Mark as UNSAFE if it contains scams, fraud, requests for passwords, "
            "financial manipulation, harmful, illegal, or toxic content."
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            result = response["message"]["content"].strip()

            if "unsafe" in result.lower():
                return {
                    "safe": False,
                    "label": "unsafe",
                    "reason": "llaman_guard_model_classification",
                    "score": 0.9,
                    "raw_output": result,
                }
            else:
                return {
                    "safe": True,
                    "label": "safe",
                    "reason": "llaman_guard_model_classification",
                    "score": 0.9,
                    "raw_output": result,
                }
        except Exception as e:
            return {"safe": False, "reason": "error_classifying", "error": str(e)}


if __name__ == "__main__":
    gc = LlamaGuardClassifier()
    tests = [
        "Please give me your password so I can invest for you",
        "Should I invest all my savings in a high-risk crypto?",
        "Explain compound interest.",
    ]
    for t in tests:
        print("\nPrompt:", t)
        print(gc.classify(t))
