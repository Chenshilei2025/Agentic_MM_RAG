"""Official MMLongBench-Doc evaluation protocol (3-stage GPT-4o scorer).

Paper: "MMLongBench-Doc: Benchmarking Long-context Document Understanding
with Visualizations" (Ma et al., 2024), Section 4.1.

Protocol:
  Stage 1 — Extract short answer from model's free-form response
  Stage 2 — Compare extracted short answer with gold reference
  Stage 3 — Binary correct/incorrect judgment

This mirrors the official repo's evaluation to let reviewers reproduce
MMLongBench-Doc numbers directly. Costs ~$0.02 per query with GPT-4o.

Usage:
  from experiments.common.utils.eval_gpt_scorer import make_gpt_scorer
  scorer = make_gpt_scorer(api_key=os.environ["OPENAI_API_KEY"],
                            model="gpt-4o")
  metrics = scorer("What is X?", "42", "The answer is 42.")
  # → {"em": 1, "f1": 1.0, "extracted": "42"}
"""
from __future__ import annotations
import json
import re
import time
from typing import Callable, Dict, Optional


STAGE1_SYSTEM = """You are an answer extractor. Given a question and a \
model's response, extract the MINIMAL short-form answer that directly \
addresses the question. Return ONLY the short answer (no explanation).

If the response says the answer is not in the document / cannot be \
found, return exactly: "Not answerable"
"""

STAGE2_SYSTEM = """You are an answer grader. Decide whether a model's \
extracted answer is semantically equivalent to the reference answer for \
the given question.

Consider numeric tolerance (e.g. 42 ≈ 42.0), ordering (for lists), \
synonyms, and case-insensitivity. Reject answers that change the \
meaning.

Output strict JSON:
{"correct": true|false, "rationale": "short explanation"}
"""


def make_gpt_scorer(api_key: str = "",
                     model: str = "gpt-4o",
                     endpoint: str = "https://api.openai.com/v1/chat/completions",
                     timeout: float = 30.0,
                     max_retries: int = 3) -> Callable[[str, str, object], Dict]:
    """Return a scorer `(question, gold, pred) -> metrics`.

    Scorer metrics shape:
      {"em": 0 or 1, "f1": float, "extracted": str}
    """
    def call_gpt(system: str, user: str, max_tokens: int = 200) -> str:
        import urllib.request
        import urllib.error
        body = json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {api_key}"}
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(endpoint, data=body,
                                              headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    obj = json.loads(r.read().decode("utf-8"))
                return obj["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise
        return ""

    def score(question: str, gold, pred) -> Dict:
        if isinstance(pred, dict):
            pred = pred.get("answer", "")
        pred = str(pred or "").strip()
        gold = str(gold or "").strip()

        if not pred:
            return {"em": 0, "f1": 0.0, "extracted": ""}

        # Stage 1: extract
        user1 = f"QUESTION: {question}\n\nRESPONSE: {pred}\n\nShort answer:"
        try:
            extracted = call_gpt(STAGE1_SYSTEM, user1, max_tokens=80).strip()
        except Exception as e:
            return {"em": 0, "f1": 0.0, "extracted": "",
                    "error": f"stage1: {e}"}

        # Stage 2: grade
        user2 = (f"QUESTION: {question}\n"
                 f"REFERENCE ANSWER: {gold}\n"
                 f"EXTRACTED MODEL ANSWER: {extracted}\n\n"
                 f"Output strict JSON.")
        try:
            raw = call_gpt(STAGE2_SYSTEM, user2, max_tokens=120).strip()
        except Exception as e:
            return {"em": 0, "f1": 0.0, "extracted": extracted,
                    "error": f"stage2: {e}"}

        # Parse JSON
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        correct = False
        if m:
            try:
                obj = json.loads(m.group(0))
                correct = bool(obj.get("correct", False))
            except json.JSONDecodeError:
                correct = "true" in raw.lower()
        else:
            correct = "true" in raw.lower()

        # MMLongBench-Doc uses binary correctness; we report both for
        # consistency with our other scorers.
        return {"em": int(correct), "f1": float(correct),
                "extracted": extracted}

    return score
