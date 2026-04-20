"""DeepSeek provider.

DeepSeek exposes an OpenAI-compatible API. This provider uses the `openai`
SDK pointed at DeepSeek's base URL. It is drop-in compatible with the rest
of the system — same BaseProvider interface, same usage_summary().

REQUIREMENTS
  pip install openai
  export DEEPSEEK_API_KEY=sk-...
  (optional) export DEEPSEEK_MODEL=deepseek-reasoner   # or deepseek-chat

MODEL NOTES
  - deepseek-reasoner : the "R1" reasoning model. Best for the Decision
                        Agent's planning role, but produces long reasoning
                        traces that count as output tokens. Costs more.
  - deepseek-chat     : faster and cheaper. Often sufficient for sub-agents.

You can use different models for DA vs sub-agents by registering two
provider instances and threading them separately — see the usage example
in scripts/live_demo_deepseek.py.
"""
import os, time, threading
from typing import Dict, List
from providers.base_provider import BaseProvider
from utils.logger import get_logger

log = get_logger("deepseek_provider")

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-reasoner")
BASE_URL = "https://api.deepseek.com"


class DeepseekProvider(BaseProvider):
    name = "deepseek"

    def __init__(self, model=None, api_key=None, max_retries=3,
                 base_delay=1.0, log_usage=True, base_url=BASE_URL):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.log_usage = log_usage
        self._client = None
        self._usage_lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0

    def _client_lazy(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "DeepseekProvider needs `openai` package. "
                    "Install with: pip install openai") from e
            if not self.api_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY not set. export DEEPSEEK_API_KEY=...")
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(self, messages, system="", max_tokens=None, **kw):
        """Adapt Anthropic-style (messages, system) to OpenAI-style."""
        client = self._client_lazy()
        # Reasoner output can be long (post-CoT). Give it room.
        if max_tokens is None:
            max_tokens = 4096 if "reasoner" in self.model.lower() else 1024

        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(messages)

        # deepseek-chat supports OpenAI's response_format={"type":"json_object"}.
        # This forces strict JSON output — dramatic reliability improvement.
        # Reasoner does NOT support it; we rely on the parser to handle prose.
        extra = {}
        if "reasoner" not in self.model.lower():
            extra["response_format"] = {"type": "json_object"}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=openai_messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    **extra,
                )
                if hasattr(resp, "usage") and resp.usage is not None:
                    with self._usage_lock:
                        self.call_count += 1
                        self.total_input_tokens += resp.usage.prompt_tokens
                        self.total_output_tokens += resp.usage.completion_tokens
                    if self.log_usage:
                        log.info(f"[llm] model={self.model} "
                                 f"in={resp.usage.prompt_tokens} "
                                 f"out={resp.usage.completion_tokens}")
                else:
                    with self._usage_lock:
                        self.call_count += 1

                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                name = type(e).__name__
                if name in ("RateLimitError", "APIConnectionError",
                            "APITimeoutError", "InternalServerError",
                            "ServiceUnavailableError"):
                    delay = self.base_delay * (2 ** attempt)
                    log.warning(f"[llm] {name} retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"DeepSeek call failed: {last_err}")

    def usage_summary(self):
        with self._usage_lock:
            return {"calls": self.call_count,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens}
