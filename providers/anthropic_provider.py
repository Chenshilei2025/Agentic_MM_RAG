"""Anthropic provider with retries and usage logging."""
import os, time, threading
from typing import Dict, List
from providers.base_provider import BaseProvider
from utils.logger import get_logger

log = get_logger("anthropic_provider")

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, model=None, api_key=None, max_retries=3,
                 base_delay=1.0, log_usage=True):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
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
            import anthropic
            if not self.api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY not set. export ANTHROPIC_API_KEY=...")
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, messages, system="", max_tokens=1024, **kw):
        client = self._client_lazy()
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = client.messages.create(
                    model=self.model, system=system or None,
                    messages=messages, max_tokens=max_tokens,
                )
                if hasattr(resp, "usage"):
                    # Atomic update — protects against concurrent sub-agent
                    # calls racing on the counters.
                    with self._usage_lock:
                        self.call_count += 1
                        self.total_input_tokens += resp.usage.input_tokens
                        self.total_output_tokens += resp.usage.output_tokens
                    if self.log_usage:
                        log.info(f"[llm] model={self.model} "
                                 f"in={resp.usage.input_tokens} "
                                 f"out={resp.usage.output_tokens}")
                else:
                    with self._usage_lock:
                        self.call_count += 1
                return "".join(b.text for b in resp.content
                               if getattr(b, "type", "") == "text")
            except Exception as e:
                last_err = e
                name = type(e).__name__
                if name in ("RateLimitError", "APIConnectionError",
                            "APITimeoutError", "InternalServerError",
                            "OverloadedError"):
                    delay = self.base_delay * (2 ** attempt)
                    log.warning(f"[llm] {name} retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"Anthropic call failed: {last_err}")

    def usage_summary(self):
        with self._usage_lock:
            return {"calls": self.call_count,
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens}
