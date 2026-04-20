"""Provider layer — the ONLY path to an LLM. Agents never call APIs directly."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseProvider(ABC):
    name: str = "base"
    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], **kw) -> str: ...

class ProviderRegistry:
    _providers: Dict[str, BaseProvider] = {}

    @classmethod
    def register(cls, provider: BaseProvider) -> None:
        cls._providers[provider.name] = provider

    @classmethod
    def get(cls, name: str) -> BaseProvider:
        if name not in cls._providers:
            raise KeyError(f"Unknown provider: {name}")
        return cls._providers[name]
