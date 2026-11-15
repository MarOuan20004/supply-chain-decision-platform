"""Interface abstraite LLMClient et implementations (Ollama, OpenAI/Mistral, Mock deterministe)."""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from supply_chain.agent.schemas import LLMMessage, ToolCallRequest
from supply_chain.config import SupplyChainSettings, settings


@dataclass
class MockToolCall:
    """Structure d'appel d'outil scripte pour les tests deterministes."""

    name: str
    arguments: dict[str, Any]
    call_id: str = ""

    def __post_init__(self) -> None:
        if not self.call_id:
            self.call_id = f"call_{uuid.uuid4().hex[:8]}"

    @property
    def tool_name(self) -> str:
        return self.name

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.name,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class MockLLMResponse:
    """Structure de reponse LLM scriptee pour tests deterministes."""

    tool_calls: list[Any] = field(default_factory=list)
    final_content: str | None = None
    tokens_used: int = 0
    role: str = "assistant"

    @property
    def content(self) -> str | None:
        return self.final_content


class LLMClient(ABC):
    """Contrat abstrait pour tout fournisseur de modele de langage supportant les outils."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Envoie l'historique des messages et les schemas d'outils, retourne la reponse LLM."""


class MockLLMClient(LLMClient):
    """Client deterministe scripte pour les tests unitaires et la CI sans reseau."""

    def __init__(
        self,
        scripted_turns: Sequence[Any] | None = None,
        scripted_responses: Sequence[Any] | None = None,
        per_call_delay_sec: float = 0.0,
    ) -> None:
        if scripted_responses is not None:
            self._turns = list(scripted_responses)
        elif scripted_turns is not None:
            self._turns = list(scripted_turns)
        else:
            self._turns = []
        self._per_call_delay_sec = float(per_call_delay_sec)
        self.call_count: int = 0
        self.recorded_messages: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        if self._per_call_delay_sec > 0.0:
            time.sleep(self._per_call_delay_sec)

        self.recorded_messages.append([dict(m) for m in messages])
        if self.call_count >= len(self._turns):
            return (
                self._turns[-1]
                if self._turns
                else MockLLMResponse(tool_calls=[], final_content=None, tokens_used=0)
            )

        turn = self._turns[self.call_count]
        self.call_count += 1
        return turn


class OllamaClient(LLMClient):
    """Client HTTP local pour Ollama (/api/chat) avec support natif du function calling."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b-instruct",
        timeout_sec: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMMessage:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0.0},
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        msg = data.get("message", {})
        raw_calls = msg.get("tool_calls") or []
        parsed_calls: list[ToolCallRequest] = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            parsed_calls.append(
                ToolCallRequest(
                    call_id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    tool_name=str(fn.get("name", "")),
                    arguments=dict(args),
                )
            )

        return LLMMessage(
            role="assistant",
            content=msg.get("content"),
            tool_calls=parsed_calls,
            tokens_used=int(data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
        )


class OpenAIClient(LLMClient):
    """Client compatible OpenAI / Mistral API (/v1/chat/completions)."""

    def __init__(
        self,
        base_url: str = "https://api.mistral.ai/v1",
        api_key: str = "",
        model: str = "mistral-small-latest",
        timeout_sec: float = 15.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_sec = timeout_sec

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMMessage:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.0,
        }
        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice_msg = data["choices"][0]["message"]
        raw_calls = choice_msg.get("tool_calls") or []
        parsed_calls: list[ToolCallRequest] = []
        for tc in raw_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                args = json.loads(args)
            parsed_calls.append(
                ToolCallRequest(
                    call_id=str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                    tool_name=str(fn.get("name", "")),
                    arguments=dict(args),
                )
            )

        usage = data.get("usage", {})
        tokens = int(usage.get("total_tokens", 0))

        return LLMMessage(
            role="assistant",
            content=choice_msg.get("content"),
            tool_calls=parsed_calls,
            tokens_used=tokens,
        )


def build_llm_client(cfg: SupplyChainSettings = settings) -> LLMClient:
    """Fabrique de client LLM pilotee par la configuration sans modifier l'orchestrateur."""
    provider = cfg.agent_llm_provider.lower().strip()
    if provider == "ollama":
        return OllamaClient(
            base_url=cfg.ollama_base_url,
            model=cfg.ollama_model,
            timeout_sec=cfg.agent_timeout_sec,
        )
    if provider == "openai":
        return OpenAIClient(
            base_url=cfg.openai_base_url,
            api_key=cfg.openai_api_key,
            model=cfg.openai_model,
            timeout_sec=cfg.agent_timeout_sec,
        )
    if provider == "mock":
        return MockLLMClient(scripted_turns=[])
    raise ValueError(
        f"Fournisseur LLM inconnu '{cfg.agent_llm_provider}'. "
        "Valeurs supportees: ['mock', 'ollama', 'openai']."
    )


def build_llm_client_from_settings(cfg: SupplyChainSettings = settings) -> LLMClient:
    """Alias retrocompatible pour build_llm_client."""
    return build_llm_client(cfg)
