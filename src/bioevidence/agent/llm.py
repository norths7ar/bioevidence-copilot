from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from bioevidence.config import Settings


JSON_FENCE_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    parsed_json: dict[str, Any] | None = None


class AgentLLMError(RuntimeError):
    pass


def create_agent_client(settings: Settings) -> OpenAI:
    if not settings.agent_api_key:
        raise AgentLLMError("BIOEVIDENCE_AGENT_API_KEY is required for agent synthesis")
    if not settings.agent_base_url:
        raise AgentLLMError("BIOEVIDENCE_AGENT_BASE_URL is required for agent synthesis")
    if not settings.agent_model:
        raise AgentLLMError("BIOEVIDENCE_AGENT_MODEL is required for agent synthesis")
    return OpenAI(api_key=settings.agent_api_key, base_url=settings.agent_base_url)


def chat_text(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


def chat_json(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    content = chat_text(
        client,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _parse_json_content(content)


def _parse_json_content(content: str) -> dict[str, Any]:
    candidates = [content]
    match = JSON_FENCE_PATTERN.search(content)
    if match:
        candidates.insert(0, match.group(1))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AgentLLMError("Agent model did not return valid JSON")
