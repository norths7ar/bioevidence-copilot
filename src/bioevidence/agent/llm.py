from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from bioevidence.config import Settings


JSON_FENCE_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    parsed_json: dict[str, Any] | None = None


class AgentLLMError(RuntimeError):
    pass


def create_agent_client(settings: Settings) -> OpenAI:
    if not settings.agent_api_key:
        raise AgentLLMError("AGENT_API_KEY is required for agent synthesis")
    if not settings.agent_base_url:
        raise AgentLLMError("AGENT_BASE_URL is required for agent synthesis")
    if not settings.agent_model:
        raise AgentLLMError("AGENT_MODEL is required for agent synthesis")
    return OpenAI(api_key=settings.agent_api_key, base_url=settings.agent_base_url)


def chat_text(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except OpenAIError as exc:
        raise AgentLLMError("Agent model request failed") from exc
    content = response.choices[0].message.content or ""
    usage = response.usage
    LOGGER.debug(
        "agent_model_completed model=%s finish_reason=%s content_chars=%d completion_tokens=%s",
        model,
        response.choices[0].finish_reason,
        len(content),
        usage.completion_tokens if usage is not None else None,
    )
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
    if not content.strip():
        raise AgentLLMError("Agent model returned empty content")
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
