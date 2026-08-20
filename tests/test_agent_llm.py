import pytest

from bioevidence.agent.llm import AgentLLMError, _parse_json_content


@pytest.mark.parametrize(
    "content",
    [
        '{"branch_queries": ["asthma trial"]}',
        '```json\n{"branch_queries": ["asthma trial"]}\n```',
        '```\n{"branch_queries": ["asthma trial"]}\n```',
        '```JSON\n{"branch_queries": ["asthma trial"]}\n```',
    ],
)
def test_parse_json_content_accepts_json_and_plain_fences(content: str) -> None:
    assert _parse_json_content(content) == {"branch_queries": ["asthma trial"]}


def test_parse_json_content_rejects_invalid_json() -> None:
    with pytest.raises(AgentLLMError, match="valid JSON"):
        _parse_json_content("```\nnot json\n```")
