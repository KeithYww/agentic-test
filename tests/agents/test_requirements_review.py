# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import config
from agents.requirements_review.main import RequirementsReviewAgent


@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "OWN_NAME", "Test Agent")
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "PORT", 8001)
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "EXTERNAL_PORT", 8001)
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "PROTOCOL", "http")
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "MODEL_NAME", "test")
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "FALLBACK_MODEL_NAME", "test")
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "THINKING_LEVEL", "LOW")
    monkeypatch.setattr(config.RequirementsReviewAgentConfig, "MAX_REQUESTS_PER_TASK", 5)
    monkeypatch.setattr(config, "AGENT_BASE_URL", "http://localhost")
    monkeypatch.setattr(config, "FEISHU_MCP_SERVER_URL", "http://feishu")
    monkeypatch.setattr(config, "MCP_SERVER_TIMEOUT_SECONDS", 30)


@patch("agents.requirements_review.main.RequirementsReviewSystemPrompt")
@patch("agents.requirements_review.main.AgentBase.__init__")
@patch("agents.requirements_review.main.get_model", return_value="test")
def test_requirements_review_agent_init(mock_get_model, mock_super_init, mock_prompt_cls, mock_config):
    mock_prompt_instance = MagicMock()
    mock_prompt_instance.get_prompt.return_value = "system prompt"
    mock_prompt_cls.return_value = mock_prompt_instance

    agent = RequirementsReviewAgent()

    mock_super_init.assert_called_once()
    _, kwargs = mock_super_init.call_args
    assert kwargs["agent_name"] == "Test Agent"
    assert kwargs["instructions"] == "system prompt"

    assert agent.get_thinking_level() == "LOW"
    assert agent.get_max_requests_per_task() == 5


def test_add_requirement_review_comment_posts_via_meego_client():
    mock_client = MagicMock()
    with patch("agents.requirements_review.main.MeegoClient", return_value=mock_client) as mock_cls:
        result = RequirementsReviewAgent.add_requirement_review_comment("WI-1", "PROJ", "feedback text")

    mock_cls.assert_called_once_with("PROJ")
    mock_client.add_work_item_comment.assert_called_once_with("PROJ", "story", "WI-1", "feedback text")
    assert "WI-1" in result


@pytest.mark.asyncio
async def test_focused_review_merges_successes_and_marks_missing_focuses():
    agent = object.__new__(RequirementsReviewAgent)
    agent.review_agent = MagicMock()
    async def run(prompt):
        if "边界条件与异常处理" in prompt and prompt.startswith("仅从"):
            raise RuntimeError("unavailable")
        result = MagicMock()
        result.output.suggested_improvements = "补充可验收规则"
        return result
    agent.review_agent.run = AsyncMock(side_effect=run)
    result = await agent._review_focused_doc("PRD")
    assert len(result.completed_focuses) == 4
    assert result.failed_focuses == ["边界条件与异常处理"]
    assert "未完成的评审维度" in result.suggested_improvements
    assert agent.review_agent.run.await_count == 6
    focused = [call.args[0] for call in agent.review_agent.run.await_args_list if call.args[0].startswith("仅从")]
    assert focused and all('{"document": "PRD"}' in prompt for prompt in focused)


@pytest.mark.asyncio
async def test_all_focused_reviews_failed_never_merge():
    agent = object.__new__(RequirementsReviewAgent)
    agent.review_agent = MagicMock()
    agent.review_agent.run = AsyncMock(side_effect=RuntimeError("unavailable"))
    with pytest.raises(RuntimeError, match="Every focused"):
        await agent._review_focused_doc("PRD")
    assert agent.review_agent.run.await_count == 5
