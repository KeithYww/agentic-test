# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live checks that every Photon relay model configured in ``config.py`` actually works.

Every ``*AgentConfig`` class (and ``OrchestratorConfig``) declares a concrete
``THINKING_LEVEL`` (e.g. ``"low"``, ``"medium"``, ``"minimal"``), so in this codebase every
real agent request goes through ``CustomLlmWrapper``'s thinking-enabled branch
(``temperature=1`` + ``thinking=<level>``, no ``top_p``). No agent config currently sets
``THINKING_LEVEL`` to ``None``, so the non-thinking branch (``temperature``/``top_p`` from
``config.py``) is only exercised here for completeness, via one representative model.

These tests make real network calls to the Photon relay (need ``config.PHOTON_API_KEY``), so
they're excluded from the default run (see the ``live_llm`` marker in ``pytest.ini``). Run them
explicitly with:
    uv run pytest tests/live -m live_llm -v
"""

import inspect
from pathlib import Path

import pytest
from dotenv import dotenv_values
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

import config
from common.llm_provider import _photon_model_profile

pytestmark = pytest.mark.live_llm

_PROMPT = "Reply with exactly the single word: ok"

# tests/conftest.py force-sets PHOTON_API_KEY=dummy for every test session so unit tests stay
# hermetic; since config.py reads it into a module-level constant at import time, config.PHOTON_API_KEY
# is always "dummy" under pytest. These tests need the real key, so read it straight from .env.
_PHOTON_API_KEY = dotenv_values(Path(__file__).resolve().parent.parent.parent / ".env").get("PHOTON_API_KEY")


def _configured_thinking_combinations() -> list[tuple[str, str]]:
    """Collects every (model_name, thinking_level) pair used by an agent config class."""
    combinations: set[tuple[str, str]] = set()
    for _, cls in inspect.getmembers(config, inspect.isclass):
        if not (hasattr(cls, "MODEL_NAME") and hasattr(cls, "THINKING_LEVEL")):
            continue
        for model_name in (cls.MODEL_NAME, getattr(cls, "FALLBACK_MODEL_NAME", None)):
            if model_name:
                combinations.add((model_name, cls.THINKING_LEVEL))
    return sorted(combinations)


def _build_model(model_name: str) -> AnthropicModel:
    provider = AnthropicProvider(api_key=_PHOTON_API_KEY, base_url=config.PHOTON_API_BASE_URL)
    return AnthropicModel(model_name, provider=provider, profile=_photon_model_profile(model_name))


@pytest.fixture(autouse=True)
def _require_photon_api_key():
    if not _PHOTON_API_KEY:
        pytest.skip("PHOTON_API_KEY is not set in .env.")


@pytest.mark.parametrize("model_name,thinking_level", _configured_thinking_combinations())
async def test_model_available_with_thinking(model_name, thinking_level):
    agent = Agent(model=_build_model(model_name), model_settings=ModelSettings(temperature=1, thinking=thinking_level))
    result = await agent.run(_PROMPT)
    assert result.output


async def test_model_available_without_thinking():
    model_name = config.OrchestratorConfig.MODEL_NAME
    settings = ModelSettings(top_p=config.TOP_P, temperature=config.TEMPERATURE)
    agent = Agent(model=_build_model(model_name), model_settings=settings)
    result = await agent.run(_PROMPT)
    assert result.output
