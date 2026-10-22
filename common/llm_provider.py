# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Builds pydantic-ai models backed by the Photon API relay (Anthropic-compatible protocol)."""

from dataclasses import replace

from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.providers.anthropic import AnthropicProvider

import config

# Photon relay model aliases whose backend rejects the legacy budget-based thinking format and
# requires the newer adaptive format instead. pydantic_ai can't detect this on its own because
# these aliases don't match its canonical Anthropic model name prefixes. Other aliases (e.g.
# "gpt-5.6-luna") work fine with the legacy format, so the override below is scoped to only
# these confirmed names instead of being applied to every model.
_MODELS_REQUIRING_ADAPTIVE_THINKING = frozenset({"claude-sonnet-5", "gpt-5.6-terra"})


class PhotonModel(AnthropicModel):
    """Identify relay models constructed by this factory for request accounting."""


def photon_candidates(model: Model) -> tuple[PhotonModel, ...]:
    """Return only relay candidates created here, never unwrap arbitrary models."""
    if isinstance(model, PhotonModel):
        return (model,)
    if isinstance(model, FallbackModel) and all(isinstance(candidate, PhotonModel) for candidate in model.models):
        return tuple(model.models)
    return ()


def _photon_model_profile(model_name: str):
    """Patches the auto-detected Anthropic profile for Photon relay model aliases that need it."""
    profile = anthropic_model_profile(model_name)
    if model_name in _MODELS_REQUIRING_ADAPTIVE_THINKING:
        return replace(profile, anthropic_supports_adaptive_thinking=True)
    if model_name in {"deepseek-v4-flash", "deepseek-v4.1-flash"}:
        # DeepSeek ignores budget_tokens; send the requested effort explicitly.
        return replace(profile, anthropic_supports_effort=True)
    return profile


def _qa_fallback_allowed(exc: Exception) -> bool:
    from common.design_runtime import current_stage

    # An API failure without a usage receipt must not hide another paid attempt.
    return current_stage.get() is None and isinstance(exc, ModelAPIError)


def get_model(primary_model_name: str, fallback_model_name: str | None = None, *, bounded_requests: bool = False) -> Model:
    """Builds a Photon API model with an optional secondary model on API errors.

    The relay is a third-party proxy in front of multiple providers, so any given model can
    intermittently error out; `FallbackModel` retries the request against `fallback_model_name`
    when that happens.
    """
    provider = AnthropicProvider(api_key=config.PHOTON_API_KEY, base_url=config.PHOTON_API_BASE_URL)
    if bounded_requests:
        provider.client.max_retries = 0
    primary_model = PhotonModel(
        primary_model_name, provider=provider, profile=_photon_model_profile(primary_model_name)
    )
    if fallback_model_name is None:
        return primary_model
    fallback_model = PhotonModel(
        fallback_model_name, provider=provider, profile=_photon_model_profile(fallback_model_name)
    )
    if bounded_requests:
        return FallbackModel(primary_model, fallback_model, fallback_on=_qa_fallback_allowed)
    return FallbackModel(primary_model, fallback_model)
