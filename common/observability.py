# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""OpenTelemetry tracing setup, exported to a self-hosted Langfuse instance.

A no-op by design when `config.OPEN_TELEMETRY_URL` is unset, matching the project's existing
"empty env var disables the feature" convention. Each process (orchestrator, each agent) calls
`init_tracing`/`shutdown_tracing` once from its own FastAPI lifespan.
"""

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.util.re import parse_env_headers
from pydantic_ai import Agent, InstrumentationSettings

import config
from common import utils

logger = utils.get_logger("observability")

tracer = trace.get_tracer("quaia")

_tracer_provider: TracerProvider | None = None


def init_tracing(service_name: str) -> None:
    """Initializes OTel tracing for the current process and instruments all pydantic-ai Agents.

    No-op if `config.OPEN_TELEMETRY_URL` is not set. `include_content=False` keeps prompt/
    completion text out of spans, consistent with `common/trace_redaction.py`'s redaction policy
    for the separate, dashboard-facing debug trace.
    """
    global _tracer_provider
    if not config.OPEN_TELEMETRY_URL:
        return

    # liberal=True tolerates a literal space in e.g. "Authorization=Basic <token>", instead of
    # requiring the token to be percent-encoded.
    headers = parse_env_headers(config.OTEL_EXPORTER_OTLP_HEADERS, liberal=True) if config.OTEL_EXPORTER_OTLP_HEADERS else None
    resource = Resource.create({SERVICE_NAME: f"{config.OTEL_SERVICE_NAME_PREFIX}-{service_name}"})
    _tracer_provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=config.OPEN_TELEMETRY_URL, headers=headers)
    _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_tracer_provider)

    Agent.instrument_all(InstrumentationSettings(include_content=False))
    logger.info(f"OpenTelemetry tracing initialized for service '{service_name}', exporting to {config.OPEN_TELEMETRY_URL}")


def shutdown_tracing() -> None:
    """Flushes and shuts down the tracer provider, if one was initialized."""
    if _tracer_provider is None:
        return
    _tracer_provider.force_flush()
    _tracer_provider.shutdown()
