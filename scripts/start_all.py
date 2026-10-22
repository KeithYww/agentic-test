"""
Starts all services in a single process / single event loop.
Each service listens on its own port; inter-service traffic uses localhost.
"""

import asyncio
import logging

import uvicorn

import config
from agents.test_case_classification.main import app as classification_app
from agents.test_case_generation.main import app as generation_app
from agents.test_case_review.main import app as review_app
from common import design_store
from orchestrator.main import orchestrator_app
from scripts.feishu_mcp_server import app as feishu_mcp_app

logger = logging.getLogger(__name__)

_SERVICES: list[tuple[str, object, int]] = [
    ("orchestrator", orchestrator_app, 8000),
    ("tc-generation", generation_app, 8002),
    ("tc-classification", classification_app, 8003),
    ("tc-review", review_app, 8004),
    ("feishu-mcp", feishu_mcp_app, 9010),
]

if config.DesignConfig.REQUIREMENT_REVIEW_ENABLED:
    from agents.requirements_review.main import app as requirements_app

    _SERVICES.insert(1, ("requirements-review", requirements_app, 8001))


async def _main() -> None:
    if config.DesignConfig.STATE_DIR:
        design_store.store = design_store.DesignStore(config.DesignConfig.STATE_DIR)
        await design_store.store.call("open")
    servers = [
        uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=port, log_config=None)
        )
        for _, app, port in _SERVICES
    ]
    for (name, _, port), _server in zip(_SERVICES, servers, strict=True):
        logger.info("Starting %s on port %d", name, port)
    try:
        await asyncio.gather(*[s.serve() for s in servers])
    finally:
        if design_store.store:
            await design_store.store.close()
            design_store.store = None


if __name__ == "__main__":
    asyncio.run(_main())
