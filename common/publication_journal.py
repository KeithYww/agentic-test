# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Forward-only publication receipts; uncertain effects are never blindly repeated."""

from collections.abc import Callable
from typing import Any

from common.design_store import DesignStore, RecoveryBlocked, digest


class PublicationJournal:
    """Share the application's store from blocking publication service threads."""

    def __init__(self, store: DesignStore, run_id: str, generation: int) -> None:
        self.store = store
        self.run_id = run_id
        self.generation = generation

    def perform(self, key: str, intent: Any, action: Callable[[], Any], validate: Callable[[Any], bool] = bool) -> Any:
        """Persist intent before sending and save only a validated remote receipt."""
        previous = self.store.call_sync("publish_begin", self.run_id, self.generation, key, intent)
        if previous is not None:
            return previous
        result = action()
        if not validate(result):
            raise RecoveryBlocked(f"remote_receipt_incomplete:{key}")
        self.store.call_sync("put", self.run_id, self.generation, "publish", key, digest(intent), result)
        return result
