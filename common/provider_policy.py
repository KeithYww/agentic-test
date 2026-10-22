# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Opt-in relay policy, separate from the existing recovery configuration contract."""

import os


class ProviderRetryPolicy:
    """Only a verified preflight contract permits extra sends."""

    PHOTON_OVERLOAD_RETRY_ENABLED = os.environ.get("PHOTON_OVERLOAD_RETRY_ENABLED", "false").lower() == "true"
    PHOTON_REJECTION_CONTRACT = os.environ.get("PHOTON_REJECTION_CONTRACT", "")


def explicit_policy() -> dict:
    """Keep absent/default settings compatible with pre-retry run manifests."""
    values = {key: value for key, value in vars(ProviderRetryPolicy).items() if key.isupper()}
    return values if any(values.values()) else {}
