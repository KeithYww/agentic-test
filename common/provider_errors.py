# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Safe provider failure codes shared by agents and the coordinator."""


class ProviderRetryError(RuntimeError):
    """Stop a provider request while preserving its safe, structured cause."""

    def __init__(self, code: str, **details: int | str | bool) -> None:
        self.code = code
        self.details = details
        messages = {
            "provider_busy_exhausted": "模型服务繁忙，本次自动尝试已停止",
            "provider_retry_deadline": "模型请求已达到等待或任务时间上限",
            "provider_usage_unknown": "模型请求结果或用量未知，已停止后续调用",
            "provider_stream_incomplete": "模型响应流连续提前结束，已按上限计费并停止，可确认后续跑",
        }
        super().__init__(f"{code}: {messages[code]}")
