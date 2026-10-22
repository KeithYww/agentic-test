# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

from urllib.parse import urlencode

import httpx

import config
from common import utils

logger = utils.get_logger("feishu_bot_notifier")


async def notify_pipeline_success(
    story_id: str, tc_count: int, bitable_url: str | None, *, pending_findings: int = 0,
    untestable: list[str] | None = None,
) -> None:
    """Send a green card to the Feishu bot group on pipeline success."""
    elements = [
        {
            "tag": "div",
            "fields": [
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**Story**\n{story_id}"}},
                {"is_short": True, "text": {"tag": "lark_md", "content": f"**用例数**\n{tc_count} 条"}},
            ],
        },
    ]
    if pending_findings:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"自动评审仍有{pending_findings}条意见，请在人工评审环节确认。"},
        })
    if untestable:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "**自动评审判定无需测试的需求项，请人工复核**\n"
                     + "\n".join(f"- {line}" for line in untestable)},
        })
    if bitable_url:
        elements.append({
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "查看多维表格"},
                    "url": bitable_url,
                    "type": "primary",
                }
            ],
        })
    else:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": "⚠️ 多维表格未生成或写回失败，请查看日志。"},
        })
    await _send({
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": "用例生成完成，待人工评审"},
                "template": "yellow" if pending_findings else "green",
            },
            "elements": elements,
        },
    })


async def notify_pipeline_failure(
    story_id: str,
    error: Exception,
    project_key: str | None = None,
    bitable_url: str | None = None,
    run_id: str | None = None,
) -> None:
    """Send a red card to the Feishu bot group on pipeline failure."""
    elements: list[dict] = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"**Story**: {story_id}\n\n**原因**: {_humanize_error(error)}",
            },
        }
    ]

    # Action buttons: story link + bitable link (if generated before the failure)
    actions = []
    if run_id and config.DASHBOARD_PUBLIC_URL:
        actions.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "Retry · 确认续跑"},
            "url": f"{config.DASHBOARD_PUBLIC_URL}/?{urlencode({'run_id': run_id})}",
            "type": "primary",
        })
    if project_key:
        actions.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "查看 Story"},
            "url": f"https://project.feishu.cn/{project_key}/story/detail/{story_id}",
            "type": "default",
        })
    if bitable_url:
        actions.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": "查看多维表格"},
            "url": bitable_url,
            "type": "primary",
        })
    if actions:
        elements.append({"tag": "action", "actions": actions})

    await _send({
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": "❌ 用例生成失败"},
                "template": "red",
            },
            "elements": elements,
        },
    })


def _humanize_error(error: Exception) -> str:
    """Translates a raw exception into a short, non-technical message for the notification card."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        url = str(error.request.url)
        if "ai_node" in url:
            return (
                f"用例已生成，但写回飞书 Story 时失败（飞书服务返回 {status} 错误）。"
                "用例文档本身应该是完好的，请联系维护人员核实。"
            )
        if status >= 500:
            return f"调用外部服务时对方服务器出错（HTTP {status}），可稍后重试。"
        return f"调用外部服务被拒绝（HTTP {status}）。"
    if isinstance(error, httpx.TimeoutException):
        return "请求超时，可能是网络或依赖服务响应过慢，可稍后重试。"
    if isinstance(error, httpx.TransportError):
        return "网络连接失败，请检查依赖服务是否可访问。"
    return str(error)[:300]


async def _send(payload: dict) -> None:
    url = config.FEISHU_BOT_WEBHOOK_URL
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
    except Exception as e:
        logger.warning(f"Feishu bot notification failed (non-fatal): {e}")
