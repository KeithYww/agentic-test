# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Isolated dev-image probe: coordinator, A2A messages, real classifier and HTTP SDK."""

import asyncio
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from a2a.helpers import get_message_text, new_text_message

import config
from agents.test_case_classification.main import TestCaseClassificationAgent
from common import design_store
from common.design_models import StageResult
from common.models import TestCase
from common.provider_errors import ProviderRetryError
from common.provider_policy import ProviderRetryPolicy
from orchestrator.test_case_design import DesignCoordinator


async def run_probe() -> dict:
    """Recover just one rejected batch and preserve an unknown error across the wire."""
    calls = []
    mode = "known"

    class Relay(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][-1]["content"]
            text = prompt if isinstance(prompt, str) else next(p["text"] for p in prompt if p["type"] == "text")
            cases = json.loads(text.split("Test cases:\n", 1)[1])
            keys = [case["key"] for case in cases]
            previous = sum(call["keys"] == keys for call in calls)
            calls.append({"keys": keys, "model": body["model"], "body": body})
            rejected = mode == "unknown" or ("TC-0" in keys and previous == 0)
            if rejected:
                status = 503
                reply = {"error": {"type": "new_api_error" if mode == "unknown" else "system_cpu_overloaded"}}
            else:
                status = 200
                output = {
                    "llm_comments": None,
                    "test_cases": [{"issue_key": c["key"], "name": c["name"], "test_type": "API",
                                    "automation_capability": "automated", "labels": [], "tool_use_comment": ""}
                                   for c in cases],
                }
                reply = {
                    "id": "reply", "type": "message", "role": "assistant", "model": body["model"],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                    "content": [{"type": "text", "text": json.dumps(output)}],
                }
                if body.get("tools"):
                    reply["stop_reason"] = "tool_use"
                    reply["content"] = [{"type": "tool_use", "id": "result",
                                         "name": body["tools"][-1]["name"], "input": output}]
            if status == 200 and body.get("stream"):
                truncated = mode == "always_truncated" or (mode == "truncated" and previous == 0)
                self._stream(reply, truncated)
                return
            encoded = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _stream(self, reply: dict, truncated: bool) -> None:
            """Stream the reply; a truncated stream ends after its thinking block, as the relay did on 2026-10-20."""
            start = {**reply, "content": [], "stop_reason": None}
            events = [{"type": "message_start", "message": start},
                      {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                      {"type": "content_block_stop", "index": 0}]
            if not truncated:
                for index, block in enumerate(reply["content"], 1):
                    if block["type"] == "tool_use":
                        events += [{"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}},
                                   {"type": "content_block_delta", "index": index,
                                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}]
                    else:
                        events += [{"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
                                   {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}}]
                    events.append({"type": "content_block_stop", "index": index})
                events += [{"type": "message_delta", "delta": {"stop_reason": reply["stop_reason"], "stop_sequence": None},
                            "usage": {"output_tokens": 2}},
                           {"type": "message_stop"}]
            encoded = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Relay)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    owner = None
    try:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(config, "PHOTON_API_BASE_URL", f"http://127.0.0.1:{server.server_port}"),
            patch.object(config, "PHOTON_API_KEY", "isolated-test-key"),
            patch.object(config, "PROMPT_INJECTION_CHECK_ENABLED", False),
            patch.object(ProviderRetryPolicy, "PHOTON_OVERLOAD_RETRY_ENABLED", True),
            patch.object(ProviderRetryPolicy, "PHOTON_REJECTION_CONTRACT", "photon-preflight-v1"),
            patch.object(config.TestCaseClassificationAgentConfig, "BATCH_SIZE", 2),
        ):
            owner = design_store.DesignStore(directory)
            await owner.call("open")
            with patch.object(design_store, "store", owner):
                agent = TestCaseClassificationAgent()
                results = []

                async def send(request):
                    message = await agent.run(new_text_message(request.model_dump_json()))
                    result = StageResult.model_validate_json(get_message_text(message))
                    results.append(result)
                    return result

                cases = [TestCase(key=f"TC-{i}", name=f"case-{i}", summary="test", comment="", labels=[],
                                  steps=[], preconditions=None, parent_issue_key="isolated-story").model_dump(mode="json")
                         for i in range(3)]
                run, _ = await owner.call("create_run", "known", {}, 6000000, 120)
                coordinator = DesignCoordinator(send, run)
                payload = {"test_cases": cases}
                classified = await coordinator.stage("classify", payload)
                assert len(classified["test_cases"]) == 3
                assert sum("TC-0" in call["keys"] for call in calls) == 2
                assert sum("TC-2" in call["keys"] for call in calls) == 1
                retried = [call for call in calls if "TC-0" in call["keys"]]
                assert retried[0]["body"] == retried[1]["body"]
                assert len({call["model"] for call in retried}) == 1
                assert await coordinator.stage("classify", payload) == classified
                assert len(calls) == 3
                state = await owner.call("run", run["id"])
                assert state["used"] == 24 and state["reserved"] == 0 and not state["usage_unknown"]

                mode = "unknown"
                run, _ = await owner.call("create_run", "unknown", {}, 6000000, 120)
                coordinator = DesignCoordinator(send, run)
                try:
                    await coordinator.stage("classify", {"test_cases": cases[:1]})
                except ProviderRetryError as exc:
                    assert exc.code == "provider_usage_unknown"
                else:
                    raise AssertionError("Unknown provider outcome must stop the coordinator")
                assert len(calls) == 4
                assert results[-1].error_code == "provider_usage_unknown"
                assert not results[-1].usage.complete
                state = await owner.call("run", run["id"])
                assert state["usage_unknown"] and state["reserved"] > 0

                # Above the SDK's non-streaming ceiling, so classification is streamed like the review stage.
                with patch.object(config.TestCaseClassificationAgentConfig, "MAX_TOKENS", 32000):
                    agent = TestCaseClassificationAgent()
                mode = "truncated"
                run, _ = await owner.call("create_run", "truncated", {}, 6000000, 120)
                coordinator = DesignCoordinator(send, run)
                classified = await coordinator.stage("classify", {"test_cases": cases[1:2]})
                assert len(classified["test_cases"]) == 1
                assert len(calls) == 6 and calls[-1]["body"] == calls[-2]["body"]
                state = await owner.call("run", run["id"])
                assert (state["used"], state["reserved"], state["usage_unknown"]) == ((10 + 32000) + 12, 0, False)

                mode = "always_truncated"
                run, _ = await owner.call("create_run", "always-truncated", {}, 6000000, 120)
                coordinator = DesignCoordinator(send, run)
                try:
                    await coordinator.stage("classify", {"test_cases": cases[2:]})
                except ProviderRetryError as exc:
                    assert exc.code == "provider_stream_incomplete"
                else:
                    raise AssertionError("A stream truncated twice must stop the coordinator")
                assert len(calls) == 8
                state = await owner.call("run", run["id"])
                assert (state["used"], state["reserved"], state["usage_unknown"]) == (2 * (10 + 32000), 0, False)
                return {"http_requests": len(calls), "known_batch_recovered": True,
                        "other_batch_replayed": False, "unknown_stopped": True,
                        "truncated_stream_resent_once": True, "repeated_truncation_stopped_with_known_usage": True}
    finally:
        if owner:
            await owner.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_probe())))
