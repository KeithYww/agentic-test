# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Mock for the Langfuse OTLP HTTP trace-ingestion endpoint.

Only asserts that spans were actually exported by the orchestrator/agents — it does not
decode the OTLP protobuf payload, just records that a request landed and how many.
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

_recorded: dict[str, int] = {"traces_received": 0}

app = FastAPI(title="otel-collector-mock")


@app.post("/v1/traces")
async def receive_traces(request: Request) -> JSONResponse:
    await request.body()
    _recorded["traces_received"] += 1
    return JSONResponse({"partialSuccess": {}})


@app.get("/__recorded")
async def get_recorded() -> JSONResponse:
    return JSONResponse(_recorded)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9020)
