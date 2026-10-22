# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Recording internal-KB and signed OSS boundaries for the real QA pipeline."""

import hashlib
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import Response

app = FastAPI()
_recorded = {"kb_queries": [], "kb_responses": [], "archives": []}


@app.post("/api/photon-kb/external/query/")
async def query(request: Request):
    _recorded["kb_queries"].append(await request.json())
    response = {"confidence": "high", "answer": "", "hits": []}
    _recorded["kb_responses"].append(response)
    return response


@app.put("/{key:path}")
async def archive(key: str, request: Request):
    body = await request.body()
    _recorded["archives"].append({"key": key, "payload": json.loads(body), "at": time.time(),
                                  "signed": request.headers.get("authorization", "").startswith("OSS4-HMAC-SHA256"),
                                  "checksum": request.headers.get("x-oss-meta-sha256")})
    return Response(status_code=200, headers={"ETag": '"'+hashlib.md5(body, usedforsecurity=False).hexdigest()+'"',
                                              "x-oss-request-id": "smoke-request"})


@app.get("/__recorded")
async def recorded():
    return _recorded
