# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Archive the final reviewed design before advancing the Feishu workflow."""

import asyncio
import hashlib
import json
import os
from urllib.parse import quote

import oss2


async def archive_design(project_key: str, story_id: str, run_id: str, payload: dict) -> str:
    """Write a versioned result using the official OSS v4 signer; fail closed."""
    endpoint = os.environ.get("OSS_ENDPOINT", "")
    bucket_name = os.environ.get("OSS_BUCKET", "")
    region = os.environ.get("OSS_REGION", "")
    if not all(
        (endpoint, bucket_name, region, os.environ.get("OSS_ACCESS_KEY_ID"), os.environ.get("OSS_ACCESS_KEY_SECRET"))
    ):
        raise RuntimeError("archive_failed: OSS configuration is incomplete")
    key = "runs/" + "/".join(quote(v, safe="") for v in (project_key, story_id, run_id)) + "/final.json"
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    digest = hashlib.sha256(data).hexdigest()
    auth = oss2.ProviderAuthV4(oss2.credentials.EnvironmentVariableCredentialsProvider())
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region, connect_timeout=30)
    # Do not retry an ambiguous write. The run ID and checksum allow manual reconciliation.
    try:
        await asyncio.to_thread(
            bucket.put_object,
            key,
            data,
            headers={"Content-Type": "application/json", "x-oss-meta-sha256": digest, "x-oss-forbid-overwrite": "true"},
        )
    except oss2.exceptions.RequestError as exc:
        raise RuntimeError(
            f"archive_unknown: run={run_id}; key={key}; sha256={digest}; reconcile before retry"
        ) from exc
    except oss2.exceptions.OssError as exc:
        raise RuntimeError(f"archive_failed: run={run_id}; key={key}") from exc
    return key
