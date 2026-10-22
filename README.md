# Agentic Test

An agentic QA framework for turning requirements into reviewed test cases, with durable execution and human handoff.

This is an independently maintained derivative of [QuAIA / agentic-qa-framework](https://github.com/partarstu/agentic-qa-framework), originally authored by Taras Paruta. The original copyright and **AGPL-3.0-only** license are retained. See [LICENSE](LICENSE) and [PUBLICATION.md](PUBLICATION.md).

[简体中文](README.zh-CN.md)

## What is included

- Requirement extraction, structured test-case generation, case/suite review, targeted repair and classification.
- SQLite checkpoints, bounded model attempts, token budgets and recovery tools.
- A2A/MCP communication, a React dashboard, progress events and trace redaction.
- Feishu document reading and per-run Bitable publishing, plus Meego, Jira/Zephyr and Xray adapters.
- Unit tests, external-boundary mocks, Docker build files and CALM architecture validation.

## Current integration boundary

This is an integration framework, **not a zero-configuration demo**. No company credentials, production endpoints, real requirements, internal run reports or original private Git history are included.

The main generation flow still needs your own compatible services:

1. An Anthropic-compatible model endpoint. `PHOTON_*` environment names and relay-specific retry contracts are retained for compatibility. Existing model aliases are examples from the source implementation; configure aliases supported by your endpoint. Public-provider compatibility has not been certified.
2. A KB endpoint implementing `POST /api/photon-kb/external/query/`, authenticated by `X-KB-Api-Key`, accepting `{"question":"...","role":"qa"}` and returning `{"answer":"...","confidence":"high"}`. An empty answer is a valid successful result; disabled or unavailable KB stops the durable generation flow. The recording mock in `tests/smoke/mocks/qa_boundary_mock.py` is a test boundary, not a production KB.
3. Your Feishu application, authenticated `lark-cli` bot identity, and document/Bitable permissions. **Generated tables are configured so anyone in your Feishu tenant with the link can edit.** Review this sharing behavior before using private requirements.
4. Your Meego project key, workflow node and field IDs. Set `MEEGO_FIELD_*`, `MEEGO_AI_PREGENERATION_NODE_KEY` and `MEEGO_PROJECT_KEY_MAP_JSON` for your own template. The checked-in defaults are synthetic examples.

Requirement review and OSS archival are disabled by default. Reaching the review limit hands remaining findings to human review; completion does not mean every quality issue was resolved. Optional prompt-injection screening is disabled by default; untrusted-content instructions do not guarantee protection against every injection.

## Development

Requires Python 3.14, [uv](https://docs.astral.sh/uv/), Node.js 20+ for UI/CALM, and `lark-cli` for Feishu access.

```bash
uv sync --frozen
cp .env.example .env
# Fill in your services, credentials, model aliases and workflow identifiers.
uv run python -m scripts.start_all
```

The dashboard listens on port 8000. Configure strong dashboard credentials, JWT secret and orchestrator API key. Keep the service on a trusted network while configuring its webhooks and access rules. For the built dashboard, build `orchestrator/ui` first or use Docker.

For durable single-instance runs, set `DESIGN_STATE_DIR` to a writable directory and retain it across restarts. Multiple processes sharing the state directory are not supported by this deployment path.

```bash
docker build -t agentic-qa-base:latest -f Dockerfile.base .
docker build --build-arg BASE_IMAGE=agentic-qa-base:latest -t agentic-test:local -f Dockerfile.all .
docker run --rm --env-file .env -p 127.0.0.1:8000:8000 -v quaia-state:/app/state -e DESIGN_STATE_DIR=/app/state agentic-test:local
```

Build files retain configurable public mirror defaults. The Helm chart is an advanced deployment template; persistent storage must be configured before relying on recovery. A public end-to-end deployment has not yet been verified. Known dependency and static-analysis findings are recorded in [SECURITY.md](SECURITY.md).

## Checks

```bash
uv run ruff check .
uv run pytest tests/ -m 'not smoke and not live_llm'
```

Live and smoke tests use real, billed model calls and require separately configured services. GitHub Actions runs ordinary checks on PRs; the real-model smoke job is manual only. CI credentials are never needed for ordinary unit tests.

```bash
cd calm
npx -y @finos/calm-cli@1.46.0 validate -p patterns/quaia.pattern.json -a architecture/quaia.arch.json -u url-mapping.json --strict -f pretty
```

## License

[GNU Affero General Public License v3.0 only](LICENSE). Preserve upstream attribution and identify modifications when redistributing. Local model weights are not bundled; obtain them under their respective licenses.
