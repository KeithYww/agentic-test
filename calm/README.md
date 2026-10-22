<!--
SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)

SPDX-License-Identifier: AGPL-3.0-only
-->

# Architecture as Code (CALM)

This directory holds the QuAIA architecture described with the
[FINOS CALM](https://calm.finos.org/) (Common Architecture Language Model)
standard. It is validated in CI as a **blocking** gate, so the model cannot
drift away from the system without the build failing.

## Layout

| Path | Purpose |
|---|---|
| `architecture/quaia.arch.json` | The architecture instance: every service/actor (`nodes`), the integration edges between them (`relationships`), and the security `controls` attached to them. |
| `patterns/quaia.pattern.json` | The governance pattern. A JSON-Schema that asserts the required nodes, relationships and controls are present. This is what makes the gate fail on drift. |
| `controls/requirements/*.json` | Control requirement schemas describing the shape of each control's configuration (authentication, prompt-injection protection, untrusted document scope). |
| `url-mapping.json` | Maps the control `requirement-url` identifiers to their local schema files so validation runs fully offline. |

## Controls enforced by the pattern

| Control | Attached to | Source mechanism |
|---|---|---|
| `orchestrator-api-key` | `orchestrator` node | `X-API-Key` on control/webhook endpoints (`ORCHESTRATOR_API_KEY`) |
| `dashboard-jwt` | `orchestrator` node | JWT on dashboard endpoints (`DASHBOARD_JWT_SECRET`) |
| `prompt-injection-guard` | every agent node | Prompt-injection screening (`PROMPT_INJECTION_CHECK_ENABLED`) |
| `untrusted-document-scope` | `test-case-generation-agent`, `test-case-review-agent`, `requirements-review-agent` | PRD/knowledge text sent only as JSON fields under `common/untrusted_content.py` policy; scope from every PRD section, never from document wording |
| `internal-service-api-key` | `embedding-service`, `prompt-guard-service` | Shared `X-API-Key` (`INTERNAL_SERVICE_API_KEY`) |

Note: `rel-requirement-review-webhook-orchestrator` (Feishu Project's native automation calling
`/requirement-ready-for-review`) intentionally carries no control — that action cannot attach custom
headers, so the endpoint is unauthenticated by design and relies on an in-memory dedup/rate-limit
guard instead (see `orchestrator/main.py::_RequirementReviewGuard`). It is modeled in the
architecture instance but, like the sibling `rel-feishu-project-webhook-orchestrator` edge, is not
asserted by the pattern.

## Running validation locally

Requires [Node.js](https://nodejs.org/) 20+.

```bash
# from the repository root
cd calm

# one-off, no global install
npx -y @finos/calm-cli@1.46.0 validate \
  -p patterns/quaia.pattern.json \
  -a architecture/quaia.arch.json \
  -u url-mapping.json \
  --strict -f pretty
```

A clean run prints `No issues found.` and exits `0`. Removing a required node,
relationship or control makes validation exit non-zero — the same check the
`Architecture (CALM)` CI job runs.

## Changing the architecture

When you add or remove an agent, service or integration edge, update
`architecture/quaia.arch.json` and, if the change is part of the contract you
want enforced, `patterns/quaia.pattern.json`. Run the command above before
committing.
