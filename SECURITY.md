# Security status

Initial source publication, 2026-10-22. This snapshot has not been certified for production deployment.

## Verification performed

- 570 ordinary unit tests passed; smoke/live model tests were excluded.
- Ruff checks passed.
- CALM strict validation passed with no issues.
- detect-secrets 1.5.0 was run without network verification. Source findings were reviewed as test-only credentials, header constants or SHA-256 fixture digests; no real credential was confirmed. This is not a guarantee that every possible secret has been detected.
- Organization-specific URLs and identifiers were replaced with synthetic examples; private operation documents and original private Git history are excluded.

## Outstanding findings

The retained dependency lockfile has known vulnerability advisories. On 2026-10-22, `uv audit --frozen --no-dev` reported 123 advisory records across the audited dependency set, including optional service dependencies and overlapping advisory identifiers. These are scanner records, not 123 independently confirmed exploitable vulnerabilities. Dependency upgrades and applicability analysis remain outstanding; no audit exceptions were added to hide the results.

Bandit reported 9 medium and 28 low findings, with no high findings. The medium results concern default temporary directories, intentional service binding to all interfaces, and model downloads without pinned revisions. These findings remain visible rather than being suppressed.

## Deployment boundaries

- The durable workflow requires a successful KB snapshot; successful empty results are valid.
- Generated Feishu tables are shared as editable by tenant members with the link.
- Configure your own model aliases, Meego workflow mappings, strong dashboard credentials, JWT secret and orchestrator key.
- Keep services behind appropriate network access controls. Some Feishu automation webhook endpoints intentionally accept requests without custom authentication headers; deduplication and rate limits do not replace a network trust boundary.
- Optional embedding/prompt-guard services and model downloads need a separate review before enabling them.
- The provided Helm template does not configure state persistence for you. Provision persistent storage before relying on recovery.

Do not include credentials, real requirements or internal request bodies in public issues. Rotate an exposed credential promptly and review Git history and other copies if exposure occurs.
