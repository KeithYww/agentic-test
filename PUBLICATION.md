# Origin and modifications

This project is an independently maintained derivative of [QuAIA / agentic-qa-framework](https://github.com/partarstu/agentic-qa-framework), originally authored by Taras Paruta. Upstream copyright notices and the AGPL-3.0-only license are retained.

Publication preparation date: 2026-10-22.

This initial public snapshot includes a requirement-to-test-case design loop, durable SQLite state, bounded model request accounting, targeted review/repair, untrusted-document handling and Feishu publishing adaptations. It starts a new repository history rather than copying private development history.

Publication changes replace organization-specific endpoints and identifiers with synthetic examples, expose model aliases and Meego field/project mappings through environment variables, replace organization branding in the dashboard, use a locally buildable Docker base, and limit real-model CI to manual runs. Internal operation records and deployment pipelines are excluded. No local model weights are distributed.

The integration semantics are retained: successful empty KB queries continue; missing/unavailable required KB does not. Existing relay protocol and retry-contract identifiers are compatibility names, not a bundled service or promise of compatibility with arbitrary providers. See README for remaining integration requirements and sharing behavior.
