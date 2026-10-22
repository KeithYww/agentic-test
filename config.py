# SPDX-FileCopyrightText: 2025-2026 Taras Paruta (partarstu@gmail.com)
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Centralized configuration for the application.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai.settings import ThinkingLevel

load_dotenv()

# Logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
GOOGLE_CLOUD_LOGGING_ENABLED = os.environ.get("GOOGLE_CLOUD_LOGGING_ENABLED", "False").lower() in ("true", "1", "t")
# When enabled, each service (orchestrator and the Python agents) additionally writes its logs to a rotating file
# under LOG_DIR, named after the service's package (e.g. orchestrator.log, requirements_review.log).
LOG_TO_FILE = os.environ.get("LOG_TO_FILE", "True").lower() in ("true", "1", "t")
LOG_DIR = os.environ.get("LOG_DIR", str(Path(__file__).resolve().parent / "logs"))

# URLs
ORCHESTRATOR_HOST = os.environ.get("ORCHESTRATOR_HOST", "localhost")
ORCHESTRATOR_PORT = int(os.environ.get("ORCHESTRATOR_PORT", "8000"))
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", f"http://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}")
DASHBOARD_PUBLIC_URL = os.environ.get("DASHBOARD_PUBLIC_URL", "").rstrip("/")
JIRA_MCP_SERVER_URL = os.environ.get("JIRA_MCP_SERVER_URL", "http://localhost:9000/sse")
FEISHU_MCP_SERVER_URL = os.environ.get("FEISHU_MCP_SERVER_URL", "http://localhost:9010/sse")
FEISHU_API_BASE_URL = os.environ.get("FEISHU_API_BASE_URL", "https://open.feishu.cn/open-apis")
ZEPHYR_BASE_URL = os.environ.get("ZEPHYR_BASE_URL")
JIRA_BASE_URL = os.environ.get("JIRA_URL")
JIRA_USER = os.environ.get("JIRA_USERNAME")
JIRA_TOKEN = os.environ.get("JIRA_API_TOKEN")

# Webhook URLs
REQUIREMENT_READY_FOR_REVIEW_WEBHOOK_URL = f"{ORCHESTRATOR_URL}/requirement-ready-for-review"
STORY_READY_FOR_TEST_CASE_GENERATION_WEBHOOK_URL = f"{ORCHESTRATOR_URL}/story-ready-for-test-case-generation"
EXECUTE_TESTS_WEBHOOK_URL = f"{ORCHESTRATOR_URL}/execute-tests"
UPDATE_RAG_DB_WEBHOOK_URL = f"{ORCHESTRATOR_URL}/update-rag-db"

# Secrets
FEISHU_APP_ID = os.environ.get("FEISHU_APP_ID")
FEISHU_APP_SECRET = os.environ.get("FEISHU_APP_SECRET")
# Shared secret guarding the internal embedding and prompt-guard services. When set, those
# services require a matching X-API-Key header and their clients send it. Left unset, the
# services stay open (they are expected to be reachable only on a private network).
INTERNAL_SERVICE_API_KEY = os.environ.get("INTERNAL_SERVICE_API_KEY")
ZEPHYR_API_TOKEN = os.environ.get("ZEPHYR_API_TOKEN")
XRAY_BASE_URL = os.environ.get("XRAY_BASE_URL")
XRAY_CLIENT_ID = os.environ.get("XRAY_CLIENT_ID")
XRAY_CLIENT_SECRET = os.environ.get("XRAY_CLIENT_SECRET")
XRAY_PRECONDITIONS_FIELD_ID = os.environ.get("XRAY_PRECONDITIONS_FIELD_ID", "Pre-conditions")

# Photon API - internal LLM relay (Anthropic-compatible protocol)
PHOTON_API_BASE_URL = os.environ.get("PHOTON_API_BASE_URL", "https://llm.example.com")
PHOTON_API_KEY = os.environ.get("PHOTON_API_KEY")

# Agent
AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost")
MCP_SERVER_ATTACHMENTS_FOLDER_PATH = os.environ.get("MCP_SERVER_ATTACHMENTS_FOLDER_PATH", "/tmp")
ATTACHMENTS_LOCAL_DESTINATION_FOLDER_PATH = os.environ.get("ATTACHMENTS_LOCAL_DESTINATION_FOLDER_PATH", "/tmp")
JIRA_ATTACHMENT_SKIP_POSTFIX = os.environ.get("JIRA_ATTACHMENT_SKIP_POSTFIX", "_SKIP")
MCP_SERVER_TIMEOUT_SECONDS = 30
SUPPORTED_ATTACHMENT_MIME_TYPES: set[str] = {
    # Images
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    # Documents
    "application/pdf",
    "text/plain",
    "application/json",
    # Audio
    "audio/mpeg",
    "audio/wav",
    "audio/flac",
    "audio/ogg",
    "audio/aac",
    "audio/aiff",
    # Video
    "video/mp4",
    "video/webm",
    "video/quicktime",
    "video/x-matroska",
    "video/x-flv",
    "video/mpeg",
    "video/x-ms-wmv",
    "video/3gpp",
}

# Test Management System
ZEPHYR_COMMENTS_CUSTOM_FIELD_NAME = "Review Comments"
ZEPHYR_CLIENT_TIMEOUT_SECONDS = 15
ZEPHYR_CUSTOM_FIELDS_JSON_FIELD_NAME = "customFields"
TEST_MANAGEMENT_SYSTEM = os.environ.get("TEST_MANAGEMENT_SYSTEM", "zephyr").lower()

# Meego (Feishu Project) — used when TEST_MANAGEMENT_SYSTEM=meego
MEEGO_BASE_URL = os.environ.get("MEEGO_BASE_URL", "https://project.feishu.cn")
MEEGO_PLUGIN_ID = os.environ.get("MEEGO_PLUGIN_ID")
MEEGO_PLUGIN_SECRET = os.environ.get("MEEGO_PLUGIN_SECRET")

FEISHU_BOT_WEBHOOK_URL = os.environ.get("FEISHU_BOT_WEBHOOK_URL", "")
MEEGO_PROJECT_KEY = os.environ.get("MEEGO_PROJECT_KEY", "example-space")
MEEGO_USER_KEY = os.environ.get("MEEGO_USER_KEY")
# The node_key of the "AI用例预生成" workflow node. Different workflow templates may assign
# different internal keys to this node; override via env var when the default doesn't match.
MEEGO_AI_PREGENERATION_NODE_KEY = os.environ.get("MEEGO_AI_PREGENERATION_NODE_KEY", "state_48")

# Feishu Bitable test-case sync (a brand-new, per-run Bitable — never writes into the team's
# existing shared table). This is now the sole test-case artifact the pipeline produces (the
# Feishu Doc report path has been removed), so it defaults to enabled.
FEISHU_BITABLE_SYNC_ENABLED = os.environ.get("FEISHU_BITABLE_SYNC_ENABLED", "True").lower() in ("true", "1", "t")
# JSON object mapping a story's "业务线" (business line) option_id to its display name, used
# only to label the generated Bitable's name — not for routing (each run gets its own base
# with a freshly-defined, clean field schema, regardless of business line).
FEISHU_BUSINESS_LINE_OPTION_NAMES_JSON = os.environ.get(
    "FEISHU_BUSINESS_LINE_OPTION_NAMES_JSON",
    "{}",
)

# Comma-separated Feishu open_ids of users who should receive edit access on every
# bot-created report document (e.g. "ou_xxx,ou_yyy"). Empty = no automatic sharing.
FEISHU_DOC_COLLABORATOR_IDS: list[str] = [
    uid.strip() for uid in os.environ.get("FEISHU_DOC_COLLABORATOR_IDS", "").split(",") if uid.strip()
]

# photon_kb (testplatform internal knowledge base) — TEMPORARY integration for test-case
# generation context. Configure your own compatible KB endpoint and key.
# The durable generation flow requires a successful query; an empty result is valid.
KB_QUERY_ENABLED = os.environ.get("KB_QUERY_ENABLED", "True").lower() in ("true", "1", "t")
KB_BASE_URL = os.environ.get("KB_BASE_URL", "https://kb.example.com")
KB_API_KEY = os.environ.get("KB_API_KEY", "")
KB_QUERY_TIMEOUT_SECONDS = float(os.environ.get("KB_QUERY_TIMEOUT_SECONDS", "60"))

# Test Reporting
TEST_REPORTER = os.environ.get("TEST_REPORTER", "allure").lower()
ALLURE_RESULTS_DIR = "allure-results"
ALLURE_REPORT_DIR = "allure-report"

# OpenTelemetry (exported to a self-hosted Langfuse instance; empty endpoint disables tracing)
OPEN_TELEMETRY_URL = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
# Raw "Key1=Value1,Key2=Value2" header string, e.g. "Authorization=Basic <base64(public_key:secret_key)>".
OTEL_EXPORTER_OTLP_HEADERS = os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
OTEL_SERVICE_NAME_PREFIX = os.environ.get("OTEL_SERVICE_NAME_PREFIX", "quaia")

# Common model config
TOP_P = 1.0
TEMPERATURE = 0.0


class DesignConfig:
    """Bounds for the local generation/review/fix workflow."""

    REQUIREMENT_REVIEW_ENABLED = os.environ.get("REQUIREMENT_REVIEW_ENABLED", "false").lower() == "true"
    OSS_ARCHIVE_ENABLED = os.environ.get("OSS_ARCHIVE_ENABLED", "false").lower() == "true"
    MAX_REVIEWS = int(os.environ.get("DESIGN_MAX_REVIEWS", "3"))
    FIX_MIN_SEVERITY = os.environ.get("DESIGN_FIX_MIN_SEVERITY", "medium")
    TIMEOUT_SECONDS = float(os.environ.get("DESIGN_TIMEOUT_SECONDS", "7200"))
    TOKEN_LIMIT = int(os.environ.get("DESIGN_TOKEN_LIMIT", "1000000"))
    MODEL_CONCURRENCY = int(os.environ.get("DESIGN_MODEL_CONCURRENCY", "4"))
    STAGE_CACHE_SIZE = int(os.environ.get("DESIGN_STAGE_CACHE_SIZE", "128"))
    MAX_NEW_CASES_PER_ROUND = int(os.environ.get("DESIGN_MAX_NEW_CASES_PER_ROUND", "10"))
    # A known context window is required; unknown Photon aliases are not guessed.
    CONTEXT_WINDOW = int(os.environ.get("DESIGN_CONTEXT_WINDOW", "0"))
    REQUIREMENT_FOCUS_COUNT = int(os.environ.get("REQUIREMENT_FOCUS_COUNT", "5"))
    STATE_DIR = os.environ.get("DESIGN_STATE_DIR", "")


class BudgetConfig:
    """Token budget (hard limit) and pricing used for cost oversight."""

    # Hard cap on the total number of tokens an agent may consume per task. When exceeded,
    # the agent run is aborted with pydantic-ai's UsageLimitExceeded. The cap is token-based
    # because pydantic-ai enforces token limits, not monetary ones.
    TOTAL_TOKENS_LIMIT_PER_TASK = int(os.environ.get("TOTAL_TOKENS_LIMIT_PER_TASK", "1000000"))

    # Indicative price in USD per 1,000,000 tokens, keyed by the relay model name.
    # Used only to estimate cost for oversight (logs + dashboard); keep these values current
    # with the provider's published pricing. Models absent from this table report a null cost.
    # TODO: confirm exact Photon API relay pricing for these models and fill in below.
    MODEL_PRICING: dict[str, dict[str, float]] = {}

# Prompt injection detection config
PROMPT_INJECTION_CHECK_ENABLED = os.environ.get("PROMPT_INJECTION_CHECK_ENABLED", "False").lower() in ("true", "1", "t")
PROMPT_GUARD_PROVIDER = os.environ.get("PROMPT_GUARD_PROVIDER", "protect_ai")
PROMPT_INJECTION_MIN_SCORE = float(os.environ.get("PROMPT_INJECTION_MIN_SCORE", "0.8"))
LOCAL_MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_models")
PROMPT_INJECTION_DETECTION_MODEL_PATH = os.path.join(LOCAL_MODELS_PATH, "prompt_detection_model")
PROMPT_INJECTION_DETECTION_MODEL_NAME = os.environ.get(
    "PROMPT_INJECTION_MODEL_NAME", "ProtectAI/deberta-v3-base-prompt-injection-v2"
)
PROMPT_GUARD_SERVICE_URL = os.environ.get("PROMPT_GUARD_SERVICE_URL")


# Orchestrator
class OrchestratorConfig:
    THINKING_LEVEL: ThinkingLevel = "low"
    AUTOMATED_TC_LABEL = "automated"
    AGENTS_DISCOVERY_INTERVAL_SECONDS = 300
    # Initial discovery (at orchestrator startup) races the other in-process agents' uvicorn
    # startup in the all-in-one deployment (all launched concurrently). A short bounded retry
    # here closes that window in seconds instead of waiting for the next periodic cycle.
    INITIAL_DISCOVERY_MAX_ATTEMPTS = 5
    INITIAL_DISCOVERY_RETRY_DELAY_SECONDS = 2.0
    AGENT_HEALTH_CHECK_INTERVAL_SECONDS = 60
    AGENT_HEALTH_CHECK_TIMEOUT_SECONDS = 10
    TASK_EXECUTION_TIMEOUT = float(os.environ.get("TASK_EXECUTION_TIMEOUT", "7200"))
    AGENT_DISCOVERY_TIMEOUT_SECONDS = 120
    INCOMING_REQUEST_WAIT_TIMEOUT = AGENT_DISCOVERY_TIMEOUT_SECONDS + 5
    MODEL_NAME = os.environ.get("ORCHESTRATOR_CONFIG_MODEL_NAME", "deepseek-v4.1-flash")
    FALLBACK_MODEL_NAME = os.environ.get("ORCHESTRATOR_CONFIG_FALLBACK_MODEL_NAME", "qwen3.8-flash")
    API_KEY = os.environ.get("ORCHESTRATOR_API_KEY")
    AGENT_DISCOVERY_PORTS = os.environ.get("AGENT_DISCOVERY_PORTS", "8001-8007")
    REMOTE_EXECUTION_AGENT_HOSTS = os.environ.get("REMOTE_EXECUTION_AGENT_HOSTS", AGENT_BASE_URL)
    # Shared bearer token expected by the execution agents' main A2A endpoint. Empty means the agents run without
    # auth (e.g. local dev), so no Authorization header is attached.
    REMOTE_EXECUTION_AGENT_AUTH_TOKEN = os.environ.get("REMOTE_EXECUTION_AGENT_AUTH_TOKEN", "")
    # Abuse guard for the unauthenticated /requirement-ready-for-review webhook (see
    # _RequirementReviewGuard in orchestrator/main.py): per-work-item dedup window and a global
    # submission rate cap, both in-memory.
    REQUIREMENT_REVIEW_DEDUP_WINDOW_SECONDS = float(os.environ.get("REQUIREMENT_REVIEW_DEDUP_WINDOW_SECONDS", "300"))
    REQUIREMENT_REVIEW_MAX_PER_MINUTE = int(os.environ.get("REQUIREMENT_REVIEW_MAX_PER_MINUTE", "5"))
    # Abuse guard for the unauthenticated /story-ready-for-test-case-generation webhook.
    TC_GEN_WEBHOOK_DEDUP_WINDOW_SECONDS = float(os.environ.get("TC_GEN_WEBHOOK_DEDUP_WINDOW_SECONDS", "300"))
    TC_GEN_WEBHOOK_MAX_PER_MINUTE = int(os.environ.get("TC_GEN_WEBHOOK_MAX_PER_MINUTE", "5"))
    # Maximum number of concurrent tasks a single agent process can handle safely.
    MAX_SLOTS_PER_AGENT = int(os.environ.get("MAX_SLOTS_PER_AGENT", "3"))
    # Maximum number of pipelines dispatched concurrently by the orchestrator.
    # Must be ≤ MAX_SLOTS_PER_AGENT to avoid overwhelming a single agent.
    MAX_CONCURRENT_PIPELINES = int(os.environ.get("MAX_CONCURRENT_PIPELINES", "3"))


# Dashboard Authentication
class DashboardAuthConfig:
    """Configuration for UI dashboard authentication."""

    USERNAME = os.environ.get("DASHBOARD_USERNAME", "")
    PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
    JWT_SECRET = os.environ.get("DASHBOARD_JWT_SECRET", "")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_HOURS = int(os.environ.get("DASHBOARD_JWT_EXPIRE_HOURS", "24"))


# Requirements Review Agent
class RequirementsReviewAgentConfig:
    THINKING_LEVEL: ThinkingLevel = "medium"
    OWN_NAME = "Requirements Reviewer"
    PORT = int(os.environ.get("PORT", "8001"))
    EXTERNAL_PORT = int(os.environ.get("EXTERNAL_PORT", PORT))
    PROTOCOL = "http"
    MODEL_NAME = os.environ.get("REQUIREMENTS_REVIEW_AGENT_CONFIG_MODEL_NAME", "claude-sonnet-5")
    FALLBACK_MODEL_NAME = os.environ.get("REQUIREMENTS_REVIEW_AGENT_CONFIG_FALLBACK_MODEL_NAME", "gpt-5.6-sol")
    MAX_REQUESTS_PER_TASK = 30


# Test Case Classification Agent
class TestCaseClassificationAgentConfig:
    THINKING_LEVEL: ThinkingLevel = "low"
    OWN_NAME = "Test Case Classification Agent"
    PORT = int(os.environ.get("PORT", "8003"))
    EXTERNAL_PORT = int(os.environ.get("EXTERNAL_PORT", PORT))
    PROTOCOL = "http"
    MODEL_NAME = os.environ.get("TEST_CASE_CLASSIFICATION_AGENT_CONFIG_MODEL_NAME", "deepseek-v4.1-flash")
    FALLBACK_MODEL_NAME = os.environ.get("TEST_CASE_CLASSIFICATION_AGENT_CONFIG_FALLBACK_MODEL_NAME", "qwen3.8-flash")
    MAX_REQUESTS_PER_TASK = 30
    MAX_TOKENS = 16000
    BATCH_SIZE = 10
    TOTAL_TOKENS_LIMIT_PER_TASK = 4_000_000


# Test Case Generation Agent
class PrdClassifierConfig:
    """Thresholds for PRD type classification and section chunking."""

    FUNCTIONAL_SECTION_THRESHOLD = int(os.environ.get("PRD_SECTION_THRESHOLD", "5"))
    SECTION_MAX_CHARS = int(os.environ.get("PRD_SECTION_MAX_CHARS", "3000"))


class TestCaseGenerationAgentConfig:
    AC_EXTRACTOR_THINKING_LEVEL: ThinkingLevel = "minimal"
    TC_GENERATOR_THINKING_LEVEL: ThinkingLevel = "low"
    TC_REPAIR_THINKING_LEVEL: ThinkingLevel = "high"
    ORCHESTRATOR_THINKING_LEVEL: ThinkingLevel = "low"
    AC_EXTRACTOR_MAX_TOKENS = int(os.environ.get("AC_EXTRACTOR_MAX_TOKENS", "32768"))
    TC_REPAIR_MAX_TOKENS = int(os.environ.get("TC_REPAIR_MAX_TOKENS", "32768"))
    TC_GENERATOR_MAX_TOKENS = int(os.environ.get("TC_GENERATOR_MAX_TOKENS", "65536"))
    OWN_NAME = "Test Case Generation Agent"
    PORT = int(os.environ.get("PORT", "8002"))
    EXTERNAL_PORT = int(os.environ.get("EXTERNAL_PORT", PORT))
    PROTOCOL = "http"
    MODEL_NAME = os.environ.get("TEST_CASE_GENERATION_AGENT_CONFIG_MODEL_NAME", "gpt-5.6-terra")
    FALLBACK_MODEL_NAME = os.environ.get("TEST_CASE_GENERATION_AGENT_CONFIG_FALLBACK_MODEL_NAME", "deepseek-v4.1-flash")
    REPAIR_MODEL_NAME = os.environ.get("TEST_CASE_GENERATION_AGENT_CONFIG_REPAIR_MODEL_NAME", "deepseek-v4-flash")
    MAX_REQUESTS_PER_TASK = 30
    # Read Phase 1 env var as fallback for zero-downtime migration
    AC_BATCH_SIZE = max(1, int(
        os.environ.get("TC_GEN_AC_BATCH_SIZE") or os.environ.get("STEPS_GENERATION_AC_BATCH_SIZE", "8")
    ))
    LLM_CONCURRENCY_LIMIT = int(os.environ.get("TC_GEN_LLM_CONCURRENCY_LIMIT", "8"))
    TC_VALIDATION_MAX_RETRIES = int(os.environ.get("TC_VALIDATION_MAX_RETRIES", "1"))


# Test Case Review Agent
class TestCaseReviewAgentConfig:
    THINKING_LEVEL: ThinkingLevel = "low"
    REVIEW_COMPLETE_STATUS_NAME = "Review Complete"
    OWN_NAME = "Test Case Review Agent"
    PORT = int(os.environ.get("PORT", "8004"))
    EXTERNAL_PORT = int(os.environ.get("EXTERNAL_PORT", PORT))
    PROTOCOL = "http"
    MODEL_NAME = os.environ.get("TEST_CASE_REVIEW_AGENT_CONFIG_MODEL_NAME", "claude-sonnet-5")
    FALLBACK_MODEL_NAME = os.environ.get("TEST_CASE_REVIEW_AGENT_CONFIG_FALLBACK_MODEL_NAME", "gpt-5.6-sol")
    MAX_REQUESTS_PER_TASK = 30
    TEST_CASE_REVIEW_BATCH_SIZE = 10
    MAX_TOKENS = int(os.environ.get("TC_REVIEW_MAX_TOKENS", "32000"))
    TOTAL_TOKENS_LIMIT_PER_TASK = 4_000_000


# Incident Creation Agent
class IncidentCreationAgentConfig:
    THINKING_LEVEL: ThinkingLevel = "low"
    OWN_NAME = "Incident Creation Agent"
    PORT = int(os.environ.get("PORT", "8007"))
    EXTERNAL_PORT = int(os.environ.get("EXTERNAL_PORT", PORT))
    PROTOCOL = "http"
    MODEL_NAME = os.environ.get("INCIDENT_CREATION_AGENT_CONFIG_MODEL_NAME", "claude-sonnet-5")
    FALLBACK_MODEL_NAME = os.environ.get("INCIDENT_CREATION_AGENT_CONFIG_FALLBACK_MODEL_NAME", "deepseek-v4.1-flash")
    MAX_REQUESTS_PER_TASK = 30
    MIN_SIMILARITY_SCORE = float(os.environ.get("INCIDENT_AGENT_MIN_SIMILARITY_SCORE", "0.7"))
    ISSUE_PRIORITY_FIELD_ID = os.environ.get("ISSUE_PRIORITY_FIELD_ID", "priority")
    ISSUE_SEVERITY_FIELD_NAME = os.environ.get("ISSUE_SEVERITY_FIELD_NAME", "customfield_10124")
    # Severity values: comma-separated list of "value:description" pairs
    SEVERITY_VALUES = os.environ.get(
        "INCIDENT_AGENT_SEVERITY_VALUES",
        "'10020':blocker or crash,'10021':functional failure,'10022':UI/UX issue,'10023':typo or minor visual issue",
    )
    # Priority values: comma-separated list of "value:description" pairs
    PRIORITY_VALUES = os.environ.get(
        "INCIDENT_AGENT_PRIORITY_VALUES", "High:immediate fix,Medium:normal release,Low:backlog"
    )
    # Jira statuses considered terminal — bugs in these statuses are excluded from duplicate detection
    TERMINAL_STATUSES = os.environ.get(
        "INCIDENT_AGENT_TERMINAL_STATUSES", "Closed,Done,Duplicate,Rejected,Won't Fix,Cannot Reproduce,Resolved"
    ).split(",")


class RetryConfig:
    MAX_RETRIES = 3
    RETRYABLE_STATUS_CODES = {404, 429, 500, 502, 503, 504}
    RETRY_BASE_DELAY_SECONDS = 5.0
    LLM_RESULTS_EXTRACTOR_RETRY_BASE_DELAY_SECONDS = float(os.environ.get("LLM_RETRY_BASE_DELAY_SECONDS", "15.0"))


class QdrantConfig:
    URL = os.environ.get("QDRANT_URL", "http://localhost")
    API_KEY = os.environ.get("QDRANT_API_KEY")
    TIMEOUT_SECONDS = int(os.environ.get("QDRANT_TIMEOUT_SECONDS", "30"))
    PORT = int(os.environ.get("QDRANT_PORT", "6333"))
    COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "jira_issues")
    TICKETS_COLLECTION_NAME = os.environ.get("QDRANT_TICKETS_COLLECTION_NAME", "jira_issues")
    METADATA_COLLECTION_NAME = os.environ.get("QDRANT_METADATA_COLLECTION_NAME", "rag_metadata")
    MIN_SIMILARITY_SCORE = float(os.environ.get("RAG_MIN_SIMILARITY_SCORE", "0.7"))
    MAX_RESULTS = int(os.environ.get("RAG_MAX_RESULTS", "5"))
    EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    EMBEDDING_MODEL_PATH = os.path.join(LOCAL_MODELS_PATH, "embedding_model")
    EMBEDDING_SERVICE_URL = os.environ.get("EMBEDDING_SERVICE_URL")
    EMBEDDING_SERVICE_TIMEOUT_SECONDS = float(os.environ.get("EMBEDDING_SERVICE_TIMEOUT_SECONDS", "120.0"))
    EMBEDDING_SERVICE_MAX_RETRIES = int(os.environ.get("EMBEDDING_SERVICE_MAX_RETRIES", "6"))
    EMBEDDING_SERVICE_RETRY_BACKOFF_CAP_SECONDS = float(
        os.environ.get("EMBEDDING_SERVICE_RETRY_BACKOFF_CAP_SECONDS", "32.0")
    )
    VALID_STATUSES = os.environ.get(
        "JIRA_VALID_STATUSES", "To Do,In Review,Ready for Development,In Progress,Done"
    ).split(",")
    BUG_ISSUE_TYPE = os.environ.get("JIRA_BUG_ISSUE_TYPE", "Bug")
