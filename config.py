"""
Flash Agent – Configuration
============================

Centralised configuration loaded from environment variables.
All agent settings in one place for easy inspection and validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env / .env.local if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        env_file = ".env.local" if Path(".env.local").exists() else ".env"
        load_dotenv(env_file, override=True)
    except ImportError:
        pass


@dataclass
class AgentConfig:
    """All agent configuration in one place."""

    # Agent identity
    agent_name: str
    agent_id: str
    agent_mode: str
    k8s_namespace: str
    k8s_node_ip: str

    # LLM Gateway
    openai_base_url: str
    openai_api_key: str
    model_alias: str
    azure_api_version: str

    # MCP Servers
    k8s_mcp_url: str
    prom_mcp_url: str
    mcp_timeout: int

    # Storage
    mcp_interactions_file: str

    # Chaos context
    chaos_namespace: str
    target_app_name: str
    include_chaos_tools: bool
    enable_prometheus_snapshot: bool  # always-on Prom enrichment in Step 2b

    # Scan behaviour
    scan_interval: int
    scan_interval_fast: int       # used while last scan reported non-Healthy state
    scan_health_threshold: int    # overall_health_score below this counts as 'fault window'
    scan_query: str

    # MCP event recency window (seconds). Cold-start floor when the agent has
    # no prior watermark, and a slack ceiling we never look beyond. Set to
    # ~3× scan_interval so a single missed scan still recovers gracefully.
    event_recency_fallback_sec: int

    # Reasoning mode ("single-shot" or "react"). single-shot pre-fetches
    # all MCP data and runs one analysis call. react opts into a multi-turn
    # tool-calling loop modeled on AIOpsLab where the LLM iteratively chooses
    # which observation tool to call next. ReAct costs 5-10x tokens per scan
    # so it is opt-in; default is single-shot.
    reasoning_mode: str
    react_max_steps: int

    # Experiment identity – injected by sidecar at runtime
    notify_id: str          # = experiment_run_id / Langfuse trace_id

    # Langfuse (optional – for Tier-2 metadata updates)
    langfuse_host: str
    langfuse_public_key: str
    langfuse_secret_key: str

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Create configuration from environment variables."""
        _load_dotenv()

        k8s_namespace = os.getenv("K8S_NAMESPACE", "")

        return cls(
            agent_name=os.getenv("AGENT_NAME", ""),
            agent_id=os.getenv("AGENT_ID", ""),
            agent_mode=os.getenv("AGENT_MODE", ""),
            k8s_namespace=k8s_namespace,
            k8s_node_ip=os.getenv("K8S_NODE_IP", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            model_alias=os.getenv("MODEL_ALIAS", ""),
            azure_api_version=os.getenv("AZURE_API_VERSION", ""),
            k8s_mcp_url=os.getenv("K8S_MCP_URL", ""),
            prom_mcp_url=os.getenv("PROM_MCP_URL", ""),
            mcp_timeout=int(os.getenv("MCP_TIMEOUT", "30")),
            mcp_interactions_file=os.getenv("MCP_INTERACTIONS_FILE", ""),
            chaos_namespace=os.getenv("CHAOS_NAMESPACE", "litmus"),
            target_app_name=os.getenv("TARGET_APP_NAME", "sock-shop"),
            include_chaos_tools=os.getenv(
                "MCP_INCLUDE_CHAOS_TOOLS", "false"
            ).lower() in ("true", "1", "yes"),
            enable_prometheus_snapshot=os.getenv(
                "ENABLE_PROMETHEUS_SNAPSHOT", "true"
            ).lower() in ("true", "1", "yes"),
            scan_interval=int(os.getenv("SCAN_INTERVAL", "0")),
            scan_interval_fast=int(os.getenv("SCAN_INTERVAL_FAST", "15")),
            scan_health_threshold=int(os.getenv("SCAN_HEALTH_THRESHOLD", "100")),
            event_recency_fallback_sec=int(
                os.getenv("EVENT_RECENCY_FALLBACK_SEC", "90")
            ),
            scan_query=os.getenv(
                "SCAN_QUERY",
                f"Analyse the operational health of all workloads in Kubernetes "
                f"namespace '{k8s_namespace}'. "
                "Identify pod failures, restarts, resource pressure, and anomalies.",
            ),
            reasoning_mode=os.getenv("AGENT_REASONING_MODE", "single-shot").strip().lower(),
            react_max_steps=int(os.getenv("AGENT_REACT_MAX_STEPS", "8")),
            notify_id=os.getenv("NOTIFY_ID", ""),
            langfuse_host=os.getenv("LANGFUSE_HOST", ""),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        )

    def validate(self) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        errors = []
        if not self.agent_name:
            errors.append("AGENT_NAME is required")
        if not self.k8s_namespace:
            errors.append("K8S_NAMESPACE is required")
        if not self.openai_base_url:
            errors.append("OPENAI_BASE_URL is required")
        if not self.model_alias:
            errors.append("MODEL_ALIAS is required")
        if not self.k8s_mcp_url:
            errors.append("K8S_MCP_URL is required")
        if not self.mcp_interactions_file:
            errors.append("MCP_INTERACTIONS_FILE is required")
        return errors
