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

    # Scan behaviour
    scan_interval: int
    scan_query: str

    # SLA thresholds (embedded in Langfuse trace metadata for certifier)
    sla_detect_sec: int
    sla_mitigate_sec: int
    sla_max_tool_calls: int

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
                "MCP_INCLUDE_CHAOS_TOOLS", "true"
            ).lower() in ("true", "1", "yes"),
            scan_interval=int(os.getenv("SCAN_INTERVAL", "0")),
            scan_query=os.getenv(
                "SCAN_QUERY",
                f"Analyse the operational health of all workloads in Kubernetes "
                f"namespace '{k8s_namespace}'. "
                "Identify pod failures, restarts, resource pressure, and anomalies.",
            ),
            sla_detect_sec=int(os.getenv("SLA_DETECT_SEC", "300")),
            sla_mitigate_sec=int(os.getenv("SLA_MITIGATE_SEC", "600")),
            sla_max_tool_calls=int(os.getenv("SLA_MAX_TOOL_CALLS", "20")),
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
