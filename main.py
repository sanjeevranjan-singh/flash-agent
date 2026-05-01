"""
Flash Agent v4.0.0 – Entry Point
==================================

Thin orchestrator harness.  Loads configuration, sets up logging,
registers signal handlers, and drives the scan loop.

All domain logic lives in dedicated modules:
  config.py          – AgentConfig dataclass
  flash_agent.py     – FlashAgent (3-step agentic pipeline)
  mcp/               – MCP JSON-RPC client & parsers
  llm/               – LLM gateway & prompt templates
  domain/            – Litmus / Argo domain helpers
  observability/     – Langfuse metadata & MCP JSONL logger
"""

from __future__ import annotations

import logging
import os
import signal
import time
from typing import Any

from config import AgentConfig
from flash_agent import FlashAgent

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("flash-agent")

# ──────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────────────────────────────────────
_shutdown = False


def _handle_signal(signum, _frame) -> None:
    global _shutdown
    logger.info("Received signal %s – shutting down gracefully", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Entry point for Flash Agent.

    Runs in two modes:
      - CronJob mode (SCAN_INTERVAL <= 0): single scan, then exit.
      - Continuous mode (SCAN_INTERVAL > 0): scan every N seconds until shutdown.

    All LLM calls go through LiteLLM proxy which handles Langfuse tracing
    automatically.  No OTEL or Langfuse SDK initialization needed here.
    """
    cfg = AgentConfig.from_env()
    errors = cfg.validate()
    if errors:
        for err in errors:
            logger.error("Config error: %s", err)
        raise SystemExit(1)

    logger.info(
        "Flash Agent v4.0.0 | agent=%s | namespace=%s | model=%s",
        cfg.agent_name, cfg.k8s_namespace, cfg.model_alias,
    )
    logger.info(
        "Kubernetes Node | IP=%s | MCP Servers: K8s=%s | Prometheus=%s",
        cfg.k8s_node_ip, cfg.k8s_mcp_url, cfg.prom_mcp_url,
    )
    logger.info(
        "Storage: ① MCP→file(%s)  ② LLM→LiteLLM→Langfuse",
        cfg.mcp_interactions_file,
    )

    agent = FlashAgent(cfg)

    if cfg.scan_interval <= 0:
        logger.info("CronJob mode – single scan")
        agent.scan(cfg.scan_query)
    else:
        logger.info(
            "Continuous mode \u2013 scan every %ds (fast=%ds when health<%d)",
            cfg.scan_interval, cfg.scan_interval_fast, cfg.scan_health_threshold,
        )
        while not _shutdown:
            analysis: Any = None
            try:
                analysis = agent.scan(cfg.scan_query)
            except Exception as exc:
                logger.exception("Scan cycle failed: %s", exc)

            # Adaptive interval: speed up while health is degraded
            interval = cfg.scan_interval
            try:
                if isinstance(analysis, dict):
                    env_state = analysis.get("environment_state", {}) or {}
                    health = analysis.get("health", {}) or {}
                    health_status = env_state.get("health_status", "Healthy")
                    score = health.get("overall_health_score", 100)
                    try:
                        score = int(score)
                    except (ValueError, TypeError):
                        score = 100
                    if health_status != "Healthy" or score < cfg.scan_health_threshold:
                        interval = cfg.scan_interval_fast
                        logger.info(
                            "Adaptive scan: degraded (status=%s score=%s) \u2192 %ds interval",
                            health_status, score, interval,
                        )
            except Exception as exc:
                logger.debug("Adaptive interval check failed: %s", exc)

            for _ in range(interval):
                if _shutdown:
                    break
                time.sleep(1)

    logger.info("Flash Agent shut down cleanly")


if __name__ == "__main__":
    main()
