"""
LLM Gateway – OpenAI client, tool selection, and analysis
===========================================================

All LLM interactions go through this module. Calls are routed through
the LiteLLM proxy which handles Langfuse tracing via success_callback.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AzureOpenAI, OpenAI

from config import AgentConfig
from mcp.parsers import build_mcp_data_summary, extract_mcp_text
from observability.langfuse import update_generation_metadata

logger = logging.getLogger("flash-agent")

# ──────────────────────────────────────────────────────────────────────────────
# Prompt loading
# ──────────────────────────────────────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a prompt text file from llm/prompts/<name>.txt."""
    return (_PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client factory
# ──────────────────────────────────────────────────────────────────────────────


def create_openai_client(cfg: AgentConfig) -> OpenAI:
    """
    Create an OpenAI-compatible client.

    When openai_base_url points to an Azure endpoint (.openai.azure.com),
    returns AzureOpenAI for direct Azure calls.  Otherwise returns the
    standard OpenAI client (typically pointed at the LiteLLM proxy).
    """
    if cfg.openai_base_url and ".openai.azure.com" in cfg.openai_base_url:
        return AzureOpenAI(
            api_key=cfg.openai_api_key,
            azure_endpoint=cfg.openai_base_url,
            api_version=cfg.azure_api_version,
            timeout=120.0,
        )
    return OpenAI(
        api_key=cfg.openai_api_key or "not-needed",
        base_url=cfg.openai_base_url,
        timeout=120.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – Tool Selection
# ══════════════════════════════════════════════════════════════════════════════


def request_tool_selection(
    cfg: AgentConfig,
    user_query: str,
    scan_id: str = "",
) -> str:
    """
    Ask the LLM which MCP tool to use (kubernetes or prometheus).

    Returns 'kubernetes' or 'prometheus'.
    """
    tool_selection_prompt = _load_prompt("tool_selection")
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": f"{tool_selection_prompt}\n\nQuery: {user_query}"},
    ]
    logger.info("Agent \u2192 LLM Gateway: requesting tool selection \u2026")

    decision = "kubernetes"
    t0 = time.time()

    try:
        resp = create_openai_client(cfg).chat.completions.create(
            model=cfg.model_alias,
            messages=messages,
            temperature=0,
            max_tokens=50,
            extra_body={
                "metadata": {
                    "generation_name": "tool_selection",
                    "scan_id": scan_id,
                    "step": "tool-selection",
                    "namespace": cfg.k8s_namespace,
                    "agent": cfg.agent_name,
                }
            },
        )
        msg = resp.choices[0].message
        raw_text = msg.content or getattr(msg, "reasoning_content", None) or ""
        output_text = raw_text.strip().lower()
        decision = "prometheus" if "prometheus" in output_text else "kubernetes"
        logger.info(
            "LLM Gateway \u2192 Agent: tool decision = %s (raw=%r)",
            decision,
            output_text,
        )

    except Exception as exc:
        logger.error(
            "LLM tool-selection failed: %s \u2013 defaulting to kubernetes", exc
        )

    duration = time.time() - t0
    logger.info("Tool selection completed in %.2fs \u2192 %s", duration, decision)

    return decision


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – LLM Data Payload Builder
# ══════════════════════════════════════════════════════════════════════════════


def build_llm_data_payload(
    mcp_data: Dict[str, Any],
    server_type: str,
    k8s_namespace: str,
) -> str:
    """
    Build a structured data payload for the LLM.

    Ensures the most important data (chaos-exporter logs with fault verdicts,
    Argo workflow list, pod status) is always included.
    Budget: ~12-15K chars total.
    """
    sections: List[str] = []
    sections.append(
        f"Data source : {server_type.upper()} MCP Tool Response\n"
        f"Namespace   : {k8s_namespace}\n"
        f"Timestamp   : {datetime.now(timezone.utc).isoformat()}\n"
    )

    # ── 1. Pod status summary (compact) ──────────────────────────────────────
    summary = build_mcp_data_summary(mcp_data)
    pods = summary.get("pods", {})
    if pods:
        sections.append(
            f"## POD STATUS\n"
            f"Total: {pods.get('total', '?')} | "
            f"By status: {json.dumps(pods.get('by_status', {}))}\n"
            f"Total restarts: {pods.get('total_restarts', 0)}\n"
            f"High-restart pods: {json.dumps(pods.get('restarted_pods', []))}"
        )

    # ── 2. Events summary ────────────────────────────────────────────────────
    events = summary.get("events", {})
    if events:
        sections.append(f"## EVENTS\n{json.dumps(events)}")

    # ── 3. Argo Workflows (sorted newest-first, latest marked) ───────────────
    argo = summary.get("argo_workflows", {})
    if argo.get("count", 0) > 0:
        latest_name = argo.get("latest", "")
        wf_lines = []
        for w in argo.get("workflows", []):
            marker = " \u2190 LATEST" if w["name"] == latest_name else ""
            wf_lines.append(
                f"  - {w['name']}  phase={w.get('phase', '?')}  "
                f"age={w.get('age', '?')}{marker}"
            )
        sections.append(
            f"## ARGO WORKFLOWS ({argo['count']} total)\n"
            f"Latest experiment workflow: {latest_name}\n"
            + "\n".join(wf_lines)
        )

    # ── 4. ChaosResults ──────────────────────────────────────────────────────
    cr = summary.get("chaosresults", {})
    if cr.get("count", 0) > 0:
        sections.append(
            f"## CHAOSRESULTS ({cr['count']} total)\n"
            f"  Names: {json.dumps(cr.get('results', []))}"
        )

    # ── 5. ChaosEngines ──────────────────────────────────────────────────────
    ce = summary.get("chaosengines", {})
    if ce.get("count", 0) > 0:
        sections.append(
            f"## CHAOSENGINES ({ce['count']} total)\n"
            f"  Names: {json.dumps(ce.get('engines', []))}"
        )

    # ── 6. chaos-exporter logs (CRITICAL — contains fault verdicts) ──────────
    pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
    if isinstance(pods_log, dict):
        for pod_name, log_data in pods_log.items():
            if "chaos-exporter" in pod_name:
                log_text = extract_mcp_text(log_data)
                if log_text:
                    verdict_lines = [
                        l.strip()
                        for l in log_text.split("\n")
                        if "FaultName=" in l
                        or "ResultVerdict=" in l
                        or "ProbeSuccessPercentage=" in l
                    ]
                    sections.append(
                        f"## CHAOS-EXPORTER LOGS ({pod_name})\n"
                        f"### Verdict lines (extracted):\n"
                        + "\n".join(verdict_lines[-30:])
                        + "\n\n"
                        f"### Full log (truncated to 6000 chars):\n"
                        + log_text[-6000:]
                    )

    # ── 7. chaos-operator logs (compact) ─────────────────────────────────────
    if isinstance(pods_log, dict):
        for pod_name, log_data in pods_log.items():
            if "chaos-operator" in pod_name:
                log_text = extract_mcp_text(log_data)
                if log_text:
                    sections.append(
                        f"## CHAOS-OPERATOR LOGS ({pod_name})\n"
                        + log_text[-2000:]
                    )

    # ── 8. Workflow pod logs (error lines only) ──────────────────────────────
    if isinstance(pods_log, dict):
        wf_log_parts: List[str] = []
        for pod_name, log_data in pods_log.items():
            if "chaos-exporter" in pod_name or "chaos-operator" in pod_name:
                continue
            log_text = extract_mcp_text(log_data)
            if log_text:
                error_lines = [
                    l.strip()
                    for l in log_text.split("\n")
                    if "error" in l.lower() or "failed" in l.lower()
                ]
                if error_lines:
                    wf_log_parts.append(
                        f"  [{pod_name}] ({len(error_lines)} error lines):\n"
                        + "\n".join(
                            f"    {l[:200]}" for l in error_lines[-5:]
                        )
                    )
        if wf_log_parts:
            sections.append(
                f"## WORKFLOW POD LOGS (error lines only)\n"
                + "\n".join(wf_log_parts)
            )

    return "\n\n".join(sections)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – LLM Analysis
# ══════════════════════════════════════════════════════════════════════════════


def request_llm_analysis(
    cfg: AgentConfig,
    mcp_data: Dict[str, Any],
    server_type: str,
    scan_id: str = "",
    agent_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Send MCP data to the LLM for deep analysis.

    Returns parsed analysis dict or None on failure.
    """
    analysis_prompt = _load_prompt("analysis")
    payload_text = build_llm_data_payload(mcp_data, server_type, cfg.k8s_namespace)

    combined_prompt = (
        f"INSTRUCTIONS:\n{analysis_prompt}\n\n"
        f"DATA TO ANALYSE:\n{payload_text}"
    )
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": combined_prompt},
    ]

    logger.info(
        "Agent \u2192 LLM Gateway: requesting analysis of %s MCP data (%d chars) \u2026",
        server_type,
        len(combined_prompt),
    )

    result: Optional[Dict[str, Any]] = None
    output_text: str = ""
    t0 = time.time()
    _gen_id = str(uuid.uuid4())

    usage: Optional[Dict[str, int]] = None
    result: Optional[Dict[str, Any]] = None

    try:
        resp = create_openai_client(cfg).chat.completions.create(
            model=cfg.model_alias,
            messages=messages,
            temperature=0.1,
            extra_body={
                "metadata": {
                    "generation_name": "llm_analysis",
                    "generation_id": _gen_id,
                    "scan_id": scan_id,
                    "step": "llm-analysis",
                    "namespace": cfg.k8s_namespace,
                    "agent": cfg.agent_name,
                    **(agent_context or {}),
                }
            },
        )
        msg = resp.choices[0].message
        output_text = (
            msg.content or getattr(msg, "reasoning_content", None) or ""
        )
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
            "completion_tokens": (
                resp.usage.completion_tokens if resp.usage else 0
            ),
        }
        # Extract JSON from response (may be wrapped in markdown code fences)
        json_text = output_text
        if "```json" in json_text:
            json_text = json_text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in json_text:
            json_text = json_text.split("```", 1)[1].split("```", 1)[0]
        result = json.loads(json_text.strip())

        # Log experiment summary
        exp = result.get("experiment_summary", {})
        faults = result.get("chaos_faults", [])
        wf_errors = result.get("workflow_errors", [])
        logger.info(
            "LLM Gateway \u2192 Agent: analysis complete | "
            "workflow=%s phase=%s | faults: %d passed, %d failed | "
            "workflow_errors=%d | issues=%d | health=%s",
            exp.get("workflow_name", "N/A"),
            exp.get("workflow_phase", "N/A"),
            exp.get("faults_passed", 0),
            exp.get("faults_failed", 0),
            len(wf_errors),
            len(result.get("issues", [])),
            result.get("health", {}).get("overall_health_score", "N/A"),
        )
        for f in faults:
            v_icon = (
                "\u2705"
                if f.get("verdict") == "Pass"
                else "\u274c" if f.get("verdict") == "Fail" else "\u23f3"
            )
            logger.info(
                "  %s %s \u2192 %s | probe=%s%% | impact: %s",
                v_icon,
                f.get("fault_name", "?"),
                f.get("verdict", "?"),
                f.get("probe_success_percentage", "?"),
                (f.get("impact_observed", "N/A") or "N/A")[:80],
            )
    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON: %s", exc)
    except Exception as exc:
        logger.error("LLM analysis call failed: %s", exc)

    duration = time.time() - t0
    logger.info("LLM analysis completed in %.2fs", duration)

    # ── Tier 2: post-call metadata enrichment ─────────────────────────────
    _post_meta: Dict[str, Any] = {}
    if usage:
        _post_meta["prompt_tokens"] = usage.get("prompt_tokens", 0)
        _post_meta["completion_tokens"] = usage.get("completion_tokens", 0)
        try:
            ptd = getattr(resp.usage, "prompt_tokens_details", None)
            _post_meta["cached_tokens"] = (
                (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
            )
        except Exception:
            _post_meta["cached_tokens"] = 0
    if result:
        _exp = result.get("experiment_summary", {})
        _post_meta["faults_passed"] = _exp.get("faults_passed", 0)
        _post_meta["faults_failed"] = _exp.get("faults_failed", 0)
    if _post_meta:
        update_generation_metadata(cfg, _gen_id, _post_meta)

    return result
