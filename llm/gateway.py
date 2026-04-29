"""
LLM Gateway – OpenAI client, tool selection, and analysis
===========================================================

All LLM interactions go through this module. Calls are routed through
the LiteLLM proxy which handles Langfuse tracing via success_callback.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import AgentConfig
from mcp.parsers import build_mcp_data_summary, extract_mcp_text
from observability.langfuse import update_generation_metadata

logger = logging.getLogger("flash-agent")


# ──────────────────────────────────────────────────────────────────────────────
# Issue 2 Fix: Fault name anonymisation
# ──────────────────────────────────────────────────────────────────────────────

def _anonymize_chaos_names(names: List[str]) -> Dict[str, str]:
    """
    Return a mapping of raw K8s chaos object name → anonymised label.

    ChaosResult / ChaosEngine names encode the fault type in their name
    (e.g. ``pod-deletelfq6l-pod-delete``).  Passing these raw names to the
    LLM leaks the ground-truth fault identity before the model has had a
    chance to reason independently.  Replace each unique name with a
    stable, zero-information label (``fault-001``, ``fault-002``, …).
    """
    mapping: Dict[str, str] = {}
    counter = 1
    for name in names:
        if name not in mapping:
            mapping[name] = f"fault-{counter:03d}"
            counter += 1
    return mapping


def _apply_name_mapping(names: List[str], mapping: Dict[str, str]) -> List[str]:
    """Replace each name using the provided mapping (pass-through if missing)."""
    return [mapping.get(n, n) for n in names]


# Per-process cache: model_alias → "max_tokens" | "max_completion_tokens".
# Populated on first successful LLM call so the detection is model-agnostic:
# we try max_tokens, catch the Azure rejection, switch to max_completion_tokens,
# and never retry again for that model alias within this process lifetime.
_model_token_param: dict[str, str] = {}


def _chat_create(client, model_alias: str, max_tokens_value: int, **kwargs):
    """Call chat.completions.create with automatic token-param detection.

    Azure o-series and gpt-4o models reject 'max_tokens' and require
    'max_completion_tokens'. We try max_tokens on the first call; on
    rejection we switch and cache the result for the process lifetime.
    """
    param = _model_token_param.get(model_alias, "max_tokens")
    try:
        return client.chat.completions.create(**{param: max_tokens_value}, **kwargs)
    except Exception as exc:
        err = str(exc)
        if param == "max_tokens" and (
            "max_completion_tokens" in err or "not supported" in err.lower()
        ):
            logger.info(
                "[token-param] model=%s rejected max_tokens – switching to "
                "max_completion_tokens (cached for process lifetime)",
                model_alias,
            )
            _model_token_param[model_alias] = "max_completion_tokens"
            return client.chat.completions.create(
                max_completion_tokens=max_tokens_value, **kwargs
            )
        raise


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
    Create an OpenAI-compatible client pointed at the LiteLLM proxy.

    All LLM calls go through LiteLLM regardless of provider — this ensures
    Langfuse tracing, retries, and sidecar metadata injection all work.
    The proxy handles Azure/OpenAI/Anthropic routing based on its config.
    """
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
        resp = _chat_create(
            create_openai_client(cfg),
            cfg.model_alias,
            50,
            model=cfg.model_alias,
            messages=messages,
            temperature=0,
            extra_body={
                "metadata": {
                    "generation_name": "tool_selection",
                    "scan_id": scan_id,
                    "step": "tool-selection",
                    "namespace": cfg.k8s_namespace,
                    "agent": cfg.agent_name,
                    **({"agent_id": cfg.agent_id} if cfg.agent_id else {}),
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
    include_chaos: bool = True,
) -> str:
    """
    Build a structured data payload for the LLM.

    Ensures the most important data (chaos-exporter logs with fault verdicts,
    Argo workflow list, pod status) is always included.
    Budget: ~12-15K chars total.

    When include_chaos is False, chaos-specific sections (Argo Workflows,
    ChaosResults, ChaosEngines, chaos-exporter/operator logs) are omitted
    to prevent data leakage in Langfuse traces (blind-observer mode).
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
    if include_chaos:
        argo = summary.get("argo_workflows", {})
        if argo.get("count", 0) > 0:
            latest_name = argo.get("latest", "")
            # Issue 2 Fix: anonymise workflow names so the epoch-suffix encoded
            # fault identity is not exposed to the LLM.
            all_wf_names = [w["name"] for w in argo.get("workflows", [])]
            wf_name_map = _anonymize_chaos_names(all_wf_names)
            anon_latest = wf_name_map.get(latest_name, latest_name)
            wf_lines = []
            for w in argo.get("workflows", []):
                anon_name = wf_name_map.get(w["name"], w["name"])
                marker = " \u2190 LATEST" if w["name"] == latest_name else ""
                wf_lines.append(
                    f"  - {anon_name}  phase={w.get('phase', '?')}  "
                    f"age={w.get('age', '?')}{marker}"
                )
            sections.append(
                f"## ARGO WORKFLOWS ({argo['count']} total)\n"
                f"Latest experiment workflow: {anon_latest}\n"
                + "\n".join(wf_lines)
            )

    # ── 4. ChaosResults ──────────────────────────────────────────────────────
    if include_chaos:
        cr = summary.get("chaosresults", {})
        if cr.get("count", 0) > 0:
            # Issue 2 Fix: anonymise ChaosResult names before passing to LLM
            anon_cr_names = _apply_name_mapping(
                cr.get("results", []),
                _anonymize_chaos_names(cr.get("results", [])),
            )
            sections.append(
                f"## CHAOSRESULTS ({cr['count']} total)\n"
                f"  Names: {json.dumps(anon_cr_names)}"
            )

    # ── 5. ChaosEngines ──────────────────────────────────────────────────────
    if include_chaos:
        ce = summary.get("chaosengines", {})
        if ce.get("count", 0) > 0:
            # Issue 2 Fix: anonymise ChaosEngine names before passing to LLM
            anon_ce_names = _apply_name_mapping(
                ce.get("engines", []),
                _anonymize_chaos_names(ce.get("engines", [])),
            )
            sections.append(
                f"## CHAOSENGINES ({ce['count']} total)\n"
                f"  Names: {json.dumps(anon_ce_names)}"
            )

    # ── 6. chaos-exporter logs (CRITICAL — contains fault verdicts) ──────────
    if include_chaos:
        pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
        if isinstance(pods_log, dict):
            for pod_name in pods_log:
                if "chaos-exporter" in pod_name:
                    log_data = pods_log[pod_name]
                    log_text = extract_mcp_text(log_data)
                    if log_text:
                        verdict_lines = [
                            l.strip()
                            for l in log_text.split("\n")
                            if "FaultName=" in l
                            or "ResultVerdict=" in l
                            or "ProbeSuccessPercentage=" in l
                        ]
                        # Issue 2 Fix: replace raw FaultName values with
                        # anonymised labels so the LLM cannot read the fault
                        # identity directly from exporter metric names.
                        raw_fault_names = re.findall(
                            r"FaultName=(\S+)", log_text
                        )
                        _fault_map = _anonymize_chaos_names(
                            list(dict.fromkeys(raw_fault_names))
                        )
                        def _sub_fault(text: str) -> str:
                            for raw, anon in _fault_map.items():
                                text = text.replace(raw, anon)
                            return text
                        verdict_lines = [_sub_fault(l) for l in verdict_lines]
                        sections.append(
                            f"## CHAOS-EXPORTER LOGS ({pod_name})\n"
                            f"### Verdict lines (extracted):\n"
                            + "\n".join(verdict_lines[-30:])
                            + "\n\n"
                            f"### Full log (truncated to 6000 chars):\n"
                            + _sub_fault(log_text[-6000:])
                        )

    # ── 7. chaos-operator logs (compact) ─────────────────────────────────────
    if include_chaos:
        pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
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
    pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
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
    payload_text = build_llm_data_payload(
        mcp_data, server_type, cfg.k8s_namespace,
        include_chaos=cfg.include_chaos_tools,
    )

    # Issue 3 Fix: split into system (instructions) + user (data) messages
    # instead of a single concatenated user message.  This gives the model
    # a clearer separation between the schema it must follow and the
    # evidence it must reason over, improving JSON adherence.
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": analysis_prompt},
        {"role": "user", "content": f"DATA TO ANALYSE:\n{payload_text}"},
    ]

    logger.info(
        "Agent \u2192 LLM Gateway: requesting analysis of %s MCP data (%d chars) \u2026",
        server_type,
        len(payload_text),
    )

    result: Optional[Dict[str, Any]] = None
    output_text: str = ""
    t0 = time.time()
    _gen_id = str(uuid.uuid4())

    usage: Optional[Dict[str, int]] = None
    result: Optional[Dict[str, Any]] = None


    try:
        resp = _chat_create(
            create_openai_client(cfg),
            cfg.model_alias,
            16384,
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
                    **({"agent_id": cfg.agent_id} if cfg.agent_id else {}),
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
