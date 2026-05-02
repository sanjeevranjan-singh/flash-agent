"""
LLM Gateway – OpenAI client, tool selection, and analysis
===========================================================

All LLM interactions go through this module. Calls are routed through
the LiteLLM proxy which handles Langfuse tracing via success_callback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import AgentConfig
from mcp.parsers import build_mcp_data_summary, extract_mcp_text, split_event_blocks
from mcp.log_dedup import greedy_compress_lines
from observability.langfuse import update_generation_metadata

logger = logging.getLogger("flash-agent")

# Match the MCP event-block timestamp line. Sample:
#   Timestamp: 2026-05-01 20:09:31 +0000 UTC
_EVENT_TS_RE = re.compile(
    r"Timestamp\s*:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*([+-]\d{4})?\s*UTC?",
    re.IGNORECASE,
)


def _parse_event_block_ts(block_text: str) -> Optional[datetime]:
    """Best-effort parse of an event block's Timestamp line. Returns UTC dt or None."""
    m = _EVENT_TS_RE.search(block_text)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


# Structural K8s/Litmus resource-type keywords that always indicate identity leakage
# regardless of which specific fault was injected.
_LEAKAGE_STRUCTURAL: frozenset[str] = frozenset({
    "chaosengine",
    "chaosresult",
    "workflow_name",
    "argo-workflow",
    "chaos-operator",
    "chaos-exporter",
})

# Pattern: matches any Litmus-style fault name  <target>-<fault-action>
# (e.g. pod-delete, node-drain, container-kill, k8s-pod-delete, disk-fill …)
# Prefixes are ONLY real LitmusChaos target categories — not generic K8s terms
# like "app" or "kubelet" which appear legitimately in logs.
_FAULT_NAME_RE = re.compile(
    r"\b(?:pod|node|container|disk|network|k8s|jvm)"
    r"-(?:delete|hog|kill|fill|loss|drain|restart|corruption|latency|"
    r"duplication|taint|stress|failure|error|partition|io|cpu|memory|network|chaos|method|gc|poweroff)\b",
    re.IGNORECASE,
)

# FaultName=<value> key-value pairs that appear in chaos-exporter / operator logs.
_FAULTNAME_KV_RE = re.compile(r"FaultName=(\S+)", re.IGNORECASE)


# Per-process cache: model_alias → "max_tokens" | "max_completion_tokens".
# Populated on first successful LLM call so the detection is model-agnostic:
# we try max_tokens, catch the Azure rejection, switch to max_completion_tokens,
# and never retry again for that model alias within this process lifetime.
_model_token_param: dict[str, str] = {}


def _sanitize_leakage_terms(text: str) -> tuple[str, list[str]]:
    """Redact chaos identity markers from LLM user payload.

    Three passes:
    1. Structural keywords (chaosengine, chaosresult, …) - exact match, always redact.
    2. Fault-name shaped strings (<target>-<fault-action>) - pattern match, covers any fault.
    3. FaultName=<value> KV pairs in logs - captures explicit fault name assignments.
    """
    if not text:
        return text, []

    found: list[str] = []
    sanitized = text

    # Pass 1 – structural resource-type keywords
    for term in _LEAKAGE_STRUCTURAL:
        if term.lower() in sanitized.lower():
            found.append(term)
            sanitized = re.sub(re.escape(term), "[REDACTED]", sanitized, flags=re.IGNORECASE)

    # Pass 2 – fault-name pattern (<target>-<fault-action>)
    matches = _FAULT_NAME_RE.findall(sanitized)
    if matches:
        found.extend(matches)
        sanitized = _FAULT_NAME_RE.sub("[FAULT-NAME]", sanitized)

    # Pass 3 – FaultName=<value> KV pairs
    kv_matches = _FAULTNAME_KV_RE.findall(sanitized)
    if kv_matches:
        found.extend([f"FaultName={v}" for v in kv_matches])
        sanitized = _FAULTNAME_KV_RE.sub("FaultName=[REDACTED]", sanitized)

    return sanitized, found


def _build_structured_analysis_user_content(
    cfg: AgentConfig,
    server_type: str,
    scan_id: str,
    payload_text: str,
    summary: Dict[str, Any],
    agent_context: Optional[Dict[str, Any]],
) -> str:
    """Build explicit sectioned user payload for better instruction adherence."""
    pods = summary.get("pods", {})
    events = summary.get("events", {})
    topology = summary.get("topology", {}) or {}
    context = agent_context or {}

    sections: List[str] = [
        "CONTEXT",
        f"- Namespace: {cfg.k8s_namespace}",
        f"- Data Source: {server_type.upper()} MCP",
        f"- Scan ID: {scan_id or 'N/A'}",
        f"- Data Sufficient: {context.get('data_sufficient', True)}",
    ]

    # Detection-gate precommit (AIOpsLab cascade pattern).
    # When the cheaper Detection stage already classified the snapshot as
    # anomalous, surface that decision so the broader analysis cannot
    # contradict it without explicit reasoning. This reduces under-reporting
    # without prescribing what the issue is.
    det_verdict = context.get("detection_gate")
    det_reason = context.get("detection_reason")
    if det_verdict:
        sections.append(
            f"- Detection Gate: anomaly_detected={det_verdict}"
            + (f" | evidence={det_reason!r}" if det_reason else "")
        )
        if det_verdict == "Yes":
            sections.append(
                "- NOTE: A separate detection pass already classified this "
                "snapshot as anomalous. health_status MUST NOT be 'Healthy' "
                "and identified_issues MUST contain at least one entry "
                "consistent with the evidence above."
            )

    # Topology: deployment-grouped pod inventory derived from observed pods.
    # Provides peer-set context for outlier detection (Channel C/F in
    # analysis prompt) without leaking fault identity. Excludes the agent's
    # own deployment so it cannot self-describe.
    if topology:
        topo_lines: List[str] = []
        # Hide the agent itself + chaos infra from the topology view: a real
        # SRE on-call would not consider their own monitoring sidecar a peer
        # of the workload, and chaos infra components must never reach the
        # LLM (blind-observer rule).  The agent's own deployment name comes
        # from cfg.agent_name so any agent (not just flash-agent) self-hides.
        _HIDE = {
            cfg.agent_name, "litellm", "litellm-proxy",
            "chaos-exporter", "chaos-operator", "chaos-runner",
            "subscriber", "event-tracker", "workflow-controller",
            "kubernetes-mcp-server", "prometheus-mcp-server",
        }
        for deploy in sorted(topology.keys()):
            if deploy in _HIDE or deploy.startswith("chaos-") or "-runner" in deploy:
                continue
            t = topology[deploy]
            ages = t.get("ages") or []
            age_summary = ""
            if ages:
                # Show min..max age and the youngest/oldest pod age separately
                # so age-skew (Channel C in prompt) is computable directly.
                age_summary = f" ages={ages[0] if len(ages)==1 else f'{min(ages, key=len)}..{max(ages, key=len)}'}"
            stat_summary = ",".join(
                f"{s}={n}" for s, n in sorted(t.get("statuses", {}).items())
            )
            topo_lines.append(
                f"  - {deploy}: pods={t['pods']} ready={t['ready']}/{t['pods']} "
                f"restarts={t['restarts']} status={{{stat_summary}}}{age_summary}"
            )
        sections.append("TOPOLOGY (deployments observed in namespace)")
        sections.extend(topo_lines)

    sections.extend([
        "",
        "METRICS",
        f"- Pods total: {pods.get('total', 0)}",
        f"- Pod status breakdown: {json.dumps(pods.get('by_status', {}))}",
        f"- Total restarts: {pods.get('total_restarts', 0)}",
        f"- Events normal/warning: {events.get('normal', 0)}/{events.get('warning', 0)}",
        "",
        "EVENTS",
        f"- Warning reasons: {json.dumps(events.get('warning_reasons', []))}",
        "",
        "LOGS AND RAW EVIDENCE",
        payload_text,
        "",
        "CONSTRAINTS",
        "- Analyze only the provided evidence.",
        "- Do not infer fault identity unless explicitly evidenced in logs/metrics.",
        "- Preserve strict JSON response schema from system prompt.",
    ])

    if not context.get("data_sufficient", True) and context.get("data_quality_note"):
        sections.extend(
            [
                "",
                "DATA QUALITY NOTE",
                f"- Focus your analysis on: {context['data_quality_note']}",
            ]
        )

    return "\n".join(sections)


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


def create_openai_client(cfg: AgentConfig, max_retries: int = 2) -> OpenAI:
    """
    Create an OpenAI-compatible client pointed at the LiteLLM proxy.

    All LLM calls go through LiteLLM regardless of provider — this ensures
    Langfuse tracing, retries, and sidecar metadata injection all work.
    The proxy handles Azure/OpenAI/Anthropic routing based on its config.

    The OpenAI SDK's transparent retries cause the LiteLLM proxy to emit a
    separate Langfuse generation per HTTP attempt, inflating trace observation
    counts and token costs. Cheap fail-open calls (detection_gate, hindsight,
    tool_selection) pass max_retries=0 so a single failed attempt simply
    triggers their fail-open default. The expensive llm_analysis call keeps
    the SDK default (max_retries=2) since a transient blip there would lose
    the entire scan's observation.
    """
    return OpenAI(
        api_key=cfg.openai_api_key or "not-needed",
        base_url=cfg.openai_base_url,
        timeout=120.0,
        max_retries=max_retries,
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
        {"role": "system", "content": tool_selection_prompt},
        {"role": "user", "content": f"Query: {user_query}"},
    ]
    logger.info("Agent \u2192 LLM Gateway: requesting tool selection \u2026")

    decision = "kubernetes"
    t0 = time.time()


    try:
        resp = _chat_create(
            create_openai_client(cfg, max_retries=0),
            cfg.model_alias,
            50,
            model=cfg.model_alias,
            messages=messages,
            temperature=0,
            extra_body={
                "metadata": {
                    "trace_id": cfg.notify_id or scan_id,
                    "session_id": scan_id,
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
# STEP 2.5 – Hindsight / Data-Quality Check  (mirrors AIOpsLab Flash pattern)
# ══════════════════════════════════════════════════════════════════════════════


def request_hindsight_check(
    cfg: AgentConfig,
    mcp_data: Dict[str, Any],
    server_type: str,
    scan_id: str = "",
) -> Dict[str, Any]:
    """
    Lightweight self-assessment after MCP collection and before LLM analysis.

    Mirrors the AIOpsLab Flash HindsightBuilder pattern: before committing to a
    full analysis LLM call, ask a fast model whether the collected data has
    enough signal.  Returns {"sufficient": True} or
    {"sufficient": False, "next_focus": "<guidance>"}.

    If the LLM call fails or returns unparseable output, defaults to
    {"sufficient": True} so the main analysis still runs.
    """
    hindsight_prompt = _load_prompt("hindsight")
    summary = build_mcp_data_summary(mcp_data, include_chaos=cfg.include_chaos_tools)
    summary_text = json.dumps(summary, indent=2)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": hindsight_prompt},
        {"role": "user", "content": f"DATA SUMMARY:\n{summary_text}"},
    ]

    try:
        resp = _chat_create(
            create_openai_client(cfg, max_retries=0),
            cfg.model_alias,
            100,
            model=cfg.model_alias,
            messages=messages,
            temperature=0,
            extra_body={
                "metadata": {
                    "trace_id": cfg.notify_id or scan_id,
                    "session_id": scan_id,
                    "generation_name": "hindsight_check",
                    "scan_id": scan_id,
                    "step": "hindsight",
                    "namespace": cfg.k8s_namespace,
                    "agent": cfg.agent_name,
                }
            },
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        result = json.loads(raw.strip())
        logger.info(
            "Hindsight check: sufficient=%s%s",
            result.get("sufficient"),
            f" | next_focus={result['next_focus']!r}" if not result.get("sufficient") else "",
        )
        return result
    except Exception as exc:
        logger.warning("Hindsight check failed (%s) – defaulting to sufficient=True", exc)
        return {"sufficient": True}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2c – Detection Gate (AIOpsLab-style binary anomaly detection)
# ══════════════════════════════════════════════════════════════════════════════


def request_detection_gate(
    cfg: AgentConfig,
    mcp_data: Dict[str, Any],
    server_type: str,
    scan_id: str = "",
) -> Dict[str, Any]:
    """
    Cheap binary anomaly check before the full structured analysis.

    Mirrors AIOpsLab's DetectionTask (clients/gpt.py + tasks/detection.py):
    a small, focused prompt that asks "is anything anomalous here?" and
    expects a Yes/No answer with a one-line justification. Splitting the
    cognitive load improves under-detection: the broad analysis prompt
    sometimes returns issues=0 because the LLM hedges; the gate poses the
    simpler question first and biases toward investigation when ANY single
    evidence channel fires.

    Returns {"anomaly_detected": "Yes"|"No", "reason": "<phrase>"}.
    On any failure, defaults to {"anomaly_detected": "Yes"} so the full
    analysis still runs (fail-open — never miss a real incident).
    """
    detection_prompt = _load_prompt("detection")
    summary = build_mcp_data_summary(mcp_data, include_chaos=cfg.include_chaos_tools)
    # Compact summary only — gate is intentionally cheap. We send the same
    # blind-observer fields the full analysis sees, but no raw log dumps.
    keep = {
        "pods": summary.get("pods", {}),
        "topology": summary.get("topology", {}),
        "events": summary.get("events", {}),
        "prometheus": summary.get("prometheus", {}),
    }
    summary_text = json.dumps(keep, indent=2, default=str)
    summary_text, _leaked = _sanitize_leakage_terms(summary_text)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": detection_prompt},
        {"role": "user", "content": f"OBSERVATION SNAPSHOT:\n{summary_text}"},
    ]

    try:
        resp = _chat_create(
            create_openai_client(cfg, max_retries=0),
            cfg.model_alias,
            150,
            model=cfg.model_alias,
            messages=messages,
            temperature=0,
            extra_body={
                "metadata": {
                    "trace_id": cfg.notify_id or scan_id,
                    "session_id": scan_id,
                    "generation_name": "detection_gate",
                    "scan_id": scan_id,
                    "step": "detection",
                    "namespace": cfg.k8s_namespace,
                    "agent": cfg.agent_name,
                }
            },
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "```json" in raw:
            raw = raw.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in raw:
            raw = raw.split("```", 1)[1].split("```", 1)[0]
        result = json.loads(raw.strip())
        ans = str(result.get("anomaly_detected", "")).strip().lower()
        normalised = "Yes" if ans in ("yes", "true", "1") else "No"
        result["anomaly_detected"] = normalised
        logger.info(
            "Detection gate: anomaly=%s | reason=%s",
            normalised,
            result.get("reason", ""),
        )
        return result
    except Exception as exc:
        logger.warning(
            "Detection gate failed (%s) – defaulting to anomaly=Yes (fail-open)",
            exc,
        )
        return {"anomaly_detected": "Yes", "reason": "detection-gate-error"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 – LLM Data Payload Builder
# ══════════════════════════════════════════════════════════════════════════════


def build_llm_data_payload(
    mcp_data: Dict[str, Any],
    server_type: str,
    k8s_namespace: str,
    include_chaos: bool = True,
    event_watermark_ts: Optional[datetime] = None,
    event_recency_fallback_sec: int = 90,
    gate_verdict: str = "",
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

    # ── 2. Events summary (with warning details if available) ─────────────────
    events = summary.get("events", {})
    if events:
        events_line = (
            f"normal={events.get('normal', 0)}  "
            f"warning={events.get('warning', 0)}"
        )
        reasons = events.get("warning_reasons", [])
        if reasons:
            events_line += f"  reasons={json.dumps(reasons)}"
        sections.append(f"## EVENTS\n{events_line}")

        # Append raw warning event blocks (up to 10) for LLM context.
        # K8s events are multi-line records (Type/Reason/Object/Message…) –
        # if we filter on a single line containing "Warning" we lose the body.
        # Walk blocks, keep blocks that mark Type: Warning, and emit the full
        # block so Reason and Message reach the LLM.
        events_text = extract_mcp_text(mcp_data.get("data", mcp_data).get("events_list", {}))
        if events_text:
            events_text = greedy_compress_lines(events_text)
            warning_blocks: List[str] = []
            for block in split_event_blocks(events_text):
                if not block:
                    continue
                block_text = "\n".join(block)
                is_warning = bool(re.search(r"\bType\s*:\s*Warning\b", block_text)) \
                             or (block[0].startswith("- ") is False and "Warning" in block[0])
                if is_warning:
                    warning_blocks.append(block_text)

            if warning_blocks:
                # ── Recency filter (MCP event watermark) ──────────────────
                # K8s `lastTimestamp` advances each time an event recurs, so
                # filtering on (now - ts) <= cutoff keeps an event that is
                # still actively firing while dropping stale "ghosts" from
                # earlier scans. Cold-start uses the configured fallback;
                # subsequent scans use max(watermark - 10s slack, fallback).
                # Fail-open if every block is filtered AND the gate said
                # "Yes": preserve the most recent 3 so the LLM still has
                # evidence — better stale data than no data when an anomaly
                # is asserted upstream.
                _now = datetime.now(timezone.utc)
                _fallback = _now - timedelta(seconds=max(1, event_recency_fallback_sec))
                if event_watermark_ts is not None:
                    _floor = event_watermark_ts - timedelta(seconds=10)
                    # Never look further back than the fallback ceiling.
                    _floor = max(_floor, _fallback)
                else:
                    _floor = _fallback

                fresh: List[str] = []
                undated: List[str] = []
                for blk in warning_blocks:
                    ts = _parse_event_block_ts(blk)
                    if ts is None:
                        # Unparseable timestamp → keep (fail-open at block level).
                        undated.append(blk)
                    elif ts >= _floor:
                        fresh.append(blk)

                filtered = fresh + undated
                if not filtered and str(gate_verdict).lower().startswith("y"):
                    filtered = warning_blocks[-3:]
                    logger.info(
                        "Watermark filter dropped all %d warning event(s); "
                        "gate=Yes — restoring last 3 to preserve evidence",
                        len(warning_blocks),
                    )
                else:
                    dropped = len(warning_blocks) - len(filtered)
                    if dropped > 0:
                        logger.info(
                            "Watermark filter: kept %d of %d warning event(s) "
                            "(floor=%s)",
                            len(filtered),
                            len(warning_blocks),
                            _floor.strftime("%H:%M:%S"),
                        )

                kept = filtered[-10:]
                if kept:
                    sections.append(
                        f"## WARNING EVENTS (latest {len(kept)})\n"
                        + "\n---\n".join(kept)
                    )

    # ── 2b. Prometheus metrics snapshot ──────────────────────────────────────
    prom = summary.get("prometheus", {})
    if prom:
        prom_lines: List[str] = []
        if "prometheus_up" in prom:
            prom_lines.append(f"prometheus_targets_up={prom['prometheus_up']}")
        if "pod_count" in prom:
            prom_lines.append(f"pod_count={prom['pod_count']}")
        if "pod_phase_counts" in prom:
            prom_lines.append(
                f"pod_phase_counts={json.dumps(prom['pod_phase_counts'])}"
            )

        # Render raw per-pod metric rows. The agent reasons about anomalies
        # (peer outliers, trends, non-zero restart counts) directly from the
        # raw values — no precomputed saturation glyph or absolute-threshold
        # marker is injected. This matches the AIOpsLab observation model:
        # the agent gets telemetry, not preconcluded verdicts.
        #
        # When a `limit` is declared we render saturation as a percentage
        # (`78% of limit`) so peer-outlier and ceiling-saturation faults
        # are visually obvious without the LLM having to divide.
        def _fmt_raw(p: Dict[str, Any], unit: str) -> str:
            value = p["value"]
            limit = p.get("limit")
            sat = p.get("saturation")
            sigma = p.get("sigma")
            base = f"  - {p['pod']}: {value:.3f}{unit}"
            if limit is not None:
                base += f" / limit {limit:.3f}{unit}"
                if sat is not None:
                    base += f" ({sat * 100:.0f}% of limit)"
            if sigma is not None:
                base += f"  [PEER-OUTLIER {sigma:.1f}σ]"
            return base

        cpu_top = prom.get("cpu_per_pod_top", [])
        if cpu_top:
            prom_lines.append("Top CPU pods (cores, current rate):")
            for p in cpu_top[:10]:
                prom_lines.append(_fmt_raw(p, " cores"))

        mem_top = prom.get("memory_per_pod_top_mb", [])
        if mem_top:
            prom_lines.append("Top memory pods (MB, working set):")
            for p in mem_top[:10]:
                prom_lines.append(_fmt_raw(p, " MB"))

        # Restart counts: explicit positive AND negative signal so the LLM
        # can disambiguate "saturation without crashes" (resource fault) vs
        # "saturation with crashes" (OOMKill / liveness-probe failure).
        restart_top = prom.get("restarts_per_pod_top", [])
        restarting = prom.get("restarting_pods", [])
        if restarting:
            prom_lines.append("Pods with restarts (cumulative, non-zero only):")
            for p in restarting[:10]:
                prom_lines.append(f"  - {p['pod']}: {int(p['value'])}")
        elif restart_top:
            prom_lines.append(
                "Pods with restarts: none (all 0) — no recent crash-loop / "
                "OOMKill in this namespace"
            )

        # CPU throttling — direct kernel-cap signal independent of usage rate.
        throttling = prom.get("throttling_pods", [])
        if throttling:
            prom_lines.append(
                "Pods being CPU-throttled (sec/sec, non-zero only — kernel "
                "is capping these pods at their CPU limit):"
            )
            for p in throttling[:10]:
                prom_lines.append(f"  - {p['pod']}: {p['value']:.3f}")

        net_top = prom.get("network_rx_per_pod_top", [])
        if net_top:
            prom_lines.append("Top network RX pods (bytes/sec):")
            for p in net_top[:5]:
                prom_lines.append(f"  - {p['pod']}: {p['value']:.0f}")

        tx_top = prom.get("network_tx_per_pod_top", [])
        if tx_top:
            prom_lines.append("Top network TX pods (bytes/sec):")
            for p in tx_top[:5]:
                prom_lines.append(f"  - {p['pod']}: {p['value']:.0f}")

        # Filesystem usage — covers disk-fill / log-flood / tmpfs faults.
        fs_top = prom.get("fs_usage_per_pod_top_mb", [])
        if fs_top:
            prom_lines.append("Top filesystem usage pods (MB):")
            for p in fs_top[:5]:
                prom_lines.append(f"  - {p['pod']}: {p['value']:.1f} MB")

        # PVC / volume utilisation — alternative path for disk-fill on
        # clusters whose cAdvisor strips pod labels off container_fs_*.
        vol_top = prom.get("volume_usage_top", [])
        if vol_top:
            prom_lines.append(
                "Top PVC / volume usage (used / capacity ratio):"
            )
            for v in vol_top[:5]:
                pct = v["ratio"] * 100
                tag = "  [NEAR-FULL]" if v["ratio"] >= 0.80 else ""
                prom_lines.append(
                    f"  - {v['pod']} (pvc={v['pvc']}): {pct:.1f}%{tag}"
                )

        # Container waiting reasons — kubelet-emitted, fires regardless of
        # whether limits exist or which cAdvisor variant is running.
        # Catches pod-delete (Terminating loop), image rollouts, and
        # CrashLoop spirals.
        waiting = prom.get("pods_waiting_by_reason", {})
        if waiting:
            prom_lines.append("Pods stuck in waiting state (kubelet status):")
            for reason, pods in waiting.items():
                pod_list = ", ".join(pods[:5])
                more = f" (+{len(pods)-5} more)" if len(pods) > 5 else ""
                prom_lines.append(f"  - {reason}: {pod_list}{more}")

        # Terminated reasons — OOMKilled is the canonical memory-hog signal,
        # present even when no memory limit is declared.
        terminated = prom.get("pods_terminated_by_reason", {})
        if terminated:
            prom_lines.append(
                "Pods with non-Completed last-terminated reason "
                "(OOMKilled / Error / etc):"
            )
            for reason, pods in terminated.items():
                pod_list = ", ".join(pods[:5])
                more = f" (+{len(pods)-5} more)" if len(pods) > 5 else ""
                prom_lines.append(f"  - {reason}: {pod_list}{more}")

        # Peer-outlier callouts — independent of resource limits. A pod
        # whose CPU or memory is N stdev above its peers is anomalous
        # even when no throttle / saturation signal exists.
        cpu_outliers = prom.get("cpu_outlier_pods", [])
        mem_outliers = prom.get("memory_outlier_pods", [])
        if cpu_outliers or mem_outliers:
            prom_lines.append(
                "Peer-outlier pods (>=2.5σ above namespace mean — direct "
                "evidence of resource-stress regardless of declared limits):"
            )
            for o in cpu_outliers[:5]:
                prom_lines.append(f"  - cpu  {o['pod']}: {o['sigma']:.1f}σ")
            for o in mem_outliers[:5]:
                prom_lines.append(f"  - mem  {o['pod']}: {o['sigma']:.1f}σ")

        # Recent restarts in last 5 min — captures pod-delete / pod-recreate
        # via increase(), which fires even when the new pod's cumulative
        # counter is 0.
        recent = prom.get("recent_restarts", [])
        if recent:
            prom_lines.append(
                "Pods restarted in last 5 min (recent recreation / "
                "crash-loop / OOM-kill — fires regardless of cumulative count):"
            )
            for p in recent[:10]:
                prom_lines.append(f"  - {p['pod']}: {p['value']:.1f}")

        # Pods Ready=false — catches network-loss and sustained probe failure
        # that don't progress to CrashLoop.
        not_ready = prom.get("pods_not_ready", [])
        if not_ready:
            prom_lines.append(
                "Pods reporting Ready=false (probe failure or peer "
                "connectivity broken — direct evidence of partition / probe-fail):"
            )
            for pod in not_ready[:10]:
                prom_lines.append(f"  - {pod}")

        if prom_lines:
            sections.append(
                "## METRICS (Prometheus, last sample)\n" + "\n".join(prom_lines)
            )

    # ── 3. Argo Workflows (sorted newest-first, latest marked) ───────────────
    if include_chaos:
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
    if include_chaos:
        cr = summary.get("chaosresults", {})
        if cr.get("count", 0) > 0:
            sections.append(
                f"## CHAOSRESULTS ({cr['count']} total)\n"
                f"  Names: {json.dumps(cr.get('results', []))}"
            )

    # ── 5. ChaosEngines ──────────────────────────────────────────────────────
    if include_chaos:
        ce = summary.get("chaosengines", {})
        if ce.get("count", 0) > 0:
            sections.append(
                f"## CHAOSENGINES ({ce['count']} total)\n"
                f"  Names: {json.dumps(ce.get('engines', []))}"
            )

    # ── 6. chaos-exporter logs (CRITICAL — contains fault verdicts) ──────────
    if include_chaos:
        pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
        if isinstance(pods_log, dict):
            for pod_name, log_data in pods_log.items():
                if "chaos-exporter" in pod_name:
                    log_text = extract_mcp_text(log_data)
                    if log_text:
                        log_text = greedy_compress_lines(log_text)
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
    if include_chaos:
        pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
        if isinstance(pods_log, dict):
            for pod_name, log_data in pods_log.items():
                if "chaos-operator" in pod_name:
                    log_text = extract_mcp_text(log_data)
                    if log_text:
                        log_text = greedy_compress_lines(log_text)
                        sections.append(
                            f"## CHAOS-OPERATOR LOGS ({pod_name})\n"
                            + log_text[-2000:]
                        )

    # ── 8. App pod logs (symptom pods: restarts, errors, crashes) ─────────────
    # Includes ALL non-chaos pods that have logs — last 30 lines plus any
    # error/crash/oom/killed/failed lines — so the LLM can determine root cause.
    pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
    if isinstance(pods_log, dict):
        app_log_parts: List[str] = []
        for pod_name, log_data in pods_log.items():
            if "chaos-exporter" in pod_name or "chaos-operator" in pod_name:
                continue
            log_text = extract_mcp_text(log_data)
            if not log_text or not log_text.strip():
                continue
            log_text = greedy_compress_lines(log_text)
            lines = [l.strip() for l in log_text.strip().split("\n") if l.strip()]
            # Prioritise diagnostic lines (errors, kills, OOM, restarts)
            diag_keywords = (
                "error", "failed", "fatal", "panic", "oomkilled",
                "killed", "crash", "exception", "timeout", "refused",
                "backoff", "readiness", "liveness",
            )
            diag_lines = [
                l for l in lines
                if any(kw in l.lower() for kw in diag_keywords)
            ]
            # Use last 30 lines as context; if diagnostic lines exist, include them first
            tail_lines = lines[-30:]
            combined = diag_lines[-10:] + ["----- recent log tail -----"] + tail_lines
            # Deduplicate while preserving order
            seen_log: set = set()
            deduped = []
            for ln in combined:
                if ln not in seen_log:
                    seen_log.add(ln)
                    deduped.append(ln)
            app_log_parts.append(
                f"  [{pod_name}]\n"
                + "\n".join(f"    {l[:220]}" for l in deduped)
            )
        if app_log_parts:
            sections.append(
                f"## APP POD LOGS (diagnostic lines + last 30 lines per pod)\n"
                + "\n\n".join(app_log_parts)
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
    event_watermark_ts: Optional[datetime] = None,
) -> Optional[Dict[str, Any]]:
    """
    Send MCP data to the LLM for deep analysis.

    Returns parsed analysis dict or None on failure.
    """
    analysis_prompt = _load_prompt("analysis")
    payload_text = build_llm_data_payload(
        mcp_data, server_type, cfg.k8s_namespace,
        include_chaos=cfg.include_chaos_tools,
        event_watermark_ts=event_watermark_ts,
        event_recency_fallback_sec=cfg.event_recency_fallback_sec,
        gate_verdict=(agent_context or {}).get("detection_gate", ""),
    )
    payload_summary = build_mcp_data_summary(
        mcp_data,
        include_chaos=cfg.include_chaos_tools,
    )

    user_content = _build_structured_analysis_user_content(
        cfg=cfg,
        server_type=server_type,
        scan_id=scan_id,
        payload_text=payload_text,
        summary=payload_summary,
        agent_context=agent_context,
    )
    user_content, leaked_terms = _sanitize_leakage_terms(user_content)
    if leaked_terms:
        logger.warning(
            "Leakage guard redacted %d term(s) before llm_analysis: %s",
            len(leaked_terms),
            sorted(set(leaked_terms)),
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": analysis_prompt},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "Agent \u2192 LLM Gateway: requesting analysis of %s MCP data (%d chars) \u2026",
        server_type,
        sum(len(m["content"]) for m in messages),
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
                    "trace_id": cfg.notify_id or scan_id,
                    "session_id": scan_id,
                    "generation_name": "llm_analysis",
                    "generation_id": _gen_id,
                    "scan_id": scan_id,
                    "step": "llm-analysis",
                    "namespace": cfg.k8s_namespace,
                    "agent": cfg.agent_name,
                    **({"agent_id": cfg.agent_id} if cfg.agent_id else {}),
                    **(agent_context or {}),
                    "redacted_terms": sorted(set(leaked_terms)) if leaked_terms else [],
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

        # Log analysis summary using current schema (environment_state +
        # identified_issues + system_errors + health). Legacy chaos_faults /
        # experiment_summary / workflow_errors fields are no longer emitted
        # by the prompt — flash-agent reports observation-grade signals,
        # not workflow verdicts.
        env_state = result.get("environment_state", {})
        issues = result.get("identified_issues", []) or []
        sys_errors = result.get("system_errors", []) or []
        logger.info(
            "LLM Gateway \u2192 Agent: analysis complete | "
            "health_status=%s | score=%s | issues=%d | system_errors=%d",
            env_state.get("health_status", "N/A"),
            result.get("health", {}).get("overall_health_score", "N/A"),
            len(issues),
            len(sys_errors),
        )
        for issue in issues:
            sev = str(issue.get("severity", "?")).upper()
            sev_icon = (
                "\u274c" if sev == "CRITICAL"
                else "\u26a0" if sev == "WARNING"
                else "\u2139"
            )
            logger.info(
                "  %s [%s] %s \u2192 %s",
                sev_icon,
                sev,
                issue.get("affected_component", "?"),
                (issue.get("issue_name", "") or "")[:80],
            )
    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON: %s", exc)
    except Exception as exc:
        logger.error("LLM analysis call failed: %s", exc)

    # ── Deterministic score post-processor ─────────────────────────────────
    # The LLM (temperature=0.1) was observed returning identical
    # overall_health_score across baseline + active-fault scans. To make the
    # score telemetry-responsive we recompute it deterministically here from
    # the LLM's own issue list + the prom summary signals, and take the
    # MORE pessimistic of (LLM, deterministic). This is purely floor-tightening:
    # we never raise the LLM's score, only lower it when evidence demands.
    if isinstance(result, dict):
        try:
            issues = result.get("identified_issues", []) or []
            prom_s = (payload_summary.get("prometheus", {}) or {})
            sev_count = {"critical": 0, "warning": 0, "info": 0}
            for it in issues:
                s = str(it.get("severity", "")).lower()
                if s in sev_count:
                    sev_count[s] += 1
            # Extra deductions from explicit prom signals
            term = prom_s.get("pods_terminated_by_reason", {}) or {}
            wait = prom_s.get("pods_waiting_by_reason", {}) or {}
            oom_n = len(term.get("OOMKilled", []) or [])
            clb_n = len(wait.get("CrashLoopBackOff", []) or [])
            cpu_o = prom_s.get("cpu_outlier_pods", []) or []
            mem_o = prom_s.get("memory_outlier_pods", []) or []
            big_outliers = sum(
                1 for o in (cpu_o + mem_o)
                if isinstance(o, dict) and (o.get("sigma") or 0) >= 3.0
            )
            vol_top = prom_s.get("volume_usage_top", []) or []
            near_full = sum(
                1 for v in vol_top
                if isinstance(v, dict) and (v.get("ratio") or 0) >= 0.80
            )
            restarting = prom_s.get("restarting_pods", []) or []
            recent_restarts_list = prom_s.get("recent_restarts", []) or []
            # Use whichever is larger — recent_restarts (5-min window via
            # increase()) catches pod-recreation that resets the cumulative
            # counter; restarting_pods catches pods whose cumulative count
            # is still non-zero. Both signals point to the same incident
            # so we don't double-count.
            recent_restarts = max(len(restarting), len(recent_restarts_list))
            deduct = (
                12 * sev_count["critical"]
                + 6 * sev_count["warning"]
                + 2 * sev_count["info"]
                + 15 * oom_n
                + 10 * clb_n
                + 10 * big_outliers
                + 10 * near_full
                + min(8 * recent_restarts, 24)  # cap restart contribution
            )
            deterministic_score = max(0, 100 - deduct)
            health = result.setdefault("health", {})
            llm_score = health.get("overall_health_score")
            try:
                llm_score_int = int(llm_score) if llm_score is not None else 100
            except (TypeError, ValueError):
                llm_score_int = 100
            final_score = min(llm_score_int, deterministic_score)
            if final_score != llm_score_int:
                logger.info(
                    "Score post-process: llm=%s deterministic=%s final=%s "
                    "(crit=%d warn=%d info=%d oom=%d clb=%d outl3s=%d nearfull=%d restarts=%d)",
                    llm_score_int, deterministic_score, final_score,
                    sev_count["critical"], sev_count["warning"], sev_count["info"],
                    oom_n, clb_n, big_outliers, near_full, recent_restarts,
                )
            health["overall_health_score"] = final_score
            # Re-derive health_status to stay consistent with the new score
            env_s = result.setdefault("environment_state", {})
            if final_score < 60:
                env_s["health_status"] = "Critical"
            elif final_score < 100:
                env_s["health_status"] = "Degraded"
            # 100 keeps whatever the LLM said (likely "Healthy")
        except Exception as exc:
            logger.warning("Score post-process skipped: %s", exc)

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
        _post_meta["issues_count"] = len(
            result.get("identified_issues", []) or []
        )
        _env = result.get("environment_state", {}) or {}
        if _env.get("health_status"):
            _post_meta["health_status"] = _env["health_status"]
    if _post_meta:
        update_generation_metadata(cfg, _gen_id, _post_meta)

    return result
