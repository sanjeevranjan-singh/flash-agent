"""
Flash Agent v3.0.0 – ITOps Kubernetes Log Metrics Agent
========================================================

Architecture (matches hand-drawn diagram):

  Tool ──► MCP Server ◄──► Agent ◄──► LLM Gateway ◄──► LLM
                              │
              ┌───────────────┼───────────────────┐
              │               │                   │
     ① MCP Req+Res     ② LLM Req+Res       ③ OTL Collector
       saved to            saved in            data flows
       SEPARATE FILE       OTL Collector       to Langfuse
       + OTL Collector

Storage Rules (per diagram notes):
  ① (Req, Response) between MCP Server and Agent
      → stored in a SEPARATE FILE  (mcp_interactions.jsonl)
      → ALSO stored in OTL Collector
  ② (Request, Response) between Agent and LLM Gateway
      → stored in OTL Collector
  ③ Finally, OTL Collector data goes to Langfuse
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Load .env file if present (prefer .env.local for local testing)
try:
    from dotenv import load_dotenv
    
    # Check if .env.local exists, otherwise use .env
    env_file = ".env.local" if Path(".env.local").exists() else ".env"
    load_dotenv(env_file, override=True)
except ImportError:
    pass

try:
    from langfuse import Langfuse
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False

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
# Configuration  (all values loaded from .env / .env.local)
# ──────────────────────────────────────────────────────────────────────────────
AGENT_NAME    = os.getenv("AGENT_NAME")
AGENT_MODE    = os.getenv("AGENT_MODE")
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE")
K8S_NODE_IP   = os.getenv("K8S_NODE_IP")

# LLM Gateway
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL_ALIAS     = os.getenv("MODEL_ALIAS")

# MCP Servers
K8S_MCP_URL  = os.getenv("K8S_MCP_URL")
PROM_MCP_URL = os.getenv("PROM_MCP_URL")
MCP_TIMEOUT  = int(os.getenv("MCP_TIMEOUT", "30"))

# OTL Collector → OTLP endpoint (rule ② and ③ – collector forwards to Langfuse)
OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
OTEL_EXPORTER_OTLP_HEADERS  = os.getenv("OTEL_EXPORTER_OTLP_HEADERS",  "")

# Langfuse direct client (rule ③ – additional rich metadata path)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "")

# ① Separate file for MCP Req+Res (JSONL – one record per line)
MCP_INTERACTIONS_FILE = os.getenv("MCP_INTERACTIONS_FILE")

# Chaos experiment context
CHAOS_NAMESPACE = os.getenv("CHAOS_NAMESPACE", "litmus")

# Scan behaviour
TRACE_TAGS    = [t.strip() for t in os.getenv("TRACE_TAGS", "").split(",") if t.strip()]
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "0"))
_scan_counter = 0  # incremented each scan for sequencing in Langfuse

# ──────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ──────────────────────────────────────────────────────────────────────────────
_shutdown = False


def _handle_signal(signum, frame) -> None:
    global _shutdown
    logger.info("Received signal %s – shutting down gracefully", signum)
    _shutdown = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ══════════════════════════════════════════════════════════════════════════════
# RULE ①  –  SEPARATE FILE:  persist MCP Server ↔ Agent Req+Res
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_mcp_log_dir() -> None:
    """Create the parent directory for the MCP interactions file if needed."""
    try:
        Path(MCP_INTERACTIONS_FILE).parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("Could not create MCP log directory: %s", exc)


def persist_mcp_interaction_to_file(
    server_type: str,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    duration_sec: float,
    scan_id: str,
) -> None:
    """
    Rule ①a – Write MCP Server ↔ Agent (Req, Response) to a SEPARATE FILE.
    Uses JSONL format: one JSON object per line for easy tail/grep/streaming.
    File path is controlled by MCP_INTERACTIONS_FILE env var.
    """
    record = {
        "scan_id":      scan_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "server_type":  server_type,
        "namespace":    K8S_NAMESPACE,
        "duration_sec": round(duration_sec, 3),
        "request":      request_payload,
        "response":     response_payload,
        "has_error":    "error" in response_payload,
    }
    try:
        _ensure_mcp_log_dir()
        with open(MCP_INTERACTIONS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("① MCP Req+Res saved to file: %s", MCP_INTERACTIONS_FILE)
    except Exception as exc:
        logger.warning("Failed to write MCP interaction file: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# OTL COLLECTOR  –  init + span helpers  (rules ① and ②)
# The OTL Collector receives both MCP and LLM spans, then forwards to Langfuse.
# ══════════════════════════════════════════════════════════════════════════════

def init_otl_collector() -> Optional[trace.Tracer]:
    """
    Initialise the OpenTelemetry (OTL) Collector OTLP exporter.

    - Rule ①b : MCP Req+Res spans are sent here.
    - Rule ②  : LLM Req+Res spans are sent here.
    - Rule ③  : The collector endpoint should be Langfuse OTLP or an OTel
                Collector that forwards to Langfuse.
    """
    if not OTEL_EXPORTER_OTLP_ENDPOINT:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT not set – OTL Collector disabled")
        return None
    try:
        headers: Dict[str, str] = {}
        if OTEL_EXPORTER_OTLP_HEADERS:
            for pair in OTEL_EXPORTER_OTLP_HEADERS.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    headers[k.strip()] = v.strip()

        resource = Resource.create({
            SERVICE_NAME:             AGENT_NAME,
            "service.version":        "3.0.0",
            "deployment.environment": AGENT_MODE,
            "k8s.namespace":          K8S_NAMESPACE,
        })
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=OTEL_EXPORTER_OTLP_ENDPOINT,
            headers=headers or None,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        tracer = trace.get_tracer(AGENT_NAME)
        logger.info("OTL Collector initialised → %s", OTEL_EXPORTER_OTLP_ENDPOINT)
        return tracer
    except Exception as exc:
        logger.warning("OTL Collector init failed: %s", exc)
        return None


def _otl_record_mcp_span(
    tracer: trace.Tracer,
    server_type: str,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    duration_sec: float,
    scan_id: str,
) -> None:
    """
    Rule ①b – Store MCP Req+Res in OTL Collector as a span
    (in addition to the separate file written by rule ①a).
    """
    with tracer.start_as_current_span(
        "mcp-interaction",
        attributes={
            "scan.id":          scan_id,
            "mcp.server_type":  server_type,
            "mcp.url":          K8S_MCP_URL if server_type == "kubernetes" else PROM_MCP_URL,
            "k8s.namespace":    K8S_NAMESPACE,
            "mcp.duration_sec": round(duration_sec, 3),
            "mcp.has_error":    "error" in response_payload,
            "mcp.request":      json.dumps(request_payload)[:1024],
            "mcp.response":     json.dumps(response_payload)[:2048],
        },
    ):
        pass
    logger.info("① MCP Req+Res also recorded in OTL Collector (rule ①b)")


def _otl_record_llm_span(
    tracer: trace.Tracer,
    span_name: str,
    messages_in: List[Dict[str, str]],
    response_text: str,
    usage: Dict[str, int],
    duration_sec: float,
    scan_id: str,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Rule ② – Store Agent ↔ LLM Gateway (Req+Res) in OTL Collector as a span.
    """
    attrs: Dict[str, Any] = {
        "scan.id":               scan_id,
        "llm.span_name":         span_name,
        "llm.model":             MODEL_ALIAS,
        "llm.gateway_url":       OPENAI_BASE_URL,
        "llm.duration_sec":      round(duration_sec, 3),
        "llm.prompt_tokens":     usage.get("prompt_tokens",     0),
        "llm.completion_tokens": usage.get("completion_tokens", 0),
        "llm.request":           json.dumps(messages_in)[:2048],
        "llm.response":          response_text[:2048],
    }
    if extra_attrs:
        attrs.update(extra_attrs)

    with tracer.start_as_current_span(span_name, attributes=attrs):
        pass
    logger.info("② LLM Req+Res (%s) recorded in OTL Collector (rule ②)", span_name)


# ══════════════════════════════════════════════════════════════════════════════
# RULE ③  –  LANGFUSE CLIENT
# OTL Collector forwards spans to Langfuse via OTLP.
# This direct client is an additional path for richer metadata / scores.
# ══════════════════════════════════════════════════════════════════════════════

def _build_langfuse_client() -> Optional["Langfuse"]:
    """
    Rule ③: OTL Collector data goes to Langfuse.
    The OTLP exporter already forwards all spans when the endpoint is set to
    Langfuse's OTLP URL.  This SDK client adds cost tracking, scores, and
    richer generation metadata on top of the OTLP path.
    """
    if not _LANGFUSE_AVAILABLE:
        logger.warning("langfuse package not installed – direct Langfuse client disabled")
        return None
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        logger.info(
            "Langfuse credentials not set – direct SDK client disabled "
            "(OTLP → Langfuse path still active if endpoint is configured)"
        )
        return None
    try:
        lf = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        logger.info("③ Langfuse direct client initialised (host=%s)", LANGFUSE_HOST)
        return lf
    except Exception as exc:
        logger.warning("Langfuse direct client init failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI / LLM Gateway client
# ──────────────────────────────────────────────────────────────────────────────

def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=OPENAI_API_KEY or "not-needed",
        base_url=OPENAI_BASE_URL,
        timeout=120.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1  –  Agent → LLM Gateway: select the MCP tool
# Stores LLM Req+Res in OTL Collector (rule ②) and Langfuse (rule ③)
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_SELECTION_SYSTEM = """\
You are an ITOps routing agent. Given a monitoring query, choose the best data source.

Available tools:
  kubernetes  – pod health, container logs, CrashLoopBackOff, OOMKilled,
                image-pull errors, pod restarts, Kubernetes resource state.
  prometheus  – time-series metrics, CPU/memory trends, request-rate,
                error-rate, latency percentiles, numeric operational data.

Reply with EXACTLY one word: kubernetes  OR  prometheus.  No explanation."""


def agent_request_tool_selection(
    user_query: str,
    otl_tracer: Optional[trace.Tracer],
    langfuse_client: Optional["Langfuse"] = None,
    scan_id: str = "",
) -> str:
    """
    Agent → LLM Gateway: ask which MCP tool to use for this scan query.

    Storage:
      Rule ② – LLM Req+Res span → OTL Collector
      Rule ③ – generation also logged to Langfuse (via Langfuse v4 API)
    Returns 'kubernetes' or 'prometheus'.
    """
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": f"{_TOOL_SELECTION_SYSTEM}\n\nQuery: {user_query}"},
    ]
    logger.info("Agent → LLM Gateway: requesting tool selection …")

    # Langfuse direct generation (rule ③) – Langfuse v4 API
    lf_generation = None
    if langfuse_client:
        try:
            lf_generation = langfuse_client.start_observation(
                name="llm-tool-selection",
                as_type="generation",
                model=MODEL_ALIAS,
                input=messages,
            )
        except Exception as exc:
            logger.warning("Langfuse generation init failed: %s", exc)

    decision    = "kubernetes"
    output_text = ""
    usage: Dict[str, int] = {}
    t0 = time.time()

    try:
        resp = _openai_client().chat.completions.create(
            model=MODEL_ALIAS,
            messages=messages,
            temperature=0,
            max_tokens=50,
        )
        msg = resp.choices[0].message
        # Some models (reasoning) return content=None; fall back to reasoning_content
        raw_text = msg.content or getattr(msg, "reasoning_content", None) or ""
        output_text = raw_text.strip().lower()
        usage = {
            "prompt_tokens":     resp.usage.prompt_tokens     if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        decision = "prometheus" if "prometheus" in output_text else "kubernetes"
        logger.info("LLM Gateway → Agent: tool decision = %s (raw=%r)", decision, output_text)

    except Exception as exc:
        logger.error("LLM tool-selection failed: %s – defaulting to kubernetes", exc)

    duration = time.time() - t0

    # Rule ②: LLM Req+Res → OTL Collector
    if otl_tracer:
        _otl_record_llm_span(
            tracer=otl_tracer,
            span_name="llm-tool-selection",
            messages_in=messages,
            response_text=output_text or decision,
            usage=usage,
            duration_sec=duration,
            scan_id=scan_id,
            extra_attrs={"llm.decision": decision},
        )

    # Rule ③: finalise Langfuse generation (v4 API: update then end)
    if lf_generation:
        try:
            lf_generation.update(
                output=output_text or decision,
                usage_details={
                    "input":  usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                },
                metadata={"decision": decision, "duration_sec": round(duration, 3)},
            )
            lf_generation.end()
        except Exception as exc:
            logger.warning("Langfuse generation.end failed: %s", exc)

    return decision


# ══════════════════════════════════════════════════════════════════════════════
# Helper: Generate fallback synthetic MCP data when pod DNS connectivity fails
# ══════════════════════════════════════════════════════════════════════════════

def generate_mcp_fallback_data(server_type, query):
    """
    Generate realistic synthetic Kubernetes/Prometheus data when MCP pod
    is unreachable (e.g., due to pod DNS connectivity issues).
    
    This allows the agent to continue gracefully and provide LLM analysis
    even when underlying MCP infrastructure fails.
    """
    import datetime
    
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    if server_type.lower() == "kubernetes":
        # Synthetic Kubernetes pod data
        return {
            "status": "fallback",
            "reason": "MCP pod cannot reach Kubernetes API (DNS connectivity issue)",
            "data": {
                "cluster": "sock-shop",
                "namespace": K8S_NAMESPACE,
                "timestamp": timestamp,
                "pods": [
                    {
                        "name": f"pod-{i}",
                        "namespace": K8S_NAMESPACE,
                        "status": "Unknown",
                        "phase": "Unknown",
                        "ready": "Unknown/Unknown",
                        "restarts": 0,
                        "reason": "MCP pod DNS connectivity issue"
                    }
                    for i in range(1, 4)
                ],
                "query_type": "operational_health",
                "query_original": query,
                "warnings": [
                    "MCP pod DNS failures - cannot resolve kubernetes.default.svc",
                    "Using synthetic data for LLM analysis",
                    "Actual cluster metrics unavailable"
                ],
                "recommendation": "Check MCP pod DNS configuration and Kubernetes cluster DNS service health"
            }
        }
    else:
        # Synthetic Prometheus metrics data
        return {
            "status": "fallback",
            "reason": "MCP pod cannot reach Kubernetes API (DNS connectivity issue)",
            "data": {
                "cluster": "sock-shop",
                "timestamp": timestamp,
                "metrics": {
                    "up": 0,
                    "node_memory_MemAvailable_bytes": 8589934592,  # 8GB placeholder
                    "node_cpu_seconds_total": 0,
                    "rate(container_cpu_usage_seconds_total[1m])": 0.05,
                    "container_memory_usage_bytes": 268435456  # 256MB placeholder
                },
                "query_type": "system_metrics",
                "query_original": query,
                "warnings": [
                    "MCP pod DNS failures - Prometheus unavailable",
                    "Using synthetic metrics for LLM analysis",
                    "Actual system metrics unavailable"
                ]
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2  –  Agent → MCP Server: call the selected tool, get Tool Response
# Stores MCP Req+Res in SEPARATE FILE (rule ①a) AND OTL Collector (rule ①b)
# ══════════════════════════════════════════════════════════════════════════════

def _mcp_jsonrpc_call(
    url: str,
    method: str,
    params: Dict[str, Any],
    session_id: Optional[str] = None,
    call_id: int = 1,
) -> tuple[Dict[str, Any], Optional[str]]:
    """
    Send a JSON-RPC 2.0 request to an MCP server and parse the SSE response.
    Returns (result_dict, session_id).
    """
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": f"{AGENT_NAME}/3.0",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    body = {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": method,
        "params": params,
    }

    resp = requests.post(url, json=body, headers=headers, timeout=MCP_TIMEOUT)
    resp.raise_for_status()

    # Capture session ID from response headers
    new_session_id = resp.headers.get("Mcp-Session-Id", session_id)

    # Parse SSE response: look for "data:" lines containing JSON-RPC result
    result: Dict[str, Any] = {}
    for line in resp.text.splitlines():
        if line.startswith("data: ") or line.startswith("data:"):
            data_str = line.split("data:", 1)[1].strip()
            if data_str:
                try:
                    parsed = json.loads(data_str)
                    if "result" in parsed:
                        result = parsed["result"]
                    elif "error" in parsed:
                        result = {"error": parsed["error"]}
                except json.JSONDecodeError:
                    pass

    return result, new_session_id


def _mcp_init_session(url: str) -> Optional[str]:
    """Initialize an MCP session and return the session ID."""
    try:
        _, session_id = _mcp_jsonrpc_call(
            url=url,
            method="initialize",
            params={
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": AGENT_NAME, "version": "3.0"},
            },
        )
        return session_id
    except Exception as exc:
        logger.warning("MCP session init failed for %s: %s", url, exc)
        return None


def _extract_active_pod_names(
    pod_list_result: Dict[str, Any],
    namespace: str,
) -> List[str]:
    """
    Parse the pods_list_in_namespace MCP result and return names of
    Running or Error pods belonging to chaos workflow runs.
    Used for Phase 2 targeted pod-log fetching.
    """
    pod_names: List[str] = []
    try:
        content = pod_list_result.get("content", [])
        if not content:
            return pod_names
        text = content[0].get("text", "")
        for line in text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue  # skip header
            parts = line.split()
            if len(parts) < 6:
                continue
            pod_name = parts[3]   # NAME column
            status   = parts[5]   # STATUS column
            # Target running or error pods from chaos / Argo workflows
            if status in ("Running", "Error") and (
                "sock-shop" in pod_name
                or "argowf-chaos" in pod_name
                or "chaos" in pod_name.lower()
            ):
                pod_names.append(pod_name)
    except Exception as exc:
        logger.warning("Failed to extract pod names for log fetching: %s", exc)
    return pod_names


def _build_mcp_trace_summary(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact but meaningful summary of MCP response data for Langfuse traces.
    Extracts key operational signals instead of dumping raw text.
    This is ONLY for trace display — does NOT affect what flows to the LLM.
    """
    import re as _re

    if "error" in response_payload:
        return {"status": "error", "error": str(response_payload.get("error", ""))[:200]}

    mcp_data = response_payload.get("data", {})
    summary: Dict[str, Any] = {"status": "ok"}

    # ── pods_list_in_namespace → pod status breakdown ────────────────────────
    pods_text = _extract_mcp_text(mcp_data.get("pods_list_in_namespace", {}))
    if pods_text:
        status_counts: Dict[str, int] = {}
        total_restarts = 0
        high_restart_pods: List[str] = []
        for line in pods_text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                pod_name = parts[3]
                status = parts[5]
                status_counts[status] = status_counts.get(status, 0) + 1
                try:
                    restarts = int(parts[6].split("(")[0]) if len(parts) > 6 else 0
                    total_restarts += restarts
                    if restarts > 0:
                        high_restart_pods.append(f"{pod_name} ({restarts})")
                except (ValueError, IndexError):
                    pass
        summary["pods"] = {
            "total": sum(status_counts.values()),
            "by_status": status_counts,
            "total_restarts": total_restarts,
        }
        if high_restart_pods:
            summary["pods"]["restarted_pods"] = high_restart_pods[:5]

    # ── events_list → event type breakdown ───────────────────────────────────
    events_text = _extract_mcp_text(mcp_data.get("events_list", {}))
    if events_text:
        if "No events found" in events_text or not events_text.strip():
            summary["events"] = {"total": 0, "note": "No events found"}
        else:
            warning_reasons: List[str] = []
            normal_count = 0
            warning_count = 0
            for line in events_text.strip().split("\n"):
                if line.startswith("NAMESPACE") or not line.strip():
                    continue
                if "Warning" in line:
                    warning_count += 1
                    for keyword in ("BackOff", "Failed", "FailedScheduling",
                                    "Unhealthy", "OOMKilling", "Evicted",
                                    "FailedMount", "FailedCreate"):
                        if keyword in line:
                            warning_reasons.append(keyword)
                            break
                elif "Normal" in line:
                    normal_count += 1
            summary["events"] = {"normal": normal_count, "warning": warning_count}
            if warning_reasons:
                summary["events"]["warning_reasons"] = warning_reasons[:5]

    # ── pods_top → resource usage or error ───────────────────────────────────
    pods_top_data = mcp_data.get("pods_top", {})
    pods_top_text = _extract_mcp_text(pods_top_data)
    if isinstance(pods_top_data, dict) and "error" in pods_top_data:
        summary["pods_top"] = {"error": str(pods_top_data["error"])[:100]}
    elif pods_top_text and "error" in pods_top_text.lower():
        summary["pods_top"] = {"error": pods_top_text[:100]}
    elif pods_top_text:
        summary["pods_top"] = {"available": True, "lines": len(pods_top_text.strip().split("\n"))}

    # ── chaosengines → engine names ──────────────────────────────────────────
    engines_text = _extract_mcp_text(mcp_data.get("chaosengines", {}))
    if engines_text and engines_text.strip():
        engine_names: List[str] = []
        for line in engines_text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                engine_names.append(parts[3])
        summary["chaosengines"] = {"count": len(engine_names), "engines": engine_names[:10]}
    else:
        summary["chaosengines"] = {"count": 0}

    # ── chaosresults → result names ──────────────────────────────────────────
    results_text = _extract_mcp_text(mcp_data.get("chaosresults", {}))
    if results_text and results_text.strip():
        result_names: List[str] = []
        for line in results_text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                result_names.append(parts[3])
        summary["chaosresults"] = {"count": len(result_names), "results": result_names[:10]}
    else:
        summary["chaosresults"] = {"count": 0}

    # ── argo_workflows → workflow name, phase, age ───────────────────────────
    argo_text = _extract_mcp_text(mcp_data.get("argo_workflows", {}))
    if argo_text and argo_text.strip():
        workflows: List[Dict[str, str]] = []
        for line in argo_text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                wf_name = parts[3]
                wf_phase = "Unknown"
                wf_age = ""
                for p in parts[4:]:
                    if p in ("Succeeded", "Failed", "Running", "Error", "Pending"):
                        wf_phase = p
                    if p and p[-1] in ("d", "h", "m", "s") and p[:-1].replace(".", "").isdigit():
                        wf_age = p
                workflows.append({"name": wf_name, "phase": wf_phase, "age": wf_age})
        # Identify experiment workflows (names ending with 10+ digit epoch suffix)
        def _wf_epoch(wf: Dict[str, str]) -> int:
            try:
                suffix = wf["name"].rsplit("-", 1)[1]
                return int(suffix) if len(suffix) >= 10 else 0
            except (ValueError, IndexError):
                return 0
        experiment_wfs = sorted(
            [w for w in workflows if _wf_epoch(w) > 0],
            key=_wf_epoch, reverse=True,
        )
        other_wfs = [w for w in workflows if _wf_epoch(w) == 0]
        # Show all experiment workflows (newest first) + up to 3 others
        sorted_wfs = experiment_wfs + other_wfs[:3]
        summary["argo_workflows"] = {
            "count": len(workflows),
            "latest": experiment_wfs[0]["name"] if experiment_wfs else "none",
            "workflows": sorted_wfs,
        }
    else:
        summary["argo_workflows"] = {"count": 0}

    # ── pods_log → log sizes + key signals per pod ───────────────────────────
    pods_log = mcp_data.get("pods_log", {})
    if isinstance(pods_log, dict) and pods_log:
        log_summary: Dict[str, Any] = {}
        for pod_name, log_data in pods_log.items():
            log_text = _extract_mcp_text(log_data)
            pod_info: Dict[str, Any] = {"size": f"{len(log_text)} chars"}

            if "chaos-exporter" in pod_name:
                faults_found = set(_re.findall(r"FaultName=(\S+)", log_text))
                verdicts_found = _re.findall(
                    r"FaultName=(\S+).*?ResultVerdict=(\S+)", log_text)
                latest: Dict[str, str] = {}
                for f, v in verdicts_found:
                    latest[f] = v
                pod_info["faults_reporting"] = sorted(faults_found)
                pod_info["latest_verdicts"] = latest
            elif "chaos-operator" in pod_name:
                pod_info["reconcile_events"] = log_text.count("Reconciling ChaosEngine")
                pod_info["errors"] = (
                    log_text.count("level=error")
                    + log_text.count('"level":"error"')
                )
            else:
                error_lines = [l.strip() for l in log_text.split("\n")
                               if "error" in l.lower()]
                if error_lines:
                    pod_info["errors"] = len(error_lines)
                    pod_info["last_error"] = error_lines[-1][:150]

            log_summary[pod_name] = pod_info
        summary["pods_log"] = log_summary

    return summary


def agent_call_mcp_server(
    server_type: str,
    query: str,
    otl_tracer: Optional[trace.Tracer],
    langfuse_client: Optional["Langfuse"] = None,
    scan_id: str = "",
) -> Dict[str, Any]:
    """
    Agent → MCP Server: fetch operational data using the selected tool.
    Uses MCP JSON-RPC 2.0 Streamable HTTP protocol.

    Storage:
      Rule ①a – Req+Res → separate JSONL file  (mcp_interactions.jsonl)
      Rule ①b – Req+Res → OTL Collector span
      Rule ③  – Langfuse span (via Langfuse v4 API)
    """
    if server_type not in ("kubernetes", "prometheus"):
        raise ValueError(f"Unknown MCP server: {server_type!r}")

    url = K8S_MCP_URL if server_type == "kubernetes" else PROM_MCP_URL

    # Build the list of MCP tool calls based on server type.
    # Each entry is (mcp_tool_name, arguments, result_key).
    # result_key avoids collisions when the same tool is called multiple times.
    if server_type == "kubernetes":
        # ── Phase 1: discovery + chaos resources + workflow status ────────
        tool_calls = [
            ("pods_list_in_namespace", {"namespace": K8S_NAMESPACE},
             "pods_list_in_namespace"),
            ("events_list",           {"namespace": K8S_NAMESPACE},
             "events_list"),
            ("pods_top",              {"namespace": K8S_NAMESPACE},
             "pods_top"),
            # Litmus ChaosEngine CRs  (MCP tool = "resources_list")
            ("resources_list", {
                "apiVersion": "litmuschaos.io/v1alpha1",
                "kind":       "ChaosEngine",
                "namespace":  CHAOS_NAMESPACE,
            }, "chaosengines"),
            # Litmus ChaosResult CRs
            ("resources_list", {
                "apiVersion": "litmuschaos.io/v1alpha1",
                "kind":       "ChaosResult",
                "namespace":  CHAOS_NAMESPACE,
            }, "chaosresults"),
            # Argo Workflow resources – experiment pipeline steps & status
            ("resources_list", {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind":       "Workflow",
                "namespace":  CHAOS_NAMESPACE,
            }, "argo_workflows"),
        ]
    else:  # prometheus
        tool_calls = [
            ("execute_query", {"query": "up"}, "prometheus_up"),
            ("execute_query",
             {"query": f'count(kube_pod_info{{namespace="{K8S_NAMESPACE}"}})'},
             "pod_count"),
        ]

    request_payload: Dict[str, Any] = {
        "server_type": server_type,
        "namespace":   K8S_NAMESPACE,
        "tools": [result_key for _, _, result_key in tool_calls],
        "tool_count": len(tool_calls),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Agent → MCP Server (%s) | url=%s | tools=%s",
                server_type, url, [f"{t[0]}→{t[2]}" for t in tool_calls])
    response_payload: Dict[str, Any] = {}
    t0 = time.time()

    try:
        # Step 1: Initialize MCP session
        session_id = _mcp_init_session(url)
        logger.info("MCP session initialized | session_id=%s", session_id)

        # Step 2: Phase 1 – call each tool via JSON-RPC tools/call
        all_results: Dict[str, Any] = {}
        for call_idx, (tool_name, tool_args, result_key) in enumerate(tool_calls, start=2):
            try:
                result, session_id = _mcp_jsonrpc_call(
                    url=url,
                    method="tools/call",
                    params={"name": tool_name, "arguments": tool_args},
                    session_id=session_id,
                    call_id=call_idx,
                )
                all_results[result_key] = result
                logger.info("  MCP tool '%s' [%s] → %d chars",
                            tool_name, result_key, len(json.dumps(result)))
            except Exception as exc:
                logger.warning("  MCP tool '%s' [%s] failed: %s",
                               tool_name, result_key, exc)
                all_results[result_key] = {"error": str(exc)}

        # Step 3: Phase 2 – targeted pod logs for running/error workflow pods
        if server_type == "kubernetes":
            pod_list_data = all_results.get("pods_list_in_namespace", {})
            active_pods = _extract_active_pod_names(pod_list_data, K8S_NAMESPACE)
            if active_pods:
                logger.info("Phase 2: fetching logs for %d active pods: %s",
                            len(active_pods), active_pods)
                pods_log_results: Dict[str, Any] = {}
                next_id = len(tool_calls) + 2
                for pod_name in active_pods[:5]:  # cap at 5 pods to limit calls
                    try:
                        log_args: Dict[str, Any] = {
                            "namespace": K8S_NAMESPACE,
                            "name": pod_name,
                        }
                        # Workflow pods have 2 containers (wait+main);
                        # chaos-exporter/operator have 1 container (no arg needed)
                        if "sock-shop-trace" in pod_name or "argowf-chaos" in pod_name:
                            log_args["container"] = "main"
                        log_result, session_id = _mcp_jsonrpc_call(
                            url=url,
                            method="tools/call",
                            params={"name": "pods_log", "arguments": log_args},
                            session_id=session_id,
                            call_id=next_id,
                        )
                        pods_log_results[pod_name] = log_result
                        logger.info("  pods_log '%s' → %d chars",
                                    pod_name, len(json.dumps(log_result)))
                    except Exception as exc:
                        logger.warning("  pods_log '%s' failed: %s", pod_name, exc)
                        pods_log_results[pod_name] = {"error": str(exc)}
                    next_id += 1
                all_results["pods_log"] = pods_log_results
            else:
                logger.info("Phase 2: no active workflow pods found for log fetching")

        response_payload = {
            "server_type": server_type,
            "namespace":   K8S_NAMESPACE,
            "data":        all_results,
        }
        logger.info(
            "MCP Server (%s) → Agent | %.2fs | tools=%s",
            server_type, time.time() - t0, list(all_results.keys()),
        )

    except requests.exceptions.Timeout:
        logger.error("MCP %s timed out after %ds", server_type, MCP_TIMEOUT)
        response_payload = generate_mcp_fallback_data(server_type, query)
    except requests.exceptions.ConnectionError as exc:
        logger.error("MCP %s unreachable: %s", server_type, exc)
        response_payload = generate_mcp_fallback_data(server_type, query)
    except requests.exceptions.HTTPError as exc:
        logger.error("MCP %s HTTP %s: %s", server_type,
                     exc.response.status_code, exc)
        response_payload = generate_mcp_fallback_data(server_type, query)
    except Exception as exc:
        logger.error("MCP %s error: %s", server_type, exc)
        response_payload = generate_mcp_fallback_data(server_type, query)

    duration = time.time() - t0

    # Rule ①a: save to SEPARATE FILE
    persist_mcp_interaction_to_file(
        server_type=server_type,
        request_payload=request_payload,
        response_payload=response_payload,
        duration_sec=duration,
        scan_id=scan_id,
    )

    # Rule ①b: also save to OTL Collector
    if otl_tracer:
        _otl_record_mcp_span(
            tracer=otl_tracer,
            server_type=server_type,
            request_payload=request_payload,
            response_payload=response_payload,
            duration_sec=duration,
            scan_id=scan_id,
        )

    # Rule ③: Langfuse MCP span (v4 API: start_observation + update + end)
    if langfuse_client:
        try:
            mcp_span = langfuse_client.start_observation(
                name=f"mcp-{server_type}-request",
                as_type="span",
                input={
                    "server": server_type,
                    "namespace": K8S_NAMESPACE,
                    "tools": [rk for _, _, rk in tool_calls],
                },
            )
            # Build compact meaningful summary for trace (full data still goes to LLM)
            trace_output = _build_mcp_trace_summary(response_payload)
            mcp_span.update(
                output=trace_output,
                metadata={
                    "server_type":  server_type,
                    "url":          url,
                    "duration_sec": round(duration, 3),
                    "has_error":    "error" in response_payload,
                },
            )
            mcp_span.end()
            logger.info("③ MCP Langfuse span recorded for %s", server_type)
        except Exception as exc:
            logger.warning("Langfuse MCP span failed: %s", exc)

    return response_payload


# ══════════════════════════════════════════════════════════════════════════════
# WORKFLOW STEP PARSER — Extract per-step status from Argo Workflow CR
# ══════════════════════════════════════════════════════════════════════════════

# The expected sequential steps from the sock-shop chaos workflow
EXPECTED_WORKFLOW_STEPS = [
    "install-application",
    "normalize-install-application-readiness",
    "apply-workload-rbac",
    "install-agent",
    "install-chaos-faults",   # parallel with load-test
    "load-test",              # parallel with install-chaos-faults
    "pod-cpu-hog",
    "pod-delete",
    "pod-network-loss",
    "pod-memory-hog",
    "disk-fill",
    "cleanup-chaos-resources",  # parallel with delete-loadtest
    "delete-loadtest",          # parallel with cleanup-chaos-resources
]

# Map chaos step names to their target app and fault details (from the workflow YAML)
CHAOS_STEP_METADATA: Dict[str, Dict[str, Any]] = {
    "pod-cpu-hog": {
        "fault_type": "pod-cpu-hog",
        "target_app": "carts",
        "target_ns": "sock-shop",
        "target_kind": "deployment",
        "duration": 30,
        "probe": "check-frontend-access-url (HTTP GET front-end:80 == 200)",
        "probe_mode": "Continuous",
        "params": {"CPU_CORES": "1", "TOTAL_CHAOS_DURATION": "30",
                   "CHAOS_KILL_COMMAND": "kill md5sum"},
    },
    "pod-delete": {
        "fault_type": "pod-delete",
        "target_app": "catalogue",
        "target_ns": "sock-shop",
        "target_kind": "deployment",
        "duration": 30,
        "probe": "check-catalogue-access-url (HTTP GET front-end:80/catalogue == 200)",
        "probe_mode": "Edge",
        "params": {"TOTAL_CHAOS_DURATION": "30", "CHAOS_INTERVAL": "10",
                   "FORCE": "false"},
    },
    "pod-network-loss": {
        "fault_type": "pod-network-loss",
        "target_app": "user-db",
        "target_ns": "sock-shop",
        "target_kind": "statefulset",
        "duration": 30,
        "probe": "check-cards-access-url (HTTP GET front-end:80/cards == 200)",
        "probe_mode": "Continuous",
        "params": {"TOTAL_CHAOS_DURATION": "30",
                   "NETWORK_PACKET_LOSS_PERCENTAGE": "100",
                   "NETWORK_INTERFACE": "eth0"},
    },
    "pod-memory-hog": {
        "fault_type": "pod-memory-hog",
        "target_app": "orders",
        "target_ns": "sock-shop",
        "target_kind": "deployment",
        "duration": 30,
        "probe": "check-frontend-access-url (HTTP GET front-end:80 == 200)",
        "probe_mode": "Continuous",
        "params": {"TOTAL_CHAOS_DURATION": "30", "MEMORY_CONSUMPTION": "500"},
    },
    "disk-fill": {
        "fault_type": "disk-fill",
        "target_app": "catalogue-db",
        "target_ns": "sock-shop",
        "target_kind": "statefulset",
        "duration": 30,
        "probe": "check-catalogue-db-cr-status (k8sProbe: pod Running, label=name=catalogue-db)",
        "probe_mode": "Continuous",
        "params": {"TOTAL_CHAOS_DURATION": "30", "FILL_PERCENTAGE": "100"},
    },
}


def _extract_mcp_text(mcp_result: Any) -> str:
    """Extract text content from an MCP tool result dict."""
    if isinstance(mcp_result, dict):
        content = mcp_result.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
    return str(mcp_result) if mcp_result else ""


def _parse_workflow_phase_from_text(argo_data: Any) -> Dict[str, str]:
    """
    Parse the Argo Workflow text table from MCP resources_list.
    Returns {workflow_name: phase} for all workflows whose name ends with
    an epoch-like numeric suffix (e.g. my-experiment-1774000085040).
    Fully dynamic — no hardcoded workflow prefix.
    """
    result: Dict[str, str] = {}
    text = _extract_mcp_text(argo_data)
    for line in text.split("\n"):
        if not line.strip() or line.startswith("NAMESPACE"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        wf_name = parts[3]
        # Check if name ends with an epoch-like suffix (10+ digit number)
        suffix = wf_name.rsplit("-", 1)[-1] if "-" in wf_name else ""
        if not (suffix.isdigit() and len(suffix) >= 10):
            continue
        # Extract phase from subsequent columns
        for candidate in parts[4:]:
            if candidate in ("Succeeded", "Failed", "Running", "Error", "Pending"):
                result[wf_name] = candidate
                break
    logger.info("Parsed workflow phases from text: %s", result)
    return result


def _get_latest_workflow(wf_phases: Dict[str, str]) -> tuple:
    """
    From a dict of {workflow_name: phase}, find the LATEST workflow
    by extracting the epoch timestamp suffix.

    Workflow names follow the pattern: <prefix>-<epoch_millis>
    Higher epoch = more recent run.

    Returns (latest_wf_name, latest_wf_phase). Falls back to ("", "Unknown").
    """
    if not wf_phases:
        return ("", "Unknown")

    best_name = ""
    best_phase = "Unknown"
    best_epoch = -1

    for wf_name, phase in wf_phases.items():
        # Extract the epoch suffix: <prefix>-1774000085040
        parts = wf_name.rsplit("-", 1)
        if len(parts) == 2:
            try:
                epoch = int(parts[1])
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_name = wf_name
                    best_phase = phase
            except ValueError:
                pass

    # Fallback: if no epoch could be parsed, use the last entry
    if not best_name and wf_phases:
        best_name = list(wf_phases.keys())[-1]
        best_phase = wf_phases[best_name]

    logger.info("Latest workflow: %s (phase=%s, epoch=%d)", best_name, best_phase, best_epoch)
    return (best_name, best_phase)


def _parse_verdicts_from_exporter_logs(mcp_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Parse REAL chaos verdicts from the chaos-exporter pod logs.
    The exporter emits lines like:
      FaultName=pod-network-loss ResultVerdict=Fail ProbeSuccessPercentage=0
    Returns {fault_name: {verdict, probe_pct, is_final}}.
    """
    import re
    verdicts: Dict[str, Dict[str, Any]] = {}

    pods_log = mcp_data.get("pods_log", {})
    exporter_text = ""
    for pod_name, log_data in pods_log.items():
        if "chaos-exporter" in pod_name:
            exporter_text = _extract_mcp_text(log_data)
            break

    if not exporter_text:
        logger.info("No chaos-exporter logs found in MCP data")
        return verdicts

    for line in exporter_text.split("\n"):
        if "FaultName=" not in line:
            continue
        fault_m = re.search(r"FaultName=(\S+)", line)
        verdict_m = re.search(r"ResultVerdict=(\S+)", line)
        probe_m = re.search(r"ProbeSuccessPercentage=(\S+)", line)
        end_m = re.search(r"EndTime=(\S+)", line)

        if not fault_m:
            continue

        fault_name = fault_m.group(1)
        verdict = verdict_m.group(1) if verdict_m else "Unknown"
        probe_pct = probe_m.group(1) if probe_m else "0"
        end_time = end_m.group(1) if end_m else "0"
        is_final = end_time not in ("0", "0.0")

        existing = verdicts.get(fault_name)
        if existing is None or (is_final and not existing.get("is_final")):
            verdicts[fault_name] = {
                "verdict": verdict,
                "probe_success_pct": probe_pct,
                "is_final": is_final,
            }

    logger.info("Parsed exporter verdicts: %s",
                {k: v["verdict"] for k, v in verdicts.items()})
    return verdicts


def _build_verdict_text_for_prompt(
    verdicts: Dict[str, Dict[str, Any]],
) -> str:
    """
    Build the pre-extracted verdicts text block to inject into the LLM prompt.
    Includes ALL expected chaos faults — marks missing ones explicitly.
    """
    chaos_steps = [s for s in EXPECTED_WORKFLOW_STEPS if s in CHAOS_STEP_METADATA]

    lines: List[str] = []
    found = 0
    missing = 0
    for fault_name in chaos_steps:
        idx = found + missing + 1
        if fault_name in verdicts:
            v = verdicts[fault_name]
            lines.append(
                f"  {idx}. {fault_name}: "
                f"verdict={v['verdict']}, "
                f"probe_success_pct={v['probe_success_pct']}, "
                f"is_final={v['is_final']}"
            )
            found += 1
        else:
            lines.append(
                f"  {idx}. {fault_name}: "
                f"verdict=NO EXPORTER DATA (not found in chaos-exporter logs)"
            )
            missing += 1

    real_passed = sum(1 for v in verdicts.values() if v["verdict"] == "Pass")
    real_failed = sum(1 for v in verdicts.values() if v["verdict"] == "Fail")

    header = (
        f"Total expected chaos faults: {len(chaos_steps)}\n"
        f"Faults with exporter verdicts: {found}\n"
        f"Faults with no exporter data: {missing}\n"
        f"Faults passed (verdict=Pass): {real_passed}\n"
        f"Faults failed (verdict=Fail): {real_failed}\n\n"
    )
    return header + "\n".join(lines) if lines else "  (no chaos faults expected)"


def _cross_validate_llm_with_verdicts(
    result: Dict[str, Any],
    verdicts: Dict[str, Dict[str, Any]],
) -> None:
    """
    Cross-validate LLM analysis against real exporter verdicts.
    Corrects experiment_summary counts and fills missing chaos_faults.
    Mutates *result* in place.
    """
    if not result:
        return

    chaos_steps = [s for s in EXPECTED_WORKFLOW_STEPS if s in CHAOS_STEP_METADATA]
    real_passed = sum(1 for v in verdicts.values() if v["verdict"] == "Pass")
    real_failed = sum(1 for v in verdicts.values() if v["verdict"] == "Fail")
    real_no_data = len(chaos_steps) - len(verdicts)

    # ── Fix experiment_summary counts ──
    exp = result.get("experiment_summary", {})
    if exp:
        old_passed = exp.get("faults_passed", "?")
        old_failed = exp.get("faults_failed", "?")
        if old_passed != real_passed or old_failed != real_failed:
            logger.warning(
                "Cross-validation: LLM fault counts mismatch — correcting: "
                "LLM said passed=%s failed=%s, real is passed=%d failed=%d no_data=%d",
                old_passed, old_failed, real_passed, real_failed, real_no_data,
            )
        exp["total_faults_executed"] = len(chaos_steps)
        exp["faults_passed"] = real_passed
        exp["faults_failed"] = real_failed
        exp.setdefault("faults_no_data", real_no_data)

        # Recalculate resilience
        if real_failed == 0 and real_no_data == 0:
            exp["overall_resilience"] = "resilient"
        elif real_failed == 0:
            exp["overall_resilience"] = "partially-resilient"
        elif real_failed <= real_passed:
            exp["overall_resilience"] = "partially-resilient"
        else:
            exp["overall_resilience"] = "fragile"

    # ── Ensure ALL expected faults appear in chaos_faults ──
    llm_faults = result.get("chaos_faults", [])
    llm_fault_names = {f.get("fault_name", "") for f in llm_faults}

    for fault_name in chaos_steps:
        if fault_name in llm_fault_names:
            # Override verdict with real data if available
            for f in llm_faults:
                if f.get("fault_name") == fault_name and fault_name in verdicts:
                    real = verdicts[fault_name]
                    if f.get("verdict") != real["verdict"]:
                        logger.warning(
                            "Cross-validation: verdict mismatch for %s: LLM=%s real=%s — correcting",
                            fault_name, f.get("verdict"), real["verdict"],
                        )
                    f["verdict"] = real["verdict"]
                    f["probe_success_percentage"] = real["probe_success_pct"]
        else:
            # Fault missing from LLM output — add it
            meta = CHAOS_STEP_METADATA.get(fault_name, {})
            if fault_name in verdicts:
                real = verdicts[fault_name]
                llm_faults.append({
                    "fault_name": fault_name,
                    "verdict": real["verdict"],
                    "probe_success_percentage": real["probe_success_pct"],
                    "target_app": meta.get("target_app"),
                    "target_kind": meta.get("target_kind"),
                    "chaos_duration_sec": meta.get("duration"),
                    "impact_observed": "LLM did not analyse this fault — added by cross-validation",
                    "recovery_observed": "insufficient data to determine recovery from provided logs",
                    "resilience_assessment": f"Verdict={real['verdict']} probe={real['probe_success_pct']}%",
                })
            else:
                llm_faults.append({
                    "fault_name": fault_name,
                    "verdict": "No Exporter Data",
                    "probe_success_percentage": None,
                    "target_app": meta.get("target_app"),
                    "target_kind": meta.get("target_kind"),
                    "chaos_duration_sec": meta.get("duration"),
                    "impact_observed": "chaos-exporter did not report this fault — logs may have rotated",
                    "recovery_observed": "N/A",
                    "resilience_assessment": "Cannot assess — no exporter data available",
                })
            logger.info("Cross-validation: added missing fault %s to LLM output", fault_name)

    result["chaos_faults"] = llm_faults
    logger.info(
        "Cross-validation complete: %d faults total (passed=%d failed=%d no_data=%d)",
        len(chaos_steps), real_passed, real_failed, real_no_data,
    )


def _record_workflow_step_spans(
    langfuse_client: "Langfuse",
    mcp_data: Dict[str, Any],
    llm_analysis: Optional[Dict[str, Any]],
    scan_id: str,
) -> None:
    """
    Create individual Langfuse child spans for each expected workflow step.
    Uses:
      - EXPECTED_WORKFLOW_STEPS (known from workflow YAML)
      - chaos-exporter logs for REAL per-fault verdicts
      - argo_workflows text table for overall workflow phase
      - LLM chaos_faults for supplementary descriptions
    """
    if not langfuse_client:
        return

    # 1. Parse real verdicts from chaos-exporter logs
    verdicts = _parse_verdicts_from_exporter_logs(mcp_data)

    # 2. Parse workflow-level phase from argo text table and pick LATEST by epoch
    wf_phases = _parse_workflow_phase_from_text(mcp_data.get("argo_workflows", {}))
    latest_wf_name, latest_wf_phase = _get_latest_workflow(wf_phases)

    # 3. Build lookup from LLM chaos_faults for impact/recovery descriptions
    llm_faults: Dict[str, Dict] = {}
    if llm_analysis:
        for cf in llm_analysis.get("chaos_faults", []):
            fname = cf.get("fault_name", "") or cf.get("step_name", "")
            if fname:
                llm_faults[fname] = cf

    # 4. Build lookup from LLM workflow_steps for non-chaos step status
    llm_steps: Dict[str, Dict] = {}
    if llm_analysis:
        for ws in llm_analysis.get("workflow_steps", []):
            sname = ws.get("step_name", "")
            if sname:
                llm_steps[sname] = ws

    created = 0
    for i, step_name in enumerate(EXPECTED_WORKFLOW_STEPS):
        is_chaos = step_name in CHAOS_STEP_METADATA

        if is_chaos:
            meta = CHAOS_STEP_METADATA[step_name]
            real = verdicts.get(meta["fault_type"], {})
            real_verdict = real.get("verdict", "N/A")
            probe_pct = real.get("probe_success_pct", "N/A")
            llm_fault = llm_faults.get(step_name, {})

            if real_verdict == "Pass":
                icon = "\u2705"
            elif real_verdict == "Fail":
                icon = "\u274c"
            elif real_verdict == "Awaited":
                icon = "\U0001f525"
            else:
                icon = "\u2b1c"

            display = (
                f"{icon} Step {i+1}: {step_name} \u2192 "
                f"{meta['target_kind']}/{meta['target_app']} [{real_verdict}]"
            )

            span_input: Dict[str, Any] = {
                "step_number": i + 1,
                "step_name": step_name,
                "fault_type": meta["fault_type"],
                "target": f"{meta['target_kind']}/{meta['target_app']}",
                "target_namespace": meta["target_ns"],
                "duration_config": f"{meta['duration']}s",
                "probe": meta["probe"],
                "probe_mode": meta["probe_mode"],
                "params": meta.get("params", {}),
            }

            span_output: Dict[str, Any] = {
                "verdict_source": "chaos-exporter logs (real)",
                "verdict": real_verdict,
                "probe_success_percentage": probe_pct,
                "chaos_duration_sec": llm_fault.get("chaos_duration_sec", "N/A"),
                "impact_observed": llm_fault.get("impact_observed", "N/A"),
                "recovery_observed": llm_fault.get("recovery_observed", "N/A"),
                "resilience_assessment": llm_fault.get("resilience_assessment", "N/A"),
                "llm_verdict": llm_fault.get("verdict", "N/A"),
                "llm_probe_pct": llm_fault.get("probe_success_percentage", "N/A"),
            }
            level = "WARNING" if real_verdict == "Fail" else "DEFAULT"

        else:
            llm_step = llm_steps.get(step_name, {})
            llm_status = llm_step.get("status", "")

            if llm_status in ("Succeeded", "Completed"):
                icon = "\u2705"
                phase = "Succeeded"
            elif llm_status == "Failed":
                icon = "\u274c"
                phase = "Failed"
            elif latest_wf_phase in ("Succeeded", "Running"):
                icon = "\u2705"
                phase = "Succeeded"
            elif latest_wf_phase == "Failed":
                icon = "\u26a0\ufe0f"
                phase = "Unknown (workflow failed)"
            else:
                icon = "\u2b1c"
                phase = latest_wf_phase

            display = f"{icon} Step {i+1}: {step_name}"
            span_input = {
                "step_number": i + 1,
                "step_name": step_name,
                "type": "infrastructure",
            }
            span_output = {
                "phase": phase,
                "workflow": latest_wf_name,
                "details": llm_step.get("details", "N/A"),
            }
            level = "WARNING" if phase == "Failed" else "DEFAULT"

        try:
            step_span = langfuse_client.start_observation(
                name=display,
                as_type="span",
                input=span_input,
                level=level,
                metadata={
                    "scan_id": scan_id,
                    "step_index": i,
                    "is_chaos_fault": is_chaos,
                    "workflow": latest_wf_name or "unknown",
                },
            )
            step_span.update(output=span_output)
            step_span.end()
            created += 1
        except Exception as exc:
            logger.warning("Failed to record Langfuse span for step %s: %s",
                           step_name, exc)

    logger.info("Recorded %d/%d workflow step spans in Langfuse",
                created, len(EXPECTED_WORKFLOW_STEPS))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3  –  Agent → LLM Gateway: deep analysis of Tool Response from MCP
# Stores LLM Req+Res in OTL Collector (rule ②) and Langfuse (rule ③)
# ══════════════════════════════════════════════════════════════════════════════

_ANALYSIS_SYSTEM = """\
You are an expert system-analysis LLM specializing in interpreting the results
of controlled fault-injection experiments. Your job is to analyze ONLY the data
provided to you. You must not rely on any prior knowledge of the experiment
design, fault names, internal workflow steps, expected sequence, or any hidden
structure.

The raw data you receive may include:
- System pod and workload states
- Execution metadata from a workflow engine
- Fault result objects produced by the experiment framework
- Logs from components that record fault execution or probe outcomes
- Application or probe logs (if available)

You must infer everything STRICTLY from the evidence present in the data.

-------------------------------------------------------------------------------
YOUR ANALYSIS TASK
-------------------------------------------------------------------------------

Analyze ONLY the most recent experiment run (identified by the latest timestamp
or highest numeric suffix in identifiers visible in the data). You must produce:

1. EXPERIMENT SUMMARY
   - Identify the latest workflow/run name (based on visible identifiers).
   - Determine its overall phase (Succeeded, Failed, Running) based ONLY on what
     the data shows.
   - Compute fault counts directly from the authoritative list above:
       total_faults_executed
       faults_passed
       faults_failed
   - Provide overall resilience classification in one of:
       "resilient" | "partially-resilient" | "fragile"
     Base this ONLY on fault outcomes and observed behavior.

2. WORKFLOW STEPS (GENERIC)
   The workflow may contain multiple execution steps, but their names or roles
   must NOT be assumed by you.

   For each execution step you discover (from workflow metadata, pod names,
   or logs):
   - step_name: use ONLY the name/identifier visible in the data
   - step_number: infer ordering based on workflow metadata
   - status: Succeeded | Failed | Skipped | Unknown
   - details: summarize what the logs or metadata show; if unclear, say
       "insufficient data"

   DO NOT use or assume any step names not present in the raw data.

3. CHAOS FAULT ANALYSIS
   For each chaos fault you can identify from the data (ChaosResult objects,
   chaos-exporter logs, workflow step names, etc.):

   - fault_name: use the identifier visible in the data
   - verdict: determine from ChaosResult verdict fields or exporter log lines
   - probe_success_percentage: extract from probe results or exporter logs, or null
   - target_app / target_kind:
       infer ONLY if the raw data clearly contains target identifiers
       otherwise set to null
   - chaos_duration_sec:
       infer from visible timestamps or duration fields if present
       otherwise null

   - impact_observed:
       Use ONLY explicit evidence from:
         * probe failures / success
         * application or system logs
         * result object messages
         * workflow/pod status transitions
       If no clear evidence exists:
         "insufficient data to determine impact from provided logs"

   - recovery_observed:
       Use ONLY explicit evidence that shows recovery:
         * probe recovery
         * application returning to normal
         * resolved error conditions
       If not clearly shown:
         "insufficient data to determine recovery from provided logs"

   - resilience_assessment:
       Provide a short, evidence-based statement about how well the
       system handled the fault (no assumptions).

4. WORKFLOW ERRORS
   If any failures occurred:
   - Identify the execution component or pod where the error occurred.
   - Extract the actual error message (short, 1–2 lines).
   - Determine the most likely root cause based on evidence.
   - Provide a concise fix suggestion.

5. ISSUES & HEALTH
   Summarize system health indicators:
   - Pod counts (healthy/unhealthy)
   - Error/warning counts
   - Overall health score (0–100)

-------------------------------------------------------------------------------
RETURN FORMAT (STRICT)
-------------------------------------------------------------------------------

Return ONLY valid JSON:

{
  "experiment_summary": {
    "workflow_name": "<latest workflow name>",
    "workflow_phase": "Succeeded|Failed|Running",
    "total_faults_executed": <int>,
    "faults_passed": <int>,
    "faults_failed": <int>,
    "overall_resilience": "<resilient|partially-resilient|fragile>"
  },
  "workflow_steps": [
    {
      "step_name": "<identifier from data>",
      "step_number": <int>,
      "status": "Succeeded|Failed|Skipped|Unknown",
      "details": "<what happened>"
    }
  ],
  "chaos_faults": [
    {
      "fault_name": "<from pre-extracted list>",
      "verdict": "<Pass|Fail|Awaited|No Observable Data>",
      "probe_success_percentage": <int|null>,
      "target_app": "<inferred or null>",
      "target_kind": "<inferred or null>",
      "chaos_duration_sec": <int|null>,
      "impact_observed": "<impact or insufficient data>",
      "recovery_observed": "<recovery or insufficient data>",
      "resilience_assessment": "<evidence-based assessment>"
    }
  ],
  "workflow_errors": [
    {
      "workflow_name": "<string>",
      "error_pod": "<identifier>",
      "error_step": "<identifier>",
      "error_message": "<actual log excerpt>",
      "root_cause": "<why it failed>",
      "fix_suggestion": "<suggested fix>"
    }
  ],
  "issues": [
    {
      "severity": "critical|warning|info",
      "affected_pod": "<id or N/A>",
      "category": "ChaosExperiment|WorkflowFailure|CrashLoop|OOM|Other",
      "related_fault": "<fault name or none>",
      "summary": "<one sentence>",
      "recommended_action": "<one sentence>"
    }
  ],
  "health": {
    "total_pods": <int>,
    "healthy_pods": <int>,
    "unhealthy_pods": <int>,
    "error_count": <int>,
    "warning_count": <int>,
    "overall_health_score": <0-100>
  }
}

-------------------------------------------------------------------------------
CRITICAL RULES
-------------------------------------------------------------------------------
1. You MUST include ALL faults from the pre-extracted list.
2. Fault counts MUST match the pre-extracted list exactly.
3. Use ONLY information visible in the data—no assumptions or guesses.
4. For missing or unclear impact/recovery, use:
   "insufficient data to determine impact from provided logs"
5. Do NOT fabricate actions, steps, fault names, or system behavior.
6. Output MUST be a valid JSON object conforming to the schema above.
"""


def _build_llm_data_payload(mcp_data: Dict[str, Any], server_type: str) -> str:
    """
    Build a structured data payload for the LLM instead of blindly truncating
    raw JSON.  Ensures the most important data (chaos-exporter logs with fault
    verdicts, Argo workflow list, pod status) is always included.

    Budget: aim for ~12-15K chars total so the combined prompt + data stays
    within free-tier context limits.
    """
    import re as _re

    sections: List[str] = []
    sections.append(
        f"Data source : {server_type.upper()} MCP Tool Response\n"
        f"Namespace   : {K8S_NAMESPACE}\n"
        f"Timestamp   : {datetime.now(timezone.utc).isoformat()}\n"
    )

    # ── 1. Pod status summary (compact) ──────────────────────────────────────
    summary = _build_mcp_trace_summary(mcp_data)
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

    # ── 3. Argo Workflows (sorted newest-first, latest marked) ────────────────
    argo = summary.get("argo_workflows", {})
    if argo.get("count", 0) > 0:
        latest_name = argo.get("latest", "")
        wf_lines = []
        for w in argo.get("workflows", []):
            marker = " ← LATEST" if w["name"] == latest_name else ""
            wf_lines.append(
                f"  - {w['name']}  phase={w.get('phase','?')}  age={w.get('age','?')}{marker}"
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
    #    Include the FULL text (up to 20K) so the LLM always sees verdict lines
    pods_log = mcp_data.get("data", mcp_data).get("pods_log", {})
    if isinstance(pods_log, dict):
        for pod_name, log_data in pods_log.items():
            if "chaos-exporter" in pod_name:
                log_text = _extract_mcp_text(log_data)
                if log_text:
                    # Extract just the verdict lines + keep full log up to 6K
                    verdict_lines = [
                        l.strip() for l in log_text.split("\n")
                        if "FaultName=" in l or "ResultVerdict=" in l
                           or "ProbeSuccessPercentage=" in l
                    ]
                    sections.append(
                        f"## CHAOS-EXPORTER LOGS ({pod_name})\n"
                        f"### Verdict lines (extracted):\n"
                        + "\n".join(verdict_lines[-30:]) + "\n\n"
                        f"### Full log (truncated to 6000 chars):\n"
                        + log_text[-6000:]
                    )

    # ── 7. chaos-operator logs (compact) ─────────────────────────────────────
    if isinstance(pods_log, dict):
        for pod_name, log_data in pods_log.items():
            if "chaos-operator" in pod_name:
                log_text = _extract_mcp_text(log_data)
                if log_text:
                    sections.append(
                        f"## CHAOS-OPERATOR LOGS ({pod_name})\n"
                        + log_text[-2000:]
                    )

    # ── 8. Workflow pod logs (error lines only — for workflow_errors) ────────
    if isinstance(pods_log, dict):
        wf_log_parts: List[str] = []
        for pod_name, log_data in pods_log.items():
            if "chaos-exporter" in pod_name or "chaos-operator" in pod_name:
                continue
            log_text = _extract_mcp_text(log_data)
            if log_text:
                error_lines = [
                    l.strip() for l in log_text.split("\n")
                    if "error" in l.lower() or "failed" in l.lower()
                ]
                if error_lines:
                    wf_log_parts.append(
                        f"  [{pod_name}] ({len(error_lines)} error lines):\n"
                        + "\n".join(f"    {l[:200]}" for l in error_lines[-5:])
                    )
        if wf_log_parts:
            sections.append(
                f"## WORKFLOW POD LOGS (error lines only)\n"
                + "\n".join(wf_log_parts)
            )

    # ── 9. Pre-extracted verdicts (from our own parser) ──────────────────────
    verdicts = _parse_verdicts_from_exporter_logs(mcp_data)
    if verdicts:
        sections.append(
            f"## PRE-EXTRACTED VERDICTS (from chaos-exporter regex parse)\n"
            + json.dumps(verdicts, indent=2)
        )

    return "\n\n".join(sections)


def agent_request_llm_analysis(
    mcp_data: Dict[str, Any],
    server_type: str,
    otl_tracer: Optional[trace.Tracer],
    langfuse_client: Optional["Langfuse"] = None,
    scan_id: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Agent → LLM Gateway: send the Tool Response from MCP for deep analysis.

    Storage:
      Rule ② – LLM Req+Res span → OTL Collector
      Rule ③ – generation also logged to Langfuse (via Langfuse v4 API)
    Returns parsed analysis dict or None on failure.
    """
    payload_text = _build_llm_data_payload(mcp_data, server_type)

    # Parse verdicts locally — used ONLY for post-LLM cross-validation, NOT sent to LLM
    verdicts = _parse_verdicts_from_exporter_logs(mcp_data)

    # Merge system instructions into user message for broad model compatibility
    combined_prompt = (
        f"INSTRUCTIONS:\n{_ANALYSIS_SYSTEM}\n\n"
        f"DATA TO ANALYSE:\n{payload_text}"
    )
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": combined_prompt},
    ]

    logger.info(
        "Agent → LLM Gateway: requesting analysis of %s MCP data (%d chars) …",
        server_type, len(combined_prompt),
    )

    # Langfuse direct generation (rule ③) – Langfuse v4 API
    lf_generation = None
    if langfuse_client:
        try:
            lf_generation = langfuse_client.start_observation(
                name="llm-analysis",
                as_type="generation",
                model=MODEL_ALIAS,
                input={
                    "prompt": "Chaos experiment analysis",
                    "data_source": f"{server_type.upper()} MCP Tool Response",
                    "namespace": K8S_NAMESPACE,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "mcp_data_chars": min(len(json.dumps(mcp_data, indent=2)), 12000),
                    "mcp_summary": _build_mcp_trace_summary(mcp_data),
                },
            )
        except Exception as exc:
            logger.warning("Langfuse analysis generation init failed: %s", exc)

    result:      Optional[Dict[str, Any]] = None
    output_text: str = ""
    usage: Dict[str, int] = {}
    t0 = time.time()

    try:
        resp = _openai_client().chat.completions.create(
            model=MODEL_ALIAS,
            messages=messages,
            temperature=0.1,
        )
        msg = resp.choices[0].message
        # Handle reasoning-only models where content may be None
        output_text = msg.content or getattr(msg, "reasoning_content", None) or ""
        usage = {
            "prompt_tokens":     resp.usage.prompt_tokens     if resp.usage else 0,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        }
        # Extract JSON from response (may be wrapped in markdown code fences)
        json_text = output_text
        if "```json" in json_text:
            json_text = json_text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in json_text:
            json_text = json_text.split("```", 1)[1].split("```", 1)[0]
        result = json.loads(json_text.strip())

        # Cross-validate LLM output against real exporter verdicts
        _cross_validate_llm_with_verdicts(result, verdicts)

        # Log experiment summary if present
        exp = result.get("experiment_summary", {})
        faults = result.get("chaos_faults", [])
        wf_errors = result.get("workflow_errors", [])
        logger.info(
            "LLM Gateway → Agent: analysis complete | "
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
        # Log each fault verdict for visibility
        for f in faults:
            v_icon = "\u2705" if f.get("verdict") == "Pass" else "\u274c" if f.get("verdict") == "Fail" else "\u23f3"
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

    # Rule ②: LLM Req+Res → OTL Collector
    if otl_tracer:
        _otl_record_llm_span(
            tracer=otl_tracer,
            span_name="llm-analysis",
            messages_in=messages,
            response_text=output_text,
            usage=usage,
            duration_sec=duration,
            scan_id=scan_id,
            extra_attrs={
                "llm.has_result":   result is not None,
                "llm.issue_count":  len(result.get("issues", [])) if result else 0,
                "llm.health_score": result.get("health", {}).get("overall_health_score", -1) if result else -1,
            },
        )

    # Rule ③: finalise Langfuse generation (v4 API: update then end)
    if lf_generation:
        try:
            # Build compact output summary for Langfuse trace display
            trace_output: Dict[str, Any] = {}
            # Expected top-level keys from our prompt's RETURN FORMAT
            _EXPECTED_KEYS = {"experiment_summary", "chaos_faults", "workflow_steps", "workflow_errors", "issues", "health"}
            _CORE_KEYS = {"experiment_summary", "chaos_faults", "workflow_steps"}
            if result:
                # Check if LLM followed our schema: need ≥2 expected keys AND at least 1 core chaos key
                matched_keys = _EXPECTED_KEYS & set(result.keys())
                has_core = bool(matched_keys & _CORE_KEYS)
                if has_core and len(matched_keys) >= 2:
                    # LLM followed our format → build compact summary
                    trace_output = {
                        "experiment": result.get("experiment_summary", {}),
                        "chaos_faults": [
                            {
                                "fault": f.get("fault_name"),
                                "verdict": f.get("verdict"),
                                "probe_pct": f.get("probe_success_percentage"),
                                "target": f.get("target_app"),
                                "impact": (f.get("impact_observed", "") or "")[:120],
                                "recovery": (f.get("recovery_observed", "") or "")[:120],
                                "resilience": (f.get("resilience_assessment", "") or "")[:80],
                            }
                            for f in result.get("chaos_faults", [])
                        ],
                        "workflow_errors": [
                            {
                                "workflow": (e.get("workflow_name", "") or "")[-30:],
                                "step": e.get("error_step"),
                                "error": (e.get("error_message", "") or "")[:120],
                                "fix": (e.get("fix_suggestion", "") or "")[:80],
                            }
                            for e in result.get("workflow_errors", [])
                        ],
                        "issue_count": len(result.get("issues", [])),
                        "health_score": result.get("health", {}).get("overall_health_score"),
                    }
                else:
                    # LLM returned valid JSON but NOT in our expected format
                    # Store raw result so we don't lose visibility
                    logger.warning(
                        "LLM result missing expected keys (found: %s, expected ≥2 of: %s). "
                        "Storing raw result in trace.",
                        list(result.keys())[:10], sorted(_EXPECTED_KEYS),
                    )
                    raw_str = json.dumps(result, default=str)
                    trace_output = {
                        "_schema_mismatch": True,
                        "_llm_keys": list(result.keys())[:15],
                        "_matched_keys": sorted(matched_keys),
                        "raw_result": raw_str[:3000],
                    }
            else:
                trace_output = {
                    "error": "LLM returned no valid JSON",
                    "raw_snippet": (output_text or "")[:500],
                }

            lf_generation.update(
                output=trace_output,
                usage_details={
                    "input":  usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                },
                metadata={
                    "has_result":       result is not None,
                    "schema_matched":   result is not None and bool((_EXPECTED_KEYS & set(result.keys())) & _CORE_KEYS) and len((_EXPECTED_KEYS & set(result.keys())) if result else set()) >= 2,
                    "result_keys":      list(result.keys())[:10] if result else [],
                    "issue_count":      len(result.get("issues", [])) if result else 0,
                    "health_score":     result.get("health", {}).get("overall_health_score") if result else None,
                    "duration_sec":     round(duration, 3),
                    "raw_output_snippet": (output_text or "")[:300],
                },
            )
            lf_generation.end()
        except Exception as exc:
            logger.warning("Langfuse analysis generation.end failed: %s", exc)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4  –  OTL Collector: emit root scan span
# Ties together the child spans already recorded under rules ① and ②.
# The collector forwards everything to Langfuse (rule ③).
# ══════════════════════════════════════════════════════════════════════════════

def otl_emit_root_scan_span(
    tracer: trace.Tracer,
    analysis: Dict[str, Any],
    server_type: str,
    duration_sec: float,
    scan_id: str,
) -> None:
    """
    Emit a root 'agent-scan' span in the OTL Collector that wraps the full
    cycle.  Child spans (mcp-interaction, llm-tool-selection, llm-analysis)
    were emitted individually during the cycle.
    The OTLP BatchSpanProcessor forwards all spans to Langfuse (rule ③).
    """
    health = analysis.get("health", {})
    issues = analysis.get("issues", [])

    with tracer.start_as_current_span(
        "agent-scan",
        attributes={
            "scan.id":              scan_id,
            "agent.name":           AGENT_NAME,
            "k8s.namespace":        K8S_NAMESPACE,
            "mcp.server_type":      server_type,
            "scan.duration_sec":    round(duration_sec, 2),
            "health.score":         health.get("overall_health_score", -1),
            "health.total_pods":    health.get("total_pods",           0),
            "health.healthy_pods":  health.get("healthy_pods",         0),
            "health.error_count":   health.get("error_count",          0),
            "health.warning_count": health.get("warning_count",        0),
            "issue.count":          len(issues),
            "tags":                 ",".join(TRACE_TAGS),
        },
    ) as root_span:
        for idx, issue in enumerate(issues):
            root_span.add_event(
                f"issue-{idx}",
                attributes={
                    "severity":           issue.get("severity",           "unknown"),
                    "affected_pod":       issue.get("affected_pod",       "unknown"),
                    "affected_container": issue.get("affected_container", "unknown"),
                    "category":           issue.get("category",           "Other"),
                    "summary":            issue.get("summary",            ""),
                    "recommended_action": issue.get("recommended_action", ""),
                },
            )

    logger.info(
        "③ OTL Collector root span emitted → forwarding to Langfuse | "
        "health=%s | issues=%d",
        health.get("overall_health_score", "N/A"),
        len(issues),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main agent workflow
# ══════════════════════════════════════════════════════════════════════════════

def agent_workflow(
    scan_query: str,
    otl_tracer: Optional[trace.Tracer],
    langfuse_client: Optional["Langfuse"],
) -> Dict[str, Any]:
    """
    Full agentic cycle – mirrors the hand-drawn diagram exactly:

      Agent Request tool  ──►  MCP Server  ──►  Tool Response from MCP
                                                        │
      Agent Req from Agent  ──►  LLM Gateway  ──►  LLM
                                                        │
                 ┌──────────────────────────────────────┘
                 │
         OTL Collector  (rules ① + ②)
                 │
                 ▼
            Langfuse  (rule ③)
    """
    global _scan_counter
    _scan_counter += 1
    scan_start = time.time()
    scan_id = (
        f"{AGENT_NAME}-{K8S_NAMESPACE}"
        f"-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    logger.info("═══ Scan #%d started | scan_id=%s ═══", _scan_counter, scan_id)

    # ── Run entire scan inside a Langfuse root span (v4 API) ─────────────────
    #    All steps (LLM + MCP) attach as children of this one root span.
    analysis = _run_scan_steps(
        scan_query=scan_query,
        scan_id=scan_id,
        scan_start=scan_start,
        otl_tracer=otl_tracer,
        langfuse_client=langfuse_client,
    )

    return analysis


def _run_scan_steps(
    scan_query: str,
    scan_id: str,
    scan_start: float,
    otl_tracer: Optional[trace.Tracer],
    langfuse_client: Optional["Langfuse"],
) -> Dict[str, Any]:
    """Execute all scan steps, optionally inside a Langfuse root span."""

    if langfuse_client:
        try:
            # Use deterministic trace_id from scan_id for correlation
            trace_id = Langfuse.create_trace_id(seed=scan_id)
            session_id = f"{AGENT_NAME}-{K8S_NAMESPACE}"

            # Open root span (v4 API – no .trace() method)
            with langfuse_client.start_as_current_observation(
                name=f"agent-scan ({scan_id[-8:]})",
                as_type="span",
                trace_context={"trace_id": trace_id},
                input={"scan_query": scan_query, "scan_id": scan_id},
                metadata={
                    "agent":            AGENT_NAME,
                    "namespace":        K8S_NAMESPACE,
                    "chaos_namespace":  CHAOS_NAMESPACE,
                    "session_id":       session_id,
                    "scan_number":      _scan_counter,
                    "scan_interval":    SCAN_INTERVAL,
                    "scan_mode":        "continuous" if SCAN_INTERVAL > 0 else "cronjob",
                    "tags":             [AGENT_NAME, K8S_NAMESPACE, f"scan-{_scan_counter}"] + TRACE_TAGS,
                },
            ) as root_span:
                logger.info("③ Langfuse root span created | trace_id=%s | session=%s | scan=#%d",
                            trace_id, session_id, _scan_counter)
                result = _execute_scan_steps(
                    scan_query=scan_query,
                    scan_id=scan_id,
                    scan_start=scan_start,
                    otl_tracer=otl_tracer,
                    langfuse_client=langfuse_client,
                )
                # Update root span with final output before context closes
                duration = time.time() - scan_start
                health = result.get("health", {}) if isinstance(result, dict) else {}
                # Build compact scan output for root span
                scan_output: Dict[str, Any] = {}
                _ROOT_EXPECTED = {"experiment_summary", "chaos_faults", "workflow_steps", "workflow_errors", "issues", "health"}
                _ROOT_CORE = {"experiment_summary", "chaos_faults", "workflow_steps"}
                if isinstance(result, dict):
                    root_matched = _ROOT_EXPECTED & set(result.keys())
                    has_root_core = bool(root_matched & _ROOT_CORE)
                    if has_root_core and len(root_matched) >= 2:
                        scan_output = {
                            "experiment": result.get("experiment_summary", {}),
                            "chaos_faults_summary": [
                                {
                                    "fault": f.get("fault_name"),
                                    "verdict": f.get("verdict"),
                                    "probe_pct": f.get("probe_success_percentage"),
                                    "impact": (f.get("impact_observed", "") or "")[:100],
                                }
                                for f in result.get("chaos_faults", [])
                            ],
                            "workflow_errors_count": len(result.get("workflow_errors", [])),
                            "issue_count": len(result.get("issues", [])),
                            "health_score": health.get("overall_health_score"),
                        }
                    else:
                        # LLM didn't follow expected schema — show raw result
                        raw_str = json.dumps(result, default=str)
                        scan_output = {
                            "_schema_mismatch": True,
                            "_llm_keys": list(result.keys())[:15],
                            "raw_result": raw_str[:2000],
                        }
                root_span.update(
                    output=scan_output,
                    metadata={
                        "duration_sec":    round(duration, 3),
                        "health_score":    health.get("overall_health_score"),
                        "issue_count":     len(result.get("issues", [])) if isinstance(result, dict) else 0,
                        "scan_number":     _scan_counter,
                    },
                )
        except Exception as exc:
            logger.warning("Langfuse root span failed: %s", exc)
            # Fall through to run without Langfuse
            result = _execute_scan_steps(
                scan_query=scan_query,
                scan_id=scan_id,
                scan_start=scan_start,
                otl_tracer=otl_tracer,
                langfuse_client=None,
            )
        finally:
            try:
                langfuse_client.flush()
                logger.info("③ Langfuse traces flushed to cloud")
            except Exception as exc:
                logger.warning("Langfuse flush failed: %s", exc)
    else:
        result = _execute_scan_steps(
            scan_query=scan_query,
            scan_id=scan_id,
            scan_start=scan_start,
            otl_tracer=otl_tracer,
            langfuse_client=None,
        )

    return result


def _execute_scan_steps(
    scan_query: str,
    scan_id: str,
    scan_start: float,
    otl_tracer: Optional[trace.Tracer],
    langfuse_client: Optional["Langfuse"],
) -> Dict[str, Any]:
    """Core scan steps – called within Langfuse root observation context."""

    # ── Step 1: Agent → LLM Gateway → which tool?  (rule ②) ─────────────────
    server_type = agent_request_tool_selection(
        user_query=scan_query,
        otl_tracer=otl_tracer,
        langfuse_client=langfuse_client,
        scan_id=scan_id,
    )

    # ── Step 2: Agent → MCP Server → Tool Response  (rules ①a + ①b) ─────────
    mcp_data = agent_call_mcp_server(
        server_type=server_type,
        query=scan_query,
        otl_tracer=otl_tracer,
        langfuse_client=langfuse_client,
        scan_id=scan_id,
    )

    if mcp_data.get("error"):
        logger.warning("MCP returned an error – analysis will reflect degraded data")

    # ── Step 3: Agent → LLM Gateway → analysis  (rule ②) ────────────────────
    analysis = agent_request_llm_analysis(
        mcp_data=mcp_data,
        server_type=server_type,
        otl_tracer=otl_tracer,
        langfuse_client=langfuse_client,
        scan_id=scan_id,
    )

    if analysis is None:
        logger.error("LLM analysis failed – returning empty result")
        # Still record workflow step spans (they use MCP data, not LLM output)
        if langfuse_client:
            try:
                mcp_inner = mcp_data.get("data", {})
                _record_workflow_step_spans(
                    langfuse_client=langfuse_client,
                    mcp_data=mcp_inner,
                    llm_analysis=None,
                    scan_id=scan_id,
                )
            except Exception as exc:
                logger.warning("Failed to record workflow step spans: %s", exc)
        return {"health": {"overall_health_score": -1}, "issues": []}

    # ── Step 3b: Record per-workflow-step Langfuse spans ─────────────────────
    #    Uses chaos-exporter logs for REAL verdicts + LLM output for descriptions
    if langfuse_client:
        try:
            mcp_inner = mcp_data.get("data", {})
            _record_workflow_step_spans(
                langfuse_client=langfuse_client,
                mcp_data=mcp_inner,
                llm_analysis=analysis,
                scan_id=scan_id,
            )
        except Exception as exc:
            logger.warning("Failed to record workflow step spans: %s", exc)

    duration = time.time() - scan_start

    # ── Step 4: OTL Collector root span → Langfuse  (rule ③) ─────────────────
    if otl_tracer:
        otl_emit_root_scan_span(
            tracer=otl_tracer,
            analysis=analysis,
            server_type=server_type,
            duration_sec=duration,
            scan_id=scan_id,
        )

    # ── Human-readable summary ────────────────────────────────────────────────
    health = analysis.get("health", {})
    issues = analysis.get("issues", [])
    logger.info(
        "═══ Scan complete | scan_id=%s | %.1fs | server=%s | "
        "health=%s | issues=%d | pods=%s ═══",
        scan_id, duration, server_type,
        health.get("overall_health_score", "?"),
        len(issues),
        health.get("total_pods", 0) or 0,
    )
    for issue in issues:
        logger.info(
            "  [%s] %s/%s — %s",
            issue.get("severity", "?").upper(),
            issue.get("affected_pod", "?"),
            issue.get("affected_container", "?"),
            issue.get("summary", ""),
        )

    return analysis


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info(
        "Flash Agent v3.0.0 | agent=%s | namespace=%s | model=%s",
        AGENT_NAME, K8S_NAMESPACE, MODEL_ALIAS,
    )
    logger.info(
        "Kubernetes Node | IP=%s | MCP Servers: K8s=%s | Prometheus=%s",
        K8S_NODE_IP, K8S_MCP_URL, PROM_MCP_URL,
    )
    logger.info(
        "Storage rules: ① MCP→file(%s)+OTL  ② LLM→OTL  ③ OTL→Langfuse",
        MCP_INTERACTIONS_FILE,
    )

    otl_tracer      = init_otl_collector()
    langfuse_client = _build_langfuse_client()

    scan_query = os.getenv(
        "SCAN_QUERY",
        f"Analyse the operational health of all workloads in Kubernetes "
        f"namespace '{K8S_NAMESPACE}'. "
        "Identify pod failures, restarts, resource pressure, and anomalies.",
    )

    if SCAN_INTERVAL <= 0:
        logger.info("CronJob mode – single scan")
        agent_workflow(scan_query, otl_tracer, langfuse_client)
    else:
        logger.info("Continuous mode – scan every %ds", SCAN_INTERVAL)
        while not _shutdown:
            try:
                agent_workflow(scan_query, otl_tracer, langfuse_client)
            except Exception as exc:
                logger.exception("Scan cycle failed: %s", exc)
            for _ in range(SCAN_INTERVAL):
                if _shutdown:
                    break
                time.sleep(1)

    # Flush OTL Collector → forwards to Langfuse (rule ③)
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
        logger.info("③ OTL Collector flushed → Langfuse")

    # Flush Langfuse direct client queue
    if langfuse_client:
        try:
            langfuse_client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed: %s", exc)

    logger.info("Flash Agent shut down cleanly")


if __name__ == "__main__":
    main()