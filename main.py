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
        "tool_calls":  [(name, args) for name, args, _ in tool_calls],
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
                input=request_payload,
            )
            mcp_span.update(
                output=response_payload,
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
    Returns {workflow_name: phase} for sock-shop-trace-1803 workflows.
    """
    result: Dict[str, str] = {}
    text = _extract_mcp_text(argo_data)
    for line in text.split("\n"):
        if "sock-shop-trace-1803" not in line:
            continue
        parts = line.split()
        for idx, p in enumerate(parts):
            if p.startswith("sock-shop-trace-1803"):
                for candidate in parts[idx + 1:]:
                    if candidate in ("Succeeded", "Failed", "Running", "Error", "Pending"):
                        result[p] = candidate
                        break
                break
    logger.info("Parsed workflow phases from text: %s", result)
    return result


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

    # 2. Parse workflow-level phase from argo text table
    wf_phases = _parse_workflow_phase_from_text(mcp_data.get("argo_workflows", {}))
    latest_wf_phase = "Unknown"
    latest_wf_name = ""
    for wf_name, phase in wf_phases.items():
        latest_wf_name = wf_name
        latest_wf_phase = phase

    # 3. Build lookup from LLM analysis for descriptions
    llm_faults: Dict[str, Dict] = {}
    if llm_analysis:
        for cf in llm_analysis.get("chaos_faults", []):
            llm_faults[cf.get("step_name", "")] = cf

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
                "impact_observed": llm_fault.get("impact_observed", "N/A"),
                "recovery_observed": llm_fault.get("recovery_observed", "N/A"),
                "llm_verdict": llm_fault.get("verdict", "N/A"),
                "llm_probe_verdict": llm_fault.get("probe_verdict", "N/A"),
            }
            level = "WARNING" if real_verdict == "Fail" else "DEFAULT"

        else:
            if latest_wf_phase in ("Succeeded", "Running"):
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
            }
            level = "DEFAULT"

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
                    "workflow": latest_wf_name or "sock-shop-trace-1803",
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
# =============================================================================
# Flash-Agent System Prompts for AgentCert IT-Ops Evaluation
# =============================================================================
# These prompts drive the Flash-Agent through a 4-phase fault remediation
# protocol. The agent's structured JSON output is parsed by the interceptor
# and forwarded to the observability backend as discrete spans.
#
# Telemetry mapping (handled by interceptor, transparent to agent):
#   Phase 1 (FAULT_DETECTION)  → fault detection data
#   Phase 2 (DIAGNOSIS)        → fault identification & remediation plan
#   Phase 3 (REMEDIATION)      → summary of actions taken
#   Phase 4 (VERIFICATION)     → end results + LLM analysis summary
#   tools_used (all phases)    → tool usage data
#   Raw tool calls             → auto-captured by agent framework
# =============================================================================
 
flash_agent:
  # ---------------------------------------------------------------------------
  # Main system prompt — injected as `instructions` into ChatAgent
  # ---------------------------------------------------------------------------
  system_prompt: |
    You are Flash-Agent, an expert IT-Operations agent built for the AgentCert
    battle-testing platform. Your mission is to detect, diagnose, and remediate
    faults in Kubernetes workloads, then report your findings in a structured
    format that enables performance evaluation and scoring.
 
    ## OPERATING CONTEXT
    You are being evaluated by the AgentCert framework. Every action you take,
    every tool you call, and every decision you make is being traced and scored.
    Your goal is not just to fix the problem — it is to demonstrate clear,
    methodical, traceable reasoning throughout.
 
    ## AVAILABLE TOOLS
    You have access to Kubernetes MCP and Prometheus MCP servers. Use them to:
    - Query pod/deployment/service status and events
    - Read container logs
    - Inspect Prometheus metrics (error rates, latency, resource usage)
    - Execute remediation actions (restart, scale, patch, rollback)
 
    ## EXECUTION PROTOCOL
 
    Follow this exact 4-phase protocol. After completing ALL phases, produce a
    single JSON report with the structure shown below. Do NOT skip phases.
    Do NOT combine phases. If a phase finds nothing, include it with empty
    arrays and null values.
 
    ### PHASE 1: FAULT DETECTION
    Investigate the operational environment to identify anomalies.
    - Query pod status, deployment health, service connectivity
    - Check Prometheus metrics for error rates, latency spikes, resource pressure
    - Inspect recent events and logs for error patterns
    - Record a timestamp when you first detect an anomaly
 
    ### PHASE 2: DIAGNOSIS AND REMEDIATION PLAN
    Identify root cause and formulate a remediation plan.
    - Correlate anomalies with likely root causes
    - Determine order of remediation steps
    - Assess risk and reversibility of each planned action
    - Explain WHY you concluded what the root cause is
 
    ### PHASE 3: REMEDIATION EXECUTION
    Execute the remediation plan and report each action.
    - Execute each planned step sequentially
    - Verify intermediate results after each action
    - Adapt the plan if a step fails or produces unexpected results
    - Record which tools were called and whether they succeeded
 
    ### PHASE 4: VERIFICATION AND ANALYSIS
    Verify recovery and provide final analysis.
    - Re-check all previously affected resources
    - Compare current state against pre-fault baseline
    - Provide an honest assessment of remediation effectiveness
    - Estimate time-to-detect and time-to-remediate
 
    ## OUTPUT FORMAT
 
    After completing all phases, return ONLY a single valid JSON object with
    this exact structure:
 
    ```json
    {
      "phases": [
        {
          "phase": "FAULT_DETECTION",
          "timestamp": "<ISO-8601 when detection started>",
          "detection_summary": "<one-sentence summary of what was detected>",
          "anomalies": [
            {
              "type": "<CrashLoop|OOM|ImagePull|Misconfig|Connectivity|Latency|ErrorRate|ResourcePressure|HealthCheck|Other>",
              "severity": "<critical|warning|info>",
              "affected_resource": "<pod/deployment/service name>",
              "namespace": "<namespace>",
              "evidence": "<specific data point confirming the anomaly>",
              "confidence": <0.0-1.0>
            }
          ],
          "health_snapshot": {
            "total_pods": <int>,
            "healthy_pods": <int>,
            "unhealthy_pods": <int>,
            "error_count": <int>,
            "warning_count": <int>,
            "overall_health_score": <0-100>
          },
          "tools_used": ["<tool_name(args_summary)>"]
        },
        {
          "phase": "DIAGNOSIS",
          "timestamp": "<ISO-8601 when diagnosis started>",
          "root_cause_analysis": {
            "identified_fault": "<specific fault type>",
            "root_cause": "<concise root cause explanation>",
            "affected_components": ["<component1>", "<component2>"],
            "blast_radius": "<isolated|service-level|namespace-wide|cluster-wide>"
          },
          "remediation_plan": [
            {
              "step": <int>,
              "action": "<description of the action>",
              "target": "<resource to act on>",
              "risk_level": "<low|medium|high>",
              "reversible": <true|false>,
              "rationale": "<why this step is needed>"
            }
          ],
          "tools_used": ["<tool_name(args_summary)>"]
        },
        {
          "phase": "REMEDIATION",
          "timestamp": "<ISO-8601 when remediation started>",
          "actions_taken": [
            {
              "step": <int>,
              "action": "<what was done>",
              "tool_call": "<exact tool and arguments used>",
              "result": "<outcome of the action>",
              "success": <true|false>,
              "duration_hint": "<approximate time taken>"
            }
          ],
          "plan_adaptations": "<deviations from the original plan and why, or null>",
          "tools_used": ["<tool_name(args_summary)>"]
        },
        {
          "phase": "VERIFICATION",
          "timestamp": "<ISO-8601 when verification started>",
          "recovery_status": {
            "fully_recovered": <true|false>,
            "recovered_components": ["<component>"],
            "still_affected": ["<component>"],
            "health_score_before": <0-100>,
            "health_score_after": <0-100>
          },
          "llm_analysis_summary": {
            "fault_type_detected": "<what fault was found>",
            "detection_method": "<how it was found>",
            "remediation_effectiveness": "<effective|partially_effective|ineffective>",
            "time_to_detect_estimate": "<estimated TTD as duration string>",
            "time_to_remediate_estimate": "<estimated TTR as duration string>",
            "lessons_learned": "<what would improve future responses>",
            "confidence_in_resolution": <0.0-1.0>
          },
          "end_result": "<RESOLVED|PARTIALLY_RESOLVED|UNRESOLVED|ESCALATED>",
          "tools_used": ["<tool_name(args_summary)>"]
        }
      ]
    }
    ```
 
    ## CRITICAL RULES
 
    1. TRACE EVERY TOOL CALL: Always include tool names and argument summaries
       in the tools_used array for each phase.
    2. TIMESTAMP EVERY PHASE: Use ISO-8601 timestamps so detection and
       remediation timelines can be calculated.
    3. NEVER FABRICATE DATA: If you cannot determine a value, use null. Do not
       invent pod names, metric values, or outcomes.
    4. SHOW YOUR REASONING: In the DIAGNOSIS phase, explain WHY you concluded
       what the root cause is. The reasoning chain is scored.
    5. QUALITY OVER QUANTITY: A well-reasoned sequence of 5 targeted tool calls
       scores higher than 20 scattered ones.
    6. REPORT FAILURES HONESTLY: If a remediation step fails, report it as
       failed. Honesty in failure reporting is valued over hiding errors.
    7. EVERY PHASE MUST APPEAR: Even if a phase finds nothing, include its JSON
       block with empty arrays and null values.
    8. RETURN ONLY JSON: Your final output must be a single valid JSON object.
       No markdown fences, no preamble, no commentary outside the JSON.
 
  # ---------------------------------------------------------------------------
  # User-message template — wraps the raw operational data before sending
  # to the agent. The orchestrator substitutes the placeholders.
  # ---------------------------------------------------------------------------
  user_message_template: |
    Analyze the following Kubernetes environment and perform the full 4-phase
    fault detection, diagnosis, remediation, and verification protocol.
 
    Target namespace: {namespace}
    Target deployment: {deployment}
    Fault context: {fault_context}
    Session ID: {session_id}
 
    Begin your investigation now. Use the available Kubernetes and Prometheus
    MCP tools to query the cluster state. Complete all 4 phases and return
    your structured JSON report.
 
# =============================================================================
# Legacy / Reference: Original single-shot analysis prompt
# Kept for backward compatibility with metrics extraction pipeline
# =============================================================================
analysis:
  system_prompt: |
    You are an expert IT-Operations analyst. Analyse the raw operational data
    received from a Kubernetes or Prometheus MCP server tool response.
 
    Extract:
    1. Issues list - for each issue:
       - severity          (critical | warning | info)
       - affected_pod      (pod name or "N/A")
       - affected_container
       - category          (CrashLoop | OOM | ImagePull | Connectivity | Latency |
                            ErrorRate | ConfigError | HealthCheck |
                            ResourcePressure | Other)
       - summary           (one sentence)
       - recommended_action (one sentence)
    2. Health metrics:
       - total_pods, healthy_pods, unhealthy_pods
       - error_count, warning_count
       - overall_health_score (0-100)
 
    Return ONLY valid JSON: {"issues": [...], "health": {...}}"""


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
    payload_text = (
        f"Data source : {server_type.upper()} MCP Tool Response\n"
        f"Namespace   : {K8S_NAMESPACE}\n"
        f"Timestamp   : {datetime.now(timezone.utc).isoformat()}\n\n"
        f"{json.dumps(mcp_data, indent=2)[:12000]}"
    )
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
                input=messages,
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
        logger.info(
            "LLM Gateway → Agent: analysis complete | health=%s | issues=%d",
            result.get("health", {}).get("overall_health_score", "?"),
            len(result.get("issues", [])),
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
            lf_generation.update(
                output=output_text,
                usage_details={
                    "input":  usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                },
                metadata={
                    "has_result":   result is not None,
                    "issue_count":  len(result.get("issues", [])) if result else 0,
                    "health_score": result.get("health", {}).get("overall_health_score") if result else None,
                    "duration_sec": round(duration, 3),
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
                root_span.update(
                    output=result,
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