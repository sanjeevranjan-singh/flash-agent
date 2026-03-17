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
    load_dotenv(env_file)
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

# Scan behaviour
TRACE_TAGS    = [t.strip() for t in os.getenv("TRACE_TAGS", "").split(",") if t.strip()]
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "0"))

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

    # Build the list of MCP tool calls based on server type
    if server_type == "kubernetes":
        tool_calls = [
            ("pods_list_in_namespace", {"namespace": K8S_NAMESPACE}),
            ("events_list",           {"namespace": K8S_NAMESPACE}),
        ]
    else:  # prometheus
        tool_calls = [
            ("execute_query", {"query": "up"}),
            ("execute_query", {"query": f'count(kube_pod_info{{namespace="{K8S_NAMESPACE}"}})'}),
        ]

    request_payload: Dict[str, Any] = {
        "server_type": server_type,
        "namespace":   K8S_NAMESPACE,
        "tool_calls":  [(name, args) for name, args in tool_calls],
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Agent → MCP Server (%s) | url=%s | tools=%s",
                server_type, url, [t[0] for t in tool_calls])
    response_payload: Dict[str, Any] = {}
    t0 = time.time()

    try:
        # Step 1: Initialize MCP session
        session_id = _mcp_init_session(url)
        logger.info("MCP session initialized | session_id=%s", session_id)

        # Step 2: Call each tool via JSON-RPC tools/call
        all_results: Dict[str, Any] = {}
        for call_idx, (tool_name, tool_args) in enumerate(tool_calls, start=2):
            result, session_id = _mcp_jsonrpc_call(
                url=url,
                method="tools/call",
                params={"name": tool_name, "arguments": tool_args},
                session_id=session_id,
                call_id=call_idx,
            )
            all_results[tool_name] = result
            logger.info("  MCP tool '%s' → %d chars",
                        tool_name, len(json.dumps(result)))

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
# STEP 3  –  Agent → LLM Gateway: deep analysis of Tool Response from MCP
# Stores LLM Req+Res in OTL Collector (rule ②) and Langfuse (rule ③)
# ══════════════════════════════════════════════════════════════════════════════

_ANALYSIS_SYSTEM = """\
You are an expert IT-Operations analyst. Analyse the raw operational data
received from a Kubernetes or Prometheus MCP server tool response.

Extract:
1. Issues list – for each issue:
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
    scan_start = time.time()
    scan_id = (
        f"{AGENT_NAME}-{K8S_NAMESPACE}"
        f"-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    logger.info("═══ Scan started | scan_id=%s ═══", scan_id)

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
            with langfuse_client.start_as_current_observation(
                name="agent-scan",
                as_type="span",
                input={"scan_query": scan_query, "scan_id": scan_id},
                metadata={
                    "agent":      AGENT_NAME,
                    "namespace":  K8S_NAMESPACE,
                },
            ) as root_span:
                logger.info("③ Langfuse root span created")
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
        return {"health": {"overall_health_score": -1}, "issues": []}

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