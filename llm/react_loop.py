"""
ReAct Loop — multi-turn tool-calling analysis (AIOpsLab pattern).

Mirrors microsoft/AIOpsLab's clients/gpt.py: the LLM iteratively chooses an
observation tool, sees the result, and decides the next action until it calls
``submit(...)`` with a structured verdict. Same JSON schema as the single-shot
analysis path so downstream code (certifier, persistence, notifier) is
unchanged.

Blind-observer rule: every tool result passes through ``_sanitize_leakage_terms``
before being appended to the message history, identical to the single-shot path.

Cost: 5-10× tokens vs single-shot. Opt-in via ``AGENT_REASONING_MODE=react``.
Budget control: ``AGENT_REACT_MAX_STEPS`` caps tool-call iterations.

Reference: microsoft/AIOpsLab clients/gpt.py + orchestrator/actions/base.py
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import AgentConfig
from mcp.client import MCPClient
from mcp.parsers import extract_mcp_text
from mcp.log_dedup import greedy_compress_lines

logger = logging.getLogger("flash-agent")


# ──────────────────────────────────────────────────────────────────────────────
# Tool schema (OpenAI function-calling format)
# ──────────────────────────────────────────────────────────────────────────────


def _build_tools_schema() -> List[Dict[str, Any]]:
    """OpenAI tool definitions exposed to the LLM in the ReAct loop."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_pods",
                "description": "List pods in the watched namespace with phase, ready, restart count, and age.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_events",
                "description": "List Kubernetes events in the watched namespace.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_pod_logs",
                "description": "Fetch the last N log lines from a specific pod.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pod_name": {"type": "string", "description": "Exact pod name from get_pods output."},
                        "lines": {"type": "integer", "description": "Number of log lines to return (default 200, max 500)."},
                    },
                    "required": ["pod_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "prom_query",
                "description": "Run a PromQL instant query against the cluster Prometheus.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "PromQL expression."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": "Terminal action: submit the final structured analysis verdict and end the loop.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "object",
                            "description": "Full analysis JSON (environment_state, identified_issues, system_errors, health).",
                        },
                    },
                    "required": ["verdict"],
                },
            },
        },
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Tool implementations (thin wrappers around MCPClient)
# ──────────────────────────────────────────────────────────────────────────────


class _ToolBox:
    """Stateful tool dispatcher. Keeps MCP sessions warm across turns."""

    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self._k8s: Optional[MCPClient] = None
        self._prom: Optional[MCPClient] = None

    def _k8s_client(self) -> MCPClient:
        if self._k8s is None:
            self._k8s = MCPClient(self.cfg.k8s_mcp_url, self.cfg.agent_name, self.cfg.mcp_timeout)
            self._k8s.initialize()
        return self._k8s

    def _prom_client(self) -> MCPClient:
        if self._prom is None:
            self._prom = MCPClient(self.cfg.prom_mcp_url, self.cfg.agent_name, self.cfg.mcp_timeout)
            self._prom.initialize()
        return self._prom

    def get_pods(self) -> str:
        try:
            r = self._k8s_client().call_tool(
                "pods_list_in_namespace",
                {"namespace": self.cfg.k8s_namespace},
            )
            return greedy_compress_lines(extract_mcp_text(r) or "")
        except Exception as exc:
            return f"[error] get_pods failed: {exc}"

    def get_events(self) -> str:
        try:
            r = self._k8s_client().call_tool(
                "events_list",
                {"namespace": self.cfg.k8s_namespace},
            )
            return greedy_compress_lines(extract_mcp_text(r) or "")
        except Exception as exc:
            return f"[error] get_events failed: {exc}"

    def get_pod_logs(self, pod_name: str, lines: int = 200) -> str:
        try:
            r = self._k8s_client().call_tool(
                "pods_log",
                {
                    "namespace": self.cfg.k8s_namespace,
                    "name": pod_name,
                },
            )
            text = greedy_compress_lines(extract_mcp_text(r) or "")
            # Tail to requested line count to keep history bounded
            try:
                n = max(10, min(int(lines or 200), 500))
            except (TypeError, ValueError):
                n = 200
            tail_lines = text.split("\n")[-n:]
            return "\n".join(tail_lines)
        except Exception as exc:
            return f"[error] get_pod_logs failed: {exc}"

    def prom_query(self, query: str) -> str:
        try:
            r = self._prom_client().call_tool("execute_query", {"query": query})
            return extract_mcp_text(r) or ""
        except Exception as exc:
            return f"[error] prom_query failed: {exc}"


# ──────────────────────────────────────────────────────────────────────────────
# Token-aware history trim (port of AIOpsLab clients/gpt.py)
# ──────────────────────────────────────────────────────────────────────────────


def _approx_tokens(text: str) -> int:
    # Rough heuristic: 4 chars per token. Avoids tiktoken dependency in the
    # base image. Good enough for budget caps.
    return max(1, len(text) // 4)


def _trim_history(messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    """Always keep the system message + the most recent messages within budget."""
    if not messages:
        return messages
    system = [m for m in messages[:1] if m.get("role") == "system"]
    rest = messages[len(system):]
    out: List[Dict[str, Any]] = list(system)
    total = sum(_approx_tokens(json.dumps(m)) for m in out)
    keep: List[Dict[str, Any]] = []
    for m in reversed(rest):
        t = _approx_tokens(json.dumps(m))
        if total + t > max_tokens:
            break
        keep.insert(0, m)
        total += t
    return out + keep


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────


_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_react_prompt(max_steps: int) -> str:
    template = (_PROMPT_DIR / "react_system.txt").read_text(encoding="utf-8")
    return template.replace("{max_steps}", str(max_steps))


def request_react_analysis(
    cfg: AgentConfig,
    scan_id: str,
    agent_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Run the ReAct multi-turn loop and return the submitted verdict.

    Returns the verdict dict on submit(), None on hard failure. If the loop
    exhausts ``react_max_steps`` without submitting, returns a degraded-but-
    valid verdict with status "Unknown" so the certifier still gets a trace.
    """
    # Lazy import to avoid circular dependency with gateway.py
    from llm.gateway import (
        _chat_create,
        create_openai_client,
        _sanitize_leakage_terms,
    )

    max_steps = max(1, int(cfg.react_max_steps or 8))
    system_prompt = _load_react_prompt(max_steps)

    # Initial user turn — same blind-observer framing as single-shot
    user_seed = (
        f"Namespace under observation: {cfg.k8s_namespace}\n"
        f"Scan ID: {scan_id}\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Begin investigation. Call ONE tool per turn. Submit when done."
    )
    if agent_context:
        det = agent_context.get("detection_gate")
        if det:
            user_seed += (
                f"\n\nA prior fast detection pass classified this snapshot as "
                f"anomaly_detected={det}"
                + (f" | evidence={agent_context.get('detection_reason')!r}" if agent_context.get("detection_reason") else "")
                + ". Investigate accordingly."
            )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_seed},
    ]

    toolbox = _ToolBox(cfg)
    tools_schema = _build_tools_schema()
    client = create_openai_client(cfg)

    final_verdict: Optional[Dict[str, Any]] = None
    base_metadata = {
        "trace_id": cfg.notify_id or scan_id,
        "session_id": scan_id,
        "scan_id": scan_id,
        "step": "react-loop",
        "namespace": cfg.k8s_namespace,
        "agent": cfg.agent_name,
        **({"agent_id": cfg.agent_id} if cfg.agent_id else {}),
    }

    for step in range(1, max_steps + 1):
        messages = _trim_history(messages, max_tokens=12000)

        try:
            resp = _chat_create(
                client,
                cfg.model_alias,
                4096,
                model=cfg.model_alias,
                messages=messages,
                tools=tools_schema,
                tool_choice="required",
                temperature=0.1,
                extra_body={
                    "metadata": {
                        **base_metadata,
                        "generation_name": f"react_step_{step}",
                        "generation_id": str(uuid.uuid4()),
                    }
                },
            )
        except Exception as exc:
            logger.error("ReAct step %d LLM call failed: %s", step, exc)
            break

        choice = resp.choices[0].message
        tool_calls = getattr(choice, "tool_calls", None) or []

        if not tool_calls:
            # Model returned text instead of a tool call — nudge once, then bail
            content = (choice.content or "").strip()
            logger.warning("ReAct step %d: no tool_call returned (content=%s)", step, content[:200])
            messages.append({"role": "assistant", "content": content or "[no content]"})
            messages.append({
                "role": "user",
                "content": "You must call exactly one tool. Do not narrate. Call a tool now.",
            })
            continue

        # Append assistant tool-call message verbatim so OpenAI sees the conversation
        messages.append({
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            logger.info("ReAct step %d → tool=%s args_keys=%s", step, name, list(args.keys()))

            if name == "submit":
                final_verdict = args.get("verdict") or {}
                # Sanitize the final verdict the same way as single-shot output
                # (defensive — verdict shouldn't contain fault-identity terms)
                serialised = json.dumps(final_verdict)
                serialised, leaked = _sanitize_leakage_terms(serialised)
                if leaked:
                    logger.warning("ReAct submit leaked %d term(s): %s", len(leaked), sorted(set(leaked)))
                    try:
                        final_verdict = json.loads(serialised)
                    except json.JSONDecodeError:
                        pass
                # Append a tool-response message so trace is well-formed
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": name,
                    "content": "submitted",
                })
                break

            # Dispatch observation tools
            if name == "get_pods":
                tool_out = toolbox.get_pods()
            elif name == "get_events":
                tool_out = toolbox.get_events()
            elif name == "get_pod_logs":
                tool_out = toolbox.get_pod_logs(
                    pod_name=str(args.get("pod_name", "")),
                    lines=int(args.get("lines", 200)),
                )
            elif name == "prom_query":
                tool_out = toolbox.prom_query(query=str(args.get("query", "")))
            else:
                tool_out = f"[error] unknown tool: {name}"

            # Cap each tool result and sanitise before appending to history
            tool_out = (tool_out or "")[:8000]
            tool_out, leaked = _sanitize_leakage_terms(tool_out)
            if leaked:
                logger.info("ReAct tool=%s redacted %d term(s)", name, len(leaked))

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": tool_out,
            })

        if final_verdict is not None:
            logger.info(
                "ReAct loop complete at step %d (submit) | issues=%d health=%s",
                step,
                len(final_verdict.get("identified_issues", []) or []),
                (final_verdict.get("health") or {}).get("overall_health_score"),
            )
            return final_verdict

    # Loop exhausted without submit() — return a degraded placeholder verdict
    # so the certifier still receives a trace (never silently drop a scan).
    logger.warning("ReAct loop exhausted %d steps without submit(); returning placeholder", max_steps)
    return {
        "environment_state": {
            "infrastructure_identifier": cfg.k8s_namespace,
            "health_status": "Unknown",
            "total_instances_running": 0,
            "instances_with_errors": 0,
            "unhealthy_instances": 0,
            "timestamp_analyzed": datetime.now(timezone.utc).isoformat(),
        },
        "identified_issues": [],
        "system_errors": [],
        "health": {
            "total_instances": 0, "healthy_instances": 0, "unhealthy_instances": 0,
            "error_count": 0, "warning_count": 0,
            "cpu_utilization_percent": None, "memory_utilization_percent": None,
            "overall_health_score": 50,
        },
        "react_meta": {"reason": "max_steps_exhausted", "max_steps": max_steps},
    }
