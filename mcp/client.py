"""
MCP Client – JSON-RPC 2.0 Streamable HTTP
============================================

Generic MCP protocol client with SSE response parsing.
No domain-specific logic — reusable by any agent implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger("flash-agent")


class MCPClient:
    """
    MCP JSON-RPC 2.0 client with SSE response parsing and session management.

    Usage::

        client = MCPClient("http://localhost:8086/mcp", "my-agent", timeout=30)
        client.initialize()
        result = client.call_tool("pods_list_in_namespace", {"namespace": "default"})
    """

    def __init__(self, url: str, agent_name: str, timeout: int = 30) -> None:
        self.url = url
        self.agent_name = agent_name
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._call_counter = 1

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    def initialize(self) -> Optional[str]:
        """Initialize MCP session and return session ID."""
        try:
            _, self._session_id = self._jsonrpc_call(
                method="initialize",
                params={
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": self.agent_name, "version": "3.0"},
                },
            )
            return self._session_id
        except Exception as exc:
            logger.warning("MCP session init failed for %s: %s", self.url, exc)
            return None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool and return the result dict."""
        self._call_counter += 1
        result, self._session_id = self._jsonrpc_call(
            method="tools/call",
            params={"name": tool_name, "arguments": arguments},
        )
        return result

    def _jsonrpc_call(
        self,
        method: str,
        params: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """
        Send a JSON-RPC 2.0 request to the MCP server and parse SSE response.
        Returns (result_dict, session_id).
        """
        parsed_url = urlparse(self.url)
        origin_host = (
            "localhost"
            if "host.docker.internal" in (parsed_url.hostname or "")
            else parsed_url.hostname
        )
        origin_port = f":{parsed_url.port}" if parsed_url.port else ""
        origin = f"{parsed_url.scheme}://{origin_host}{origin_port}"

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": f"{self.agent_name}/3.0",
            "Origin": origin,
            "Host": f"{origin_host}{origin_port}",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        body = {
            "jsonrpc": "2.0",
            "id": self._call_counter,
            "method": method,
            "params": params,
        }

        resp = requests.post(
            self.url, json=body, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()

        new_session_id = resp.headers.get("Mcp-Session-Id", self._session_id)

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


def generate_fallback_data(
    server_type: str, query: str, namespace: str
) -> Dict[str, Any]:
    """
    Generate synthetic data when MCP server is unreachable.

    Allows the agent to continue gracefully and provide LLM analysis
    even when underlying MCP infrastructure fails.
    """
    import datetime as _dt

    timestamp = _dt.datetime.utcnow().isoformat() + "Z"

    if server_type.lower() == "kubernetes":
        return {
            "status": "fallback",
            "reason": "MCP pod cannot reach Kubernetes API (DNS connectivity issue)",
            "data": {
                "cluster": namespace,
                "namespace": namespace,
                "timestamp": timestamp,
                "pods": [
                    {
                        "name": f"pod-{i}",
                        "namespace": namespace,
                        "status": "Unknown",
                        "phase": "Unknown",
                        "ready": "Unknown/Unknown",
                        "restarts": 0,
                        "reason": "MCP pod DNS connectivity issue",
                    }
                    for i in range(1, 4)
                ],
                "query_type": "operational_health",
                "query_original": query,
                "warnings": [
                    "MCP pod DNS failures - cannot resolve kubernetes.default.svc",
                    "Using synthetic data for LLM analysis",
                    "Actual cluster metrics unavailable",
                ],
                "recommendation": "Check MCP pod DNS configuration and Kubernetes cluster DNS service health",
            },
        }
    else:
        return {
            "status": "fallback",
            "reason": "MCP pod cannot reach Kubernetes API (DNS connectivity issue)",
            "data": {
                "cluster": namespace,
                "timestamp": timestamp,
                "metrics": {
                    "up": 0,
                    "node_memory_MemAvailable_bytes": 8589934592,  # 8GB placeholder
                    "node_cpu_seconds_total": 0,
                    "rate(container_cpu_usage_seconds_total[1m])": 0.05,
                    "container_memory_usage_bytes": 268435456,  # 256MB placeholder
                },
                "query_type": "system_metrics",
                "query_original": query,
                "warnings": [
                    "MCP pod DNS failures - Prometheus unavailable",
                    "Using synthetic metrics for LLM analysis",
                    "Actual system metrics unavailable",
                ],
            },
        }
