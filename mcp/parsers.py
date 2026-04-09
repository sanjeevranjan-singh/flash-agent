"""
MCP Parsers – MCP response text parsing and summarisation
==========================================================

Generic functions that transform raw MCP tool responses into
structured data for agent processing and LLM prompt construction.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

import logging

logger = logging.getLogger("flash-agent")


def extract_mcp_text(mcp_result: Any) -> str:
    """Extract text content from an MCP tool result dict."""
    if isinstance(mcp_result, dict):
        content = mcp_result.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
    return str(mcp_result) if mcp_result else ""


def extract_active_pod_names(
    pod_list_result: Dict[str, Any],
    namespace: str,
) -> List[str]:
    """
    Parse the pods_list_in_namespace MCP result and return names of
    Running or Error pods belonging to chaos workflow runs.
    Used for Phase 2 targeted pod-log fetching.
    """
    target_app_pattern = os.getenv("TARGET_APP_POD_PATTERN", "sock-shop")
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
            pod_name = parts[3]  # NAME column
            status = parts[5]  # STATUS column
            # Target running or error pods from chaos / Argo workflows
            if status in ("Running", "Error") and (
                target_app_pattern in pod_name
                or "argowf-chaos" in pod_name
                or "chaos" in pod_name.lower()
            ):
                pod_names.append(pod_name)
    except Exception as exc:
        logger.warning("Failed to extract pod names for log fetching: %s", exc)
    return pod_names


def build_mcp_data_summary(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact structured summary of the MCP response data.

    Used by the LLM payload builder to extract pod status, events,
    workflow metadata and chaos results for the LLM analysis prompt.
    """
    if "error" in response_payload:
        return {
            "status": "error",
            "error": str(response_payload.get("error", ""))[:200],
        }

    mcp_data = response_payload.get("data", {})
    summary: Dict[str, Any] = {"status": "ok"}

    # ── pods_list_in_namespace → pod status breakdown ────────────────────────
    pods_text = extract_mcp_text(mcp_data.get("pods_list_in_namespace", {}))
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
                    restarts = (
                        int(parts[6].split("(")[0]) if len(parts) > 6 else 0
                    )
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
    events_text = extract_mcp_text(mcp_data.get("events_list", {}))
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
                    for keyword in (
                        "BackOff",
                        "Failed",
                        "FailedScheduling",
                        "Unhealthy",
                        "OOMKilling",
                        "Evicted",
                        "FailedMount",
                        "FailedCreate",
                    ):
                        if keyword in line:
                            warning_reasons.append(keyword)
                            break
                elif "Normal" in line:
                    normal_count += 1
            summary["events"] = {
                "normal": normal_count,
                "warning": warning_count,
            }
            if warning_reasons:
                summary["events"]["warning_reasons"] = warning_reasons[:5]

    # ── pods_top → resource usage or error ───────────────────────────────────
    pods_top_data = mcp_data.get("pods_top", {})
    pods_top_text = extract_mcp_text(pods_top_data)
    if isinstance(pods_top_data, dict) and "error" in pods_top_data:
        summary["pods_top"] = {"error": str(pods_top_data["error"])[:100]}
    elif pods_top_text and "error" in pods_top_text.lower():
        summary["pods_top"] = {"error": pods_top_text[:100]}
    elif pods_top_text:
        summary["pods_top"] = {
            "available": True,
            "lines": len(pods_top_text.strip().split("\n")),
        }

    # ── chaosengines → engine names ──────────────────────────────────────────
    engines_text = extract_mcp_text(mcp_data.get("chaosengines", {}))
    if engines_text and engines_text.strip():
        engine_names: List[str] = []
        for line in engines_text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                engine_names.append(parts[3])
        summary["chaosengines"] = {
            "count": len(engine_names),
            "engines": engine_names[:10],
        }
    else:
        summary["chaosengines"] = {"count": 0}

    # ── chaosresults → result names ──────────────────────────────────────────
    results_text = extract_mcp_text(mcp_data.get("chaosresults", {}))
    if results_text and results_text.strip():
        result_names: List[str] = []
        for line in results_text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                result_names.append(parts[3])
        summary["chaosresults"] = {
            "count": len(result_names),
            "results": result_names[:10],
        }
    else:
        summary["chaosresults"] = {"count": 0}

    # ── argo_workflows → workflow name, phase, age ───────────────────────────
    argo_text = extract_mcp_text(mcp_data.get("argo_workflows", {}))
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
                    if p in (
                        "Succeeded",
                        "Failed",
                        "Running",
                        "Error",
                        "Pending",
                    ):
                        wf_phase = p
                    if (
                        p
                        and p[-1] in ("d", "h", "m", "s")
                        and p[:-1].replace(".", "").isdigit()
                    ):
                        wf_age = p
                workflows.append(
                    {"name": wf_name, "phase": wf_phase, "age": wf_age}
                )

        # Identify experiment workflows (names ending with 10+ digit epoch suffix)
        def _wf_epoch(wf: Dict[str, str]) -> int:
            try:
                suffix = wf["name"].rsplit("-", 1)[1]
                return int(suffix) if len(suffix) >= 10 else 0
            except (ValueError, IndexError):
                return 0

        experiment_wfs = sorted(
            [w for w in workflows if _wf_epoch(w) > 0],
            key=_wf_epoch,
            reverse=True,
        )
        other_wfs = [w for w in workflows if _wf_epoch(w) == 0]
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
            log_text = extract_mcp_text(log_data)
            pod_info: Dict[str, Any] = {"size": f"{len(log_text)} chars"}

            if "chaos-exporter" in pod_name:
                faults_found = set(re.findall(r"FaultName=(\S+)", log_text))
                verdicts_found = re.findall(
                    r"FaultName=(\S+).*?ResultVerdict=(\S+)", log_text
                )
                latest: Dict[str, str] = {}
                for f, v in verdicts_found:
                    latest[f] = v
                pod_info["faults_reporting"] = sorted(faults_found)
                pod_info["latest_verdicts"] = latest
            elif "chaos-operator" in pod_name:
                pod_info["reconcile_events"] = log_text.count(
                    "Reconciling ChaosEngine"
                )
                pod_info["errors"] = log_text.count(
                    "level=error"
                ) + log_text.count('"level":"error"')
            else:
                error_lines = [
                    l.strip()
                    for l in log_text.split("\n")
                    if "error" in l.lower()
                ]
                if error_lines:
                    pod_info["errors"] = len(error_lines)
                    pod_info["last_error"] = error_lines[-1][:150]

            log_summary[pod_name] = pod_info
        summary["pods_log"] = log_summary

    return summary
