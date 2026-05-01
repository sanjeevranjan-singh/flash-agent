"""
MCP Parsers – MCP response text parsing and summarisation
==========================================================

Generic functions that transform raw MCP tool responses into
structured data for agent processing and LLM prompt construction.
"""

from __future__ import annotations

import json
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
    target_app_name: str = "",
) -> List[str]:
    """
    Parse the pods_list_in_namespace MCP result and return names of pods
    to fetch logs for in Phase 2.

    Priority 1 (always): any pod in a non-Running/non-Completed state
    (Error, CrashLoopBackOff, OOMKilled, Terminating, Pending, etc.)
    Priority 2 (always): any Running pod with restarts > 0
    Priority 3: chaos/argo workflow pods + pods matching target_app_name
    Priority 4 (topology): any remaining pods in the watched namespace, in the
    order they appear in the pod list, so the agent always has SOME log evidence
    to reason from even when no pod is visibly degraded.  Industry SREs sample
    logs from every workload during incident triage; flash-agent must too,
    otherwise external faults (cpu-hog, network-loss) that do not restart the
    pod produce zero log signal and the LLM cannot detect them.

    target_app_name is passed from cfg.target_app_name (set via TARGET_APP_NAME
    env var, defaults to the watched namespace). This ensures the function works
    for any application under test, not just sock-shop.
    """
    # Fall back to namespace as identifier when no explicit app name given
    if not target_app_name:
        target_app_name = namespace
    unhealthy: List[str] = []
    restarted: List[str] = []
    workflow: List[str] = []
    topology: List[str] = []

    HEALTHY_STATES = {"Running", "Completed", "Succeeded"}

    try:
        content = pod_list_result.get("content", [])
        if not content:
            return []
        text = content[0].get("text", "")
        for line in text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            pod_name = parts[3]   # NAME column
            status   = parts[5]   # STATUS column
            restarts_raw = parts[6] if len(parts) > 6 else "0"
            try:
                restarts = int(restarts_raw.split("/")[0])
            except (ValueError, IndexError):
                restarts = 0

            is_chaos = (
                "argowf-chaos" in pod_name
                or "chaos" in pod_name.lower()
                or target_app_name in pod_name
            )

            if status not in HEALTHY_STATES:
                unhealthy.append(pod_name)
            elif restarts > 0:
                restarted.append(pod_name)
            elif is_chaos:
                workflow.append(pod_name)
            else:
                # Healthy app pod – still sample its logs so the agent
                # has visibility into the workload during chaos that
                # never crashes the pod (cpu-hog, network-loss, …).
                topology.append(pod_name)

    except Exception as exc:
        logger.warning("Failed to extract pod names for log fetching: %s", exc)
        return []

    # Combine with priority order, deduplicate, limit to 8 total
    seen: set = set()
    result: List[str] = []
    for pod in unhealthy + restarted + workflow + topology:
        if pod not in seen:
            seen.add(pod)
            result.append(pod)
        if len(result) >= 8:
            break
    return result


def split_event_blocks(events_text: str) -> List[List[str]]:
    """
    Split MCP events_list output into one logical block per event.

    Handles three shapes returned by kubernetes-mcp-server:
      (A) YAML-list shape (most common): each event is a list item that begins
          with a line starting with "- " at column 0; subsequent fields are
          indented (e.g. "  Reason: Unhealthy"). NO blank lines between items.
      (B) Blank-line separated YAML-ish blocks.
      (C) kubectl-style table: one line per event with TYPE/REASON/OBJECT/MSG.

    Previously, only (B) and (C) were handled. The live cluster returns (A),
    so the splitter saw every "Reason:", "Type:", "Message:" line as its OWN
    one-line "block" and the warning detection only matched isolated
    "Type: Warning" lines while their Reason/Message lived in different blocks.
    """
    raw = (events_text or "").strip()
    if not raw:
        return []

    lines = raw.splitlines()

    # Detect YAML-list shape: any line starts with "- " at col 0 (and is not the
    # leading "# ..." comment that some MCP servers prepend).
    yaml_list_idx = [
        i for i, l in enumerate(lines) if l.startswith("- ")
    ]
    if yaml_list_idx:
        blocks: List[List[str]] = []
        current: List[str] = []
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("- "):
                if current:
                    blocks.append(current)
                current = [line]
            else:
                if current:
                    current.append(line)
                # else: stray pre-amble, skip
        if current:
            blocks.append(current)
        return blocks

    # Blank-line separated blocks
    if "\n\n" in raw:
        blocks = []
        for chunk in raw.split("\n\n"):
            block = [l for l in chunk.splitlines() if l.strip()]
            if block:
                blocks.append(block)
        return blocks

    # Table shape: one line per event
    blocks = []
    for line in lines:
        if not line.strip() or line.startswith("NAMESPACE") or line.lstrip().startswith("#"):
            continue
        blocks.append([line])
    return blocks


def build_mcp_data_summary(response_payload: Dict[str, Any], include_chaos: bool = True) -> Dict[str, Any]:
    """
    Build a compact structured summary of the MCP response data.

    Used by the LLM payload builder to extract pod status, events,
    workflow metadata and chaos results for the LLM analysis prompt.

    When include_chaos is False, chaos-specific sections (ChaosEngines,
    ChaosResults, Argo Workflows) are omitted to prevent fault identity
    leakage in blind-observer mode.
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
        # Topology: group pods by deployment (derived from `name=` label, or
        # fallback to stripping the trailing -<rs-hash>-<pod-hash> from name).
        # AIOpsLab provides a static `app_summary` to its agents; we cannot
        # rely on a static description (any app may be deployed) so we derive
        # topology from observed pods. Strictly observation-only: same data
        # an SRE would see from `kubectl get pods -o wide --show-labels`.
        topology: Dict[str, Dict[str, Any]] = {}

        _DEPLOY_HASH_RE = re.compile(r"-[0-9a-f]{6,10}-[0-9a-z]{5}$")

        def _deployment_of(pod_name: str, labels_blob: str) -> str:
            # Prefer `name=<X>` label (sock-shop convention); fallback to
            # `app.kubernetes.io/name=<X>`; final fallback is name regex strip.
            for key in ("name=", "app.kubernetes.io/name=", "app="):
                if key in labels_blob:
                    val = labels_blob.split(key, 1)[1].split(",", 1)[0].strip()
                    if val and val != "flash-agent":
                        return val
                    if val == "flash-agent":
                        return val
            stripped = _DEPLOY_HASH_RE.sub("", pod_name)
            return stripped or pod_name

        for line in pods_text.strip().split("\n"):
            if line.startswith("NAMESPACE") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            pod_name = parts[3] if len(parts) > 3 else ""
            status = parts[5] if len(parts) > 5 else "Unknown"
            status_counts[status] = status_counts.get(status, 0) + 1
            try:
                restarts = int(parts[6].split("(")[0]) if len(parts) > 6 else 0
            except (ValueError, IndexError):
                restarts = 0
            total_restarts += restarts
            if restarts > 0:
                high_restart_pods.append(f"{pod_name} ({restarts})")
            age = parts[7] if len(parts) > 7 else "?"
            labels = parts[-1] if len(parts) > 8 else ""
            deploy = _deployment_of(pod_name, labels)
            d = topology.setdefault(deploy, {
                "pods": 0, "running": 0, "ready": 0,
                "restarts": 0, "ages": [], "statuses": {},
            })
            d["pods"] += 1
            if status == "Running":
                d["running"] += 1
            d["restarts"] += restarts
            d["ages"].append(age)
            d["statuses"][status] = d["statuses"].get(status, 0) + 1
            ready_field = parts[4] if len(parts) > 4 else "0/0"
            if "/" in ready_field:
                try:
                    r_now, r_total = ready_field.split("/")
                    if r_now == r_total and int(r_total) > 0:
                        d["ready"] += 1
                except ValueError:
                    pass

        summary["pods"] = {
            "total": sum(status_counts.values()),
            "by_status": status_counts,
            "total_restarts": total_restarts,
        }
        if high_restart_pods:
            summary["pods"]["restarted_pods"] = high_restart_pods[:5]
        if topology:
            summary["topology"] = topology

    # ── events_list → event type breakdown ───────────────────────────────────
    # The MCP server can return events in two shapes:
    #   (A) kubectl-style table: one line per event with TYPE/REASON/OBJECT/MSG
    #   (B) YAML-ish blocks: multi-line records separated by a blank line, each
    #       line in the form "Field: value" (Type, Reason, Object, Message…).
    # The previous version only handled (A) – it filtered single lines that
    # contained "Warning" and matched a hardcoded keyword allow-list.  Under
    # shape (B) every real reason (Unhealthy, BackOff, FailedScheduling, …) is
    # on a SEPARATE line from "Type: Warning", so the filter discarded all body
    # information and the LLM saw only "Type: Warning" repeated.  This block
    # parser handles both shapes by walking event records.
    events_text = extract_mcp_text(mcp_data.get("events_list", {}))
    if events_text:
        if "No events found" in events_text or not events_text.strip():
            summary["events"] = {"total": 0, "note": "No events found"}
        else:
            normal_count = 0
            warning_count = 0
            warning_reasons: List[str] = []

            blocks = split_event_blocks(events_text)

            for block in blocks:
                joined = "\n".join(block)
                # Classify Warning vs Normal at block scope (handles all shapes)
                is_warning = bool(re.search(r"\bType\s*:\s*Warning\b", joined)) \
                             or "Warning" in block[0]
                is_normal  = bool(re.search(r"\bType\s*:\s*Normal\b", joined)) \
                             or "Normal" in block[0]

                if is_warning:
                    warning_count += 1
                    # Extract Reason (YAML-ish "Reason: X") OR pull a known
                    # reason keyword from the table line.  We do NOT redact
                    # the reason – it is operational signal, not fault identity.
                    reason_match = re.search(r"\bReason\s*:\s*([A-Za-z0-9_-]+)", joined)
                    if reason_match:
                        warning_reasons.append(reason_match.group(1))
                    else:
                        for kw in (
                            "BackOff", "Failed", "FailedScheduling",
                            "Unhealthy", "OOMKilling", "Evicted",
                            "FailedMount", "FailedCreate", "ProbeWarning",
                            "ProbeFailed", "NodeNotReady", "Killing",
                        ):
                            if kw in joined:
                                warning_reasons.append(kw)
                                break
                elif is_normal:
                    normal_count += 1

            summary["events"] = {
                "normal": normal_count,
                "warning": warning_count,
            }
            if warning_reasons:
                # Preserve order, dedup, cap at 10 distinct reasons
                seen_r: set = set()
                deduped = []
                for r in warning_reasons:
                    if r not in seen_r:
                        seen_r.add(r)
                        deduped.append(r)
                summary["events"]["warning_reasons"] = deduped[:10]

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
    if include_chaos:
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
    if include_chaos:
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
    if not include_chaos:
        summary["argo_workflows"] = {"count": 0}
    argo_text = extract_mcp_text(mcp_data.get("argo_workflows", {})) if include_chaos else ""
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

    # ── prometheus → CPU/memory/restart top pods ─────────────────────────────
    prom_data = mcp_data.get("prometheus", {})
    if isinstance(prom_data, dict) and prom_data:
        prom_summary = _summarize_prometheus(prom_data)
        if prom_summary:
            summary["prometheus"] = prom_summary

    return summary


def _parse_prom_instant(raw: Any) -> List[Dict[str, Any]]:
    """Parse a Prometheus instant-query MCP response into [{labels, value}]."""
    if raw is None:
        return []
    if isinstance(raw, dict) and set(raw.keys()) == {"error"}:
        return []

    payload: Any = None
    # Case 1: MCP-wrapped {"content": [{"type":"text","text":"<json>"}]}
    if isinstance(raw, dict) and "content" in raw:
        text = extract_mcp_text(raw)
        if text:
            try:
                payload = json.loads(text)
            except (ValueError, TypeError):
                return []
    # Case 2: already a parsed dict (e.g. {"data":{"result":[...]}})
    elif isinstance(raw, dict):
        payload = raw
    # Case 3: a JSON string
    elif isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return []
    else:
        return []

    # Unwrap common shapes: {"data":{"result":[...]}} or {"result":[...]}
    result = None
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], dict):
            result = payload["data"].get("result")
        elif "result" in payload:
            result = payload.get("result")
    if not isinstance(result, list):
        return []

    out: List[Dict[str, Any]] = []
    for entry in result:
        if not isinstance(entry, dict):
            continue
        metric = entry.get("metric", {}) or {}
        value = entry.get("value")
        try:
            num = float(value[1]) if isinstance(value, (list, tuple)) and len(value) >= 2 else None
        except (ValueError, TypeError):
            num = None
        if num is None:
            continue
        out.append({"labels": metric, "value": num})
    return out


def _top_pods(rows: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    rows_sorted = sorted(rows, key=lambda r: r["value"], reverse=True)
    out: List[Dict[str, Any]] = []
    for r in rows_sorted[:limit]:
        pod = r["labels"].get("pod") or r["labels"].get("instance") or "unknown"
        out.append({"pod": pod, "value": round(r["value"], 4)})
    return out


def _summarize_prometheus(prom_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a compact summary of Prometheus snapshot keyed by result_key.

    Computes per-pod saturation = current_usage / declared_limit when limits
    are defined on the workload. This makes anomaly detection domain-agnostic:
    we don't need to know whether the workload is sock-shop, mongo, or kafka -
    every pod is judged against its own declared resource envelope.
    """
    summary: Dict[str, Any] = {}

    up_rows = _parse_prom_instant(prom_data.get("prometheus_up"))
    if up_rows:
        summary["prometheus_up"] = sum(1 for r in up_rows if r["value"] >= 1.0)

    pod_count_rows = _parse_prom_instant(prom_data.get("pod_count"))
    if pod_count_rows:
        summary["pod_count"] = int(pod_count_rows[0]["value"])

    # ── per-pod limits (used for saturation calc) ─────────────────────────
    cpu_limits: Dict[str, float] = {}
    for r in _parse_prom_instant(prom_data.get("cpu_limit_per_pod")):
        pod = r["labels"].get("pod")
        if pod and r["value"] > 0:
            cpu_limits[pod] = r["value"]

    mem_limits: Dict[str, float] = {}
    for r in _parse_prom_instant(prom_data.get("memory_limit_per_pod")):
        pod = r["labels"].get("pod")
        if pod and r["value"] > 0:
            mem_limits[pod] = r["value"]

    # ── CPU usage + saturation ────────────────────────────────────────────
    cpu_rows = _parse_prom_instant(prom_data.get("cpu_per_pod"))
    if cpu_rows:
        enriched = []
        for r in cpu_rows:
            pod = r["labels"].get("pod", "")
            limit = cpu_limits.get(pod)
            sat = (r["value"] / limit) if limit else None
            enriched.append({
                "labels": r["labels"],
                "value": r["value"],
                "limit": limit,
                "saturation": sat,
            })
        summary["cpu_per_pod_top"] = _top_pods_with_sat(enriched)

    # ── memory usage + saturation ─────────────────────────────────────────
    mem_rows = _parse_prom_instant(prom_data.get("memory_per_pod"))
    if mem_rows:
        enriched = []
        for r in mem_rows:
            pod = r["labels"].get("pod", "")
            limit = mem_limits.get(pod)
            sat = (r["value"] / limit) if limit else None
            enriched.append({
                "labels": r["labels"],
                "value": r["value"] / (1024 * 1024),  # MB
                "limit": (limit / (1024 * 1024)) if limit else None,
                "saturation": sat,
            })
        summary["memory_per_pod_top_mb"] = _top_pods_with_sat(enriched)

    restart_rows = _parse_prom_instant(prom_data.get("restarts_per_pod"))
    if restart_rows:
        summary["restarts_per_pod_top"] = _top_pods(restart_rows)
        summary["restarting_pods"] = [
            p for p in summary["restarts_per_pod_top"] if p["value"] > 0
        ]

    phase_rows = _parse_prom_instant(prom_data.get("pod_phase_counts"))
    if phase_rows:
        phase_counts: Dict[str, int] = {}
        for r in phase_rows:
            phase = r["labels"].get("phase", "Unknown")
            phase_counts[phase] = phase_counts.get(phase, 0) + int(r["value"])
        summary["pod_phase_counts"] = phase_counts

    net_rows = _parse_prom_instant(prom_data.get("network_rx_per_pod"))
    if net_rows:
        summary["network_rx_per_pod_top"] = _top_pods(net_rows)

    return summary


def _top_pods_with_sat(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Like _top_pods but preserves limit/saturation fields."""
    out: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: x["value"], reverse=True):
        pod = r["labels"].get("pod")
        if not pod:
            continue
        out.append({
            "pod": pod,
            "value": round(r["value"], 4),
            "limit": round(r["limit"], 4) if r.get("limit") is not None else None,
            "saturation": round(r["saturation"], 4) if r.get("saturation") is not None else None,
        })
    return out
