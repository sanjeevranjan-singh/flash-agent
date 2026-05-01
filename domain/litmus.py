"""
Domain – Litmus Chaos / Argo Workflow Logic
=============================================

Litmus-specific domain functions: CRD tool calls, workflow ID extraction,
experiment phase parsing. Isolates all Litmus/Argo knowledge so the core
agent remains domain-agnostic.

A customer replacing flash-agent with their own agent would replace this
module (or not use it at all).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

from mcp.parsers import extract_mcp_text

logger = logging.getLogger("flash-agent")


def get_kubernetes_tool_calls(
    k8s_namespace: str,
    chaos_namespace: str,
    include_chaos: bool = True,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Return the MCP tool call list for Kubernetes data collection.

    Each entry is (mcp_tool_name, arguments_dict, result_key).
    Includes generic K8s calls + optionally Litmus CRD + Argo Workflow calls.

    When include_chaos is False, chaos/Argo tools are omitted so that
    their names never appear in Langfuse trace metadata (data-leakage
    prevention for blind-observer mode).
    """
    calls: List[Tuple[str, Dict[str, Any], str]] = [
        (
            "pods_list_in_namespace",
            {"namespace": k8s_namespace},
            "pods_list_in_namespace",
        ),
        ("events_list", {"namespace": k8s_namespace}, "events_list"),
        ("pods_top", {"namespace": k8s_namespace}, "pods_top"),
    ]
    if include_chaos:
        calls.extend([
            # Litmus ChaosEngine CRs
            (
                "resources_list",
                {
                    "apiVersion": "litmuschaos.io/v1alpha1",
                    "kind": "ChaosEngine",
                    "namespace": chaos_namespace,
                },
                "chaosengines",
            ),
            # Litmus ChaosResult CRs
            (
                "resources_list",
                {
                    "apiVersion": "litmuschaos.io/v1alpha1",
                    "kind": "ChaosResult",
                    "namespace": chaos_namespace,
                },
                "chaosresults",
            ),
            # Argo Workflow resources
            (
                "resources_list",
                {
                    "apiVersion": "argoproj.io/v1alpha1",
                    "kind": "Workflow",
                    "namespace": chaos_namespace,
                },
                "argo_workflows",
            ),
        ])
    return calls


def get_prometheus_tool_calls(
    k8s_namespace: str,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Return the MCP tool call list for Prometheus snapshot collection.

    Provides per-pod CPU rate, memory working-set, restart count, and pod-phase
    distribution scoped to the agent's namespace. These metrics are merged with
    the kubernetes MCP snapshot on every scan so the LLM has time-series signal
    for fault detection (CPU spikes, memory growth) which kubectl alone cannot
    reveal between scan cycles.
    """
    ns = k8s_namespace
    return [
        ("execute_query", {"query": "up"}, "prometheus_up"),
        (
            "execute_query",
            {"query": f'count(kube_pod_info{{namespace="{ns}"}})'},
            "pod_count",
        ),
        # CPU rate per pod (cores), 1-minute window.
        #
        # cAdvisor emits two possible series shapes per pod:
        #   (A) per-container series with `container=<name>`, `image=<ref>`
        #       e.g. {pod="carts", container="carts", image="weaveworksdemos/carts"}
        #   (B) pod-aggregate series with `container=""`, `image=""`
        #       e.g. {pod="carts", container="", image="", cpu="total"}
        #
        # Production clusters running kube-prometheus-stack typically emit
        # both. Minikube/dev clusters often emit only (B). The kube-
        # prometheus-stack standard is `image!="", container!="POD"` which
        # selects (A) and excludes the pause container; if (A) is absent
        # this returns empty.
        #
        # We use PromQL `or` to fall back from (A) to (B), giving correct
        # results on any cluster shape without double-counting.
        (
            "execute_query",
            {"query": (
                f'sum by (pod) ('
                f'rate(container_cpu_usage_seconds_total{{namespace="{ns}",image!="",container!="POD"}}[1m])'
                f')'
                f' or '
                f'sum by (pod) ('
                f'rate(container_cpu_usage_seconds_total{{namespace="{ns}",container=""}}[1m])'
                f')'
            )},
            "cpu_per_pod",
        ),
        # Memory working set per pod (bytes) — same dual-shape handling.
        (
            "execute_query",
            {"query": (
                f'sum by (pod) ('
                f'container_memory_working_set_bytes{{namespace="{ns}",image!="",container!="POD"}}'
                f')'
                f' or '
                f'sum by (pod) ('
                f'container_memory_working_set_bytes{{namespace="{ns}",container=""}}'
                f')'
            )},
            "memory_per_pod",
        ),
        # Cumulative container restarts per pod
        (
            "execute_query",
            {"query": (
                f'sum by (pod) ('
                f'kube_pod_container_status_restarts_total{{namespace="{ns}"}}'
                f')'
            )},
            "restarts_per_pod",
        ),
        # Pod-phase distribution
        (
            "execute_query",
            {"query": f'count by (phase) (kube_pod_status_phase{{namespace="{ns}"}})'},
            "pod_phase_counts",
        ),
        # Network receive bytes rate per pod (bytes/sec)
        (
            "execute_query",
            {"query": (
                f'sum by (pod) ('
                f'rate(container_network_receive_bytes_total{{namespace="{ns}"}}[1m])'
                f')'
            )},
            "network_rx_per_pod",
        ),
        # Declared CPU limit per pod (cores). Used to compute saturation
        # = usage / limit. Domain-agnostic anomaly detection.
        (
            "execute_query",
            {"query": (
                f'sum by (pod) ('
                f'kube_pod_container_resource_limits{{namespace="{ns}",resource="cpu"}}'
                f')'
            )},
            "cpu_limit_per_pod",
        ),
        # Declared memory limit per pod (bytes).
        (
            "execute_query",
            {"query": (
                f'sum by (pod) ('
                f'kube_pod_container_resource_limits{{namespace="{ns}",resource="memory"}}'
                f')'
            )},
            "memory_limit_per_pod",
        ),
    ]


def extract_workflow_ids_from_resource(
    mcp_result: Any,
) -> Dict[str, str]:
    """
    Extract experiment IDs from a full Argo Workflow resource returned by
    MCP resources_get.  Parses metadata.labels and metadata.uid.

    Returns dict with keys: workflow_id (experimentId), workflow_run_id (runId),
    revision_id, infra_id, subject, workflow_name.
    """
    ids: Dict[str, str] = {}
    text = extract_mcp_text(mcp_result)
    if not text:
        return ids

    # Strategy: try JSON → YAML → regex (most to least structured)
    resource: Dict[str, Any] = {}
    parsed = False

    # 1. Try JSON
    if text.strip().startswith("{"):
        try:
            resource = json.loads(text)
            parsed = True
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Try YAML (if not already parsed)
    if not parsed:
        try:
            import yaml

            resource = yaml.safe_load(text) or {}
            parsed = bool(resource and isinstance(resource, dict))
        except Exception:
            parsed = False

    # 3. Extract from parsed dict
    if parsed and isinstance(resource, dict):
        metadata = resource.get("metadata", {})
        if isinstance(metadata, dict):
            labels = metadata.get("labels", {}) or {}
            ids["uid"] = metadata.get("uid", "")
            ids["workflow_run_id"] = metadata.get("uid", "")
            ids["workflow_id"] = labels.get("workflow_id", "")
            ids["revision_id"] = labels.get("revision_id", "")
            ids["infra_id"] = labels.get("infra_id", "")
            ids["subject"] = labels.get("subject", "")
            ids["workflow_name"] = metadata.get("name", "")

    # 4. Fallback: regex extraction from raw text
    if not ids.get("workflow_id"):
        for key in ("workflow_id", "revision_id", "infra_id", "subject"):
            m = re.search(rf"{key}:\s*(\S+)", text)
            if m:
                ids[key] = m.group(1).strip("\"'")
    if not ids.get("workflow_run_id"):
        uid_m = re.search(r"uid:\s*([0-9a-f-]{36})", text)
        if uid_m:
            ids["uid"] = uid_m.group(1)
            ids["workflow_run_id"] = uid_m.group(1)
    if not ids.get("workflow_name"):
        name_m = re.search(r"^\s+name:\s*(\S+)", text, re.MULTILINE)
        if name_m:
            ids["workflow_name"] = name_m.group(1).strip("\"'")

    return {k: v for k, v in ids.items() if v}


def parse_workflow_phase_from_text(argo_data: Any) -> Dict[str, str]:
    """
    Parse the Argo Workflow text table from MCP resources_list.
    Returns {workflow_name: phase} for all workflows whose name ends with
    an epoch-like numeric suffix (e.g. my-experiment-1774000085040).
    Fully dynamic — no hardcoded workflow prefix.
    """
    result: Dict[str, str] = {}
    text = extract_mcp_text(argo_data)
    for line in text.split("\n"):
        if not line.strip() or line.startswith("NAMESPACE"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        wf_name = parts[3]
        suffix = wf_name.rsplit("-", 1)[-1] if "-" in wf_name else ""
        if not (suffix.isdigit() and len(suffix) >= 10):
            continue
        for candidate in parts[4:]:
            if candidate in (
                "Succeeded",
                "Failed",
                "Running",
                "Error",
                "Pending",
            ):
                result[wf_name] = candidate
                break
    logger.info("Parsed workflow phases from text: %s", result)
    return result


def get_latest_workflow(
    wf_phases: Dict[str, str],
) -> Tuple[str, str]:
    """
    From a dict of {workflow_name: phase}, find the LATEST workflow
    by extracting the epoch timestamp suffix.

    Returns (latest_wf_name, latest_wf_phase). Falls back to ("", "Unknown").
    """
    if not wf_phases:
        return ("", "Unknown")

    best_name = ""
    best_phase = "Unknown"
    best_epoch = -1

    for wf_name, phase in wf_phases.items():
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

    if not best_name and wf_phases:
        best_name = list(wf_phases.keys())[-1]
        best_phase = wf_phases[best_name]

    logger.info(
        "Latest workflow: %s (phase=%s, epoch=%d)",
        best_name,
        best_phase,
        best_epoch,
    )
    return (best_name, best_phase)
