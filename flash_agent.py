"""
Flash Agent v4.0.0 – FlashAgent Implementation
=================================================

Implements AgentInterface for ITOps Kubernetes analysis.
Orchestrates the 3-step agentic pipeline:

  1. Tool Selection  (LLM decides: kubernetes or prometheus)
  2. MCP Data Collection  (3-phase: discovery -> logs -> IDs)
  3. LLM Analysis  (deep analysis of collected data)

This module is the "glue" — it coordinates calls to the MCP client,
domain-specific parsers, LLM gateway, and observability hooks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests

from config import AgentConfig
from domain.litmus import (
    extract_workflow_ids_from_resource,
    get_kubernetes_tool_calls,
    get_latest_workflow,
    get_prometheus_tool_calls,
    parse_workflow_phase_from_text,
)
from llm.gateway import (
    request_llm_analysis,
    request_tool_selection,
    request_hindsight_check,
    request_detection_gate,
)
from mcp.client import MCPClient, generate_fallback_data
from mcp.parsers import build_mcp_data_summary, extract_active_pod_names
from observability.mcp_logger import persist_mcp_interaction

logger = logging.getLogger("flash-agent")


class FlashAgent:
    """
    Flash Agent – ITOps Kubernetes analysis agent.

    Implements the AgentInterface protocol. The orchestrator (main.py)
    calls scan() on each cycle; everything inside is the agent's business.
    """

    def __init__(self, cfg: AgentConfig) -> None:
        self.cfg = cfg
        self._scan_counter = 0

    # ══════════════════════════════════════════════════════════════════════════
    # Public Interface (AgentInterface protocol)
    # ══════════════════════════════════════════════════════════════════════════

    def scan(self, query: str) -> Dict[str, Any]:
        """Execute one full analysis scan cycle."""
        self._scan_counter += 1
        scan_start = time.time()
        scan_id = (
            f"{self.cfg.agent_name}-{self.cfg.k8s_namespace}"
            f"-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        )
        logger.info(
            "\u2550\u2550\u2550 Scan #%d started | scan_id=%s \u2550\u2550\u2550",
            self._scan_counter,
            scan_id,
        )

        analysis = self._execute_scan_steps(
            scan_query=query,
            scan_id=scan_id,
            scan_start=scan_start,
        )
        return analysis

    def health_check(self) -> bool:
        """Return True if the agent is ready to accept scan requests."""
        return bool(self.cfg.agent_name and self.cfg.k8s_namespace)

    def get_capabilities(self) -> List[str]:
        """Return list of capability identifiers this agent supports."""
        return ["kubernetes", "prometheus", "litmus-chaos"]

    # ══════════════════════════════════════════════════════════════════════════
    # Private: Scan Orchestration
    # ══════════════════════════════════════════════════════════════════════════

    def _execute_scan_steps(
        self,
        scan_query: str,
        scan_id: str,
        scan_start: float,
    ) -> Dict[str, Any]:
        """
        Core scan steps — orchestrates the full MCP + LLM cycle.

        Steps:
          1. Ask LLM which MCP tool to use (kubernetes or prometheus)
          2. Call MCP server to fetch operational data
          3. Send data to LLM for deep analysis
          4. Inject experiment IDs from MCP Phase 3 into result
        """
        # ── Step 1: Tool Selection ───────────────────────────────────────────
        server_type = request_tool_selection(
            cfg=self.cfg,
            user_query=scan_query,
            scan_id=scan_id,
        )

        # ── Step 2: MCP Data Collection ─────────────────────────────────────
        mcp_data = self._collect_mcp_data(
            server_type=server_type,
            query=scan_query,
            scan_id=scan_id,
        )

        if mcp_data.get("error"):
            logger.warning(
                "MCP returned an error \u2013 analysis will reflect degraded data"
            )

        # ── Step 2b: Always augment with Prometheus snapshot when available ──
        # The kubernetes MCP is the primary source, but Prometheus adds
        # time-series CPU/memory/restart signal that kubectl alone misses
        # between scan windows. A failed prom snapshot is non-fatal.
        # Skip when:
        #   - PROM_MCP_URL is unset, OR
        #   - the toggle ENABLE_PROMETHEUS_SNAPSHOT=false, OR
        #   - LLM already selected the prometheus server (data already present).
        if (
            self.cfg.prom_mcp_url
            and self.cfg.enable_prometheus_snapshot
            and server_type != "prometheus"
        ):
            try:
                prom_snapshot = self._collect_prometheus_snapshot(scan_id)
                if prom_snapshot:
                    mcp_data.setdefault("data", {})["prometheus"] = prom_snapshot
            except Exception as exc:
                logger.warning("Prometheus snapshot failed: %s", exc)

        # ── Build agent context for Langfuse trace enrichment ────────────────
        _summary = build_mcp_data_summary(mcp_data)
        _pods_info = _summary.get("pods", {})
        _events_info = _summary.get("events", {})
        _wf_info = _summary.get("argo_workflows", {})
        _chaos_info = _summary.get("chaosengines", {})

        agent_context: Dict[str, Any] = {
            "scan_number": self._scan_counter,
            "mcp_server_type": server_type,
            "mcp_duration_sec": mcp_data.get("_mcp_duration_sec", 0),
            "mcp_data_keys": mcp_data.get("_mcp_data_keys", []),
            "pods_total": _pods_info.get("total", 0),
            "pods_by_status": _pods_info.get("by_status", {}),
            "pods_total_restarts": _pods_info.get("total_restarts", 0),
            "high_restart_pods": _pods_info.get("restarted_pods", []),
            "mcp_errors": mcp_data.get("error", None),
            "events_normal": _events_info.get("normal", 0),
            "events_warning": _events_info.get("warning", 0),
        }
        # Only include chaos/workflow metadata when chaos tools are enabled,
        # to prevent leaking chaos engine names into Langfuse trace metadata.
        if self.cfg.include_chaos_tools:
            agent_context["workflows_total"] = _wf_info.get("count", 0)
            agent_context["workflow_latest"] = _wf_info.get("latest", "")
            agent_context["chaos_engines_total"] = _chaos_info.get("count", 0)
            agent_context["chaos_engine_names"] = _chaos_info.get("engines", [])

        # ── Hindsight / data-quality annotation (Flash pattern) ────────────
        # Mirrors AIOpsLab Flash's HindsightBuilder: assess data quality before
        # the analysis call and annotate agent_context with the result.
        # Unlike a skip gate, this NEVER blocks the analysis — the agent must
        # always produce an observation so the certifier has a trace to score.
        # If data is thin the LLM will correctly report "no issues detected" or
        # "insufficient telemetry", which is itself a valid observation.
        hindsight = request_hindsight_check(
            cfg=self.cfg,
            mcp_data=mcp_data,
            server_type=server_type,
            scan_id=scan_id,
        )
        agent_context["data_sufficient"] = hindsight.get("sufficient", True)
        if not hindsight.get("sufficient", True):
            agent_context["data_quality_note"] = hindsight.get(
                "next_focus", "insufficient data"
            )
            logger.info(
                "Hindsight: data quality low (%s) — proceeding with analysis anyway",
                agent_context["data_quality_note"],
            )

        # ── Step 2c: Detection gate (AIOpsLab-style binary check) ──────────
        # Cheap binary anomaly check on the same blind-observer data. Fail-open:
        # if the gate says "No" we still run full analysis to satisfy the
        # certifier-trace requirement, but we record the gate verdict for
        # telemetry. The gate's main value is making the broader analysis
        # less likely to under-report when ANY single evidence channel fires.
        detection = request_detection_gate(
            cfg=self.cfg,
            mcp_data=mcp_data,
            server_type=server_type,
            scan_id=scan_id,
        )
        agent_context["detection_gate"] = detection.get("anomaly_detected", "Yes")
        agent_context["detection_reason"] = detection.get("reason", "")

        # ── Step 3: LLM Analysis ────────────────────────────────────────────
        if self.cfg.reasoning_mode == "react":
            # Multi-turn ReAct loop (AIOpsLab clients/gpt.py pattern). The
            # LLM iteratively chooses observation tools and submits a final
            # verdict via submit(). Same JSON schema as single-shot so all
            # downstream code is unchanged. 5-10x token cost; opt-in only.
            from llm.react_loop import request_react_analysis
            logger.info(
                "Running ReAct analysis (max_steps=%d)", self.cfg.react_max_steps
            )
            analysis = request_react_analysis(
                cfg=self.cfg,
                scan_id=scan_id,
                agent_context=agent_context,
            )
        else:
            analysis = request_llm_analysis(
                cfg=self.cfg,
                mcp_data=mcp_data,
                server_type=server_type,
                scan_id=scan_id,
                agent_context=agent_context,
            )

        if analysis is None:
            logger.error("LLM analysis failed \u2013 returning empty result")
            return {"health": {"overall_health_score": -1}, "identified_issues": []}

        # ── Inject experiment IDs from MCP Phase 3 ───────────────────────────
        mcp_inner_ids = mcp_data.get("data", {})
        workflow_ids = mcp_inner_ids.get("workflow_ids", {})
        if workflow_ids:
            experiment_id = workflow_ids.get("workflow_id", "")
            experiment_run_id = workflow_ids.get("workflow_run_id", "")
            analysis["experiment_info"] = {
                "experiment_id": experiment_id,
                "experiment_run_id": experiment_run_id,
                "revision_id": workflow_ids.get("revision_id", ""),
                "infra_id": workflow_ids.get("infra_id", ""),
                "subject": workflow_ids.get("subject", ""),
                "workflow_name": workflow_ids.get("workflow_name", ""),
            }
            logger.info(
                "Injected experiment IDs: experiment_id=%s experiment_run_id=%s",
                experiment_id or "N/A",
                experiment_run_id or "N/A",
            )
        else:
            analysis["experiment_info"] = {}
            logger.info(
                "No workflow IDs available to inject (Phase 3 may have failed)"
            )

        duration = time.time() - scan_start

        # ── Human-readable summary ───────────────────────────────────────────
        health = analysis.get("health", {})
        env_state = analysis.get("environment_state", {})
        issues = analysis.get("identified_issues", []) or []
        logger.info(
            "\u2550\u2550\u2550 Scan complete | scan_id=%s | %.1fs | server=%s | "
            "health=%s score=%s | issues=%d | pods=%s \u2550\u2550\u2550",
            scan_id,
            duration,
            server_type,
            env_state.get("health_status", "?"),
            health.get("overall_health_score", "?"),
            len(issues),
            health.get("total_instances", 0) or 0,
        )
        for issue in issues:
            logger.info(
                "  [%s] %s \u2014 %s",
                str(issue.get("severity", "?")).upper(),
                issue.get("affected_component", "?"),
                issue.get("issue_name", ""),
            )

        return analysis

    # ══════════════════════════════════════════════════════════════════════════
    # Private: MCP Data Collection (Phase 1 / 2 / 3)
    # ══════════════════════════════════════════════════════════════════════════

    def _collect_mcp_data(
        self,
        server_type: str,
        query: str,
        scan_id: str = "",
    ) -> Dict[str, Any]:
        """
        Fetch operational data from MCP server using the selected tool.

        Phase 1: Discovery (pods, events, resources, workflows)
        Phase 2: Targeted pod logs for running/error workflow pods
        Phase 3: Full Workflow resource for experiment ID extraction
        """
        if server_type not in ("kubernetes", "prometheus"):
            raise ValueError(f"Unknown MCP server: {server_type!r}")

        url = (
            self.cfg.k8s_mcp_url
            if server_type == "kubernetes"
            else self.cfg.prom_mcp_url
        )

        # Get domain-specific tool call list
        if server_type == "kubernetes":
            tool_calls = get_kubernetes_tool_calls(
                self.cfg.k8s_namespace,
                self.cfg.chaos_namespace,
                include_chaos=self.cfg.include_chaos_tools,
            )
        else:
            tool_calls = get_prometheus_tool_calls(self.cfg.k8s_namespace)

        request_payload: Dict[str, Any] = {
            "server_type": server_type,
            "namespace": self.cfg.k8s_namespace,
            "tools": [result_key for _, _, result_key in tool_calls],
            "tool_count": len(tool_calls),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Agent \u2192 MCP Server (%s) | url=%s | tools=%s",
            server_type,
            url,
            [f"{t[0]}\u2192{t[2]}" for t in tool_calls],
        )
        response_payload: Dict[str, Any] = {}
        t0 = time.time()

        try:
            client = MCPClient(url, self.cfg.agent_name, self.cfg.mcp_timeout)
            session_id = client.initialize()
            logger.info("MCP session initialized | session_id=%s", session_id)

            # Phase 1: call each tool
            all_results: Dict[str, Any] = {}
            for tool_name, tool_args, result_key in tool_calls:
                try:
                    result = client.call_tool(tool_name, tool_args)
                    all_results[result_key] = result
                    logger.info(
                        "  MCP tool '%s' [%s] \u2192 %d chars",
                        tool_name,
                        result_key,
                        len(json.dumps(result)),
                    )
                except Exception as exc:
                    logger.warning(
                        "  MCP tool '%s' [%s] failed: %s",
                        tool_name,
                        result_key,
                        exc,
                    )
                    all_results[result_key] = {"error": str(exc)}

            # Phase 2: targeted pod logs
            if server_type == "kubernetes":
                pod_list_data = all_results.get(
                    "pods_list_in_namespace", {}
                )
                active_pods = extract_active_pod_names(
                    pod_list_data, self.cfg.k8s_namespace,
                    target_app_name=self.cfg.target_app_name,
                )
                if active_pods:
                    logger.info(
                        "Phase 2: fetching logs for %d active pods: %s",
                        len(active_pods),
                        active_pods,
                    )
                    pods_log_results: Dict[str, Any] = {}
                    for pod_name in active_pods[:8]:
                        try:
                            log_args: Dict[str, Any] = {
                                "namespace": self.cfg.k8s_namespace,
                                "name": pod_name,
                            }
                            if (
                                f"{self.cfg.target_app_name}-trace" in pod_name
                                or "argowf-chaos" in pod_name
                            ):
                                log_args["container"] = "main"
                            log_result = client.call_tool(
                                "pods_log", log_args
                            )
                            pods_log_results[pod_name] = log_result
                            logger.info(
                                "  pods_log '%s' \u2192 %d chars",
                                pod_name,
                                len(json.dumps(log_result)),
                            )
                        except Exception as exc:
                            logger.warning(
                                "  pods_log '%s' failed: %s", pod_name, exc
                            )
                            pods_log_results[pod_name] = {
                                "error": str(exc)
                            }
                    all_results["pods_log"] = pods_log_results
                else:
                    logger.info(
                        "Phase 2: no active workflow pods found for log fetching"
                    )

                # Phase 3: Workflow resource for experiment IDs
                wf_phases = parse_workflow_phase_from_text(
                    all_results.get("argo_workflows", {})
                )
                latest_wf_name, _ = get_latest_workflow(wf_phases)
                if latest_wf_name:
                    logger.info(
                        "Phase 3: fetching full Workflow resource for '%s' to extract IDs",
                        latest_wf_name,
                    )
                    try:
                        wf_detail_result = client.call_tool(
                            "resources_get",
                            {
                                "apiVersion": "argoproj.io/v1alpha1",
                                "kind": "Workflow",
                                "namespace": self.cfg.chaos_namespace,
                                "name": latest_wf_name,
                            },
                        )
                        all_results["workflow_detail"] = wf_detail_result
                        workflow_ids = extract_workflow_ids_from_resource(
                            wf_detail_result
                        )
                        all_results["workflow_ids"] = workflow_ids
                        logger.info(
                            "  Workflow IDs extracted: %s", workflow_ids
                        )
                    except Exception as exc:
                        logger.warning(
                            "  Phase 3 resources_get failed: %s", exc
                        )
                        all_results["workflow_ids"] = {}
                else:
                    logger.info(
                        "Phase 3: no latest workflow found \u2014 skipping ID extraction"
                    )
                    all_results["workflow_ids"] = {}

            response_payload = {
                "server_type": server_type,
                "namespace": self.cfg.k8s_namespace,
                "data": all_results,
            }
            logger.info(
                "MCP Server (%s) \u2192 Agent | %.2fs | tools=%s",
                server_type,
                time.time() - t0,
                list(all_results.keys()),
            )

        except requests.exceptions.Timeout:
            logger.error(
                "MCP %s timed out after %ds",
                server_type,
                self.cfg.mcp_timeout,
            )
            response_payload = generate_fallback_data(
                server_type, query, self.cfg.k8s_namespace
            )
        except requests.exceptions.ConnectionError as exc:
            logger.error("MCP %s unreachable: %s", server_type, exc)
            response_payload = generate_fallback_data(
                server_type, query, self.cfg.k8s_namespace
            )
        except requests.exceptions.HTTPError as exc:
            logger.error(
                "MCP %s HTTP %s: %s",
                server_type,
                exc.response.status_code,
                exc,
            )
            response_payload = generate_fallback_data(
                server_type, query, self.cfg.k8s_namespace
            )
        except Exception as exc:
            logger.error("MCP %s error: %s", server_type, exc)
            response_payload = generate_fallback_data(
                server_type, query, self.cfg.k8s_namespace
            )

        duration = time.time() - t0

        response_payload["_mcp_duration_sec"] = round(duration, 2)
        response_payload["_mcp_data_keys"] = list(
            response_payload.get("data", {}).keys()
        )

        # Save MCP interaction to JSONL audit file
        persist_mcp_interaction(
            cfg=self.cfg,
            server_type=server_type,
            request_payload=request_payload,
            response_payload=response_payload,
            duration_sec=duration,
            scan_id=scan_id,
        )

        return response_payload

    def _collect_prometheus_snapshot(self, scan_id: str = "") -> Dict[str, Any]:
        """
        Lightweight Prometheus snapshot used to enrich the kubernetes MCP
        data with CPU/memory/restart time-series signal.

        Iterates the queries from ``get_prometheus_tool_calls`` and returns
        a dict keyed by ``result_key``. Per-tool failures are captured as
        ``{"error": str}`` so a single failed query does not abort the
        whole snapshot.
        """
        if not self.cfg.prom_mcp_url:
            return {}

        url = self.cfg.prom_mcp_url
        tool_calls = get_prometheus_tool_calls(self.cfg.k8s_namespace)
        request_payload: Dict[str, Any] = {
            "server_type": "prometheus",
            "namespace": self.cfg.k8s_namespace,
            "tools": [result_key for _, _, result_key in tool_calls],
            "tool_count": len(tool_calls),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Agent \u2192 Prometheus MCP | url=%s | tools=%s",
            url,
            [f"{t[0]}\u2192{t[2]}" for t in tool_calls],
        )

        all_results: Dict[str, Any] = {}
        t0 = time.time()
        try:
            client = MCPClient(url, self.cfg.agent_name, self.cfg.mcp_timeout)
            client.initialize()
            for tool_name, tool_args, result_key in tool_calls:
                try:
                    result = client.call_tool(tool_name, tool_args)
                    all_results[result_key] = result
                    logger.info(
                        "  Prom tool '%s' [%s] \u2192 %d chars",
                        tool_name,
                        result_key,
                        len(json.dumps(result)),
                    )
                except Exception as exc:
                    logger.warning(
                        "  Prom tool '%s' [%s] failed: %s",
                        tool_name,
                        result_key,
                        exc,
                    )
                    all_results[result_key] = {"error": str(exc)}
        except Exception as exc:
            logger.warning("Prometheus MCP unavailable: %s", exc)
            return {}

        duration = time.time() - t0
        response_payload = {
            "server_type": "prometheus",
            "namespace": self.cfg.k8s_namespace,
            "data": all_results,
            "_mcp_duration_sec": round(duration, 2),
            "_mcp_data_keys": list(all_results.keys()),
        }
        logger.info(
            "Prometheus MCP \u2192 Agent | %.2fs | tools=%s",
            duration,
            list(all_results.keys()),
        )

        try:
            persist_mcp_interaction(
                cfg=self.cfg,
                server_type="prometheus",
                request_payload=request_payload,
                response_payload=response_payload,
                duration_sec=duration,
                scan_id=scan_id,
            )
        except Exception as exc:
            logger.debug("persist_mcp_interaction (prom) failed: %s", exc)

        return all_results
