# Flash Agent v3.0.0 – Issue Fix Log

**Date:** March 16, 2026  
**File:** `test.py` (agent-mcp-llm.py adapted for local testing)

---

## Issue 1: Invalid Model Name – 400 Bad Request

**Error:**
```
HTTP/1.1 400 Bad Request
{'error': {'message': "Invalid model name passed in model=gpt-4o. Call `/v1/models` to view available models for your key."}}
```

**Root Cause:**  
The code had `MODEL_ALIAS` defaulting to `gpt-4o`, but the LiteLLM proxy was configured with OpenRouter as the backend. The available models in LiteLLM were:
- `gemini-3-flash`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `auto-free`

`gpt-4o` did not exist in the LiteLLM config, so it was rejected with a 400 error.

**Fix:**  
Changed the default `MODEL_ALIAS` from `gpt-4o` to `auto-free` (a working model in the LiteLLM config).

```python
# Before
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "gpt-4o")

# After
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "auto-free")
```

---

## Issue 2: Gemini API Key Invalid – 401 Unauthorized

**Error:**
```
HTTP/1.1 401 Unauthorized
litellm.AuthenticationError: GeminiException - API key not valid. Please pass a valid API key.
Received Model Group=gemini-2.5-flash
```

**Root Cause:**  
After switching to `gemini-2.5-flash`, LiteLLM routed the request directly to Google's Gemini API (googleapis.com) instead of through OpenRouter. The Gemini API key configured in LiteLLM was invalid or expired.

**Fix:**  
Switched `MODEL_ALIAS` to `auto-free`, which routes through OpenRouter's free tier and does not require a separate Gemini API key.

---

## Issue 3: MCP Server 400 Bad Request – Wrong Protocol

**Error:**
```
MCP kubernetes HTTP 400: 400 Client Error: Bad Request for url: http://localhost:8086/mcp
```

**Root Cause:**  
The code was sending a **custom JSON payload** to the MCP servers:
```json
{"query": "...", "namespace": "default", "agent": "flash-agent"}
```

But MCP servers expect the **JSON-RPC 2.0** protocol with proper message format:
```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "pods_list_in_namespace", "arguments": {"namespace": "default"}}}
```

Additionally, the Prometheus MCP server requires a **session ID** (obtained via an `initialize` handshake) in the `Mcp-Session-Id` header.

**Fix:**  
Rewrote the MCP communication layer with three new functions:

1. **`_mcp_jsonrpc_call()`** – Sends JSON-RPC 2.0 requests and parses SSE (Server-Sent Events) responses. Handles session ID propagation via headers.

2. **`_mcp_init_session()`** – Performs the MCP `initialize` handshake to obtain a session ID.

3. **`agent_call_mcp_server()`** – Rewritten to:
   - Initialize an MCP session first
   - Call specific MCP tools via `tools/call` method:
     - **Kubernetes MCP:** `pods_list_in_namespace` + `events_list`
     - **Prometheus MCP:** `execute_query` (PromQL queries)
   - Parse SSE response format (`data: {...}` lines)

---

## Issue 4: LLM Returns `content=None` – NoneType Error

**Error:**
```
LLM tool-selection failed: 'NoneType' object has no attribute 'strip' – defaulting to kubernetes
```

**Root Cause:**  
The `auto-free` model routed to a **reasoning model** (e.g., via OpenRouter) that returns its output in the `reasoning_content` field instead of `content`. The code called `.content.strip()` without checking for `None`.

**Fix:**  
Added fallback logic to check both `content` and `reasoning_content`:

```python
# Before
output_text = resp.choices[0].message.content.strip().lower()

# After
msg = resp.choices[0].message
raw_text = msg.content or getattr(msg, "reasoning_content", None) or ""
output_text = raw_text.strip().lower()
```

Also increased `max_tokens` from `8` to `50` to give reasoning models enough space.

---

## Issue 5: Model Doesn't Support System Role – 400 Bad Request

**Error:**
```
HTTP/1.1 400 Bad Request
OpenrouterException - "Developer instruction is not enabled for models/gemma-3n-e2b-it"
```

**Root Cause:**  
The `auto-free` model routed to `gemma-3n-e2b-it` (Google Gemma), which does **not support the `system` role** in chat messages. The code was sending:
```json
[
  {"role": "system", "content": "You are an expert..."},
  {"role": "user", "content": "Data to analyse..."}
]
```

Additionally, `response_format: {"type": "json_object"}` is not universally supported across all models.

**Fix:**  
1. **Merged system prompt into user message** for both tool selection and analysis:
   ```python
   # Before
   messages = [
       {"role": "system", "content": _ANALYSIS_SYSTEM},
       {"role": "user",   "content": payload_text},
   ]

   # After
   combined_prompt = f"INSTRUCTIONS:\n{_ANALYSIS_SYSTEM}\n\nDATA TO ANALYSE:\n{payload_text}"
   messages = [
       {"role": "user", "content": combined_prompt},
   ]
   ```

2. **Removed `response_format`** parameter from the API call.

3. **Added JSON extraction** from markdown code fences (models often wrap JSON in ` ```json ... ``` `):
   ```python
   if "```json" in json_text:
       json_text = json_text.split("```json", 1)[1].split("```", 1)[0]
   ```

---

## Issue 6: Logging Format Error – `%d` with NoneType

**Error:**
```
TypeError: %d format: a real number is required, not NoneType
Arguments: ('flash-agent-default-...', 6.56, 'kubernetes', 100, 0, None)
```

**Root Cause:**  
The LLM returned `"total_pods": null` in the JSON response. The log format string used `%d` (integer format) for `total_pods`, which failed when the value was `None`.

**Fix:**  
Changed the format specifier and added a fallback:

```python
# Before
"health=%s | issues=%d | pods=%d ═══",
...
health.get("total_pods", 0),

# After
"health=%s | issues=%d | pods=%s ═══",
...
health.get("total_pods", 0) or 0,
```

---

## Summary

| # | Issue | HTTP Code | Root Cause | Fix |
|---|-------|-----------|------------|-----|
| 1 | Invalid model name | 400 | `gpt-4o` not in LiteLLM config | Changed to `auto-free` |
| 2 | Gemini API key invalid | 401 | LiteLLM routing to Gemini directly | Used `auto-free` via OpenRouter |
| 3 | MCP wrong protocol | 400 | Custom JSON instead of JSON-RPC 2.0 | Rewrote MCP layer with proper protocol |
| 4 | `content=None` from LLM | — | Reasoning model returns `reasoning_content` | Fallback to `reasoning_content` |
| 5 | System role unsupported | 400 | `gemma-3n-e2b-it` rejects system role | Merged into user message |
| 6 | Logging NoneType error | — | LLM returned `null` for `total_pods` | Added null-safe formatting |

---

## Final Result

After all fixes, the full agent flow works end-to-end:

```
✅ LLM tool selection    → kubernetes      (200 OK)
✅ MCP session init      → connected
✅ MCP pods_list         → data received
✅ MCP events_list       → data received
✅ LLM analysis          → health=100      (200 OK)
✅ Scan complete         → ~4.6s
```

---

## Issue 7: Agent-to-MCP Traces Not Appearing in Langfuse Cloud

**Date:** March 17, 2026

**Symptom:**
Only 2 traces were visible in Langfuse Cloud — both were LLM request/response traces (tool selection and analysis). No MCP (Agent ↔ MCP Server) traces were present. The agent was calling MCP tools successfully, but those interactions were invisible in Langfuse.

**Investigation:**

The 2 existing traces in Langfuse had `"service.name": "litellm-proxy"` and `"scope": {"name": "agentcert"}` — meaning they were **sent by the LiteLLM proxy** (the LLM Gateway), **not by the Flash Agent** itself. LiteLLM has a built-in Langfuse/OTLP integration that auto-logs every LLM call it proxies.

The Flash Agent (`test.py`) was sending **zero traces** to Langfuse due to two issues:

### Problem A: `langfuse` Python package not installed

Terminal logs showed:
```
langfuse package not installed – direct Langfuse client disabled
```
The `langfuse` package was missing from both the local `.venv` and the Docker container.

### Problem B: Langfuse v2 API used with Langfuse v4 installed

After installing `langfuse`, the terminal showed:
```
Langfuse root trace creation failed: 'Langfuse' object has no attribute 'trace'
```
The code was written for **Langfuse SDK v2** which used:
- `langfuse_client.trace()` — to create a trace
- `trace.generation()` — to create a generation child
- `trace.span()` — to create a span child
- `generation.end(output=..., usage=..., metadata=...)` — to finalize

But **Langfuse SDK v4** has a completely different API:
- `langfuse_client.start_as_current_observation(name=..., as_type="span")` — context manager for root span
- `langfuse_client.start_observation(name=..., as_type="generation")` — to create child observations
- `observation.update(output=..., usage_details=..., metadata=...)` — to set output/metadata
- `observation.end()` — to finalize (no parameters)

### Problem C: Each step created separate disconnected traces

The original code created **3 separate Langfuse traces** with different IDs:
- `{scan_id}-tool-select` for LLM tool selection
- `{scan_id}-mcp-{type}` for MCP call
- `{scan_id}-analysis` for LLM analysis

This meant even if they worked, they would appear as 3 unrelated traces in Langfuse instead of a single unified trace with child spans.

---

**Fix (multi-part):**

### Fix 7a: Install `langfuse` package

**Local environment:**
```bash
pip install langfuse
```
Installed `langfuse==4.0.0` in the local `.venv`.

**Docker container — added to `requirements.txt`:**
```diff
  kubernetes>=28.0.0
  openai>=1.0.0
  opentelemetry-api>=1.20.0
  opentelemetry-sdk>=1.20.0
  opentelemetry-exporter-otlp-proto-http>=1.20.0
+ langfuse>=2.0.0
```

**Rebuilt Docker image:**
```bash
docker build -t agentcert/flash-agent:latest -f Dockerfile ..
```
Verified: `docker run --rm --entrypoint python agentcert/flash-agent:latest -c "import langfuse; print('OK')"` → success.

### Fix 7b: Set Langfuse credentials in `.env`

Added to `.env`:
```env
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Fix 7c: Migrate all Langfuse code from v2 API to v4 API

Rewrote all Langfuse integration in `test.py` to use the Langfuse v4 SDK API.

**1. Root trace → Root span with context manager**

```python
# Before (v2 API – BROKEN)
lf_trace = langfuse_client.trace(
    id=scan_id,
    name="agent-scan",
    metadata={...},
    tags=TRACE_TAGS,
)

# After (v4 API – WORKING)
with langfuse_client.start_as_current_observation(
    name="agent-scan",
    as_type="span",
    input={"scan_query": scan_query, "scan_id": scan_id},
    metadata={"agent": AGENT_NAME, "namespace": K8S_NAMESPACE},
) as root_span:
    # all child observations auto-attach to this root
    ...
    root_span.update(output=result, metadata={...})
```

**2. LLM generation spans**

```python
# Before (v2 API – BROKEN)
lf_generation = lf_trace.generation(
    name="llm-tool-selection",
    model=MODEL_ALIAS,
    input=messages,
)
lf_generation.end(
    output=output_text,
    usage={"input": prompt_tokens, "output": completion_tokens},
    metadata={...},
)

# After (v4 API – WORKING)
lf_generation = langfuse_client.start_observation(
    name="llm-tool-selection",
    as_type="generation",
    model=MODEL_ALIAS,
    input=messages,
)
lf_generation.update(
    output=output_text,
    usage_details={"input": prompt_tokens, "output": completion_tokens},
    metadata={...},
)
lf_generation.end()
```

**3. MCP span (the missing trace)**

```python
# Before (v2 API – BROKEN)
lf_trace.span(
    name=f"mcp-{server_type}-request",
    input=request_payload,
    output=response_payload,
    metadata={...},
)

# After (v4 API – WORKING)
mcp_span = langfuse_client.start_observation(
    name=f"mcp-{server_type}-request",
    as_type="span",
    input=request_payload,
)
mcp_span.update(output=response_payload, metadata={...})
mcp_span.end()
```

### Fix 7d: Unified trace architecture

Changed from 3 separate disconnected traces to a **single root span** (`agent-scan`) with all steps as children:

- Moved root trace creation into `agent_workflow()` using a `with` context manager
- Changed function signatures from `lf_trace` parameter to `langfuse_client` parameter
- Each step (`agent_request_tool_selection`, `agent_call_mcp_server`, `agent_request_llm_analysis`) now calls `langfuse_client.start_observation()` which auto-attaches as a child of the current root span
- Added `langfuse_client.flush()` after each scan cycle to ensure data is sent

---

**Verification:**

Terminal logs after fix:
```
③ Langfuse direct client initialised (host=https://cloud.langfuse.com)
③ Langfuse root span created
③ MCP Langfuse span recorded for prometheus      ← MCP TRACE NOW VISIBLE
③ Langfuse traces flushed to cloud
```

No errors. Confirmed across multiple scan cycles with both Kubernetes and Prometheus MCP servers.

**Langfuse Cloud dashboard now shows:**
```
agent-scan (root span)
  ├── llm-tool-selection        (generation)
  ├── mcp-prometheus-request    (span)        ← Agent ↔ MCP traces visible!
  └── llm-analysis              (generation)
```

---

## Updated Summary

| # | Issue | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | Invalid model name (400) | `gpt-4o` not in LiteLLM config | Changed to `auto-free` |
| 2 | Gemini API key invalid (401) | LiteLLM routing to Gemini directly | Used `auto-free` via OpenRouter |
| 3 | MCP wrong protocol (400) | Custom JSON instead of JSON-RPC 2.0 | Rewrote MCP layer with proper protocol |
| 4 | `content=None` from LLM | Reasoning model returns `reasoning_content` | Fallback to `reasoning_content` |
| 5 | System role unsupported (400) | `gemma-3n-e2b-it` rejects system role | Merged into user message |
| 6 | Logging NoneType error | LLM returned `null` for `total_pods` | Added null-safe formatting |
| 7 | MCP traces missing in Langfuse | `langfuse` not installed + v2 API used with v4 SDK | Installed langfuse, migrated to v4 API, unified trace architecture |

---

## Issue 8: Agent Scanning Wrong Namespace – No Pods Found

**Date:** March 17, 2026

**Symptom:**
The Kubernetes MCP server was running on `localhost:8086` and the `pod-list-namespace` tool worked correctly in MCP Inspector when called with `namespace: "sock-shop"`. However, the agent's Langfuse traces showed it was querying the `default` namespace and reporting zero pods:

```json
{
  "tool_calls": [["pods_list_in_namespace", {"namespace": "default"}], ["events_list", {"namespace": "default"}]],
  "data": {
    "pods_list_in_namespace": {"content": [{"type": "text", "text": ""}]},
    "events_list": {"content": [{"type": "text", "text": "# No events found"}]}
  }
}
```

The LLM analysis then correctly concluded "No pods are running in the default namespace" — which was technically accurate, since all workloads were deployed in `sock-shop`, not `default`.

**Root Cause:**
Two values in the `.env` file were configured with the wrong namespace:

1. **`K8S_NAMESPACE=default`** — This variable is used by `agent_call_mcp_server()` to pass the namespace argument to MCP tool calls (`pods_list_in_namespace`, `events_list`).

2. **`SCAN_QUERY`** — The scan query text also hardcoded `'default'`:
   ```
   SCAN_QUERY=Analyse the operational health of all workloads in Kubernetes namespace 'default'. Identify pod failures, restarts, resource pressure, and anomalies.
   ```
   This query is sent to the LLM for tool selection (Step 1) and included in the analysis prompt (Step 3), reinforcing the wrong namespace throughout the pipeline.

**Fix:**
Updated both values in `.env` to target the correct namespace:

```diff
  # Kubernetes
- K8S_NAMESPACE=default
+ K8S_NAMESPACE=sock-shop
  K8S_NODE_IP=192.168.65.3
```

```diff
  # Scan Behaviour
  TRACE_TAGS=flash-agent
  SCAN_INTERVAL=300
- SCAN_QUERY=Analyse the operational health of all workloads in Kubernetes namespace 'default'. Identify pod failures, restarts, resource pressure, and anomalies.
+ SCAN_QUERY=Analyse the operational health of all workloads in Kubernetes namespace 'sock-shop'. Identify pod failures, restarts, resource pressure, and anomalies.
```

**Verification:**
After the fix, the MCP tool calls now use `{"namespace": "sock-shop"}` and return actual pod data from the sock-shop microservices deployment.

---

## Updated Summary

| # | Issue | Root Cause | Fix |
|---|-------|------------|-----|
| 1 | Invalid model name (400) | `gpt-4o` not in LiteLLM config | Changed to `auto-free` |
| 2 | Gemini API key invalid (401) | LiteLLM routing to Gemini directly | Used `auto-free` via OpenRouter |
| 3 | MCP wrong protocol (400) | Custom JSON instead of JSON-RPC 2.0 | Rewrote MCP layer with proper protocol |
| 4 | `content=None` from LLM | Reasoning model returns `reasoning_content` | Fallback to `reasoning_content` |
| 5 | System role unsupported (400) | `gemma-3n-e2b-it` rejects system role | Merged into user message |
| 6 | Logging NoneType error | LLM returned `null` for `total_pods` | Added null-safe formatting |
| 7 | MCP traces missing in Langfuse | `langfuse` not installed + v2 API used with v4 SDK | Installed langfuse, migrated to v4 API, unified trace architecture |
| 8 | Agent scanning wrong namespace | `.env` had `K8S_NAMESPACE=default` instead of `sock-shop` | Updated `K8S_NAMESPACE` and `SCAN_QUERY` in `.env` |

---

## Final Result

After all fixes, the full agent flow works end-to-end with full observability:

```
✅ LLM tool selection    → kubernetes/prometheus  (200 OK)
✅ MCP session init      → connected
✅ MCP tool calls        → data received (sock-shop namespace)
✅ LLM analysis          → health score computed   (200 OK)
✅ Scan complete         → ~7-40s (varies by model)
✅ Langfuse root span    → agent-scan trace created
✅ Langfuse MCP span     → mcp-{type}-request logged
✅ Langfuse LLM spans    → llm-tool-selection + llm-analysis logged
✅ Langfuse flush        → traces sent to cloud
```
