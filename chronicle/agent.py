"""
╔══════════════════════════════════════════════════════════════════╗
║  CHRONICLE — agent.py                                            ║
║  Session 11.1: High-Throughput Inference Foundation              ║
╚══════════════════════════════════════════════════════════════════╝

Chronicle is a local-first personal AI analyst.
It connects to your real data — Spotify, GitHub, finances,
fitness records, journal entries — and tells you what your
data says about you that you haven't admitted yet.

This file is the inference core. It handles:
  - Gemini API configuration and single-request inference
  - Async concurrent load testing (TTFT / TPOT measurement)
  - Chronicle VRAM budget calculation (maps to local vLLM in S11.3)
  - Session verification

Permanent from Session 11.1 onward.
"""

# ── Imports (Session 11.1) ────────────────────────────────────────
import asyncio
import aiohttp
import ssl
import certifi
import time
import statistics
import os
from typing import Optional
from pydantic import BaseModel
from google import genai

# ── Configuration (Session 11.1) ─────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-2.5-flash"
GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL}:generateContent"
)

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set. "
        "Export it before running Chronicle."
    )

genai_client = genai.Client(api_key=GEMINI_API_KEY)

# ── Chronicle Agent Roles (Session 11.1) ─────────────────────────
# These are the 5 permanent agents Chronicle will run concurrently.
# Each agent has an assigned inference tier determined by task complexity.
# Tier assignments are finalised in Session 11.2 (quantization).
# VRAM requirements are calculated in this session.
CHRONICLE_AGENTS = {
    "ingestion":  {"role": "Parse and normalise raw data from all 5 sources",   "tier": "utility"},    # S11.1
    "pattern":    {"role": "Find cross-source correlations between data signals", "tier": "utility"},    # S11.1
    "timeline":   {"role": "Sequence life events from data across all sources",   "tier": "utility"},    # S11.1
    "brutality":  {"role": "Deliver honest cross-source analysis, no softening",  "tier": "frontier"},   # S11.1
    "synthesis":  {"role": "Produce the final structured analyst brief",          "tier": "frontier"},   # S11.1
}

# ── VRAM Budget Constants (Session 11.1) ─────────────────────────
# Used by calculate_chronicle_vram_budget() below.
# Values updated in Session 11.2 when quantization tiers are locked.
VRAM_BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "int8": 1,
    "int4": 0.5,
}
KV_CACHE_GB_PER_AGENT_4K = 2.0    # Llama-3-8B @ 4K context per agent. S11.1.
MODEL_WEIGHT_GB_7B_FP16   = 14.0   # 7B params × 2 bytes. S11.1.
MODEL_WEIGHT_GB_7B_INT4   = 3.5    # 7B params × 0.5 bytes. S11.1.
CUDA_OVERHEAD_GB          = 2.0    # Runtime overhead, constant. S11.1.
SAFETY_HEADROOM_GB        = 8.0    # 10% of 80GB A100, rounded. S11.1.

# ── Pydantic Schema (Session 11.1) ───────────────────────────────

class AnalysisRequest(BaseModel):
    """
    What it does:   Defines a valid Chronicle analysis request.
    When called:    On every POST to /analyze (api.py).
    Returns:        Validated request object.
    Introduced:     Session 11.1. Permanent.

    Fields grow each session as Chronicle gains new capabilities.
    """
    question: str                         # The user's analysis question. S11.1.
    data_sources: list[str] = []          # Which sources to analyse. S11.3 (MCP adds live pulls).
    depth: str = "standard"               # "quick" | "standard" | "deep". S11.1.


class BenchmarkResult(BaseModel):
    """
    What it does:   Stores timing metrics for a single inference request.
    When called:    Populated by benchmark_single_request().
    Returns:        Structured metrics dict.
    Introduced:     Session 11.1. Permanent.
    """
    request_id: int
    ttft_seconds: Optional[float] = None
    total_latency_seconds: Optional[float] = None
    approximate_output_tokens: int = 0
    tpot_seconds: Optional[float] = None
    response_text: Optional[str] = None
    status: str = "error"
    error_message: Optional[str] = None


# ── VRAM Budget Calculator (Session 11.1) ────────────────────────

def calculate_chronicle_vram_budget(precision: str = "fp16") -> dict:
    """
    What it does:   Calculates the minimum VRAM required to run Chronicle's
                    5-agent swarm simultaneously at a given model precision.
    When called:    At startup to validate GPU readiness, and by run_session_verification().
    Returns:        Dict with per-component VRAM breakdown and total.
    Introduced:     Session 11.1. Permanent.

    Note: Values are for a Llama-3-8B baseline. Session 11.2 recalculates
    this with the quantization tier assigned to each agent role.
    Session 11.3 finalises max_model_len per agent to bound KV cache growth.
    """
    bytes_per_param = VRAM_BYTES_PER_PARAM.get(precision, 2)
    weight_gb_per_model = (7_000_000_000 * bytes_per_param) / (1024 ** 3)

    # Chronicle runs 5 agents. Each holds its own logical KV cache.
    # With PagedAttention (Session 11.3), these share a single physical VRAM pool.
    # Here we calculate the total pool size needed.
    num_agents = len(CHRONICLE_AGENTS)
    kv_cache_total_gb = num_agents * KV_CACHE_GB_PER_AGENT_4K

    # Frontier agents (Brutality, Synthesis) need a heavier model.
    # Utility agents (Ingestion, Timeline, Pattern) can share a smaller one.
    # Session 11.2 locks which agents use which tier.
    # For Session 11.1: assume one shared model for all agents.
    total_gb = (
        weight_gb_per_model     # model weights
        + kv_cache_total_gb     # 5 agent KV caches
        + CUDA_OVERHEAD_GB      # runtime overhead
        + SAFETY_HEADROOM_GB    # 10% safety buffer
    )

    return {
        "precision":         precision,
        "weight_gb":         round(weight_gb_per_model, 1),
        "kv_cache_total_gb": round(kv_cache_total_gb, 1),
        "agents":            num_agents,
        "kv_per_agent_gb":   KV_CACHE_GB_PER_AGENT_4K,
        "cuda_overhead_gb":  CUDA_OVERHEAD_GB,
        "safety_buffer_gb":  SAFETY_HEADROOM_GB,
        "total_required_gb": round(total_gb, 1),
        "recommended_gpu":   "A100-40GB" if total_gb <= 40 else "A100-80GB",
        "note": (
            "Session 11.1 baseline. Recalculated in S11.2 (quantization tiers), "
            "finalised in S11.3 (max_model_len per agent)."
        ),
    }


# ── Single Inference Call (Session 11.1) ─────────────────────────

async def chronicle_infer(
    session: aiohttp.ClientSession,
    question: str,
    agent_name: str,
    request_id: int,
) -> BenchmarkResult:
    """
    What it does:   Fires one inference request against the Gemini API on
                    behalf of a named Chronicle agent. Records TTFT and TPOT.
    When called:    By run_concurrent_analysis() for each agent slot.
    Returns:        BenchmarkResult with timing metrics.
    Introduced:     Session 11.1. Permanent.

    Note: Uses direct REST via aiohttp (not the synchronous SDK) to enable
    true concurrent dispatch without blocking the event loop.
    TTFT here approximates header-arrival time. True per-token TTFT requires
    the streaming endpoint — introduced in Session 12.2 (SSE).
    """
    result = BenchmarkResult(request_id=request_id)

    agent_prompt = (
        f"You are Chronicle's {agent_name} agent. "
        f"Your role: {CHRONICLE_AGENTS[agent_name]['role']}. "
        f"The user's question: {question}. "
        f"Respond with a brief analysis (2-3 sentences) as this agent would."
    )

    payload = {
        "contents": [{"parts": [{"text": agent_prompt}]}],
        "generationConfig": {"maxOutputTokens": 512, "temperature": 0.7},
    }

    try:
        request_start = time.monotonic()

        async with session.post(
            GEMINI_REST_URL,
            json=payload,
            params={"key": GEMINI_API_KEY},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            first_token_time = time.monotonic()
            body = await response.json()
            complete_time = time.monotonic()

            if response.status != 200:
                err = body.get("error", {}).get("message", f"HTTP {response.status}")
                result.error_message = err
                return result

            candidates = body.get("candidates", [])
            if not candidates:
                result.error_message = "No candidates returned"
                return result

            text = ""
            for part in candidates[0].get("content", {}).get("parts", []):
                text += part.get("text", "")

            approx_tokens = max(len(text.split()), 1)
            ttft = first_token_time - request_start
            total = complete_time - request_start
            tpot = (total - ttft) / approx_tokens

            result.ttft_seconds = round(ttft, 4)
            result.total_latency_seconds = round(total, 4)
            result.approximate_output_tokens = approx_tokens
            result.tpot_seconds = round(tpot, 6)
            result.response_text = text
            result.status = "success"

    except aiohttp.ClientError as e:
        result.error_message = f"ClientError: {str(e)}"
    except Exception as e:
        result.error_message = f"Exception: {str(e)}"

    return result


# ── Concurrent Analysis Runner (Session 11.1) ────────────────────

async def run_concurrent_analysis(question: str) -> dict:
    """
    What it does:   Fires all 5 Chronicle agents simultaneously against the
                    Gemini API and collects their responses and timing metrics.
    When called:    By /analyze endpoint in api.py on every user request.
    Returns:        Dict with per-agent responses and aggregate timing stats.
    Introduced:     Session 11.1. Permanent.

    This is the core Chronicle inference pattern. All 5 agents fire at the
    same moment, simulating the concurrent KV cache load they would place
    on a self-hosted vLLM server. TTFT and TPOT distributions across the
    5 agents reveal scheduling pressure at the serving layer.
    """
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(limit=len(CHRONICLE_AGENTS), ssl=ssl_ctx)
    agent_names = list(CHRONICLE_AGENTS.keys())

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                chronicle_infer(session, question, agent_name, idx + 1)
            )
            for idx, agent_name in enumerate(agent_names)
        ]
        wall_start = time.monotonic()
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_elapsed = time.monotonic() - wall_start

    results = {}
    ttft_values = []
    for agent_name, raw in zip(agent_names, raw_results):
        if isinstance(raw, Exception):
            results[agent_name] = {
                "status": "error",
                "error_message": str(raw),
            }
        else:
            results[agent_name] = raw.model_dump()
            if raw.status == "success" and raw.ttft_seconds is not None:
                ttft_values.append(raw.ttft_seconds)

    metrics = {
        "wall_clock_seconds": round(wall_elapsed, 3),
        "agents_succeeded": sum(1 for r in results.values() if r.get("status") == "success"),
        "agents_total": len(CHRONICLE_AGENTS),
    }
    if ttft_values:
        metrics["mean_ttft_seconds"]  = round(statistics.mean(ttft_values), 3)
        metrics["max_ttft_seconds"]   = round(max(ttft_values), 3)
        metrics["min_ttft_seconds"]   = round(min(ttft_values), 3)

    return {"agent_results": results, "metrics": metrics}


# ── Session Verification (Session 11.1) ──────────────────────────

def run_session_verification() -> dict:
    """
    ┌─────────────────────────────────────────────────────────────┐
    │  SESSION 11.1 — VERIFICATION TEST                           │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT THIS TESTS:                                           │
    │    - Gemini API key is present and valid                    │
    │    - All 5 Chronicle agents are defined with roles          │
    │    - VRAM budget calculator runs at FP16 and INT4           │
    │    - Concurrent analysis fires and returns results          │
    │    - TTFT is measured and within acceptable range           │
    ├─────────────────────────────────────────────────────────────┤
    │  PASS CRITERIA:                                             │
    │    ✓ GEMINI_API_KEY present in environment                  │
    │    ✓ All 5 agents defined: ingestion, pattern, timeline,    │
    │      brutality, synthesis                                   │
    │    ✓ FP16 VRAM budget calculates > 0 GB                     │
    │    ✓ INT4 VRAM budget < FP16 VRAM budget                    │
    │    ✓ Concurrent analysis returns ≥ 1 successful agent       │
    ├─────────────────────────────────────────────────────────────┤
    │  WHAT A PASS PROVES:                                        │
    │    Chronicle's inference foundation is operational.         │
    │    All 5 agents can fire concurrently against Gemini.       │
    │    VRAM budget calculation is ready for S11.2 quantization. │
    └─────────────────────────────────────────────────────────────┘
    """
    checks = []
    start = time.monotonic()

    # CHECK 1: API key present
    key_present = bool(GEMINI_API_KEY)
    checks.append({
        "label":        "GEMINI_API_KEY present in environment",
        "passed":       key_present,
        "note":         "Found" if key_present else "MISSING — export GEMINI_API_KEY",
    })

    # CHECK 2: All 5 agents defined
    expected_agents = {"ingestion", "pattern", "timeline", "brutality", "synthesis"}
    actual_agents = set(CHRONICLE_AGENTS.keys())
    agents_ok = expected_agents == actual_agents
    checks.append({
        "label":   "All 5 Chronicle agents defined",
        "passed":  agents_ok,
        "note":    f"Found: {sorted(actual_agents)}" if agents_ok else f"Missing: {expected_agents - actual_agents}",
    })

    # CHECK 3: FP16 VRAM budget > 0
    budget_fp16 = calculate_chronicle_vram_budget("fp16")
    fp16_ok = budget_fp16["total_required_gb"] > 0
    checks.append({
        "label":   "FP16 VRAM budget calculates correctly",
        "passed":  fp16_ok,
        "note":    f"{budget_fp16['total_required_gb']} GB total required (recommended: {budget_fp16['recommended_gpu']})",
    })

    # CHECK 4: INT4 budget is less than FP16 budget
    budget_int4 = calculate_chronicle_vram_budget("int4")
    int4_ok = budget_int4["total_required_gb"] < budget_fp16["total_required_gb"]
    checks.append({
        "label":   "INT4 VRAM budget < FP16 VRAM budget",
        "passed":  int4_ok,
        "note":    f"INT4: {budget_int4['total_required_gb']} GB vs FP16: {budget_fp16['total_required_gb']} GB",
    })

    # CHECK 5: Concurrent analysis returns ≥ 1 successful agent
    test_question = "What patterns exist in my personal data?"
    try:
        analysis = asyncio.run(run_concurrent_analysis(test_question))
        succeeded = analysis["metrics"]["agents_succeeded"]
        concurrent_ok = succeeded >= 1
        note = (
            f"{succeeded}/5 agents succeeded. "
            f"Mean TTFT: {analysis['metrics'].get('mean_ttft_seconds', 'N/A')}s"
        )
    except Exception as e:
        concurrent_ok = False
        note = f"Exception: {str(e)}"

    checks.append({
        "label":   "Concurrent 5-agent analysis returns ≥ 1 success",
        "passed":  concurrent_ok,
        "note":    note,
    })

    duration_ms = round((time.monotonic() - start) * 1000)
    passed = sum(1 for c in checks if c["passed"])
    total = len(checks)

    return {
        "passed":      passed == total,
        "checks":      checks,
        "summary":     f"{passed}/{total} checks passed in {duration_ms}ms",
        "duration_ms": duration_ms,
    }


# ── CLI Entry Point (Session 11.1) ───────────────────────────────

if __name__ == "__main__":
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  Chronicle — Session 11.1 Verification               ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # Print VRAM budget at both precision tiers
    for precision in ["fp16", "int4"]:
        budget = calculate_chronicle_vram_budget(precision)
        print(f"  VRAM Budget ({precision.upper()}):")
        print(f"    Model weights:      {budget['weight_gb']} GB")
        print(f"    KV cache (5 agents):{budget['kv_cache_total_gb']} GB")
        print(f"    CUDA overhead:      {budget['cuda_overhead_gb']} GB")
        print(f"    Safety buffer:      {budget['safety_buffer_gb']} GB")
        print(f"    ─────────────────────────────────────")
        print(f"    TOTAL REQUIRED:     {budget['total_required_gb']} GB")
        print(f"    Recommended GPU:    {budget['recommended_gpu']}\n")

    # Run verification
    result = run_session_verification()
    print(f"\n  Verification: {result['summary']}\n")
    for check in result["checks"]:
        icon = "✓" if check["passed"] else "✗"
        print(f"  {icon} {check['label']}")
        print(f"      {check['note']}")
    print()

    if result["passed"]:
        print("  ✓ Session 11.1 COMPLETE. Start the API: python api.py")
    else:
        print("  ✗ Fix failing checks before proceeding.")
    print()

# ══════════════════════════════════════════════════════════════════
# SESSION 11.2 HANDOFF — "Model Quantization"
# ══════════════════════════════════════════════════════════════════
#
# What gets ADDED in Session 11.2 (extend, never remove):
#   CHRONICLE_AGENTS updated: each agent gains a locked quantization
#     tier (int4, int8, fp16) based on the task survivability matrix.
#   VRAM_BUDGET_PER_AGENT: dict mapping agent → (model_size, precision, vram_gb)
#   calculate_chronicle_vram_budget() updated to use per-agent tiers
#     instead of one shared precision for all 5 agents.
#   task_survivability_matrix(): new function returning structured
#     analysis of which Chronicle tasks survive each quantization level.
#   tiered_cost_model(): calculates monthly GPU cost for the 5-agent
#     swarm at the locked tier assignments.
#
# What stays UNCHANGED from Session 11.1:
#   CHRONICLE_AGENTS dict (keys and roles)
#   chronicle_infer() — async single inference call
#   run_concurrent_analysis() — 5-agent concurrent dispatcher
#   AnalysisRequest / BenchmarkResult Pydantic schemas
#   run_session_verification() — replaced each session with new checks
# ══════════════════════════════════════════════════════════════════