"""
╔══════════════════════════════════════════════════════════════════╗
║  CHRONICLE — api.py                                              ║
║  Session 11.1: HTTP Gateway                                      ║
╚══════════════════════════════════════════════════════════════════╝

FastAPI gateway that exposes Chronicle's inference layer over HTTP.
This file grows each session as Chronicle gains new capabilities.

Session 11.1: /health, /analyze, /vram-budget
Session 12.1: MCP ingestion endpoints (added, not replaced)
Session 12.2: SSE streaming (added, not replaced)
Session 12.3: 202 async pattern (added, not replaced)

Permanent from Session 11.1 onward.
"""

# ── Imports (Session 11.1) ────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uvicorn

from chronicle.agent import (
    AnalysisRequest,
    run_concurrent_analysis,
    calculate_chronicle_vram_budget,
    CHRONICLE_AGENTS,
)

# ── App Setup (Session 11.1) ──────────────────────────────────────
app = FastAPI(
    title="Chronicle API",
    description="Local-first personal AI analyst. Session 11.1 — Inference Foundation.",
    version="11.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Endpoints (Session 11.1) ──────────────────────────────────────

@app.get("/health")
async def health():
    """
    Session 11.1 health check.
    Returns API version, active agents, and VRAM budget summary.
    Grows each session: new capabilities appended to 'capabilities' list.
    """
    budget = calculate_chronicle_vram_budget("fp16")
    return {
        "status":       "ok",
        "session":      "11.1",
        "version":      "11.1.0",
        "agents":       list(CHRONICLE_AGENTS.keys()),
        "vram_summary": {
            "total_required_gb": budget["total_required_gb"],
            "recommended_gpu":   budget["recommended_gpu"],
            "precision":         "fp16",
        },
        "capabilities": [
            "concurrent_5_agent_inference",   # S11.1
            # "quantized_model_routing",       # S11.2 — not yet
            # "mcp_live_data_ingestion",       # S12.1 — not yet
            # "sse_streaming",                 # S12.2 — not yet
            # "async_job_queue",               # S12.3 — not yet
        ],
    }


@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    """
    Session 11.1: Runs all 5 Chronicle agents concurrently against
    the user's question and returns their responses with timing metrics.

    Session 12.2 will replace this with an SSE streaming response.
    Session 12.3 will add a 202 async variant for deep analyses.
    """
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    try:
        result = await run_concurrent_analysis(request.question)
        return {
            "question":      request.question,
            "agent_results": result["agent_results"],
            "metrics":       result["metrics"],
            "session":       "11.1",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vram-budget")
async def vram_budget(precision: str = "fp16"):
    """
    Session 11.1: Returns Chronicle's VRAM budget at a given precision tier.
    Used by the dashboard panel to show infrastructure cost in real time.
    Precision options: fp32, fp16, int8, int4
    """
    valid = {"fp32", "fp16", "int8", "int4"}
    if precision not in valid:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid precision '{precision}'. Must be one of: {sorted(valid)}",
        )
    return calculate_chronicle_vram_budget(precision)


# ── Server Entry Point (Session 11.1) ────────────────────────────

if __name__ == "__main__":
    print("\n  Chronicle API — Session 11.1")
    print("  Starting on http://localhost:8000")
    print("  Swagger UI: http://localhost:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)