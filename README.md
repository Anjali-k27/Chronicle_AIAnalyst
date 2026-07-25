# Chronicle_AIAnalyst# Chronicle — Personal AI Analyst

Chronicle is a local-first personal AI analyst that connects to your real data — Spotify, GitHub, finances, fitness records, journal entries — and tells you what your data says about you that you haven't admitted yet.

It runs a swarm of 5 specialised agents concurrently, each with a distinct analytical role, firing simultaneously against a language model and returning structured insights with latency metrics.

---

## What Chronicle Becomes (Final Vision)

By the final session, Chronicle is a fully self-hosted inference system that:

- Pulls live data from 5 sources via MCP (Spotify, GitHub, bank transactions, fitness tracker, journal)
- Routes each analysis task to the right quantization tier (INT4 for utility agents, FP16 for frontier agents) to minimise VRAM cost without losing output quality
- Serves 5 concurrent agents on a local vLLM instance with PagedAttention and continuous batching
- Streams responses token-by-token to the browser via SSE
- Queues deep analyses asynchronously and notifies when complete
- Traces every inference call with OpenTelemetry and displays spans in the dashboard
- Shows a live per-agent cost ledger tracking GPU spend by task

The full deployment runs on a single A100-40GB GPU. No cloud inference. No API fees after setup.

---

## Session Roadmap

| Session | Title | What gets built |
|---------|-------|-----------------|
| **S11.1** | **Inference Foundation** ← you are here | 5-agent concurrent inference via Gemini API. VRAM budget calculator. TTFT/TPOT measurement. Split UI with live metrics dashboard. |
| S11.2 | Model Quantization | Task survivability matrix (which agents tolerate INT4/INT8). Per-agent tier assignments. VRAM budget recalculated by tier. Monthly GPU cost model. |
| S11.3 | GPU Allocation | vLLM server config. PagedAttention KV cache pool. `max_model_len` per agent. Continuous batching. Local inference replacing Gemini API. |
| S12.1 | FastAPI + MCP | MCP server for each data source. Live data ingestion endpoints. Agents receive real Spotify/GitHub/finance/fitness/journal data. |
| S12.2 | SSE Streaming | Streaming inference via Server-Sent Events. Real TTFT and TPOT per token. UI renders token-by-token. |
| S12.3 | Async Job Queue | 202 Accepted pattern for deep analyses. Background job queue. Webhook/poll notification when complete. |
| S13.1 | OpenTelemetry Tracing | Trace every agent call. Span viewer in dashboard. P50/P95 latency breakdowns. |
| S14.2 | Cost Ledger | Per-agent GPU spend tracking. Accumulated cost by data source. Budget alerts. |

---

## Architecture (Session 11.1)

```
Browser (index.html)
        │
        │  POST /analyze   GET /vram-budget   GET /health
        ▼
  FastAPI Gateway (api.py)
        │
        │  asyncio.gather — all 5 agents fire simultaneously
        ▼
  ┌─────────────────────────────────────────────┐
  │  ingestion  pattern  timeline  brutality  synthesis  │
  │       (5 concurrent aiohttp requests)               │
  └─────────────────────────────────────────────┘
        │
        ▼
  Gemini API  (gemini-2.5-flash)
  [Replaced with local vLLM in Session 11.3]
```

**Five agents, one question, fired simultaneously:**

| Agent | Role | Tier |
|-------|------|------|
| `ingestion` | Parse and normalise raw data from all 5 sources | utility |
| `pattern` | Find cross-source correlations between data signals | utility |
| `timeline` | Sequence life events from data across all sources | utility |
| `brutality` | Deliver honest cross-source analysis, no softening | frontier |
| `synthesis` | Produce the final structured analyst brief | frontier |

---

## Project Structure

```
chronicle/
├── agent.py          # Inference core: 5 agents, VRAM calculator, async runner
├── api.py            # FastAPI gateway: /health, /analyze, /vram-budget
├── index.html        # Split UI: chat panel + live metrics dashboard
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container for the API server
├── docker-compose.yml# Runs API + UI together
├── .env              # Your Gemini API key (never committed)
└── .gitignore
```

---

## Prerequisites

- Python 3.10+ (local) or Docker Desktop (Docker path)
- A Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com)

---

## Quick Start — Local

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd <repo-name>/chronicle
```

**2. Create your `.env` file**

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Verify the setup**

```bash
export $(cat .env | xargs) && python agent.py
```

Expected output:

```
╔══════════════════════════════════════════════════════╗
║  Chronicle — Session 11.1 Verification               ║
╚══════════════════════════════════════════════════════╝

  VRAM Budget (FP16):
    Model weights:      13.0 GB
    KV cache (5 agents):10.0 GB
    CUDA overhead:      2.0 GB
    Safety buffer:      8.0 GB
    ─────────────────────────────────────
    TOTAL REQUIRED:     33.0 GB
    Recommended GPU:    A100-40GB

  Verification: 5/5 checks passed in ~4000ms

  ✓ GEMINI_API_KEY present in environment
  ✓ All 5 Chronicle agents defined
  ✓ FP16 VRAM budget calculates correctly
  ✓ INT4 VRAM budget < FP16 VRAM budget
  ✓ Concurrent 5-agent analysis returns ≥ 1 success

  ✓ Session 11.1 COMPLETE. Start the API: python api.py
```

All 5 checks must pass before proceeding.

**5. Start the API**

```bash
export $(cat .env | xargs) && python api.py
```

The API starts on `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

**6. Open the UI**

```bash
open index.html
```

Or double-click `index.html` in Finder. The UI connects to the API automatically.

---

## Quick Start — Docker

**1. Clone the repo**

```bash
git clone <your-repo-url>
cd <repo-name>/chronicle
```

**2. Create your `.env` file**

```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

**3. Build and start**

```bash
docker-compose up --build
```

This starts two containers:
- **API** → `http://localhost:8000` (FastAPI + all 5 agents)
- **UI** → `http://localhost:3000` (nginx serving `index.html`)

Open `http://localhost:3000` in your browser.

**4. Stop**

```bash
docker-compose down
```

**Rebuild after code changes:**

```bash
docker-compose up --build --force-recreate
```

---

## How It Works — Step by Step

### Step 1: You submit a question

You type a question in the browser (or click one of the three example chips: Spotify, Finance, GitHub). The browser sends a `POST /analyze` request to the FastAPI gateway with your question.

### Step 2: The gateway dispatches 5 agents simultaneously

`run_concurrent_analysis()` in `agent.py` creates 5 async tasks — one per agent — and fires them all at the same instant using `asyncio.gather()`. Each task calls `chronicle_infer()`, which sends an HTTP POST to the Gemini REST API via `aiohttp`.

Each agent receives a different system prompt that frames the question through its specific analytical lens.

### Step 3: TTFT and TPOT are measured

For each agent, Chronicle records:
- **TTFT** (Time to First Token): time from request dispatch to API response headers arriving
- **TPOT** (Time Per Output Token): time per token after the first — meaningful only with streaming (Session 12.2); shows as ~0ms in S11.1 because the full response arrives at once

### Step 4: Responses return to the browser

The gateway assembles all 5 agent responses and timing metrics into a single JSON payload and returns it to the browser. The UI renders each agent's analysis in the chat panel and updates the metrics dashboard (wall clock, mean TTFT, min/max TTFT, success count).

### Step 5: VRAM budget is calculated

`calculate_chronicle_vram_budget()` computes how much GPU memory Chronicle's 5-agent swarm would need on a self-hosted vLLM server. The dashboard lets you toggle between FP32, FP16, INT8, and INT4 to see how quantization reduces the memory requirement. This calculation is finalised in Session 11.2.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Session info, agent list, VRAM summary |
| `POST` | `/analyze` | Run all 5 agents concurrently against your question |
| `GET` | `/vram-budget?precision=fp16` | VRAM breakdown at a given precision tier |
| `GET` | `/docs` | Swagger UI (interactive API explorer) |

**`POST /analyze` request body:**

```json
{
  "question": "What does my Spotify data say about my emotional state?",
  "data_sources": [],
  "depth": "standard"
}
```

**`POST /analyze` response:**

```json
{
  "question": "...",
  "session": "11.1",
  "metrics": {
    "wall_clock_seconds": 4.1,
    "agents_succeeded": 5,
    "agents_total": 5,
    "mean_ttft_seconds": 3.8,
    "min_ttft_seconds": 3.4,
    "max_ttft_seconds": 4.1
  },
  "agent_results": {
    "ingestion":  { "status": "success", "response_text": "...", "ttft_seconds": 4.1, ... },
    "pattern":    { "status": "success", "response_text": "...", "ttft_seconds": 3.4, ... },
    "timeline":   { "status": "success", "response_text": "...", "ttft_seconds": 3.5, ... },
    "brutality":  { "status": "success", "response_text": "...", "ttft_seconds": 3.5, ... },
    "synthesis":  { "status": "success", "response_text": "...", "ttft_seconds": 4.1, ... }
  }
}
```

---

## Testing Example Prompts

The UI includes three pre-built example prompts. Click any chip to pre-fill the input:

**Spotify · mood**
> My Spotify data, last 30 days: top artists — Radiohead (42 plays), Kendrick Lamar (38), Mac DeMarco (31). Listening peaks 11pm–2am on weekdays. Skip rate highest Monday mornings...

**Finance · habits**
> My last 60 days of transactions: Uber Eats £340, gym membership cancelled in March, coffee shops £180 (avg 2x daily), 3 Amazon orders placed past midnight, savings rate dropped from 18% to 4%...

**GitHub · work**
> My GitHub activity last 90 days: 847 commits total, 73% pushed between 9pm–1am, 0 commits on weekends, 14 force-pushes to main, 6 PRs open >30 days...

---

## VRAM Budget Reference

Calculated by `calculate_chronicle_vram_budget()` for a Llama-3-8B baseline with 5 agents at 4K context each:

| Precision | Model weights | KV cache (5 agents) | CUDA overhead | Safety buffer | Total | GPU |
|-----------|--------------|---------------------|---------------|---------------|-------|-----|
| FP32 | 26.1 GB | 10.0 GB | 2.0 GB | 8.0 GB | 46.1 GB | A100-80GB |
| FP16 | 13.0 GB | 10.0 GB | 2.0 GB | 8.0 GB | 33.0 GB | A100-40GB |
| INT8 | 6.5 GB | 10.0 GB | 2.0 GB | 8.0 GB | 26.5 GB | A100-40GB |
| INT4 | 3.3 GB | 10.0 GB | 2.0 GB | 8.0 GB | 23.3 GB | A100-40GB |

Per-agent tier assignments (INT4 for utility, FP16 for frontier) are locked in Session 11.2, which reduces total VRAM significantly below the single-tier FP16 estimate.

---

## Troubleshooting

**`GEMINI_API_KEY environment variable is not set`**
You need to export the key before running. Use `export $(cat .env | xargs)` or set it in your shell profile.

**`SSL: CERTIFICATE_VERIFY_FAILED`**
Chronicle uses `certifi` to handle macOS certificate verification. If you see this after a fresh install, run `pip install certifi` and retry.

**`[Errno 48] address already in use` on port 8000**
Another process is on port 8000. Kill it: `lsof -ti :8000 | xargs kill -9`

**UI shows "API offline — start python api.py"**
The API isn't running or crashed. Check the terminal running `api.py` for errors, then restart with `export $(cat .env | xargs) && python api.py`.

**0/5 agents succeeded**
Usually an SSL or API key issue. Run `python agent.py` to see the per-check failure reason from the verification suite.
