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