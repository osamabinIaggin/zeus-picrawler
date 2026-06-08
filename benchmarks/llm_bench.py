#!/usr/bin/env python3
"""
LLM Latency Benchmark for Zeus
Times: has_internet(), OpenAI call, Ollama call
"""

import os
import json
import time
import requests

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # set via env, never hard-code
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL   = "gpt-4o-mini"
OLLAMA_URL     = "http://localhost:11434/api/generate"
TEXT_MODEL     = "llama3.2:3b"
TEST_QUERY     = "what is 2 plus 2"

SYSTEM_PROMPT = """\
You are an AI spider robot named Zeus (PiCrawler). You have four legs, a camera, \
and an ultrasonic distance sensor. You interact with people through conversation.

You MUST always respond with a single valid JSON object and NOTHING else.

Example:
{"actions": ["wave_hand"], "answer": "Hello! I am Zeus, your friendly spider robot!"}

Valid actions (use exact spelling):
sit, stand, wave_hand, shake_hand, fighting, excited, play_dead, nod, shake_head, \
look_left, look_right, look_up, look_down, warm_up, push_up

Rules:
- "actions" MUST be a JSON array, can be empty []
- "answer" MUST be a short string reply
- No extra keys, no markdown, no code fences, no text outside the JSON

Tone: cheerful, witty, playful robotic humour.\
"""

def sep(label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print('='*60)

def tick():
    return time.perf_counter()

def elapsed(t0):
    return time.perf_counter() - t0

# ── 1. Benchmark has_internet() ───────────────────────────────────────────────
sep("BENCHMARK 1: has_internet() — 5 iterations")
times = []
for i in range(5):
    t0 = tick()
    try:
        requests.head("https://api.openai.com", timeout=3)
        ok = True
    except Exception:
        ok = False
    dt = elapsed(t0)
    times.append(dt)
    print(f"  Run {i+1}: {dt*1000:.0f} ms  (internet={ok})")

avg_internet = sum(times) / len(times)
print(f"\n  Average: {avg_internet*1000:.0f} ms")
print(f"  Min:     {min(times)*1000:.0f} ms")
print(f"  Max:     {max(times)*1000:.0f} ms")

# ── 2. Benchmark OpenAI call ──────────────────────────────────────────────────
sep("BENCHMARK 2: OpenAI gpt-4o-mini — 3 iterations")
openai_times = []
for i in range(3):
    t0 = tick()
    try:
        resp = requests.post(OPENAI_URL, json={
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": TEST_QUERY},
            ],
            "max_tokens": 150,
        }, headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }, timeout=15)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        dt = elapsed(t0)
        openai_times.append(dt)
        # Show token usage if available
        usage = resp.json().get("usage", {})
        print(f"  Run {i+1}: {dt*1000:.0f} ms  | tokens: prompt={usage.get('prompt_tokens','?')} completion={usage.get('completion_tokens','?')}")
        print(f"    Response: {raw[:120]}")
    except Exception as e:
        dt = elapsed(t0)
        openai_times.append(dt)
        print(f"  Run {i+1}: FAILED in {dt*1000:.0f} ms — {e}")

if openai_times:
    avg_openai = sum(openai_times) / len(openai_times)
    print(f"\n  Average: {avg_openai*1000:.0f} ms")
    print(f"  Min:     {min(openai_times)*1000:.0f} ms")
    print(f"  Max:     {max(openai_times)*1000:.0f} ms")

# ── 3. Benchmark Ollama call ──────────────────────────────────────────────────
sep("BENCHMARK 3: Ollama llama3.2:3b — 3 iterations")
ollama_times = []
for i in range(3):
    t0 = tick()
    try:
        prompt = SYSTEM_PROMPT + "\n\nHuman: " + TEST_QUERY + "\nZeus:"
        resp = requests.post(OLLAMA_URL, json={
            "model": TEXT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 130},
        }, timeout=180)
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
        dt = elapsed(t0)
        ollama_times.append(dt)
        data = resp.json()
        prompt_tok = data.get("prompt_eval_count", "?")
        completion_tok = data.get("eval_count", "?")
        eval_dur_ns = data.get("eval_duration", 0)
        eval_dur_s = eval_dur_ns / 1e9 if eval_dur_ns else 0
        load_dur_ns = data.get("load_duration", 0)
        load_dur_s = load_dur_ns / 1e9 if load_dur_ns else 0
        print(f"  Run {i+1}: {dt*1000:.0f} ms total")
        print(f"    load: {load_dur_s*1000:.0f} ms  |  eval: {eval_dur_s*1000:.0f} ms  |  tokens: prompt={prompt_tok} completion={completion_tok}")
        print(f"    Response: {raw[:120]}")
    except Exception as e:
        dt = elapsed(t0)
        ollama_times.append(dt)
        print(f"  Run {i+1}: FAILED in {dt*1000:.0f} ms — {e}")

if ollama_times:
    avg_ollama = sum(ollama_times) / len(ollama_times)
    print(f"\n  Average: {avg_ollama*1000:.0f} ms")
    print(f"  Min:     {min(ollama_times)*1000:.0f} ms")
    print(f"  Max:     {max(ollama_times)*1000:.0f} ms")

# ── 4. Summary ────────────────────────────────────────────────────────────────
sep("SUMMARY")
print(f"  has_internet() avg:   {avg_internet*1000:.0f} ms  (called on EVERY LLM request!)")
if openai_times:
    print(f"  OpenAI avg:           {avg_openai*1000:.0f} ms")
    print(f"  OpenAI + internet():  {(avg_openai + avg_internet)*1000:.0f} ms  (current total)")
if ollama_times:
    print(f"  Ollama avg:           {avg_ollama*1000:.0f} ms")
    print(f"  Ollama + internet():  {(avg_ollama + avg_internet)*1000:.0f} ms  (current total)")
print(f"\n  => Caching internet status (60s) saves ~{avg_internet*1000:.0f} ms per request")
