# Benchmarks

How the model and prompt choices ended up the way they did. The Pi 5 is the
slow part, so I measured this stuff instead of guessing and hoping.

- **[`benchmark_llm.py`](benchmark_llm.py)** — the contract test. Sends a candidate Ollama model the *exact* production system prompt and options, then checks every reply for three things: valid JSON, only real actions, and the right persona — while timing it. Run this before you swap a new local model into `zeus.py`.
- **[`llm_bench.py`](llm_bench.py)** — the latency test. Times `has_internet()`, an OpenAI call, and an Ollama call side by side, with token counts.

```bash
export OPENAI_API_KEY=sk-...      # only needed for the OpenAI comparison
python3 benchmarks/benchmark_llm.py
python3 benchmarks/llm_bench.py
```

What these turned up is basically the changelog at the top of `zeus.py`: shorter
prompts, capping `num_predict`, keeping the model warm in RAM with
`keep_alive=-1`, and caching the internet check so it isn't re-run every turn.
