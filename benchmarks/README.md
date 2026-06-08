# Benchmarks

How the model and prompt choices in Zeus were made. The Pi 5 is the bottleneck,
so every prompt token and model swap was measured rather than guessed.

| File | What it measures |
|------|------------------|
| [`benchmark_llm.py`](benchmark_llm.py) | **Contract test:** sends the *exact* production system prompt + options to a candidate Ollama model and checks, per prompt, that the reply is (1) valid JSON, (2) uses only real actions, and (3) keeps the persona — while timing latency. Use this to vet a new local model before putting it in `zeus.py`. |
| [`llm_bench.py`](llm_bench.py) | **Latency test:** times `has_internet()`, an OpenAI call, and an Ollama call head-to-head, with token accounting. |

## Running

```bash
export OPENAI_API_KEY=sk-...      # only needed for the OpenAI comparison in llm_bench.py
python3 benchmarks/benchmark_llm.py
python3 benchmarks/llm_bench.py
```

> Findings from these benchmarks drove the real latency wins in `zeus.py`:
> shorter prompts, `num_predict` caps, `keep_alive=-1` to keep the model warm in
> RAM, and caching the internet check.
