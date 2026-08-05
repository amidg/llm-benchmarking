# llm-benchmarking

Benchmark and health-check local LLMs behind any OpenAI-compatible endpoint —
llama.cpp / llama-swap, LiteLLM, Ollama, vLLM, or a hosted API.

Two stdlib-only Python scripts. **No `pip install`, no venv, no dependencies.**
Python 3.10+.

| script | what it does |
|---|---|
| `smoke.py` | fast health check of every tier and alias. Run after any config change. |
| `llmbench.py` | compare models: throughput, tool use, quality, long context, vision. |

## Quick start

```bash
git clone git@github.com:amidg/llm-benchmarking.git
cd llm-benchmarking

./smoke.py                                          # is everything alive?
./llmbench.py --models model-a model-b              # compare two models
```

## smoke.py

```bash
./smoke.py                 # full check: tiers, aliases, vision, approval shape
./smoke.py --quick         # just list what each tier serves and what is resident
./smoke.py --endpoint-only # skip the gateway, hit the inference server directly
```

Exits non-zero on failure, so it works in a git hook or a cron job.

Beyond "does it respond", it checks two things that break *silently*:

- **Empty content with a filled reasoning trace.** A thinking model given a small
  `max_tokens` spends the whole budget reasoning and returns `content: ""`. Looks like a
  dead endpoint; isn't.
- **The one-word approval shape.** Callers that send `max_tokens=16` and expect a single
  word (`APPROVE`/`DENY`/`ESCALATE`) get an empty string from a thinking model. This
  replays exactly that request.

A missing embedding alias is reported as *skip*, not failure.

## llmbench.py

```bash
# default suite (thinking, approval, perf, tools, quality)
./llmbench.py --models model-a model-b

# everything, including the slow long-context needle test
./llmbench.py --models a b --tests all --json out.json

# a secondary tier
./llmbench.py --endpoint http://localhost:8082/v1 --models small-model

# through a gateway that needs a key
./llmbench.py --endpoint http://localhost:4000/v1 --api-key "$LITELLM_MASTER_KEY" \
              --models local-main local-tiny

# CI gate — exits 1 on a tool-use or approval regression
./llmbench.py --models local-main --fail-on-regression
```

Useful flags: `--rounds N`, `--perf-lines N` (320 ≈ 6.5K tokens, 2800 ≈ 58K),
`--long-lines N` (1800 ≈ 50K), `--gen-tokens N`, `--timeout SECONDS`.

## Configuration

Defaults: `localhost:8081` and `:8082` for inference tiers, `localhost:4000` for the
gateway. Override with `--endpoint`.

`smoke.py` needs a gateway key. It looks, in order, at:

1. `$LITELLM_MASTER_KEY`
2. `$LITELLM_ENV` — path to a `litellm.env` file
3. `./litellm.env`
4. `litellm.env` beside the script, then one directory up

```bash
export LITELLM_MASTER_KEY=sk-...
# or point at an existing stack's env file
export LITELLM_ENV=~/Projects/dockerfiles/ai/litellm.env
```

## Reading the results

**Only compare models measured in the same run.** Absolute throughput drifts 20-30% with
load and thermals — one model measured anywhere from 467 to 641 tok/s prefill on identical
config. `llmbench.py` interleaves models across rounds so drift hits all of them equally.
The ratios are stable; the absolutes are not.

**`decode` and `deep` are different numbers, and the gap is the point.** `decode`
generates from a fresh short prompt (~0 context) — the `tg128`-equivalent that almost
every number quoted online means. `deep` continues from the long prefill, which is what an
agent turn actually does. On CPU-offloaded MoE these diverge sharply, so **judge an agent
tier on `deep`.** A config tuned on `decode` alone can look excellent and still feel slow.

**Prefill and decode trade against each other.** Changing quantization or offload usually
buys one and costs the other — one measured swap gained 49% decode and lost 22% prefill.
Judge with your workload's prompt-to-generation ratio, not one number.

**A 3/4 where a *different* case fails each run is a token budget problem, not a defect.**
Thinking models can spend an entire budget reasoning and return `content: ""` with
`finish_reason: length`, which scores identically to a wrong answer. `tools` and `quality`
budget 1200 tokens and report `empty` separately from a real miss. If you add a test, give
it a thinking-model-sized budget or it will measure reasoning length, not capability.

**A `quality` failure at `temperature > 0` may be sampling noise.** One 2B model answered
17×4 as 56 at `presence_penalty 1.5` and 68 at temperature 0. The battery pins
`temperature=0` for this reason.

## What each test catches

| test | catches |
|---|---|
| `thinking` | reasoning eating the token budget; **reasoning leaking into `content`** when a model uses tags the server does not parse |
| `approval` | the `max_tokens=16` one-word shape — a thinking model returns empty |
| `perf` | prefill and decode, at **two context depths** |
| `tools` | 4 cases: single call, selection among 4 schemas, tool-result round-trip, and correctly *not* calling a tool. One easy call passes on everything and discriminates nothing. |
| `quality` | arithmetic, JSON extraction, classification, terse output — at `temperature 0` |
| `longctx` | needle retrieval at 60% depth, plus prefill throughput at real depth |
| `vision` | a model can pass every text test and still **crash the server** on an image |

## Portability

Nothing here is specific to one stack — any OpenAI-compatible
`/v1/chat/completions`, `/v1/embeddings` and `/v1/models` will work. Two caveats:

- `perf` needs the server to return a `timings` object (llama.cpp does; most others don't).
- `smoke.py` assumes the alias names `local-main`, `local-vision`, `local-tiny`,
  `local-embed`. Edit `CHAT_ALIASES` / `EMBED_ALIASES` at the top of the file for a
  different naming scheme.

See [`AGENTS.md`](AGENTS.md) for detailed operating instructions aimed at LLMs and coding
agents.
