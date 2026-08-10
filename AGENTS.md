# AGENTS.md — llm-benchmarking

Operating instructions for LLMs and coding agents working in this repo. `README.md` is the
human quick-start; this file is how to run the tools correctly and how to avoid drawing
wrong conclusions from them.

## What this repo is

Two stdlib-only Python scripts that benchmark and health-check LLMs behind an
OpenAI-compatible endpoint. **No dependencies, no venv, no build step.** Python 3.10+.

```
smoke.py        health check — tiers, aliases, vision, the one-word approval shape
llmbench.py     comparison   — throughput, tool use, quality, long context, vision
parallel.py     scaling    — throughput/latency under parallel load (llama.cpp vs vLLM)
```

Both are executable and self-documenting (`--help`). Neither writes to the system; the only
side effect is loading models on the server under test.

## Rules

1. **Never quote a number from a run that measured only one model.** Absolute throughput
   drifts 20-30% with load and thermals. Only same-run comparisons mean anything.
2. **Judge agent tiers on `deep`, not `decode`.** They are different measurements (below).
3. **Vision must be tested with a real image.** A model can pass every text test and still
   crash the server on an image.
4. **Do not lower a test's `max_tokens` to save time.** Thinking models return empty
   content when the budget runs out, which scores as a wrong answer.
5. **Report what failed, not just the score.** `3/4` is meaningless without which case.
6. **Never edit a config to make a benchmark pass.** Fix the config for the right reason,
   then re-measure.

## Running

```bash
./smoke.py                                        # after any config or container change
./smoke.py --quick                                # no generation, just what is loaded
./smoke.py --endpoint-only                        # skip the gateway

./llmbench.py --models a b                        # default suite
./llmbench.py --models a b --tests perf --rounds 3
./llmbench.py --models a b --tests all --json out.json
./llmbench.py --models a --fail-on-regression     # exits 1 on tools/approval regression

./parallel.py --models a b                        # default suite
./parallel.py --models a b --conc-levels 1 4 16   # custom levels
./parallel.py --models a b --interleave           # test models concurrently
./parallel.py --models a b --tests short tools    # subset of tests
./parallel.py --models a --fail-on-regression     # exits 1 on throughput regression
```

Tests: `thinking`, `approval`, `perf`, `tools`, `quality`, `longctx`, `vision`, or `all`.
Default suite is `thinking approval perf tools quality`.

Flags that matter: `--endpoint`, `--api-key`, `--rounds`, `--perf-lines`, `--long-lines`,
`--gen-tokens`, `--timeout`, `--json`.

**Long runs belong in the background.** A full `--tests all` across two large models takes
20-40 minutes, and model swaps dominate. Start it in the background and poll the output
file rather than blocking.

## Configuration

Defaults are `localhost:8081` / `:8082` (inference tiers) and `localhost:4000` (gateway).

`smoke.py` resolves the gateway key in this order:

1. `$LITELLM_MASTER_KEY`
2. `$LITELLM_ENV` — explicit path to a `litellm.env`
3. `./litellm.env`
4. `litellm.env` beside the script, then one directory up

If the key is missing, `smoke.py` fails fast with a clear message — that is not a stack
failure.

Alias names (`local-main`, `local-vision`, `local-tiny`, `local-embed`) are hardcoded in
`CHAT_ALIASES` / `EMBED_ALIASES` at the top of `smoke.py`. Edit those for a different
scheme. A missing embedding alias reports *skip*, not failure.

## Interpreting results — the traps

### `decode` vs `deep`

- **`decode`** — generation from a fresh short prompt, ~0 context. Equivalent to
  `llama-bench`'s `tg128`, and what almost every figure quoted online means.
- **`deep`** — generation continuing from the ~6.5K-token prefill. What an agent turn
  actually does, since agents send a large system prompt every time.

Both come from a single `perf` round. On CPU-offloaded MoE they diverge sharply — one
measured config showed 36.6 shallow against 31.4 deep. **Mixing these up has cost real
time:** a stack once had a hand-measured deep figure sitting in a table next to
`llmbench` shallow numbers, which made an already-fast tier look 2x slower than it was and
triggered an entire optimisation effort against a non-existent problem.

### Empty content is not a wrong answer

Thinking models emit reasoning first. If `max_tokens` runs out mid-reasoning the response
has `content: ""` and `finish_reason: length`. A naive scorer counts that as a failure.

**Signature: a 3/4 where a *different* case fails on each run.** That is budget noise, not
a defect. Measured on one model with an identical prompt: 830 reasoning chars → passed,
1649 → failed at 400 tokens; at 1200 it passed every time, once needing 482 completion
tokens.

`tools` and `quality` budget 1200 and report `empty` separately from a real miss. Any test
you add must do the same.

### Prompt caching corrupts repeated measurements

Sending the *same* prompt twice to the same loaded model hits the server's prompt cache —
the second call reports a tiny `prompt_n` (e.g. `ptok=4`) and a meaningless prefill number.
`llmbench.py` avoids this by alternating models across rounds, which flushes the cache. If
you write a custom single-model loop, **only round 1 is a valid prefill measurement.**

### Sampling noise at `temperature > 0`

`quality` pins `temperature=0`. `tools` does not, because tool calling should be robust at
the server's default sampling — so a single `tools` failure may be stochastic. Re-run
before concluding a regression, and probe the specific case directly.

### Prefill is usually GPU-bound

If you are tuning CPU-side settings (threads, affinity, memory placement), prefill will
barely move while decode does. Do not tune on a prefill number.

## Adding a test

Tests live in `llmbench.py` as `t_<name>(ep, model, key, timeout)` returning a dict, wired
into the `TESTS` map. Requirements:

- **Budget 1200+ tokens** for anything that must read a model's answer.
- **Distinguish empty from wrong** — set an `empty` flag when `content` is blank.
- **Discriminate.** A test everything passes is worse than no test; it adds runtime and
  false confidence. The `tools` battery has four cases precisely because one easy call
  passed on every model tried.
- Add a row to the "what each test catches" table in `README.md` describing the bug it
  exists to catch, not what it does.

## Troubleshooting

**Connection reset, or a container reporting `(unhealthy)`.** `localhost` resolves to
`::1` first; servers bound to IPv4 `0.0.0.0` accept then reset through a container runtime's
IPv6 forward. Use `127.0.0.1`.

**Everything times out right after starting the stack.** Gateways can take 30-60s to become
ready and return empty bodies until then. Poll the gateway's health endpoint first. On
Intel/SYCL backends a cold kernel-cache compile can take minutes on the first request.

**A model "hangs" on load.** Large GGUFs take time and log formats change between upstream
versions — greps for old strings make a healthy server look dead. Check the process is
alive and memory is climbing before assuming a failure.

**`perf` returns `no timings returned`.** The server does not emit a `timings` object.
llama.cpp does; most other backends do not. Other tests still work.

**Benchmarks are slower than expected.** Model swaps dominate multi-model runs. Reduce
`--rounds`, or unload between runs if the server exposes an unload endpoint.
