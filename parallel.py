#!/usr/bin/env python3
"""Concurrent benchmark — measure how throughput and latency scale with parallel requests.

Stdlib only — no pip install, no venv. Runs against any OpenAI-compatible endpoint.

    ./concurrent.py --models model-a model-b                          # default suite
    ./concurrent.py --models a b --conc-levels 1 4 16 64             # custom levels
    ./concurrent.py --models a b --requests 50 --conc-levels 5       # 5 rounds per level
    ./concurrent.py --models a b --interleave                        # test models concurrently
    ./concurrent.py --endpoint http://localhost:8082/v1 ...          # the iGPU tier
    ./concurrent.py --models a b --json out.json                     # machine-readable

WHY THIS EXISTS — single-request benchmarks don't tell you what happens when 32 agents
hit the server simultaneously. llama.cpp (threaded) and vLLM (continuous batching)
scale very differently. This script measures that.

    llama.cpp: throughput plateaus then falls as thread contention and CPU overhead
               dominate. TTFT climbs linearly.
    vLLM:      throughput grows with concurrency (continuous batching), then flattens
               when GPU compute saturates. TTFT stays flat or drops.

Exit code is 0 unless --fail-on-regression is given and throughput at high concurrency
drops below a threshold of single-request throughput.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

DEFAULT_ENDPOINT = "http://localhost:8081/v1"


# --------------------------------------------------------------------------- io
def post(endpoint: str, path: str, body: dict, api_key: str | None, timeout: int) -> dict:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except Exception:
            return {"error": {"message": f"HTTP {e.code}"}}
    except Exception as e:
        return {"error": {"message": f"{type(e).__name__}: {e}"}}


def chat(endpoint, model, messages, api_key, timeout, **kw) -> dict:
    body = {"model": model, "messages": messages,
            "chat_template_kwargs": {"reasoning_effort": "low"}, **kw}
    return post(endpoint, "/chat/completions", body, api_key, timeout)


def timings(resp: dict) -> dict:
    t = resp.get("timings") or {}
    # llama.cpp puts token counts in usage, not timings
    usage = resp.get("usage") or {}
    t["completion_tokens"] = usage.get("completion_tokens")
    t["prompt_tokens"] = usage.get("prompt_tokens")
    return t


def err(resp: dict) -> str | None:
    if "error" in resp:
        e = resp["error"]
        return str(e.get("message", e)) if isinstance(e, dict) else str(e)
    return None


# ---------------------------------------------------------------- test payloads
# Three payload types representing common agent workflows.
# Each is self-contained so requests are independent and safe to fire in parallel.

SHORT_PROMPT = [
    {"role": "system",
     "content": ("You are a customer support assistant. Classify the request below "
                 "and provide a brief response.")},
    {"role": "user",
     "content": ("My order #48291 hasn't arrived yet. It was supposed to come by "
                 "Friday. Can you check the status? The tracking number is "
                 "TRK-9928374-X and the estimated delivery was August 1st. "
                 "I need this resolved ASAP as I'm out of the product.")},
]

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"},
                                      "unit": {"type": "string",
                                               "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city", "unit"]}}},
    {"type": "function", "function": {
        "name": "search_email", "description": "Search the user's email",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "limit": {"type": "integer"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_calendar_event", "description": "Create a calendar event",
        "parameters": {"type": "object",
                       "properties": {"title": {"type": "string"},
                                      "iso_start": {"type": "string"},
                                      "duration_minutes": {"type": "integer"}},
                       "required": ["title", "iso_start", "duration_minutes"]}}},
]

TOOL_PROMPT = [
    {"role": "system",
     "content": ("You are an assistant with access to tools. Use them when needed "
                 "and summarize the results.")},
    {"role": "user",
     "content": ("I need help planning my day. First, check the weather in Tokyo for "
                 "tomorrow in celsius. Then search my email for any messages from "
                 "'conference@ai-summit.org' with limit 3. Finally, create a calendar "
                 "event called 'AI Summit Keynote' starting 2026-08-10T09:00:00Z "
                 "for 60 minutes. Summarize what you found and did.")},
]


def _doc(i: int) -> str:
    """Generate a single ~200-token synthetic research document."""
    return (
        f"=== Document {i} ===\n"
        f"This document discusses topic-{i % 12} in the context of modern "
        f"computing systems. The primary findings indicate that approach "
        f"{chr(65 + (i % 26))}-{i} achieves significantly better results "
        f"than baseline methods. Specifically, we observe a {30 + (i * 7) % 60}% "
        f"improvement in throughput and a {10 + (i * 3) % 40}% reduction in "
        f"error rates. The methodology involves combining techniques from "
        f"references [{i % 5 + 1}, {(i + 1) % 5 + 1}, {(i + 2) % 5 + 1}]. "
        f"Our experiments were conducted on a dataset of {10000 + (i * 500) % 90000} "
        f"samples collected across {3 + (i % 8)} different environments. "
        f"The statistical significance was verified using a two-tailed "
        f"t-test with p < 0.01. We also conducted ablation studies showing "
        f"that component A contributes {40 + (i * 11) % 50}% to the overall "
        f"gain while component B accounts for the remainder. The "
        f"computational complexity is O(n log n) where n is the input size, "
        f"making this approach scalable to large datasets. Memory usage "
        f"stays below {512 + (i * 64) % 2048}MB for typical workloads. "
        f"Edge cases were handled by implementing a fallback mechanism "
        f"that degrades gracefully. We believe this work opens new "
        f"directions for research in distributed systems and "
        f"high-performance computing.\n"
    )


def _longctx_prompt() -> list[dict]:
    """Generate a ~15K token prompt: 80 synthetic research documents."""
    docs = "\n\n".join(_doc(i) for i in range(1, 81))
    return [
        {"role": "system",
         "content": ("You are a technical analyst. Read the retrieved documents "
                     "carefully and provide a comprehensive summary with specific "
                     "details and citations from each document.")},
        {"role": "user",
         "content": (
             "Here are retrieved documents from our knowledge base. "
             "Provide a comprehensive summary covering the main themes, "
             "key findings, and specific technical details from each "
             "document. Cite document numbers when referencing findings.\n\n"
             + docs
             + "\n\nPlease summarize all 80 documents, covering the main "
             "themes and specific technical details from each."
         )},
    ]


# ---------------------------------------------------------------------- percentile
def pct(values: list[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


# ---------------------------------------------------------------- test payloads by type
def _get_payload(test_name: str) -> tuple[list, list | None, float]:
    """Return (messages, tools, temperature) for a test type."""
    if test_name == "short":
        return SHORT_PROMPT, None, 0.7
    elif test_name == "tools":
        return TOOL_PROMPT, TOOLS, 0.7
    elif test_name == "longctx":
        return _longctx_prompt(), None, 0.0
    raise ValueError(f"unknown test: {test_name}")


# ---------------------------------------------------------------- concurrent runner
def _single_request(
    endpoint: str, model: str, messages: list, api_key: str | None,
    timeout: int, tools: list | None, gen_tokens: int, temperature: float,
) -> dict:
    """Fire one request and measure its timing."""
    kw: dict[str, Any] = {"max_tokens": gen_tokens, "temperature": temperature}
    if tools:
        kw["tools"] = tools
        kw["tool_choice"] = "auto"
    t0 = time.monotonic()
    resp = chat(endpoint, model, messages, api_key, timeout, **kw)
    elapsed = time.monotonic() - t0
    t = timings(resp)
    ttft = 0.0
    if t.get("prompt_ms") and t.get("first_token_time"):
        ttft = t["first_token_time"] / 1000.0
    return {
        "ok": err(resp) is None,
        "error": err(resp),
        "elapsed": elapsed,
        "ttft": ttft,
        "prompt_tokens": t.get("prompt_n"),
        "completion_tokens": t.get("completion_tokens"),
    }


def _run_one_model_at_level(
    model: str, test_name: str, args: argparse.Namespace,
) -> dict:
    """Run all requests for one model at one concurrency level. Returns metrics."""
    messages, tools, temperature = _get_payload(test_name)
    conc = args.conc_level
    n_requests = args.requests
    results: list[dict] = []
    max_workers = max(conc, 1)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _single_request, args.endpoint, model, messages, args.api_key,
                args.timeout, tools, args.gen_tokens, temperature,
            ): i
            for i in range(n_requests)
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    total = len(results)
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    latencies = [r["elapsed"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r["ttft"] > 0]
    tps_per_request = [r["completion_tokens"] / r["elapsed"]
                       for r in ok if r["completion_tokens"] and r["elapsed"] > 0]

    total_tokens = sum(r["completion_tokens"] for r in ok if r["completion_tokens"])
    total_time = sum(r["elapsed"] for r in ok)
    throughput = total_tokens / total_time if total_time > 0 else 0.0

    return {
        "total_requests": total,
        "ok": len(ok),
        "errors": len(failed),
        "error_rate": len(failed) / total if total else 0.0,
        "throughput_tps": round(throughput, 2),
        "latency_p50": round(pct(latencies, 50), 3) if latencies else None,
        "latency_p90": round(pct(latencies, 90), 3) if latencies else None,
        "latency_p99": round(pct(latencies, 99), 3) if latencies else None,
        "latency_mean": round(statistics.mean(latencies), 3) if latencies else None,
        "ttft_p50": round(pct(ttfts, 50), 3) if ttfts else None,
        "ttft_p90": round(pct(ttfts, 90), 3) if ttfts else None,
        "per_request_tps_p50": round(pct(tps_per_request, 50), 2) if tps_per_request else None,
        "per_request_tps_p90": round(pct(tps_per_request, 90), 2) if tps_per_request else None,
    }


def _run_sequential(models, test_name, args, conc_levels) -> dict:
    """Sequential mode: run each model fully before moving to the next."""
    level_results: list[dict] = []
    for conc in conc_levels:
        args.conc_level = conc
        t0 = time.monotonic()
        print(f"    conc={conc:<4}  {args.requests} requests...", flush=True)
        metrics = _run_one_model_at_level(models[0], test_name, args)
        metrics["concurrency"] = conc
        metrics["p50"] = metrics.get("latency_p50")
        metrics["p90"] = metrics.get("latency_p90")
        metrics["p99"] = metrics.get("latency_p99")
        level_results.append(metrics)
        elapsed = time.time() - t0

        tp = metrics.get("throughput_tps", 0)
        p99 = metrics.get("latency_p99")
        err_rate = metrics.get("error_rate", 0)
        tt50 = metrics.get("ttft_p50")

        tp_s = f"{tp:6.1f} t/s" if tp > 0 else "     -"
        p99_s = f"{p99:.3f}s" if p99 is not None else "     -"
        err_s = f"{err_rate*100:.1f}%" if err_rate > 0 else "   0.0%"
        tt_s = f"{tt50:.3f}s" if tt50 is not None else "      -"

        print(f"      tp={tp_s}  p99={p99_s}  "
              f"errors={err_s}  ttft_p50={tt_s}  "
              f"[{elapsed:.0f}s]", flush=True)

    return {
        "levels": [l["concurrency"] for l in level_results],
        "levels_data": level_results,
    }


def _run_interleaved(models, test_name, args, conc_levels) -> dict:
    """Interleaved mode: at each concurrency level, run all models simultaneously.

    Uses a single ThreadPoolExecutor sized to handle all models' requests at once.
    This shares thermal/load drift between models, like llmbench.py rounds.
    """
    all_results: list[dict] = []

    for conc in conc_levels:
        args.conc_level = conc
        n_models = len(models)
        # Total workers = conc * n_models so each model gets its own concurrent slot
        total_workers = conc * n_models
        print(f"\n    conc={conc}  {args.requests} req/model across {n_models} models "
              f"({total_workers} workers)", flush=True)

        t0 = time.monotonic()
        model_results: dict[str, list[dict]] = {m: [] for m in models}

        with ThreadPoolExecutor(max_workers=total_workers) as pool:
            futures_by_model: dict[str, list] = {m: [] for m in models}

            for m in models:
                messages, tools, temperature = _get_payload(test_name)
                for req_i in range(args.requests):
                    fut = pool.submit(
                        _single_request, args.endpoint, m, messages,
                        args.api_key, args.timeout, tools,
                        args.gen_tokens, temperature,
                    )
                    futures_by_model[m].append(fut)

            # Collect results per model
            for m in models:
                for fut in futures_by_model[m]:
                    model_results[m].append(fut.result())

        wall = time.time() - t0

        # Aggregate results for each model at this level
        for m in models:
            results = model_results[m]
            total = len(results)
            ok = [r for r in results if r["ok"]]
            failed = [r for r in results if not r["ok"]]

            latencies = [r["elapsed"] for r in ok]
            ttfts = [r["ttft"] for r in ok if r["ttft"] > 0]
            tps_per_request = [r["completion_tokens"] / r["elapsed"]
                               for r in ok if r["completion_tokens"] and r["elapsed"] > 0]

            total_tokens = sum(r["completion_tokens"] for r in ok if r["completion_tokens"])
            total_time = sum(r["elapsed"] for r in ok)
            throughput = total_tokens / total_time if total_time > 0 else 0.0

            metrics = {
                "concurrency": conc,
                "p50": round(pct(latencies, 50), 3) if latencies else None,
                "p90": round(pct(latencies, 90), 3) if latencies else None,
                "p99": round(pct(latencies, 99), 3) if latencies else None,
                "throughput_tps": round(throughput, 2),
                "error_rate": len(failed) / total if total else 0.0,
                "ttft_p50": round(pct(ttfts, 50), 3) if ttfts else None,
                "ttft_p90": round(pct(ttfts, 90), 3) if ttfts else None,
            }
            all_results.append(metrics)

            tp = metrics["throughput_tps"]
            p99 = metrics["p99"]
            err_rate = metrics["error_rate"]
            tt50 = metrics["ttft_p50"]

            tp_s = f"{tp:6.1f} t/s" if tp > 0 else "     -"
            p99_s = f"{p99:.3f}s" if p99 is not None else "     -"
            err_s = f"{err_rate*100:.1f}%" if err_rate > 0 else "   0.0%"
            tt_s = f"{tt50:.3f}s" if tt50 is not None else "      -"

            print(f"      {m:<26} tp={tp_s}  p99={p99_s}  "
                  f"errors={err_s}  ttft_p50={tt_s}  [{wall:.0f}s]", flush=True)

    return {
        "levels": list(conc_levels),
        "levels_data": all_results,
    }


# ---------------------------------------------------------------- test definitions (for sequential runner)
def _test_short(ep, model, key, timeout, conc, n_req, gen_tok) -> dict:
    """Short prompt + generation. Chat/summarization workloads."""
    return _run_one_model_at_level(model, "short",
                                   type('Args', (), {
                                       'endpoint': ep, 'api_key': key,
                                       'timeout': timeout, 'conc_level': conc,
                                       'requests': n_req, 'gen_tokens': gen_tok,
                                   })())


def _test_tools(ep, model, key, timeout, conc, n_req, gen_tok) -> dict:
    """Tool-use scenario. Agent workflows with function calling."""
    return _run_one_model_at_level(model, "tools",
                                   type('Args', (), {
                                       'endpoint': ep, 'api_key': key,
                                       'timeout': timeout, 'conc_level': conc,
                                       'requests': n_req, 'gen_tokens': gen_tok,
                                   })())


def _test_longctx(ep, model, key, timeout, conc, n_req, gen_tok) -> dict:
    """Long context retrieval (~15K tokens). RAG workloads."""
    return _run_one_model_at_level(model, "longctx",
                                   type('Args', (), {
                                       'endpoint': ep, 'api_key': key,
                                       'timeout': timeout, 'conc_level': conc,
                                       'requests': n_req, 'gen_tokens': gen_tok,
                                   })())


CONCURRENT_TESTS = {
    "short":    lambda a, m: _test_short(a.endpoint, m, a.api_key, a.timeout,
                                          a.conc_level, a.requests, a.gen_tokens),
    "tools":    lambda a, m: _test_tools(a.endpoint, m, a.api_key, a.timeout,
                                          a.conc_level, a.requests, a.gen_tokens),
    "longctx":  lambda a, m: _test_longctx(a.endpoint, m, a.api_key, a.timeout,
                                            a.conc_level, a.requests, a.gen_tokens),
}
DEFAULT_CONCURRENT_TESTS = ["short", "tools", "longctx"]
DEFAULT_CONC_LEVELS = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------- ui
def _fmt(v, dec=1) -> str:
    if v is None:
        return "-"
    return f"{v:.{dec}f}"


def summary_table(results: dict) -> str:
    """Compact comparison: throughput and p99 at each concurrency level."""
    rows = []
    models = list(results)
    if not models:
        return "(no results)"

    conc_levels: list[int] = []
    for m in models:
        for t in results[m]:
            conc_levels.extend(results[m][t].get("levels", []))
    conc_levels = sorted(set(conc_levels))

    head = f"{'model':<26} {'test':<8}"
    for c in conc_levels:
        head += f" {'tp@{c}':>8} {'p99@{c}':>8}"
    rows.append(head)
    rows.append("-" * len(head))

    for m in models:
        for tname in sorted(results[m]):
            tdata = results[m][tname]
            row = f"{m:<26} {tname:<8}"
            for c in conc_levels:
                lvl = next((l for l in tdata.get("levels_data", [])
                            if l["concurrency"] == c), None)
                if lvl:
                    row += (f" {lvl['throughput_tps']:8.1f} "
                            f"{_fmt(lvl['p99'], 3):>8}")
                else:
                    row += " " * 17
            rows.append(row)
    return "\n".join(rows)


def detail_section(model: str, test_name: str, data: dict) -> str:
    """Detailed metrics for one model + test type."""
    lines = [f"\n  {model} / {test_name}:"]
    lines.append(f"    conc    rps     p50     p90     p99  err%  ttft_p50  ttft_p90")
    lines.append(f"    {'─' * 66}")
    for lvl in data.get("levels_data", []):
        c = lvl["concurrency"]
        tp = f"{lvl['throughput_tps']:6.1f}"
        p50 = _fmt(lvl["p50"], 3)
        p90 = _fmt(lvl["p90"], 3)
        p99 = _fmt(lvl["p99"], 3)
        er = f"{lvl['error_rate']*100:5.1f}" if lvl["error_rate"] > 0 else "  0.0"
        tt50 = _fmt(lvl.get("ttft_p50"), 3)
        tt90 = _fmt(lvl.get("ttft_p90"), 3)
        lines.append(f"    {c:<7} {tp}   {p50}   {p90}   {p99}  {er}%  {tt50}  {tt90}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Concurrent benchmark — throughput and latency under parallel load.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "EXAMPLES\n"
            "  ./parallel.py --models a b\n"
            "  ./parallel.py --models a b --conc-levels 1 2 4 8 16\n"
            "  ./parallel.py --models a --requests 100 --conc-levels 8\n"
            "  ./parallel.py --models a --tests short tools --gen-tokens 256\n"
            "  ./parallel.py --models a b --interleave  # test models concurrently\n"
        ),
    )
    ap.add_argument("--models", nargs="+", required=True,
                    help="model names to benchmark")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help=f"default {DEFAULT_ENDPOINT}")
    ap.add_argument("--api-key", default=None, help="API key (needed via LiteLLM gateway)")
    ap.add_argument("--tests", nargs="+", default=DEFAULT_CONCURRENT_TESTS,
                    choices=list(CONCURRENT_TESTS),
                    help="which test types to run (default: short tools longctx)")
    ap.add_argument("--conc-levels", nargs="+", type=int, default=None,
                    help="concurrency levels to test (default: 1 2 4 8 16 32)")
    ap.add_argument("--requests", type=int, default=20,
                    help="requests per concurrency level per test (default: 20)")
    ap.add_argument("--gen-tokens", type=int, default=256,
                    help="max generation tokens per request (default: 256)")
    ap.add_argument("--timeout", type=int, default=300,
                    help="per-request timeout in seconds (default: 300)")
    ap.add_argument("--interleave", action="store_true",
                    help="run models concurrently at each level (shares thermal "
                         "drift, like llmbench.py rounds)")
    ap.add_argument("--json", metavar="PATH", help="write raw results here")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 1 if throughput at highest concurrency drops "
                         "below 50%% of single-request throughput")

    a = ap.parse_args()

    conc_levels = a.conc_levels if a.conc_levels else DEFAULT_CONC_LEVELS
    conc_levels = sorted(set(c for c in conc_levels if c > 0))
    if not conc_levels:
        print("ERROR: no valid concurrency levels specified", file=sys.stderr)
        return 1

    results: dict[str, dict[str, dict]] = {m: {} for m in a.models}

    for test_name in a.tests:
        print(f"\n### {test_name}", flush=True)

        if a.interleave:
            print(f"\n  (interleaved: {len(a.models)} models, {len(conc_levels)} "
                  f"levels, {a.requests} requests each)", flush=True)
            # Interleaved: run all models together once per test type
            level_data = _run_interleaved(a.models, test_name, a, conc_levels)
            for m in a.models:
                results[m][test_name] = level_data
        else:
            # Sequential: each model runs independently
            for m in a.models:
                level_data = _run_sequential([m], test_name, a, conc_levels)
                results[m][test_name] = level_data

    # ---- output
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_table(results))

    for m in a.models:
        for tname in results[m]:
            print(detail_section(m, tname, results[m][tname]))

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nraw results -> {a.json}")

    if a.fail_on_regression:
        for m in a.models:
            for tname in results[m]:
                lvl_data = results[m][tname]["levels_data"]
                if len(lvl_data) < 2:
                    continue
                single_tp = next(
                    (l["throughput_tps"] for l in lvl_data if l["concurrency"] == 1),
                    None,
                )
                if single_tp is None or single_tp == 0:
                    continue
                highest = lvl_data[-1]
                if highest["throughput_tps"] < single_tp * 0.5:
                    print(
                        f"\nREGRESSION: {m}/{tname} throughput at "
                        f"conc={highest['concurrency']} is "
                        f"{highest['throughput_tps']:.1f} t/s "
                        f"(< 50% of single-request {single_tp:.1f} t/s)",
                        flush=True,
                    )
                    return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
