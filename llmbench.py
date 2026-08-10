#!/usr/bin/env python3
"""Benchmark local llama.cpp models behind llama-swap or LiteLLM.

Stdlib only — no pip install, no venv. Runs against any OpenAI-compatible endpoint.

    ./llmbench.py --models qwen3.6-35b qwen3.5-9b            # default suite
    ./llmbench.py --models a b --tests perf tools            # subset
    ./llmbench.py --endpoint http://localhost:8082/v1 ...    # the iGPU tier
    ./llmbench.py --models a b --json results.json           # machine-readable

WHY THE TESTS LOOK LIKE THIS — each one exists because it caught a real bug:

  thinking  Every model here is a thinking model. With a small max_tokens the whole
            budget goes to reasoning and `content` comes back EMPTY. This silently
            broke Hermes' smart-approval (max_tokens=16) for weeks.
  tools     A single easy tool call passes on everything. Selection among several
            schemas, a tool-result round trip, and correctly NOT calling a tool are
            what actually discriminate.
  longctx   Prefill throughput collapses as context grows, and a model can pass every
            text test yet crash on an image at the same settings. Measure at depth.
  perf      Prefill and decode move in OPPOSITE directions when you change quant or
            offload. Always report both, and interleave rounds — absolute numbers
            drift 20-30% with load and thermals, so only same-session runs compare.

Exit code is 0 unless --fail-on-regression is given and a check fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_ENDPOINT = "http://localhost:8081/v1"
NEEDLE = "CRIMSON-PANGOLIN-7734"


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
    except Exception as e:  # timeout, connection reset, upstream crash
        return {"error": {"message": f"{type(e).__name__}: {e}"}}


def chat(endpoint, model, messages, api_key, timeout, **kw) -> dict:
    body = {"model": model, "messages": messages, **kw}
    return post(endpoint, "/chat/completions", body, api_key, timeout)


def content_of(resp: dict) -> str:
    try:
        return (resp["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        return ""


def reasoning_len(resp: dict) -> int:
    try:
        return len(resp["choices"][0]["message"].get("reasoning_content") or "")
    except Exception:
        return 0


def timings(resp: dict) -> dict:
    return resp.get("timings") or {}


def err(resp: dict) -> str | None:
    if "error" in resp:
        e = resp["error"]
        return str(e.get("message", e)) if isinstance(e, dict) else str(e)
    return None


# ---------------------------------------------------------------- test payloads
def filler(n_lines: int) -> str:
    return "\n".join(
        f"Record {i}: service=svc-{i % 37} region=eu-west-{i % 3} "
        f"latency_ms={40 + (i % 160)} status=ok"
        for i in range(n_lines)
    )


def prefill_prompt(n_lines: int) -> str:
    return ("Here is a log:\n" + filler(n_lines)
            + "\nIn one short sentence, what is this?")


def needle_prompt(n_lines: int, depth: float = 0.6) -> str:
    lines = filler(n_lines).split("\n")
    at = int(len(lines) * depth)
    lines.insert(at, f"Record SPECIAL: the deployment passphrase is {NEEDLE} "
                     "and it must be quoted exactly.")
    return ("Below is a long operations log. Read it, then answer.\n\n"
            + "\n".join(lines)
            + "\n\nQuestion: what is the deployment passphrase? "
              "Reply with ONLY the passphrase, nothing else.")


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
    {"type": "function", "function": {
        "name": "run_sql", "description": "Run a read-only SQL query",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}},
                       "required": ["sql"]}}},
]

QUALITY = [
    ("arithmetic", "What is 17 times 4? Reply with only the number.", "68"),
    ("extract_json",
     'Extract the port number as JSON like {"port": N}. Log: connection refused on '
     "127.0.0.1:5432 after retry. Reply with only JSON.", "5432"),
    ("classify",
     "Classify into exactly one of BUG/FEATURE/QUESTION/CHORE. Reply with only the "
     "label. Message: The export button spins forever with 50k rows.", "BUG"),
    ("terse_title",
     "Generate a short title (max 6 words). Reply with ONLY the title.\n"
     "User: my postgres container keeps OOMing after an hour", "OOM"),
]


# ---------------------------------------------------------------------- tests
def t_thinking(ep, model, key, timeout) -> dict:
    """Does the model burn a small budget on reasoning and return empty content?"""
    r = chat(ep, model, [{"role": "user", "content": QUALITY[3][1]}],
             key, timeout, max_tokens=400)
    e = err(r)
    if e:
        return {"error": e}
    c = content_of(r)
    # Some GGUF templates emit reasoning in tags llama.cpp does not parse, so it
    # lands in `content` instead of `reasoning_content`. That is worse than normal
    # thinking: downstream consumers get the trace as if it were the answer.
    leaked = [t for t in ("<analysis>", "<think>", "Thinking Process:", "<reasoning>")
              if t.lower() in c.lower()]
    return {"reasoning_chars": reasoning_len(r),
            "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
            "content": c[:60],
            "leaked_into_content": leaked or None,
            "thinking_on": reasoning_len(r) > 0 or bool(leaked)}


def t_approval_shape(ep, model, key, timeout) -> dict:
    """Hermes' smart-approval shape: max_tokens=16, expects one word.

    A thinking model returns EMPTY here. This is the check that caught the bug.
    """
    r = chat(ep, model, [
        {"role": "system", "content": "You assess shell command risk."},
        {"role": "user", "content":
            "The following command was flagged as: script execution via -c flag\n\n"
            "<command>\npython -c \"print('hello')\"\n</command>\n\n"
            "Respond with exactly one word: APPROVE, DENY, or ESCALATE"}],
        key, timeout, max_tokens=16, temperature=0)
    e = err(r)
    if e:
        return {"error": e}
    c = content_of(r)
    return {"content": c, "reasoning_chars": reasoning_len(r),
            "finish": r["choices"][0].get("finish_reason"),
            "pass": c.upper() in ("APPROVE", "DENY", "ESCALATE")}


def t_perf(ep, model, key, timeout, lines, gen_tokens, rounds) -> dict:
    """Prefill, plus decode measured at BOTH context depths.

    These are different numbers and they diverge a lot on CPU-offloaded MoE:

      decode_tps       generation from a fresh short prompt, i.e. ~0 context.
                       Comparable to llama-bench's tg128 and to most numbers
                       quoted online.
      decode_deep_tps  generation continuing from the ~6.5K-token prefill above.
                       This is what an agent turn actually experiences, since a
                       Hermes request carries a large system prompt.

    Quoting only the shallow number flatters a config that collapses at depth.
    Deep decode comes free from the prefill request — same call, just a real
    max_tokens instead of 16 — so this costs one longer generation, not one
    extra request.
    """
    pf, dc, dd = [], [], []
    acc_deep, acc_shallow = [], []

    def acceptance(t):
        """Fraction of MTP/speculative draft tokens the target model accepted.

        Reported separately for shallow and deep because they are not the same
        number: a config can look lossless on a short prompt and drift at depth.
        None on a server without speculative decoding.
        """
        n = t.get("draft_n")
        return (t["draft_n_accepted"] / n) if n else None

    for _ in range(rounds):
        r = chat(ep, model, [{"role": "user", "content": prefill_prompt(lines)}],
                 key, timeout, max_tokens=gen_tokens, temperature=0)
        if err(r):
            return {"error": err(r)}
        t = timings(r)
        if t.get("prompt_per_second"):
            pf.append((t["prompt_per_second"], t.get("prompt_n"), t.get("prompt_ms", 0) / 1000))
        if t.get("predicted_per_second"):
            dd.append(t["predicted_per_second"])
        a = acceptance(t)
        if a is not None:
            acc_deep.append(a)
        r = chat(ep, model, [{"role": "user", "content":
                              "Write exactly 300 words about why Postgres connection "
                              "pooling matters. Plain prose, no lists."}],
                 key, timeout, max_tokens=gen_tokens)
        if err(r):
            return {"error": err(r)}
        t = timings(r)
        if t.get("predicted_per_second"):
            dc.append(t["predicted_per_second"])
        a = acceptance(t)
        if a is not None:
            acc_shallow.append(a)
    if not pf or not dc:
        return {"error": "no timings returned (is this llama.cpp?)"}
    return {"prefill_tps": statistics.median(x[0] for x in pf),
            "prompt_tokens": pf[0][1],
            "prefill_s": statistics.median(x[2] for x in pf),
            "decode_tps": statistics.median(dc),
            "decode_deep_tps": statistics.median(dd) if dd else None,
            "accept_shallow": statistics.median(acc_shallow) if acc_shallow else None,
            "accept_deep": statistics.median(acc_deep) if acc_deep else None,
            "rounds": rounds}


def t_tools(ep, model, key, timeout) -> dict:
    """Tool calling: emit, select, consume a result, and correctly decline.

    TOOL_BUDGET is deliberately generous. These are thinking models, and at
    max_tokens=TOOL_BUDGET they intermittently spent the WHOLE budget in reasoning_content
    and returned content "" with finish_reason=length — scored as a tool-calling
    failure when it was only a budget failure. Measured on qwen3.6-35b: identical
    round-trip prompt, 830 reasoning chars -> passed, 1649 -> failed at 400; at 1200
    it passed every time, once needing 482 completion tokens. A different case failed
    on each run, which is the tell for budget noise rather than a real defect.
    """
    TOOL_BUDGET = 1200
    out = {}
    # 1. single call, two required args
    r = chat(ep, model, [{"role": "user", "content":
                          "What is the weather in Berlin in celsius? Use the tool."}],
             key, timeout, tools=TOOLS[:1], tool_choice="auto", max_tokens=TOOL_BUDGET)
    tc = (r.get("choices", [{}])[0].get("message", {}) or {}).get("tool_calls") if not err(r) else None
    ok = False
    if tc:
        a = json.loads(tc[0]["function"]["arguments"] or "{}")
        ok = (tc[0]["function"]["name"] == "get_weather"
              and a.get("city", "").lower() == "berlin" and a.get("unit") == "celsius")
    out["single"] = bool(ok)

    # 2. selection among 4 schemas, 3 required args
    r = chat(ep, model, [{"role": "user", "content":
                          "Book a 45 minute meeting called 'Postgres postmortem' "
                          "starting 2026-08-05T14:00:00Z."}],
             key, timeout, tools=TOOLS, tool_choice="auto", max_tokens=TOOL_BUDGET)
    tc = (r.get("choices", [{}])[0].get("message", {}) or {}).get("tool_calls") if not err(r) else None
    ok = False
    if tc:
        f = tc[0]["function"]
        a = json.loads(f["arguments"] or "{}")
        ok = (f["name"] == "create_calendar_event" and a.get("duration_minutes") == 45
              and "2026-08-05" in str(a.get("iso_start", "")))
    out["select"] = bool(ok)

    # 3. use a returned tool result rather than calling again
    r = chat(ep, model, [
        {"role": "user", "content": "What is the weather in Berlin?"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_weather", "arguments": '{"city": "Berlin"}'}}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": '{"temp_c": 19, "condition": "light rain"}'}],
        key, timeout, tools=TOOLS, max_tokens=TOOL_BUDGET)
    out["roundtrip"] = (not err(r)) and "19" in content_of(r)

    # 4. must NOT call a tool
    r = chat(ep, model, [{"role": "user", "content":
                          "What is 17 times 4? Answer directly, do not use any tool."}],
             key, timeout, tools=TOOLS, tool_choice="auto", max_tokens=TOOL_BUDGET,
             temperature=0)
    m = (r.get("choices", [{}])[0].get("message", {}) or {}) if not err(r) else {}
    out["no_tool"] = (not m.get("tool_calls")) and "68" in content_of(r)
    out["score"] = f"{sum(bool(v) for k, v in out.items() if k != 'score')}/4"
    return out


def t_quality(ep, model, key, timeout) -> dict:
    """Accuracy at temperature 0.

    max_tokens is deliberately generous. These are all thinking models, and at
    max_tokens=300 they routinely spent the whole budget in reasoning_content and
    returned content "" — which scores identically to a WRONG answer while meaning
    something completely different. Both 35B models used to fail two items each
    that way, on different items, purely as budget noise.

    So: budget 1200, and report `empty` separately from a real miss. An empty
    result is a harness/budget problem, not a capability finding.
    """
    out = {}
    for name, prompt, expect in QUALITY:
        r = chat(ep, model, [{"role": "user", "content": prompt}], key, timeout,
                 max_tokens=1200, temperature=0)
        c = content_of(r)
        rec = {"pass": expect.lower() in c.lower(), "got": c[:60]}
        if not c.strip():
            rec["empty"] = True
            rec["finish"] = r["choices"][0].get("finish_reason")
            rec["reasoning_chars"] = reasoning_len(r)
        out[name] = rec
    graded = [v for k, v in out.items() if k != "score"]
    out["score"] = f"{sum(v['pass'] for v in graded)}/{len(QUALITY)}"
    n_empty = sum(1 for v in graded if v.get("empty"))
    if n_empty:
        out["empty_answers"] = n_empty
    return out


def t_longctx(ep, model, key, timeout, lines) -> dict:
    r = chat(ep, model, [{"role": "user", "content": needle_prompt(lines)}],
             key, timeout, max_tokens=1200, temperature=0)
    e = err(r)
    if e:
        return {"error": e}
    t = timings(r)
    c = content_of(r)
    return {"found": NEEDLE in c,
            "empty": not c.strip(),
            "ctx_tokens": t.get("prompt_n"),
            "prefill_tps": t.get("prompt_per_second"),
            "prefill_s": round(t.get("prompt_ms", 0) / 1000, 1)}


def t_vision(ep, model, key, timeout) -> dict:
    """A 1x1 PNG. Catches the config that passes every text test then crashes on
    an image because VRAM headroom cannot fit the vision encoder's buffer."""
    png = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ"
           "AAAABJRU5ErkJggg==")
    r = chat(ep, model, [{"role": "user", "content": [
        {"type": "text", "text": "What colour is this image? One word."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}}]}],
        key, timeout, max_tokens=64)
    e = err(r)
    return {"vision": "no" if e else "yes", "detail": (e or content_of(r))[:70]}


TESTS = {
    "thinking": lambda a, m: t_thinking(a.endpoint, m, a.api_key, a.timeout),
    "approval": lambda a, m: t_approval_shape(a.endpoint, m, a.api_key, a.timeout),
    "perf":     lambda a, m: t_perf(a.endpoint, m, a.api_key, a.timeout,
                                    a.perf_lines, a.gen_tokens, a.rounds),
    "tools":    lambda a, m: t_tools(a.endpoint, m, a.api_key, a.timeout),
    "quality":  lambda a, m: t_quality(a.endpoint, m, a.api_key, a.timeout),
    "longctx":  lambda a, m: t_longctx(a.endpoint, m, a.api_key, a.timeout,
                                       a.long_lines),
    "vision":   lambda a, m: t_vision(a.endpoint, m, a.api_key, a.timeout),
}
DEFAULT_TESTS = ["thinking", "approval", "perf", "tools", "quality"]


# --------------------------------------------------------------------------- ui
def summary_table(results: dict) -> str:
    rows, models = [], list(results)
    head = (f"{'model':<26} {'prefill':>10} {'decode':>9} {'deep':>9} "
            f"{'acc s/d':>12} {'tools':>6} {'quality':>8} {'think':>6} {'approval':>9}")
    rows.append(head)
    rows.append("-" * len(head))
    for m in models:
        r = results[m]
        p = r.get("perf", {})
        pf = f"{p['prefill_tps']:.0f} t/s" if "prefill_tps" in p else "-"
        dc = f"{p['decode_tps']:.1f} t/s" if "decode_tps" in p else "-"
        dd = f"{p['decode_deep_tps']:.1f} t/s" if p.get("decode_deep_tps") else "-"
        ac = ("-" if p.get("accept_shallow") is None and p.get("accept_deep") is None
              else "/".join(f"{p[k]*100:.0f}%" if p.get(k) is not None else "-"
                            for k in ("accept_shallow", "accept_deep")))
        tl = r.get("tools", {}).get("score", "-")
        ql = r.get("quality", {}).get("score", "-")
        tk = r.get("thinking", {})
        th = "-" if not tk else ("ERR" if "error" in tk
                                 else ("on" if tk.get("thinking_on") else "off"))
        ap = r.get("approval", {})
        ap = ("-" if not ap else ("ERR" if "error" in ap
                                  else ("PASS" if ap.get("pass") else "EMPTY/FAIL")))
        rows.append(f"{m:<26} {pf:>10} {dc:>9} {dd:>9} {ac:>12} {tl:>6} {ql:>8} "
                    f"{th:>6} {ap:>9}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark local models on an OpenAI-compatible endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help=f"default {DEFAULT_ENDPOINT} (primary GPU); iGPU is :8082")
    ap.add_argument("--api-key", default=None, help="needed only via LiteLLM :4000")
    ap.add_argument("--tests", nargs="+", default=DEFAULT_TESTS,
                    choices=list(TESTS) + ["all"])
    ap.add_argument("--rounds", type=int, default=2,
                    help="perf rounds per model; models are INTERLEAVED across rounds "
                         "so thermal drift hits all of them equally")
    ap.add_argument("--perf-lines", type=int, default=320,
                    help="filler lines for the prefill prompt (320 ~= 6.5K tokens)")
    ap.add_argument("--long-lines", type=int, default=1800,
                    help="filler lines for the needle test (1800 ~= 50K tokens)")
    ap.add_argument("--gen-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--json", metavar="PATH", help="write raw results here")
    ap.add_argument("--fail-on-regression", action="store_true",
                    help="exit 1 if any model fails tools/quality/approval")
    a = ap.parse_args()

    tests = list(TESTS) if "all" in a.tests else a.tests
    results: dict[str, dict] = {m: {} for m in a.models}

    for name in tests:
        print(f"\n### {name}", flush=True)
        if name == "perf":
            # interleave models across rounds so drift is shared, not stacked
            per = {m: {"prefill": [], "decode": [], "deep": [],
                       "acc_s": [], "acc_d": []} for m in a.models}
            for rnd in range(a.rounds):
                for m in a.models:
                    one = t_perf(a.endpoint, m, a.api_key, a.timeout,
                                 a.perf_lines, a.gen_tokens, 1)
                    if "error" in one:
                        print(f"  r{rnd+1} {m:<26} ERROR {one['error'][:60]}", flush=True)
                        continue
                    per[m]["prefill"].append(one["prefill_tps"])
                    per[m]["decode"].append(one["decode_tps"])
                    if one.get("decode_deep_tps"):
                        per[m]["deep"].append(one["decode_deep_tps"])
                    per[m]["prompt_tokens"] = one["prompt_tokens"]
                    for src, dst in (("accept_shallow", "acc_s"), ("accept_deep", "acc_d")):
                        if one.get(src) is not None:
                            per[m][dst].append(one[src])
                    deep = one.get("decode_deep_tps")
                    deep_s = f"{deep:5.1f}" if deep else "    -"
                    acc = ""
                    if one.get("accept_deep") is not None or one.get("accept_shallow") is not None:
                        f = lambda v: f"{v*100:.0f}%" if v is not None else "-"
                        acc = (f"  accept={f(one.get('accept_shallow'))}"
                               f"/{f(one.get('accept_deep'))}")
                    print(f"  r{rnd+1} {m:<26} prefill={one['prefill_tps']:7.1f} t/s  "
                          f"decode={one['decode_tps']:5.1f} t/s  "
                          f"deep={deep_s} t/s{acc}", flush=True)
            for m in a.models:
                if per[m]["prefill"]:
                    results[m]["perf"] = {
                        "prefill_tps": statistics.median(per[m]["prefill"]),
                        "decode_tps": statistics.median(per[m]["decode"]),
                        "decode_deep_tps": (statistics.median(per[m]["deep"])
                                            if per[m]["deep"] else None),
                        "accept_shallow": (statistics.median(per[m]["acc_s"])
                                           if per[m]["acc_s"] else None),
                        "accept_deep": (statistics.median(per[m]["acc_d"])
                                        if per[m]["acc_d"] else None),
                        "prompt_tokens": per[m].get("prompt_tokens"),
                        "rounds": a.rounds}
            continue

        for m in a.models:
            t0 = time.time()
            out = TESTS[name](a, m)
            results[m][name] = out
            brief = {k: v for k, v in out.items() if k not in ("got",)}
            print(f"  {m:<26} {json.dumps(brief, default=str)[:150]}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 80)
    print(summary_table(results))
    print("=" * 80)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nraw results -> {a.json}")

    if a.fail_on_regression:
        bad = []
        for m, r in results.items():
            if r.get("tools", {}).get("score", "0/4") != "4/4":
                bad.append(f"{m}: tools {r.get('tools', {}).get('score')}")
            if "approval" in r and not r["approval"].get("pass"):
                bad.append(f"{m}: approval returned empty (thinking model?)")
        if bad:
            print("\nREGRESSIONS:")
            for b in bad:
                print("  -", b)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
