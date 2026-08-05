#!/usr/bin/env python3
"""Fast health check for the whole local inference stack. Stdlib only.

    ./smoke.py                     # check every tier + every alias
    ./smoke.py --quick             # skip generation, just list what is loaded
    ./smoke.py --endpoint-only     # skip LiteLLM, check llama-swap directly

Run after any config change, container restart, or model swap. Takes ~1-2 minutes
warm, longer if models need loading.

Exits 1 if anything fails, so it works in a pre-commit hook or a cron.

Reads LITELLM_MASTER_KEY from the environment, or LITELLM_ENV / ./litellm.env.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TIERS = {
    "primary GPU  (:8081)": "http://localhost:8081/v1",
    "secondary GPU(:8082)": "http://localhost:8082/v1",
}
GATEWAY = "http://localhost:4000/v1"
# Aliases every machine should define. local-embed is laptop-only by design —
# the desktop has no embedding GGUF, so its absence there is expected, not a bug.
CHAT_ALIASES = ["local-main", "local-vision", "local-tiny"]
EMBED_ALIASES = ["local-embed"]

PNG_1PX = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhK"
           "mMIQAAAABJRU5ErkJggg==")


def env_candidates() -> list[Path]:
    """Where to look for a litellm.env, in priority order.

    This repo is standalone, so it cannot assume it sits next to the stack it
    benchmarks. Point LITELLM_ENV at the file, run from a directory that has one,
    or just export LITELLM_MASTER_KEY directly.
    """
    here = Path(__file__).resolve().parent
    explicit = os.environ.get("LITELLM_ENV")
    return ([Path(explicit)] if explicit else []) + [
        Path.cwd() / "litellm.env",
        here / "litellm.env",
        here.parent / "litellm.env",
    ]


def master_key() -> str | None:
    k = os.environ.get("LITELLM_MASTER_KEY")
    if k:
        return k
    for env in env_candidates():
        if env.exists():
            m = re.search(r"^LITELLM_MASTER_KEY=(.*)$", env.read_text(), re.M)
            if m:
                return m.group(1).strip()
    return None


def req(url: str, body: dict | None, key: str | None, timeout: int):
    r = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.load(resp), None
    except urllib.error.HTTPError as e:
        try:
            return json.load(e), None
        except Exception:
            return None, f"HTTP {e.code}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ok(msg):
    print(f"  \033[32mok\033[0m   {msg}")


def bad(msg):
    print(f"  \033[31mFAIL\033[0m {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip generation calls")
    ap.add_argument("--endpoint-only", action="store_true", help="skip the LiteLLM gateway")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    failures = 0

    print("\n== llama-swap tiers ==")
    live = {}
    for label, ep in TIERS.items():
        data, e = req(ep + "/models", None, None, 15)
        if e or not data:
            bad(f"{label} unreachable ({e})")
            failures += 1
            continue
        ids = [m["id"] for m in data.get("data", [])]
        loaded = [m["id"] for m in data.get("data", [])
                  if (m.get("status") or {}).get("value") == "loaded"]
        live[label] = ids
        ok(f"{label} {len(ids)} models: {', '.join(ids)}")
        if loaded:
            print(f"       resident now: {', '.join(loaded)}")

    if a.quick:
        print("\n(--quick: skipping generation)")
        return 1 if failures else 0

    if not a.endpoint_only:
        key = master_key()
        if not key:
            bad("LITELLM_MASTER_KEY not found (env, LITELLM_ENV, or ./litellm.env)")
            return 1

        print("\n== gateway aliases ==")
        for alias in CHAT_ALIASES:
            t0 = time.time()
            data, e = req(GATEWAY + "/chat/completions",
                          {"model": alias,
                           "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                           "max_tokens": 400}, key, a.timeout)
            if e or not data or "error" in (data or {}):
                bad(f"{alias:<14} {e or str(data.get('error'))[:60]}")
                failures += 1
                continue
            c = (data["choices"][0]["message"].get("content") or "").strip()
            # Empty content with a filled reasoning trace = thinking ate the budget.
            rc = len(data["choices"][0]["message"].get("reasoning_content") or "")
            if not c:
                bad(f"{alias:<14} EMPTY content (reasoning={rc} chars) — raise max_tokens "
                    "or use a non-thinking model")
                failures += 1
            else:
                ok(f"{alias:<14} {c[:30]!r}  [{time.time()-t0:.1f}s]")

        for alias in EMBED_ALIASES:
            data, e = req(GATEWAY + "/embeddings",
                          {"model": alias, "input": "smoke test"}, key, a.timeout)
            if e or not data or "error" in (data or {}):
                print(f"  \033[33mskip\033[0m {alias:<14} not available here "
                      "(expected on the desktop)")
            else:
                ok(f"{alias:<14} {len(data['data'][0]['embedding'])} dims")

        print("\n== vision ==")
        data, e = req(GATEWAY + "/chat/completions",
                      {"model": "local-vision", "max_tokens": 400,
                       "messages": [{"role": "user", "content": [
                           {"type": "text", "text": "What colour is this? One word."},
                           {"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{PNG_1PX}"}}]}]},
                      key, a.timeout)
        if e or not data or "error" in (data or {}):
            bad(f"local-vision rejected an image: {e or str(data.get('error'))[:70]}")
            failures += 1
        else:
            ok(f"local-vision accepted an image -> "
               f"{(data['choices'][0]['message'].get('content') or '')[:30]!r}")

        print("\n== approval shape (max_tokens=16, must return one word) ==")
        data, e = req(GATEWAY + "/chat/completions",
                      {"model": "local-tiny", "max_tokens": 16, "temperature": 0,
                       "messages": [
                           {"role": "system", "content": "You assess shell command risk."},
                           {"role": "user", "content":
                            "Flagged: script execution via -c flag\n<command>\n"
                            "python -c \"print('hi')\"\n</command>\nRespond with exactly "
                            "one word: APPROVE, DENY, or ESCALATE"}]}, key, a.timeout)
        if e or not data or "error" in (data or {}):
            bad(f"approval probe failed: {e}")
            failures += 1
        else:
            c = (data["choices"][0]["message"].get("content") or "").strip().upper()
            if c in ("APPROVE", "DENY", "ESCALATE"):
                ok(f"local-tiny returned {c!r} within 16 tokens")
            else:
                bad(f"local-tiny returned {c!r} — a thinking model will return empty "
                    "here and silently break Hermes smart-approval")
                failures += 1

    print()
    if failures:
        print(f"\033[31m{failures} check(s) failed\033[0m")
        return 1
    print("\033[32mall checks passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
