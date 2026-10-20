#!/usr/bin/env python3
"""Coding arm: small tasks with hidden tests, scored by first-attempt success.

Stdlib only — no pip install, no venv. Runs against any OpenAI-compatible endpoint.

    ./coding.py --models tiel-coder-35b-a3b-ud-q3_k_xl
    ./coding.py --models a b --only lru_cache parse_duration
    ./coding.py --models a b --endpoint http://localhost:8080/v1 --json coding.json

WHY THIS IS SEPARATE FROM llmbench.py:

  llmbench measures capability (tools, quality, long context) and throughput. It says
  nothing about whether a model can produce a working file. This arm asks for code, runs
  a hidden unittest suite the model never sees, and allows repairs from the failing
  output — the loop an agent actually runs.

  Quality is the score; time and completion tokens are reported beside it because they
  are what an agent pays per turn. A model that passes in 300 tokens and 7 s is not
  equivalent to one that passes in 6,500 tokens and 86 s (measured Tiel vs Qwen3.6,
  2026-10-19: both 5/5 pass@1, Tiel 4.7x faster and 6x leaner).

  Tasks are small on purpose: they fit one file and one hidden suite, and they run on a
  single 8 GB card. They saturate — both models passed 5/5 first try — so treat a tie as
  "no evidence of a quality difference", not as proof of equality on real agentic work.

Rules this script follows (see AGENTS.md):
  - models are interleaved task by task, so thermal drift is shared;
  - temperature is 0 and max_tokens is large (a thinking model must not lose its answer
    to the budget);
  - failures report which task and the test output, never just a score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = "http://localhost:8081/v1"
SYSTEM = ("You are a senior Python engineer. Reply with the complete file in a single "
          "```python code block and no prose.")

TASKS = [
    {
        "id": "stats_bugfix",
        "filename": "stats.py",
        "initial": '''def mean(xs):
    return sum(xs) // len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2] + s[n // 2 + 1]) / 2
''',
        "prompt": """Fix the two bugs in this file. Keep the function names and signatures.

```python
def mean(xs):
    return sum(xs) // len(xs)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2] + s[n // 2 + 1]) / 2
```
""",
        "test": '''import unittest
from stats import mean, median

class T(unittest.TestCase):
    def test_mean_float(self):
        self.assertEqual(mean([1, 2]), 1.5)

    def test_mean_repeating(self):
        self.assertAlmostEqual(mean([3, 3, 4]), 10 / 3)

    def test_median_even(self):
        self.assertEqual(median([1, 2, 3, 4]), 2.5)

    def test_median_odd(self):
        self.assertEqual(median([7, 1, 3]), 3)

    def test_median_unsorted_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_mean_zero(self):
        self.assertEqual(mean([0]), 0.0)
''',
    },
    {
        "id": "merge_intervals",
        "filename": "solution.py",
        "initial": "",
        "prompt": """Implement `merge_intervals(intervals)` in solution.py.

- `intervals` is a list of [start, end] pairs of ints, possibly unsorted.
- Merge every pair that overlaps or touches (e.g. [1,2] and [2,3] merge into [1,3]).
- Return a new list of [start, end] pairs, sorted by start.
- Return [] for an empty input.
""",
        "test": '''import unittest
from solution import merge_intervals

class T(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])

    def test_single(self):
        self.assertEqual(merge_intervals([[1, 2]]), [[1, 2]])

    def test_classic(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]),
                         [[1, 6], [8, 10], [15, 18]])

    def test_touching(self):
        self.assertEqual(merge_intervals([[1, 2], [2, 3]]), [[1, 3]])

    def test_nested(self):
        self.assertEqual(merge_intervals([[1, 10], [2, 3]]), [[1, 10]])

    def test_unsorted(self):
        self.assertEqual(merge_intervals([[5, 6], [1, 3], [2, 4]]), [[1, 4], [5, 6]])
''',
    },
    {
        "id": "parse_duration",
        "filename": "solution.py",
        "initial": "",
        "prompt": """Implement `parse_duration(text)` in solution.py.

Parse a duration string made of one or more `<int><unit>` components, where unit is
`h`, `m` or `s`. Components must be in descending order (h, then m, then s), each
unit at most once. Return the total number of seconds as an int. Raise ValueError
for anything else (empty string, unknown unit, repeated unit, wrong order, missing
number).

Examples: "45s" -> 45, "2h" -> 7200, "1h30m" -> 5400, "1h30m15s" -> 5415, "0s" -> 0.
""",
        "test": '''import unittest
from solution import parse_duration

class T(unittest.TestCase):
    def test_valid(self):
        for text, seconds in [("45s", 45), ("2h", 7200), ("90m", 5400),
                              ("1h30m", 5400), ("1h30m15s", 5415), ("0s", 0)]:
            with self.subTest(text=text):
                self.assertEqual(parse_duration(text), seconds)

    def test_invalid(self):
        for text in ["", "10", "1m1h", "1x", "1h30", "h1", "1h1h", "1.5h"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_duration(text)
''',
    },
    {
        "id": "csv_bugfix",
        "filename": "solution.py",
        "initial": '''def parse_csv_line(line):
    return line.split(",")
''',
        "prompt": """Fix `parse_csv_line(line)` in solution.py so it parses one CSV line.

Fields are separated by commas. A field may be wrapped in double quotes; inside
quotes a comma is literal and a doubled quote ("") means one quote character.
Unquoted fields are returned as-is. No newlines occur in the input. Return a list
of field strings.

```python
def parse_csv_line(line):
    return line.split(",")
```
""",
        "test": '''import unittest
from solution import parse_csv_line

class T(unittest.TestCase):
    def test_parse(self):
        for line, fields in [
            ("a,b,c", ["a", "b", "c"]),
            ('"a,b",c', ["a,b", "c"]),
            ('"he said ""hi""",2', ['he said "hi"', "2"]),
            (",", ["", ""]),
            ('"",x', ["", "x"]),
            ('"a""b"', ['a"b']),
            ("plain", ["plain"]),
        ]:
            with self.subTest(line=line):
                self.assertEqual(parse_csv_line(line), fields)
''',
    },
    {
        "id": "lru_cache",
        "filename": "solution.py",
        "initial": "",
        "prompt": """Implement `LRUCache` in solution.py with O(1) get and put.

- `LRUCache(capacity)`; capacity may be 0.
- `get(key)` returns the value, or -1 when the key is absent.
- `put(key, value)` inserts or updates; a get or put makes the key the most recently used.
- When full, evict the least recently used key. Updating an existing key never evicts.
""",
        "test": '''import unittest
from solution import LRUCache

class T(unittest.TestCase):
    def test_classic(self):
        c = LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        self.assertEqual(c.get(1), 1)
        c.put(3, 3)
        self.assertEqual(c.get(2), -1)
        self.assertEqual(c.get(3), 3)
        c.put(4, 4)
        self.assertEqual(c.get(1), -1)
        self.assertEqual(c.get(3), 3)
        self.assertEqual(c.get(4), 4)

    def test_capacity_one(self):
        c = LRUCache(1)
        c.put(1, 1)
        c.put(2, 2)
        self.assertEqual(c.get(1), -1)
        self.assertEqual(c.get(2), 2)

    def test_capacity_zero(self):
        c = LRUCache(0)
        c.put(1, 1)
        self.assertEqual(c.get(1), -1)

    def test_update_is_mru(self):
        c = LRUCache(2)
        c.put(1, 1)
        c.put(2, 2)
        c.put(1, 10)
        c.put(3, 3)
        self.assertEqual(c.get(2), -1)
        self.assertEqual(c.get(1), 10)

    def test_update_no_evict(self):
        c = LRUCache(2)
        c.put(1, 1)
        c.put(1, 2)
        self.assertEqual(c.get(1), 2)
        c.put(2, 2)
        self.assertEqual(c.get(1), 2)
        self.assertEqual(c.get(2), 2)
''',
    },
]


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
    except Exception as e:  # connection reset, timeout
        return {"error": {"message": f"{type(e).__name__}: {e}"}}


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def run_task(endpoint: str, model: str, task: dict, api_key: str | None,
             timeout: int, attempts: int, max_tokens: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="coding-") as tmp:
        target = os.path.join(tmp, task["filename"])
        test_path = os.path.join(tmp, "test_hidden.py")
        with open(test_path, "w") as f:
            f.write(task["test"])
        with open(target, "w") as f:
            f.write(task["initial"])

        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": task["prompt"]}]
        started = time.time()
        tokens = 0
        passed = False
        output = ""
        for attempt in range(1, attempts + 1):
            resp = post(endpoint, "/chat/completions", {
                "model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0,
            }, api_key, timeout)
            if "error" in resp:
                return {"task": task["id"], "pass": False, "attempts": attempt,
                        "seconds": round(time.time() - started, 1), "tokens": tokens,
                        "error": str(resp["error"])[:200]}
            usage = resp.get("usage") or {}
            tokens += usage.get("completion_tokens") or 0
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
            with open(target, "w") as f:
                f.write(extract_code(content))
            p = subprocess.run(
                [sys.executable, "-m", "unittest", "-q", "test_hidden"],
                cwd=tmp, capture_output=True, text=True,
            )
            output = (p.stdout + p.stderr)[-2500:]
            if p.returncode == 0:
                passed = True
                break
            if attempt < attempts:
                messages += [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content":
                     "The hidden tests failed:\n\n" + output +
                     "\nFix the file. Return the complete corrected file in one ```python block."},
                ]
        out = {"task": task["id"], "pass": passed, "attempts": attempt,
               "seconds": round(time.time() - started, 1), "tokens": tokens}
        if not passed:
            out["failure_tail"] = output[-600:]
        return out


def summary_table(results: dict[str, dict]) -> str:
    lines = [f"{'model':<28} {'pass@1':>7} {'passed':>7} {'time':>7} {'tokens':>8}"]
    lines.append("-" * 62)
    for model, r in results.items():
        tasks = r["tasks"]
        passed = sum(1 for t in tasks if t["pass"])
        first = sum(1 for t in tasks if t["pass"] and t["attempts"] == 1)
        secs = sum(t["seconds"] for t in tasks)
        toks = sum(t["tokens"] for t in tasks)
        lines.append(f"{model:<28} {first:>4}/{len(tasks):<2} {passed:>4}/{len(tasks):<2} "
                     f"{secs:>6.1f}s {toks:>8}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--api-key")
    ap.add_argument("--only", nargs="+", default=[],
                    help="task ids to run (default: all)")
    ap.add_argument("--repair-attempts", type=int, default=3,
                    help="max attempts per task (default 3)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--json")
    ap.add_argument("--fail-on-failure", action="store_true",
                    help="exit 1 if any task fails")
    a = ap.parse_args()

    tasks = [t for t in TASKS if not a.only or t["id"] in a.only]
    unknown = set(a.only) - {t["id"] for t in TASKS}
    if unknown:
        ap.error(f"unknown task(s): {', '.join(sorted(unknown))}")

    results: dict[str, dict] = {m: {"tasks": []} for m in a.models}
    for task in tasks:
        print(f"\n### {task['id']}", flush=True)
        for model in a.models:
            r = run_task(a.endpoint, model, task, a.api_key, a.timeout,
                         a.repair_attempts, a.max_tokens)
            results[model]["tasks"].append(r)
            if "error" in r:
                print(f"  {model:<26} ERROR {r['error'][:70]}", flush=True)
            else:
                mark = "pass" if r["pass"] else "FAIL"
                print(f"  {model:<26} {mark} attempt {r['attempts']} "
                      f"{r['seconds']:5.1f}s {r['tokens']:6d} tok", flush=True)
                if not r["pass"]:
                    print(f"    {r['failure_tail'].splitlines()[-1][:120]}", flush=True)

    print("\n" + "=" * 62)
    print(summary_table(results))
    print("=" * 62)

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nraw results -> {a.json}")

    if a.fail_on_failure and any(not t["pass"] for r in results.values() for t in r["tasks"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
