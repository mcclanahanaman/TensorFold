"""Run a Thor client tool with request marks (FR-OPS-17, ROWS-2): the wall-clock start and end of every request it sends.

    python tools/thor/marks_run.py MARKS.jsonl TOOL.py FUNCTION [the tool's arguments ...]

TOOL.py is imported as a module and its FUNCTION (serve_ab.py: chat; natural_prefill.py: request) wrapped, so every
call appends one JSON line to MARKS.jsonl: {"seq" (0, 1, ... in call order), "tool" (TOOL's stem), "t0", "t1"
(time.time() at the call and at its return), "ok" (returned without raising), "chars" (the prompt text's characters:
a long string argument, or the messages' contents), "max_tokens"}. Then the tool's main() runs with the given
arguments; its output and files are unchanged. Concurrent calls (serve_ab's decode streams) each get a line.
Standard library only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path


def prompt_chars(args, kwargs) -> int:
    n = 0
    for v in [*args, *kwargs.values()]:
        if isinstance(v, str) and len(v) > 200:
            n += len(v)
        elif isinstance(v, list):
            n += sum(len(m["content"]) for m in v if isinstance(m, dict) and isinstance(m.get("content"), str))
    return n


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    marks, tool, name = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), sys.argv[3]
    sys.path.insert(0, str(tool.parent))
    spec = importlib.util.spec_from_file_location(tool.stem, tool)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[tool.stem] = mod
    spec.loader.exec_module(mod)
    orig = getattr(mod, name)
    lock, seq = threading.Lock(), [0]
    out = marks.open("a", buffering=1)

    def wrapped(*args, **kwargs):
        with lock:
            n = seq[0]
            seq[0] += 1
        t0, ok = time.time(), False
        try:
            got = orig(*args, **kwargs)
            ok = True
            return got
        finally:
            mt = kwargs.get("max_tokens", args[3] if len(args) > 3 and isinstance(args[3], int) else None)
            rec = {"seq": n, "tool": tool.stem, "t0": round(t0, 4), "t1": round(time.time(), 4), "ok": ok,
                   "chars": prompt_chars(args, kwargs), "max_tokens": mt}
            with lock:
                out.write(json.dumps(rec) + "\n")

    setattr(mod, name, wrapped)
    sys.argv = [str(tool), *sys.argv[4:]]
    mod.main()


if __name__ == "__main__":
    main()
