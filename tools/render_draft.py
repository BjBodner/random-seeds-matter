#!/usr/bin/env python3
"""Render article/draft_src.md -> article/draft.md, pulling code blocks from the repo.

A line `<<code experiments/e1_split.py e1>>` becomes a ```python block that starts with
`# file: experiments/e1_split.py` followed by the code between `# --- article: e1 ---` and the
next `# --- end ---` in that file. The article therefore always shows the code that ran.

Usage: python tools/render_draft.py [--src article/draft_src.md] [--out article/draft.md]
"""
import argparse
import os
import re

TAG = re.compile(r"^<<code (\S+) (\S+)>>\s*$", re.MULTILINE)


def region(path, name):
    text = open(path, encoding="utf-8").read()
    m = re.search(rf"^# --- article: {re.escape(name)} ---\n(.*?)^# --- end ---", text, re.DOTALL | re.MULTILINE)
    if not m:
        raise SystemExit(f"region '{name}' not found in {path}")
    return m.group(1).rstrip("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="article/draft_src.md")
    ap.add_argument("--out", default="article/draft.md")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    src = open(a.src, encoding="utf-8").read()
    out = TAG.sub(lambda m: f"```python\n# file: {m.group(1)}\n{region(os.path.join(a.repo, m.group(1)), m.group(2))}\n```",
                  src)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"wrote {a.out} ({len(TAG.findall(src))} code blocks)")


if __name__ == "__main__":
    main()
