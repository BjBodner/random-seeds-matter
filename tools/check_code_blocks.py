#!/usr/bin/env python3
"""Fail if an article code block drifts from the repo code it claims to come from.

Usage: python tools/check_code_blocks.py article/draft.md [--repo .]

Every ```python block must start with `# file: <path>`; the rest of the block must appear
verbatim (whitespace at line ends ignored) in that file. Together with tools/check_numbers.py
this guarantees the article shows the exact code that produced results/.
"""
import argparse
import os
import re
import sys

BLOCK = re.compile(r"^```(?:python|py)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
MARKER = re.compile(r"^#\s*file:\s*(\S+)\s*$")


def norm(text):
    return "\n".join(line.rstrip() for line in text.strip("\n").splitlines())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draft")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()
    md = open(a.draft, encoding="utf-8").read()
    problems, n = [], 0
    for m in BLOCK.finditer(md):
        n += 1
        lines = m.group(1).splitlines()
        mk = MARKER.match(lines[0]) if lines else None
        if not mk:
            problems.append(f"block {n}: missing '# file:' marker (starts: {lines[0][:60] if lines else ''!r})")
            continue
        path = os.path.join(a.repo, mk.group(1))
        if not os.path.exists(path):
            problems.append(f"block {n}: {mk.group(1)} does not exist")
            continue
        body = norm("\n".join(lines[1:]))
        if body not in norm(open(path, encoding="utf-8").read()):
            problems.append(f"block {n}: not a verbatim excerpt of {mk.group(1)}")
    if problems:
        print(f"FAIL: {len(problems)} of {n} code blocks:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"OK: all {n} python blocks are verbatim excerpts of their '# file:' source")


if __name__ == "__main__":
    main()
