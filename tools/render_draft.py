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
CITE = re.compile(r"\[(@[\w]+(?:\s*;\s*@[\w]+)*)\]")


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
    ap.add_argument("--refs", default="article/references.tsv")
    ap.add_argument("--repo-url", default="https://github.com/BjBodner/random-seeds-matter")
    a = ap.parse_args()
    src = open(a.src, encoding="utf-8").read()
    out = TAG.sub(lambda m: f"```python\n# file: {m.group(1)}\n{region(os.path.join(a.repo, m.group(1)), m.group(2))}\n```",
                  src)

    # citations: [@a; @b] -> [1, 2], numbered in order of first use; <<references>> -> the list
    refs = dict(line.rstrip("\n").split("\t", 1) for line in open(a.refs, encoding="utf-8") if line.strip())
    order = []

    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        for k in keys:
            if k not in refs:
                raise SystemExit(f"unknown citation key: {k}")
            if k not in order:
                order.append(k)
        return "[" + ", ".join(str(order.index(k) + 1) for k in keys) + "]"

    out = CITE.sub(cite, out)
    ref_list = "\n\n".join(f"[{i}] {refs[k]}" for i, k in enumerate(order, 1))
    out = out.replace("<<references>>", ref_list)
    if a.repo_url:
        out = out.replace("{{REPO_URL}}", a.repo_url)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    with open(os.path.join(os.path.dirname(a.out), "citations.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# References\n\nTDS format `[N] Name, Title (Year), Source`, numbered in order of first use.\n\n"
                + ref_list + "\n")
    print(f"wrote {a.out} ({len(TAG.findall(src))} code blocks, {len(order)} references)")


if __name__ == "__main__":
    main()
