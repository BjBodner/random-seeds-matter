#!/usr/bin/env python3
"""Fail if a number quoted in the article's prose doesn't appear in captured results.

Usage: python tools/check_numbers.py article/draft.md results/ [--allow 2026,3,5,10]

- Scans prose only (code fences, inline code, links/URLs, image refs, and the
  references section are stripped).
- Checks decimals (0.956), percentages (12.5%), multipliers (5.7x / 5.7×), and
  integers >= 11 (smaller integers are usually counts like "3 models"; list any
  you want checked anyway by putting them in results).
- A number passes if it appears verbatim in any results/*.txt or *.csv file.
  Rounded-for-prose values must therefore be printed in that rounded form by
  the experiment code too — that's intentional.
"""
import argparse, glob, os, re, sys

def prose_only(md: str) -> str:
    md = re.sub(r"```.*?```", " ", md, flags=re.DOTALL)
    md = re.sub(r"`[^`]*`", " ", md)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"\]\([^)]*\)", "]", md)
    md = re.sub(r"https?://\S+", " ", md)
    md = re.split(r"\n#+\s*(sources|references)[^\n]*\n", md, flags=re.I)[0]
    return md

NUM = re.compile(r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(\s?%|\s?[x×])?(?![\w])")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("draft"); ap.add_argument("results")
    ap.add_argument("--allow", default="", help="comma list of numbers to ignore (years, etc.)")
    a = ap.parse_args()
    text = prose_only(open(a.draft, encoding="utf-8").read())
    corpus = ""
    for p in glob.glob(os.path.join(a.results, "**", "*"), recursive=True):
        if p.endswith((".txt", ".csv", ".json")):
            corpus += open(p, encoding="utf-8", errors="ignore").read() + "\n"
    allow = {s.strip() for s in a.allow.split(",") if s.strip()}
    allow |= {str(y) for y in range(1950, 2031)}  # years
    missing = []
    for m in NUM.finditer(text):
        raw, suffix = m.group(1), (m.group(2) or "").strip()
        val = raw.replace(",", "")
        if val in allow:
            continue
        is_dec = "." in val
        if not (is_dec or suffix or int(val) >= 11):
            continue
        if val not in corpus and raw not in corpus:
            ctx = text[max(0, m.start() - 40): m.end() + 40].replace("\n", " ")
            missing.append(f"{raw}{suffix}  …{ctx}…")
    if missing:
        print(f"FAIL: {len(missing)} number(s) in prose not found in {a.results}:")
        for x in missing:
            print("  -", x)
        sys.exit(1)
    print("OK: every checked number in the prose appears in results/")

if __name__ == "__main__":
    main()
