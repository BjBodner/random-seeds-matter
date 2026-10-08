#!/usr/bin/env bash
# Build final.html (dark) + source bundle, working around the known gotchas.
# Usage (from repo root):
#   tools/build_article.sh <slug> "<Title>" "<Subtitle>" "<Mon D, YYYY>" "<N min read>"
# Expects: article/draft.md, article/citations.md, article/assets_dark.css,
#          article/BUNDLE_README.md (enriched README), images/, results/, experiments/ (or code/)
set -euo pipefail

SLUG="$1"; TITLE="$2"; DECK="$3"; DATE="$4"; READTIME="$5"
ROOT="$(pwd)"
# prefer the v3 writer scripts (proper markdown renderer, explicit theme); fall back to the kit copy
SKILL="${SKILL_SCRIPTS:-$HOME/.claude/skills/benjy-medium-writer/scripts}"
[ -f "$SKILL/build_html.py" ] || SKILL="$ROOT/.claude/skills/benjy-medium-writer/scripts"
PY="${PY:-python}"
CSS="$ROOT/article/assets_dark.css"
WS="$ROOT/build/workspace"

[ -f "$CSS" ] || { echo "ERROR: $CSS missing (cp templates/assets_dark.css article/)"; exit 1; }
[ -f "$ROOT/article/BUNDLE_README.md" ] || { echo "ERROR: article/BUNDLE_README.md missing"; exit 1; }

# 0. numbers check
"$PY" "$ROOT/tools/check_numbers.py" "$ROOT/article/draft.md" "$ROOT/results"

# 1. stage a skill-shaped workspace (bundle.py expects draft.md, code/, images/, results/)
rm -rf "$ROOT/build"; mkdir -p "$WS"
cp "$ROOT/article/draft.md" "$ROOT/article/citations.md" "$WS/"
cp -r "$ROOT/images" "$ROOT/results" "$WS/"
if [ -d "$ROOT/experiments" ]; then cp -r "$ROOT/experiments" "$WS/code"; else cp -r "$ROOT/code" "$WS/code"; fi
[ -f "$ROOT/seeds_matter.py" ] && cp "$ROOT/seeds_matter.py" "$WS/code/"
[ -f "$ROOT/requirements.txt" ] && cp "$ROOT/requirements.txt" "$WS/code/"
find "$WS" -name "__pycache__" -prune -exec rm -rf {} +

# 2. HTML with the DARK css (gotcha 1); the template renders title + deck, so drop them from the body
"$PY" - "$WS/draft.md" "$WS/draft_body.md" <<'PY'
import re, sys
md = open(sys.argv[1], encoding="utf-8").read()
md = re.sub(r"\A\s*# [^\n]+\n+(\*[^\n]+\*\n+)?", "", md, count=1)
open(sys.argv[2], "w", encoding="utf-8", newline="\n").write(md)
PY
"$PY" "$SKILL/build_html.py" "$WS/draft_body.md" "$WS/final.html" \
  --title "$TITLE" --deck "$DECK" --author "Benjamin Bodner" \
  --date "$DATE" --readtime "$READTIME" --css "$CSS" --base-dir "$WS"
grep -q "background:#121212" "$WS/final.html" || { echo "ERROR: dark CSS not applied"; exit 1; }
if grep -q '<img src="images/' "$WS/final.html"; then echo "ERROR: un-embedded image(s) in HTML"; exit 1; fi

# 3. bundle, then fix README / css / pycache (gotchas 2-4)
rm -f "$WS/draft_body.md"
# 3. bundle: the v3 bundle.py keeps a README.md found in the workspace (gotcha 2), copies the
#    CSS with --css (gotcha 4) and adds PREREGISTRATION / DECISIONS when present
cp "$ROOT/article/BUNDLE_README.md" "$WS/README.md"
cp "$ROOT/PREREGISTRATION.md" "$ROOT/article/DECISIONS.md" "$WS/" 2>/dev/null || true
find "$WS" -name "__pycache__" -prune -exec rm -rf {} +
rm -f "$ROOT/build/${SLUG}_bundle.zip"
"$PY" "$SKILL/bundle.py" "$WS" "$SLUG" --out "$ROOT/build" --title "$TITLE" --css "$CSS"
"$PY" - "$ROOT/build/${SLUG}_bundle.zip" <<'PY'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
bad = [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
readme = [n for n in names if n.endswith("README.md")]
assert not bad, f"__pycache__ in bundle: {bad[:3]}"
assert readme, "README.md missing from bundle"
print(f"bundle OK: {len(names)} files, README present, no __pycache__")
PY

# 4. publish into article/
cp "$WS/final.html" "$ROOT/article/final.html"
cp "$ROOT/build/${SLUG}_bundle.zip" "$ROOT/article/"

# 5. placeholder scan
if grep -RniE "placeholder|\bTODO\b|\bTBD\b|lorem|\{\{" "$ROOT/article/draft.md" "$ROOT/article/BUNDLE_README.md"; then
  echo "WARNING: placeholders found above — must be zero before shipping"
fi
echo "OK: article/final.html and article/${SLUG}_bundle.zip"
