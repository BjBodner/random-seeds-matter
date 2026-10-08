#!/usr/bin/env bash
# Build final.html (dark) + source bundle, working around the known gotchas.
# Usage (from repo root):
#   tools/build_article.sh <slug> "<Title>" "<Subtitle>" "<Mon D, YYYY>" "<N min read>"
# Expects: article/draft.md, article/citations.md, article/assets_dark.css,
#          article/BUNDLE_README.md (enriched README), images/, results/, experiments/ (or code/)
set -euo pipefail

SLUG="$1"; TITLE="$2"; DECK="$3"; DATE="$4"; READTIME="$5"
ROOT="$(pwd)"
SKILL="$ROOT/.claude/skills/benjy-medium-writer/scripts"
CSS="$ROOT/article/assets_dark.css"
WS="$ROOT/build/workspace"

[ -f "$CSS" ] || { echo "ERROR: $CSS missing (cp templates/assets_dark.css article/)"; exit 1; }
[ -f "$ROOT/article/BUNDLE_README.md" ] || { echo "ERROR: article/BUNDLE_README.md missing"; exit 1; }

# 0. numbers check
python "$ROOT/tools/check_numbers.py" "$ROOT/article/draft.md" "$ROOT/results"

# 1. stage a skill-shaped workspace (bundle.py expects draft.md, code/, images/, results/)
rm -rf "$ROOT/build"; mkdir -p "$WS"
cp "$ROOT/article/draft.md" "$ROOT/article/citations.md" "$WS/"
cp -r "$ROOT/images" "$ROOT/results" "$WS/"
if [ -d "$ROOT/experiments" ]; then cp -r "$ROOT/experiments" "$WS/code"; else cp -r "$ROOT/code" "$WS/code"; fi
[ -f "$ROOT/seeds_matter.py" ] && cp "$ROOT/seeds_matter.py" "$WS/code/"
[ -f "$ROOT/requirements.txt" ] && cp "$ROOT/requirements.txt" "$WS/code/"
find "$WS" -name "__pycache__" -prune -exec rm -rf {} +

# 2. HTML with the DARK css (gotcha 1)
python "$SKILL/build_html.py" "$WS/draft.md" "$WS/final.html" \
  --title "$TITLE" --deck "$DECK" --author "Benjamin Bodner" \
  --date "$DATE" --readtime "$READTIME" --css "$CSS" --base-dir "$WS"
grep -q "background:#121212" "$WS/final.html" || { echo "ERROR: dark CSS not applied"; exit 1; }
if grep -q '<img src="images/' "$WS/final.html"; then echo "ERROR: un-embedded image(s) in HTML"; exit 1; fi

# 3. bundle, then fix README / css / pycache (gotchas 2-4)
python "$SKILL/bundle.py" "$WS" "$SLUG" --out "$ROOT/build" --title "$TITLE"
STAGE="$ROOT/build/${SLUG}_bundle"
cp "$ROOT/article/BUNDLE_README.md" "$STAGE/README.md"
cp "$CSS" "$STAGE/assets_dark.css"
find "$STAGE" -name "__pycache__" -prune -exec rm -rf {} +
rm -f "$ROOT/build/${SLUG}_bundle.zip"
(cd "$ROOT/build" && python - "$SLUG" <<'PY'
import os, sys, zipfile
slug = sys.argv[1]; stage = f"{slug}_bundle"
with zipfile.ZipFile(f"{slug}_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(stage):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            z.write(os.path.join(root, fn))
PY
)

# 4. publish into article/
cp "$WS/final.html" "$ROOT/article/final.html"
cp "$ROOT/build/${SLUG}_bundle.zip" "$ROOT/article/"

# 5. placeholder scan
if grep -RniE "placeholder|\bTODO\b|\bTBD\b|lorem" "$ROOT/article/draft.md" "$ROOT/article/BUNDLE_README.md"; then
  echo "WARNING: placeholders found above — must be zero before shipping"
fi
echo "OK: article/final.html and article/${SLUG}_bundle.zip"
