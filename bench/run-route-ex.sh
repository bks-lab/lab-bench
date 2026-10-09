#!/usr/bin/env bash
# Full run of the route1 and ex1 families, the side that reaches the Jev API
# and Ollama. The GLiNER arms run on the GPU machine with bench/run-route-ex.bat.
#
#   TYPESAFE_API_KEY=... OLLAMA_HOST=http://localhost:11434 bash bench/run-route-ex.sh
#   bash bench/run-route-ex.sh score     # after the GLiNER results are copied into results/
#
# LOCAL_MODEL and LOCAL_ARM pick the local model (default: Winnow-12B, the
# best pv1 local arm, see results/route1/method.md). HOST_LABEL is what the
# `host` field of the local rows says (default: local, so no machine name
# from OLLAMA_HOST ends up in committed rows). The score step fails when no
# arm has rows.
set -euo pipefail
cd "$(dirname "$0")/.."

LOCAL_MODEL="${LOCAL_MODEL:-hf.co/EldanRing/Winnow-12B:Q8_0}"
LOCAL_ARM="${LOCAL_ARM:-winnow-12b}"
OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"
HOST_LABEL="${HOST_LABEL:-local}"

score() {
  node bench/score-route.mjs --md results/route1/score.md
  node bench/score-extract.mjs --md results/ex1/score.md
}

if [ "${1:-}" = "score" ]; then score; exit 0; fi

if [ -z "${TYPESAFE_API_KEY:-}" ]; then echo "TYPESAFE_API_KEY is not set" >&2; exit 1; fi
curl -sf "$OLLAMA/api/version" >/dev/null || { echo "Ollama does not answer at $OLLAMA" >&2; exit 1; }

node bench/run-jev.mjs --pv route1 --runs 1
node bench/run-local.mjs --pv route1 --model "$LOCAL_MODEL" --arm "$LOCAL_ARM" --template gemma4 --prefix '' --host "$OLLAMA" --host-label "$HOST_LABEL" --runs 1 --skip-over-26
node bench/run-local-extract.mjs --model "$LOCAL_MODEL" --arm "$LOCAL_ARM" --template gemma4 --host "$OLLAMA" --host-label "$HOST_LABEL"
echo "Mac side done. Copy results/route1/ and results/ex1/ from the GLiNER machine, then: bash bench/run-route-ex.sh score"
