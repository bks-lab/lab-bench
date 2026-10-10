#!/usr/bin/env bash
# Set or lift a HOLD on GPU jobs (the current job is not affected).
#   bench/queue/hold.sh set <name> "<reason>" [release_after_idle_min]
#     with a release time the hold lifts itself once the GPU was free that long,
#     but only after an Ollama model was seen loaded (the held-for work ran)
#   bench/queue/hold.sh lift <name>
#   bench/queue/hold.sh pause | resume      the PAUSE switch
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/common.sh"
case "${1:-}" in
  set)
    tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
    python3 -c 'import json,sys; d={"reason":sys.argv[1]}; a=sys.argv[2:]; d.update({"release_after_idle_min":float(a[0]),"require_busy_first":True} if a else {}); print(json.dumps(d))' "$3" ${4:+"$4"} > "$tmp"
    scp -q "$tmp" "$JMB_HOST:$JMB_ROOT_SCP/HOLD-$2" ;;
  lift) ssh "$JMB_HOST" "del $JMB_ROOT_WIN\\HOLD-$2" ;;
  pause) ssh "$JMB_HOST" "type nul > $JMB_ROOT_WIN\\PAUSE" ;;
  resume) ssh "$JMB_HOST" "del $JMB_ROOT_WIN\\PAUSE" ;;
  *) sed -n '2,8p' "$0"; exit 2 ;;
esac
