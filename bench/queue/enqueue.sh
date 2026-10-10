#!/usr/bin/env bash
# Enqueue job templates on the PC queue, in the order given.
#   bench/queue/enqueue.sh [--ref REF] [--by NAME] [--dry-run] bench/queue/jobs/c1/*.json
# REF defaults to origin/main. The commit is uploaded once as src/<sha>.tar
# (git archive); the runner extracts it. Each template gets id, key, repo_ref,
# requested_by and enqueued_at; ids are <yyyymmdd-hhmmss>-<test>-<arm>, one
# second apart so the name order is the run order.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/common.sh"
repo="$(git -C "$here" rev-parse --show-toplevel)"
ref="origin/main"; by="${USER:-agent}"; dry=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) ref="$2"; shift 2 ;;
    --by) by="$2"; shift 2 ;;
    --dry-run) dry=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] || { echo "no job templates given" >&2; exit 2; }
git -C "$repo" fetch -q origin || true
sha="$(git -C "$repo" rev-parse --verify "$ref^{commit}")"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

if [[ $dry -eq 0 ]]; then
  if ! ssh "$JMB_HOST" "if exist $JMB_ROOT_WIN\\src\\$sha (echo have) else if exist $JMB_ROOT_WIN\\src\\$sha.tar (echo have)" | grep -q have; then
    git -C "$repo" archive --format=tar -o "$tmp/src.tar" "$sha"
    scp -q "$tmp/src.tar" "$JMB_HOST:$JMB_ROOT_SCP/src/$sha.tar.part"
    ssh "$JMB_HOST" "move /Y $JMB_ROOT_WIN\\src\\$sha.tar.part $JMB_ROOT_WIN\\src\\$sha.tar >nul"
    echo "uploaded source $sha"
  fi
fi

for t in "$@"; do
  id="$(date +%Y%m%d-%H%M%S)"
  out="$(python3 - "$t" "$id" "$sha" "$by" <<'PY'
import json, sys, datetime
path, stamp, sha, by = sys.argv[1:]
j = json.load(open(path, encoding="utf-8"))
for k in ("test", "arm", "cmd"):
    if k not in j:
        sys.exit(f"{path}: missing {k}")
j["id"] = f"{stamp}-{j['test'].lower()}-{j['arm']}"
j["key"] = f"{j['test']}-{j['arm']}"
j["repo_ref"] = sha
j.setdefault("requested_by", by)
j["template"] = path
j["enqueued_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
print(j["id"])
print(json.dumps(j, indent=1, ensure_ascii=False))
PY
)"
  jid="$(head -1 <<<"$out")"
  tail -n +2 <<<"$out" > "$tmp/$jid.json"
  if [[ $dry -eq 1 ]]; then
    echo "would enqueue $jid"; cat "$tmp/$jid.json"
  else
    scp -q "$tmp/$jid.json" "$JMB_HOST:$JMB_ROOT_SCP/incoming/$jid.json.part"
    ssh "$JMB_HOST" "move /Y $JMB_ROOT_WIN\\incoming\\$jid.json.part $JMB_ROOT_WIN\\incoming\\$jid.json >nul"
    echo "enqueued $jid"
  fi
  sleep 1
done
