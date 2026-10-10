#!/usr/bin/env bash
# Copy one job's out dir, log and status from the PC into a local folder.
#   bench/queue/fetch.sh <job id> [dest dir, default .cache/queue/<id>]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/common.sh"
id="$1"; dest="${2:-$(git -C "$here" rev-parse --show-toplevel)/.cache/queue/$id}"
mkdir -p "$dest"
scp -q -r "$JMB_HOST:$JMB_ROOT_SCP/out/$id/*" "$dest/" || true
scp -q "$JMB_HOST:$JMB_ROOT_SCP/logs/$id.log" "$dest/" || true
scp -q "$JMB_HOST:$JMB_ROOT_SCP/done/$id.status.json" "$dest/" 2>/dev/null \
  || scp -q "$JMB_HOST:$JMB_ROOT_SCP/failed/$id.status.json" "$dest/" || true
ls -la "$dest"
