#!/usr/bin/env bash
# Print the PC queue status: runner, holds, queue with expected GPU minutes, last ten results.
#   bench/queue/status.sh            status
#   bench/queue/status.sh gate       one reading of the GPU gate
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/common.sh"
ssh "$JMB_HOST" "\"$JMB_PY_WIN\" $JMB_ROOT_WIN\\runner.py ${1:-status}"
