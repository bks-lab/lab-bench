#!/usr/bin/env bash
# Install or update the queue runner on the PC and (re)start the scheduled task.
#   bench/queue/install.sh            copy runner.py, register task bench-queue if missing, start it
#   bench/queue/install.sh --update   copy runner.py, end and restart the task (current job is lost:
#                                     only when the queue is idle or paused)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
. "$here/common.sh"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
ssh "$JMB_HOST" "if not exist $JMB_ROOT_WIN mkdir $JMB_ROOT_WIN"
scp -q "$here/runner.py" "$JMB_HOST:$JMB_ROOT_SCP/runner.py"
# The task XML is a template: the SID of the PC account, pythonw.exe next to
# JMB_PY_WIN and the queue root are filled in here, so none of them is
# committed. schtasks wants the XML as UTF-16.
sid="$(ssh "$JMB_HOST" "whoami /user /fo csv /nh" | tr -d '\r"' | awk -F, '{print $2}')"
[[ "$sid" == S-1-* ]] || { echo "could not read the account SID from $JMB_HOST (got: $sid)" >&2; exit 1; }
pythonw="${JMB_PYW_WIN:-${JMB_PY_WIN%python.exe}pythonw.exe}"
python3 - "$here/bench-queue.task.xml" "$tmp/bench-queue.xml" "$sid" "$pythonw" "$JMB_ROOT_WIN" <<'PY2'
import sys
src, dst, sid, pyw, root = sys.argv[1:]
s = open(src, encoding="utf-8").read()
s = s.replace("@USER_SID@", sid).replace("@PYTHONW@", pyw).replace("@ROOT@", root)
assert "@" not in s.split("-->", 1)[1], "unfilled placeholder in task XML"
open(dst, "w", encoding="utf-16").write(s)
PY2
scp -q "$tmp/bench-queue.xml" "$JMB_HOST:$JMB_ROOT_SCP/bench-queue.task.xml"
cat > "$tmp/install.bat" <<BAT
@echo off
schtasks /query /tn bench-queue >nul 2>&1
if errorlevel 1 (
  schtasks /create /tn bench-queue /xml $JMB_ROOT_WIN\\bench-queue.task.xml
)
if "%1"=="--update" (
  schtasks /end /tn bench-queue
  ping -n 8 127.0.0.1 >nul
)
schtasks /run /tn bench-queue
BAT
sed -i '' 's/$/\r/' "$tmp/install.bat"
scp -q "$tmp/install.bat" "$JMB_HOST:$JMB_ROOT_SCP/install.bat"
ssh "$JMB_HOST" "$JMB_ROOT_WIN\\install.bat ${1:-}"
sleep 5
"$here/status.sh" | head -5
