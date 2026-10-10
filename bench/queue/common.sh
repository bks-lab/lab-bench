# Shared settings for the Mac side of the PC job queue (sourced, not run).
# Machine-specific values (ssh target, Python path on the PC) are not
# committed: they come from the environment or from bench/local.env
# (gitignored, copy bench/local.env.example).
_jmb_local="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/local.env"
# shellcheck disable=SC1090
[[ -f "$_jmb_local" ]] && . "$_jmb_local"
: "${JMB_HOST:?set JMB_HOST (ssh target of the GPU PC, user@host) in bench/local.env or the environment}"
: "${JMB_PY_WIN:?set JMB_PY_WIN (python.exe on the PC) in bench/local.env or the environment}"
JMB_ROOT_WIN="${JMB_ROOT_WIN:-D:\\jmb-queue}"
JMB_ROOT_SCP="${JMB_ROOT_SCP:-D:/jmb-queue}"
