#!/usr/bin/env bash
# Build the four Hugging Face repositories from this checkout and upload them.
# Needs HF_TOKEN in the environment (write access to the bks-lab organisation)
# and the hf CLI (pip install huggingface_hub). CI runs this on every push to
# main, see .github/workflows/hf-sync.yml.
#
#   bks-lab/lab-bench-results          dataset   hf/build.py
#   bks-lab/lab-bench                  Space     hf/space/ + the results tables
#   bks-lab/xrechnung-invoice-fields   dataset   hf/c3/build.py
#   bks-lab/xrechnung-invoice-fields-viewer  Space  hf/c3/space/ + invoices, predictions
#
# Stale files under data/, pages/, xml/ and pdf/ are deleted on upload, so a
# row or page that left the repository also leaves the Hub.
set -euo pipefail
: "${HF_TOKEN:?HF_TOKEN is not set}"
root=$(cd "$(dirname "$0")/.." && pwd)
commit=$(git -C "$root" rev-parse HEAD)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
msg="lab-bench ${commit}"

python3 "$root/hf/build.py" --out "$out/results" --commit "$commit"
python3 "$root/hf/c3/build.py" --out "$out/c3" --commit "$commit"

mkdir -p "$out/results-space/data" "$out/c3-space/data"
cp "$root"/hf/space/* "$out/results-space/"
cp "$out"/results/data/*.jsonl "$out/results-space/data/"
cp "$root"/hf/c3/space/* "$out/c3-space/"
cp "$out/c3/invoices.jsonl" "$out/c3/predictions.jsonl" "$out/c3-space/data/"

hf upload bks-lab/lab-bench-results "$out/results" . --repo-type dataset \
  --delete 'data/*' --commit-message "$msg"
hf upload bks-lab/lab-bench "$out/results-space" . --repo-type space \
  --delete 'data/*' --commit-message "$msg"
hf upload bks-lab/xrechnung-invoice-fields "$out/c3" . --repo-type dataset \
  --delete 'pages/*' --delete 'xml/*' --delete 'pdf/*' --commit-message "$msg"
hf upload bks-lab/xrechnung-invoice-fields-viewer "$out/c3-space" . --repo-type space \
  --delete 'data/*' --commit-message "$msg"
echo "published ${commit}"
