#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/.hf_home
export HF_HUB_DISABLE_XET=1
/venv/main/bin/python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="imagenet-1k",
    repo_type="dataset",
    allow_patterns=["data/validation-*.parquet"],
    max_workers=4,
)
print(f"ImageNet validation snapshot: {path}", flush=True)
PY
