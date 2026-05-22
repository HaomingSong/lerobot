#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="/mnt/hwfile/songhaoming/workspace/lerobot_review"
CHECKOUT="${WORKSPACE_ROOT}/lerobot-pr_3269_fix"
OUTPUT_DIR="${WORKSPACE_ROOT}/assets/outputs/pr-3629"
RUNTIME_DIR="${WORKSPACE_ROOT}/assets/runtime"

mkdir -p \
  "${OUTPUT_DIR}" \
  "${RUNTIME_DIR}/hf" \
  "${RUNTIME_DIR}/hf/hub" \
  "${RUNTIME_DIR}/transformers" \
  "${RUNTIME_DIR}/tmp" \
  "${RUNTIME_DIR}/apptainer-tmp" \
  "${RUNTIME_DIR}/xdg-cache" \
  "${RUNTIME_DIR}/torch" \
  "${RUNTIME_DIR}/pip-cache"

export HF_HOME="${RUNTIME_DIR}/hf"
export HF_HUB_CACHE="${RUNTIME_DIR}/hf/hub"
export TRANSFORMERS_CACHE="${RUNTIME_DIR}/transformers"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME="${RUNTIME_DIR}/xdg-cache"
export PIP_CACHE_DIR="${RUNTIME_DIR}/pip-cache"
export TMPDIR="${RUNTIME_DIR}/tmp"
export APPTAINER_CACHEDIR="${RUNTIME_DIR}/apptainer-cache"
export APPTAINER_TMPDIR="${RUNTIME_DIR}/apptainer-tmp"
export LEROBOT_APPTAINER_TMPDIR="${RUNTIME_DIR}/tmp"
export LEROBOT_WORKSPACE="${CHECKOUT}"
export PYTHONPATH="${CHECKOUT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${CHECKOUT}"

echo "hostname=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unknown}"
echo "date=$(date --iso-8601=seconds)"
echo "commit=$(git rev-parse HEAD)"

/mnt/hwfile/songhaoming/bin/lerobot-apptainer \
  python scripts/topreward_probe/write_qwen3vl_processor_intermediate_report_zh.py \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size 2 \
    --num-frames 4 \
    --height 64 \
    --width 64

echo "date_done=$(date --iso-8601=seconds)"
