#!/bin/bash
# run_py_hold.sh HOLD_JOBID LOGNAME script.py [args...]  -- run a project python script inside a running interactive hold
# (srun --overlap, detached), with env.sh sourced, the qwen-image-edit env and the third_party extras on PYTHONPATH.
set -uo pipefail
HOLD="$1"; LOG="$2"; shift 2
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
SRUN="$(command -v srun || echo /cm/local/apps/slurm/current/bin/srun)"
mkdir -p "${ROOT}/logs"
nohup "${SRUN}" --jobid="${HOLD}" --overlap --ntasks=1 --cpus-per-task="${CPUS:-8}" --job-name="py-${LOG}" \
    bash -c 'cd "'"${ROOT}"'"; _d="$(pwd)"; while [ "$_d" != "/" ] && [ ! -f "$_d/env.sh" ]; do _d="$(dirname "$_d")"; done; . "$_d/env.sh"; set +u; export PYTHONPATH="${SI_ROOT}/third_party/pyextra_qie:${SI_ROOT}/third_party/ml-flextok:${SI_ROOT}/third_party/1d-tokenizer:${SI_REPO}"; export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; echo "[py-hold] $(hostname) $(date +%T) $*"; exec "${CONDA_ROOT}/envs/${PYENV:-qwen-image-edit}/bin/python" "$@"' _ "$@" > "${ROOT}/logs/${LOG}.log" 2>&1 &
echo "[launch] hold ${HOLD} ${LOG}: $* -> logs/${LOG}.log (pid $!)"
