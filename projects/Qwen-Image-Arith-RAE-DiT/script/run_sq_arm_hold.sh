#!/bin/bash
# run_sq_arm_hold.sh HOLD_JOBID ARM RUN_NAME [extra trainer args...]
# Launches sq_arm_step.sh (trainer + eval watcher) as a detached `srun --overlap` step inside a running interactive hold.
#   HOLD=8794233 bash script/run_sq_arm_hold.sh 8794233 siglip2 sq_siglip2
set -uo pipefail
HOLD="$1"; ARM="$2"; RUN_NAME="$3"; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
SRUN="$(command -v srun || echo /cm/local/apps/slurm/current/bin/srun)"
mkdir -p "${ROOT}/logs"
nohup "${SRUN}" --jobid="${HOLD}" --overlap --ntasks=1 --cpus-per-task="${CPUS:-8}" --job-name="sq-${ARM}" \
    bash "${HERE}/sq_arm_step.sh" "${ARM}" "${RUN_NAME}" "$@" > "${ROOT}/logs/sq_${RUN_NAME}_step.log" 2>&1 &
echo "[launch] hold ${HOLD} arm ${ARM} run ${RUN_NAME} -> logs/sq_${RUN_NAME}_{step,train,watch}.log (srun client pid $!)"
