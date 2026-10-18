#!/bin/bash
# sq_arm_step.sh ARM RUN_NAME [extra trainer args...]
# One srun step inside an interactive hold: trains the square-canvas DiT-only row for ARM (background) and runs the
# eval watcher (foreground) on the same GPU; the watcher writes STOP at test OCR >= THRESH, the trainer exits on it.
#   THRESH (0.95), N_TEST (0 = all 1000), SAVE_EVERY (100), BATCH (64), LR (2e-4), DEPTH (12), HIDDEN (1152), HEADS (16),
#   SHIFT ('rule'), EPOCHS (300), MAX_STEPS (unset), SMOKE_STEPS (unset: normal run; N -> --max_steps N --smoke-free quick run)
_d="${SI_ENV_SH:-}"; if [ -z "$_d" ]; then _d="$(cd "$(dirname "$0")" && pwd)"; while [ "$_d" != "/" ] && [ ! -f "$_d/env.sh" ]; do _d="$(dirname "$_d")"; done; _d="$_d/env.sh"; fi
. "$_d"
set -uo pipefail
ARM="$1"; RUN_NAME="$2"; shift 2
ROOT=${SI_REPO}/projects/Qwen-Image-Arith-RAE-DiT
RUN="${ROOT}/runs/${RUN_NAME}"
export PYTHONPATH="${SI_REPO}" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=offline WANDB_SILENT=true
PY="${CONDA_ROOT}/envs/qwen-image-edit/bin/python"
mkdir -p "${RUN}" "${ROOT}/logs"
cd "${ROOT}"
echo "[step] $(hostname) arm=${ARM} run=${RUN} gpu=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1) $(date +%T)"
STATS="${ROOT}/stats/sq256_same/${ARM}.pt"
[ -f "${STATS}" ] || { echo "[step] computing latent stats for ${ARM}"; "${PY}" compute_latent_stats.py --arm "${ARM}" || exit 1; }

TRAIN_ARGS=(--arm "${ARM}" --out_dir "${RUN}" --batch_size "${BATCH:-64}" --lr "${LR:-2e-4}" --depth "${DEPTH:-12}"
            --hidden "${HIDDEN:-1152}" --heads "${HEADS:-16}" --sigma_shift "${SHIFT:-rule}" --save_every "${SAVE_EVERY:-100}"
            --epochs "${EPOCHS:-300}" --warmup_steps "${WARMUP:-500}" --arch "${ARCH:-chcat}")
[ -n "${MAX_STEPS:-}" ] && TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
"${PY}" train_i2i_sq_dit.py "${TRAIN_ARGS[@]}" "$@" > "${ROOT}/logs/sq_${RUN_NAME}_train.log" 2>&1 &
TPID=$!
echo "[step] trainer pid ${TPID}; log logs/sq_${RUN_NAME}_train.log"
sleep 60
if ! kill -0 ${TPID} 2>/dev/null; then echo "[step] trainer died early:"; tail -20 "${ROOT}/logs/sq_${RUN_NAME}_train.log"; exit 1; fi
bash "${ROOT}/script/sq_watch_eval.sh" "${RUN}" "${THRESH:-0.95}" "${N_TEST:-0}" > "${ROOT}/logs/sq_${RUN_NAME}_watch.log" 2>&1
echo "[step] watcher exited $(date +%T); waiting for trainer ${TPID}"
wait ${TPID}; echo "[step] trainer exit $? $(date +%T)"
tail -3 "${RUN}/eval/ledger.tsv"
