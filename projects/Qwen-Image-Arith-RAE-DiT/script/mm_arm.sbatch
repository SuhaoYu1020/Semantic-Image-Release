#!/bin/bash
#SBATCH --job-name=mm-arm
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# One tokenizer arm of the MMDiT panel as a batch job (trainer + eval watcher on the job's GPU), for when no interactive
# hold is available.  Env: ARM (required), RUN_NAME (default mm_<ARM>), ARCH (mmdit), SHIFT (15), N_TEST (200), THRESH (0.95).
_d="${SI_ENV_SH:-}"; if [ -z "$_d" ]; then _d="${SLURM_SUBMIT_DIR:-$PWD}"; while [ "$_d" != "/" ] && [ ! -f "$_d/env.sh" ]; do _d="$(dirname "$_d")"; done; _d="$_d/env.sh"; fi
. "$_d"
set -uo pipefail
ROOT=${SI_REPO}/projects/Qwen-Image-Arith-RAE-DiT; cd "${ROOT}"
: "${ARM:?set ARM}"; RUN_NAME="${RUN_NAME:-mm_${ARM}}"
export ARCH="${ARCH:-mmdit}" SHIFT="${SHIFT:-15}" N_TEST="${N_TEST:-200}" THRESH="${THRESH:-0.95}"
echo "[mm-arm] $(hostname) arm=${ARM} run=${RUN_NAME} arch=${ARCH} shift=${SHIFT} gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) $(date +%T)"
bash script/sq_arm_step.sh "${ARM}" "${RUN_NAME}"
echo "[mm-arm] DONE $(date +%T)"
