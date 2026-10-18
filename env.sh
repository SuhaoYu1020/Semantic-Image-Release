#!/usr/bin/env bash
# ============================================================================
# env.sh — single source of truth for every path/resource this repo uses.
#
#   source /path/to/repo/env.sh
#
# Nothing here is machine-specific *by force*: every variable is set only if
# unset, so you override by (a) exporting before sourcing, or (b) editing
# machine.env (gitignored, one per server — copy machine.env.example).
#
# Layout this assumes (the repo is the `code/` dir inside a workspace):
#
#   $SI_ROOT/                  workspace  (default: parent of this repo)
#   ├── code/                  <- $SI_REPO, this git repo
#   ├── data/                  <- $SI_DATA    (from HF, see scripts/fetch_data.sh)
#   ├── ckpts/                 <- $SI_CKPTS   (from HF, see scripts/fetch_base_weights.py)
#   ├── runs/                  <- $SI_RUNS    (training output, never in git)
#   └── hf_cache/              <- $HF_HOME
# ============================================================================

# ---- roots -----------------------------------------------------------------
SI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export SI_REPO
export SI_ROOT="${SI_ROOT:-$(dirname "$SI_REPO")}"

# ---- per-machine overrides (not in git) ------------------------------------
if [ -f "$SI_REPO/machine.env" ]; then
  # shellcheck disable=SC1091
  . "$SI_REPO/machine.env"
fi

# ---- data / weights / outputs ----------------------------------------------
export SI_DATA="${SI_DATA:-$SI_ROOT/data}"
export SI_CKPTS="${SI_CKPTS:-$SI_ROOT/ckpts}"
export SI_RUNS="${SI_RUNS:-$SI_ROOT/runs}"
export HF_HOME="${HF_HOME:-$SI_ROOT/hf_cache}"

# huggingface_hub looks for the auth token at $HF_HOME/token. Pointing HF_HOME
# at a project-local cache therefore hides a token stored in the default
# location, and every private-repo download 401s. Carry it across explicitly.
if [ -z "${HF_TOKEN:-}" ]; then
  for _t in "$HF_HOME/token" "${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/token"; do
    if [ -r "$_t" ]; then HF_TOKEN="$(cat "$_t")"; export HF_TOKEN; break; fi
  done
  unset _t
fi

# ---- conda -----------------------------------------------------------------
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export SI_ENV="${SI_ENV:-qwen-image-edit}"        # training / rendering
export SI_ENV_OCR="${SI_ENV_OCR:-eval-ocr-b200}"  # Qwen3-VL OCR scoring

# OCR scorer used by score_render_ocr.py (the metric of record).
# A local dir wins over the hub id, so this works on offline compute nodes.
export SI_OCR_MODEL="${SI_OCR_MODEL:-$SI_CKPTS/Qwen3-VL-2B-Instruct}"

# ---- SLURM (used by scripts/sub.sh, NOT by #SBATCH directives) -------------
# #SBATCH lines are comments parsed before any shell runs, so they cannot read
# these. scripts/sub.sh passes them as CLI flags, which override the directives.
export SI_PARTITION="${SI_PARTITION:-dgx-b200}"
export SI_ACCOUNT="${SI_ACCOUNT:-jgu32-lab}"
export SI_GPU_TYPE="${SI_GPU_TYPE-B200}"           # "" => untyped  --gres=gpu:N (dash, not :-, so an EMPTY machine.env override survives)
export SI_CPUS_PER_GPU="${SI_CPUS_PER_GPU:-28}"    # cluster CLI filter minimum
export SI_MEM_PER_GPU_G="${SI_MEM_PER_GPU_G:-224}" # cluster CLI filter maximum
export SI_TIME="${SI_TIME:-7-00:00:00}"

# ---- runtime defaults ------------------------------------------------------
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"         # compute nodes have no internet
export WANDB_SILENT="${WANDB_SILENT:-true}"
export PYTHONPATH="$SI_REPO${PYTHONPATH:+:$PYTHONPATH}"

# ---- helpers ---------------------------------------------------------------
# si_activate [env-name] : activate the conda env and put its bin first on PATH
si_activate() {
  local e="${1:-$SI_ENV}"
  # shellcheck disable=SC1091
  . "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$e"
  export PATH="$CONDA_ROOT/envs/$e/bin:$PATH"   # ninja/nvcc JIT needs this
}

# si_gres N : emit the --gres value for N GPUs on this cluster
si_gres() {
  local n="${1:-1}"
  if [ -n "$SI_GPU_TYPE" ]; then echo "gpu:${SI_GPU_TYPE}:${n}"; else echo "gpu:${n}"; fi
}
