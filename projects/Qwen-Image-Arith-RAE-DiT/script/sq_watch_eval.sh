#!/bin/bash
# sq_watch_eval.sh RUN_DIR [THRESH=0.95] [N_TEST=0]
# Watches a train_i2i_sq_dit.py run dir: whenever a new snapshot (step_*/meta.json) appears, renders the test split with
# render_eval_sq_dit.py (qwen-image-edit env), scores the answer-line crop with score_render_ocr_crop.py (eval-ocr-b200
# env), appends to RUN_DIR/eval/ledger.tsv and, once test OCR >= THRESH, writes RUN_DIR/STOP (the trainer exits on it).
# Always evaluates the NEWEST unevaluated snapshot first (eval is slower than 100 training steps); older ones are left
# unevaluated unless BACKFILL=1.  Exits when STOP was written (and the final/ dir, if present, is scored) or when the
# trainer wrote DONE.json and nothing is left to score.
_d="${SI_ENV_SH:-}"; if [ -z "$_d" ]; then _d="$(cd "$(dirname "$0")" && pwd)"; while [ "$_d" != "/" ] && [ ! -f "$_d/env.sh" ]; do _d="$(dirname "$_d")"; done; _d="$_d/env.sh"; fi
. "$_d"
set -uo pipefail
RUN="$1"; THRESH="${2:-0.95}"; N_TEST="${3:-0}"
ROOT=${SI_REPO}/projects/Qwen-Image-Arith-RAE-DiT
export PYTHONPATH="${SI_REPO}" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY_TRAIN="${CONDA_ROOT}/envs/qwen-image-edit/bin/python"
PY_OCR="${CONDA_ROOT}/envs/eval-ocr-b200/bin/python"
mkdir -p "${RUN}/eval"
LEDGER="${RUN}/eval/ledger.tsv"
[ -f "${LEDGER}" ] || printf 'tag\tstep\tocr_acc\tgt_ocr_acc\tn\tlatent_rmse\trender_sec\tocr_sec\twhen\n' > "${LEDGER}"
echo "[watch] $(hostname) run=${RUN} thresh=${THRESH} n_test=${N_TEST} $(date +%T)"

evaluated() { awk -F'\t' -v t="$1" 'NR>1 && $1==t {f=1} END{exit !f}' "${LEDGER}"; }

score_one() {   # $1 = ckpt dir, $2 = tag, $3 = n_test ; echoes acc
  # NOTE: one `local a=.. b="$a"` line expands ALL words before any assignment, so `out` would see the CALLER's
  # tag, not this call's -> split the declarations (this bit the first _full confirmation run).
  local ck="$1" tag="$2"
  local n="${3:-$N_TEST}"
  local out="${RUN}/eval/${tag}"
  local t0 t1
  t0=$(date +%s)
  "${PY_TRAIN}" "${ROOT}/render_eval_sq_dit.py" --ckpt "${ck}" --out_dir "${out}" --n_test "${n}" --n_train -1 ${EVAL_STEPS:+--steps ${EVAL_STEPS}} \
      > "${out}.render.log" 2>&1 || { echo "[watch] RENDER FAILED for ${tag}: $(tail -3 "${out}.render.log")"; return 1; }
  t1=$(date +%s)
  local man="${out}/ocr_manifest.json"
  printf '{"root": "%s", "configs": [{"n": "%s", "group": "sq", "label": "%s", "dir": "eval/%s"}]}\n' "${RUN}" "${tag}" "${tag}" "${tag}" > "${man}"
  "${PY_OCR}" "${SI_REPO}/projects/Qwen-Image-Arith-ScratchTE/score_render_ocr_crop.py" --model "${SI_OCR_MODEL}" \
      --manifest "${man}" --out_dir "${out}/ocr" --no_full > "${out}.ocr.log" 2>&1 || { echo "[watch] OCR FAILED for ${tag}: $(tail -3 "${out}.ocr.log")"; return 1; }
  local t2; t2=$(date +%s)
  "${PY_TRAIN}" - "${out}" "${tag}" "${LEDGER}" $((t1-t0)) $((t2-t1)) <<'EOF'
import json, sys, time, os
out, tag, ledger, rsec, osec = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
s = json.load(open(os.path.join(out, "ocr", "summary.json")))
row = [r for r in s["rows"] if r["n"] == tag][0]
b = row.get("test") or row.get("train") or {}
rep = json.load(open(os.path.join(out, "report.json")))
step = rep["meta"].get("global_step")
acc, gt, n = b.get("ocr_acc"), b.get("gt_ocr_acc"), b.get("n")
with open(ledger, "a") as f:
    f.write(f"{tag}\t{step}\t{acc}\t{gt}\t{n}\t{rep.get('latent_rmse_mean')}\t{rsec}\t{osec}\t{time.strftime('%H:%M:%S')}\n")
print(f"ACC {acc} GT {gt} N {n} STEP {step}")
EOF
}

while true; do
  cands=()
  for d in "${RUN}"/step_*; do
    [ -f "$d/meta.json" ] || continue
    tag=$(basename "$d"); evaluated "${tag}" || cands+=("$d")
  done
  if [ ${#cands[@]} -gt 0 ]; then
    # newest first (numeric step)
    pick=$(printf '%s\n' "${cands[@]}" | awk -F'step_' '{print $NF"\t"$0}' | sort -n | tail -1 | cut -f2)
    tag=$(basename "${pick}")
    echo "[watch] $(date +%T) scoring ${tag} (${#cands[@]} unevaluated)"
    res=$(score_one "${pick}" "${tag}" "${N_TEST}") || { sleep 20; continue; }
    echo "[watch] ${tag}: ${res}"
    acc=$(echo "${res}" | awk '{print $2}')
    if [ "${N_TEST}" != "0" ] && "${PY_TRAIN}" -c "import sys; sys.exit(0 if float('${acc}') >= float('${THRESH}') else 1)"; then
      # quick subset cleared the bar -> confirm on the FULL test split before stopping
      echo "[watch] ${tag}: quick ${acc} >= ${THRESH}; confirming on the full test split"
      res=$(score_one "${pick}" "${tag}_full" 0) || { sleep 20; continue; }
      echo "[watch] ${tag}_full: ${res}"
      acc=$(echo "${res}" | awk '{print $2}')
    fi
    if "${PY_TRAIN}" -c "import sys; sys.exit(0 if float('${acc}') >= float('${THRESH}') else 1)"; then
      echo "[watch] *** ${tag} test OCR ${acc} >= ${THRESH} -> writing STOP ***"
      date +%T > "${RUN}/STOP"
      echo "${tag}" > "${RUN}/BEST_TAG"
      break
    fi
    [ "${BACKFILL:-0}" = "1" ] || true
    continue
  fi
  if [ -f "${RUN}/DONE.json" ]; then
    if [ -f "${RUN}/final/meta.json" ] && ! evaluated final; then
      res=$(score_one "${RUN}/final" final 0) && echo "[watch] final: ${res}"
    fi
    echo "[watch] trainer finished and nothing left to score -> exit $(date +%T)"; break
  fi
  sleep 30
done
# score the final snapshot too if the trainer wrote one after STOP
for i in $(seq 1 60); do [ -f "${RUN}/DONE.json" ] && break; sleep 10; done
if [ -f "${RUN}/final/meta.json" ] && ! evaluated final; then
  res=$(score_one "${RUN}/final" final 0) && echo "[watch] final: ${res}"
fi
echo "[watch] done $(date +%T)"; cat "${LEDGER}"
