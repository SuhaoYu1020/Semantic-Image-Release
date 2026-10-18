#!/bin/bash
# 100-epoch hard one-to-many curriculum, phase 2: when the parent arm reaches the switch snapshot, start the shift-1 continuation
# (weights only, fresh optimizer, warmup 100) for the remaining budget on the hold with the most free GPU memory.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"   # the project dir (script/..)
HOLDS=(${HOLDS:-8966589 8966590 8966591 8966592 8966593}); HARD=${SI_DATA:-$(dirname "$(dirname "$(dirname "$PWD")")")/data}/arithmetic_add_unordered_sq256_same_anshard; B=112500
SRUN="$(command -v srun || echo /cm/local/apps/slurm/current/bin/srun)"
freest_hold() { local best="" bestfree=-1; for h in "${HOLDS[@]}"; do u=$(timeout 60 "$SRUN" --jobid=$h --overlap --ntasks=1 --cpus-per-task=1 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tail -n1); [ -z "$u" ] && continue; f=$((183000 - u)); if [ "$f" -gt "$bestfree" ]; then best=$h; bestfree=$f; fi; done; echo "$best $bestfree"; }
# spec: source_run switch_step new_run arm   (switch at 24 / 50 / 76 epochs = multiples of the 2250-step snapshot cadence)
SPECS=(
 "mm_om100_s15_vae 27000 mm_om100_s15to1_24_vae vae" "mm_om100_s15_siglip2 27000 mm_om100_s15to1_24_siglip2 siglip2"
 "mm_om100_s15_vae 56250 mm_om100_s15to1_50_vae vae" "mm_om100_s15_siglip2 56250 mm_om100_s15to1_50_siglip2 siglip2"
 "mm_om100_reg_vae 56250 mm_om100_regto1_50_vae vae" "mm_om100_reg_siglip2 56250 mm_om100_regto1_50_siglip2 siglip2"
 "mm_om100_s15_vae 85500 mm_om100_s15to1_76_vae vae" "mm_om100_s15_siglip2 85500 mm_om100_s15to1_76_siglip2 siglip2"
)
pending=("${SPECS[@]}")
while [ ${#pending[@]} -gt 0 ]; do
  rest=()
  for spec in "${pending[@]}"; do set -- $spec; src=$1; st=$2; new=$3; arm=$4
    if [ -d runs/$new ]; then continue; fi
    if [ -f runs/$src/step_$st/meta.json ] && [ -f runs/$src/step_$st/dit/model.safetensors ]; then
      read h free < <(freest_hold)
      if [ -n "$h" ] && [ "$free" -gt 45000 ]; then
        echo "[$(date +%T)] $new from $src/step_$st -> hold $h (free ${free} MiB)"
        ARCH=mmdit SHIFT=1 MAX_STEPS=$((B - st)) SAVE_EVERY=2250 N_TEST=50 THRESH=2 WARMUP=100 bash script/run_sq_arm_hold.sh $h $arm $new --data_root $HARD --lr_horizon_epochs 100 --seed 42 --init_ckpt runs/$src/step_$st | grep -v launch
        sleep 90
      else
        echo "[$(date +%T)] $new ready but no hold with >45 GB free (best $free MiB); retry later"; rest+=("$spec")
      fi
    else rest+=("$spec"); fi
  done
  pending=("${rest[@]}"); [ ${#pending[@]} -gt 0 ] && sleep 900
done
echo "[$(date +%T)] all 100-epoch phase-2 continuations launched"
