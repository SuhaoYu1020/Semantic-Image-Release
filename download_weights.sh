#!/usr/bin/env bash
# Download every external weight this package needs, into the folder names the code expects.
#   bash download_weights.sh            # everything (VAE arm + SigLIP2 arm + OCR model), ~10.5 GB
#   bash download_weights.sh vae ocr    # only what the VAE arm and the OCR scorer need, ~4.8 GB
# Target root = $SI_CKPTS (from env.sh) or ./ckpts.  Needs `pip install -U huggingface_hub` (gives huggingface-cli / hf).
# All repos are public; a token is only needed if your network requires one for the Hub.
set -euo pipefail
CK="${SI_CKPTS:-$PWD/ckpts}"; mkdir -p "$CK"
if command -v hf >/dev/null 2>&1; then DL="hf download"; elif command -v huggingface-cli >/dev/null 2>&1; then DL="huggingface-cli download"; else echo "install huggingface_hub first: pip install -U huggingface_hub" >&2; exit 1; fi
want="${*:-vae siglip2 ocr}"
for w in $want; do
  case "$w" in
    vae)      # Qwen-Image VAE: recon_gate.VaeCodec loads AutoencoderKLQwenImage.from_pretrained("$SI_CKPTS/Qwen-Image", subfolder="vae")
              echo "[vae] Qwen/Qwen-Image (vae/* only) -> $CK/Qwen-Image"; $DL Qwen/Qwen-Image --include "vae/*" --local-dir "$CK/Qwen-Image" ;;
    siglip2)  # SigLIP2 so400m/14 encoder + Scale-RAE decoder: scalerae_codec.SPECS["siglip2_so400m"] encoder_dir / decoder_dir
              echo "[siglip2] google/siglip2-so400m-patch14-224 -> $CK/siglip2-so400m-patch14-224"; $DL google/siglip2-so400m-patch14-224 --local-dir "$CK/siglip2-so400m-patch14-224"
              echo "[siglip2] nyu-visionx/siglip2_decoder -> $CK/ScaleRAE-siglip2_decoder"; $DL nyu-visionx/siglip2_decoder --local-dir "$CK/ScaleRAE-siglip2_decoder" ;;
    ocr)      # Qwen3-VL-2B-Instruct for score_render_ocr_crop.py; point SI_OCR_MODEL at the folder (or leave it as the hub id when online)
              echo "[ocr] Qwen/Qwen3-VL-2B-Instruct -> $CK/Qwen3-VL-2B-Instruct"; $DL Qwen/Qwen3-VL-2B-Instruct --local-dir "$CK/Qwen3-VL-2B-Instruct"
              echo "    export SI_OCR_MODEL=$CK/Qwen3-VL-2B-Instruct" ;;
    *) echo "unknown target '$w' (vae | siglip2 | ocr)" >&2; exit 2 ;;
  esac
done
echo "[done] weights under $CK  (set SI_CKPTS=$CK in env.sh)"
ls -1 "$CK"
