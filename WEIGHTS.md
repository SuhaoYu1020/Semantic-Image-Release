# Weights and other external files

All weights are public Hugging Face repositories. The code loads them from fixed sub-directories of `$SI_CKPTS`
(set in `env.sh`), so download them to exactly these folder names. `bash download_weights.sh` (package root) runs all
the commands below; `huggingface-cli` comes with `pip install huggingface_hub`, `hf download` is the newer spelling.

## Required for the VAE arm (`--arm vae`)

| what | repo | local folder | size |
|---|---|---|---|
| Qwen-Image VAE (only the `vae/` subfolder is loaded by `diffusers.AutoencoderKLQwenImage`) | `Qwen/Qwen-Image` | `$SI_CKPTS/Qwen-Image/vae/` | 243 MB |

```bash
huggingface-cli download Qwen/Qwen-Image --include "vae/*" --local-dir "$SI_CKPTS/Qwen-Image"
```

`recon_gate.VaeCodec` calls `AutoencoderKLQwenImage.from_pretrained(f"{SI_CKPTS}/Qwen-Image", subfolder="vae")`, so
`$SI_CKPTS/Qwen-Image/vae/config.json` and the safetensors file must exist; nothing else from Qwen-Image is needed.

## Required for the SigLIP2 arm (`--arm siglip2`)

| what | repo | local folder | size |
|---|---|---|---|
| SigLIP2 so400m/14 224 encoder | `google/siglip2-so400m-patch14-224` | `$SI_CKPTS/siglip2-so400m-patch14-224/` | 4.3 GB |
| Scale-RAE SigLIP2 decoder (`config.json`, `model.pt`) | `nyu-visionx/siglip2_decoder` | `$SI_CKPTS/ScaleRAE-siglip2_decoder/` | 1.6 GB |

```bash
huggingface-cli download google/siglip2-so400m-patch14-224 --local-dir "$SI_CKPTS/siglip2-so400m-patch14-224"
huggingface-cli download nyu-visionx/siglip2_decoder        --local-dir "$SI_CKPTS/ScaleRAE-siglip2_decoder"
```

The folder names are the `encoder_dir` / `decoder_dir` entries of `SPECS["siglip2_so400m"]` in `scalerae_codec.py`.
The decoder is the one released with "Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders"
(Scale-RAE, arXiv 2601.16208); it is used frozen, exactly as released.

## Required for OCR scoring

| what | repo | how it is referenced |
|---|---|---|
| Qwen3-VL-2B-Instruct | `Qwen/Qwen3-VL-2B-Instruct` | `SI_OCR_MODEL` in `env.sh` (a hub id works; a local snapshot path avoids re-downloads on offline nodes) |

```bash
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir "$SI_CKPTS/Qwen3-VL-2B-Instruct"
export SI_OCR_MODEL="$SI_CKPTS/Qwen3-VL-2B-Instruct"
```

## Latent statistics (included)

`projects/Qwen-Image-Arith-RAE-DiT/stats/sq256_same/{vae,siglip2}.pt` are the per-channel mean / std used to
standardise the latents (computed on the one-to-one training renders with `compute_latent_stats.py`). The trainer
picks `stats/sq256_same/<arm>.pt` by default (`--stats` to override, `--no_stats` to disable).

## Fonts (included under `fonts_extra/`)

The hard one-to-many variants use 8 answer fonts: DejaVuSansMono-Bold, DejaVuSansMono-Oblique and
LiberationMono-Regular from the system (`/usr/share/fonts/truetype/{dejavu,liberation}/`, packages `fonts-dejavu`
and `fonts-liberation`), plus five Google Fonts (OFL) shipped here in `fonts_extra/`: Caveat, PermanentMarker,
PressStart2P, Orbitron, Pacifico. They are needed to regenerate the variants (`make_square_gate_set.py --answer_fonts`)
and by the glyph-fidelity scorer, which rebuilds the template bank from the font list stored in
`arithmetic_add_unordered_sq256_same_ansfont_hard_k0/stats.json`. That list holds our cluster's absolute paths;
`om_fidelity_score.fonts()` falls back to `$SI_DATA/fonts_extra/<name>` and the system font directories when a path
does not exist, so copy `fonts_extra/` into your `$SI_DATA` (or edit the paths in `stats.json`).

## Data

The dataset zip `arith_sq256_1to1_1toMany_2026-10-17.zip` unpacks into `$SI_DATA/` (it contains
`arithmetic_add_unordered_sq256_same/`, the eight `…_ansfont_hard_k*/` folders and `…_anshard/`). Its `README.md`
describes the layout.
