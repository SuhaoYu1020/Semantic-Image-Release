# Latent arithmetic with a small MMDiT: training and evaluation code (VAE and SigLIP2 arms)

Exported 2026-10-17 from `semantic-image/code/`. This is the subset of the project needed to train the flow-matching
MMDiT on the square arithmetic renders in a fixed visual tokenizer's latent space (Qwen-Image VAE or SigLIP2) and to run
every evaluation used in the write-up: OCR accuracy, one-step x̂₀ read-out, the latent answer reader, glyph fidelity, and
the σ-schedule curriculum on the hard one-to-many task.

The directory layout mirrors the repository, because the scripts locate each other by relative path:

```
env.sh                                   cluster paths and env vars (edit SI_ROOT / SI_DATA / SI_CKPTS / CONDA_ROOT / SI_OCR_MODEL)
projects/Qwen-Image-Arith-RAE-DiT/       everything below
projects/Qwen-Image-Arith-Emit/score_render_ocr.py          OCR model loader + parsing (Qwen3-VL), imported by the crop scorer
projects/Qwen-Image-Arith-ScratchTE/score_render_ocr_crop.py  answer-line crop OCR scorer
```

## 1. Task, model, data

* Task: a 256×256 canvas shows `Q: a + b = ?` (a, b in 0..99); the target writes `A: <sum>` under it. One-to-one:
  one target per question (standard font). Hard one-to-many: eight valid targets per question, the answer digits in a
  random one of 8 fonts, rotated ±30°, shifted ±10 px.
* Model: two-stream MMDiT (12 blocks × 1152, 16 heads, 561M params), trained from scratch with flow matching
  (velocity target, logit-normal σ with a shift). Conditioning = the clean source-image tokens in a second stream
  (`--arch mmdit`); no text. Both streams live in the tokenizer's latent space.
* Latent spaces (`sq_codecs.py`, `--arm`): `vae` = Qwen-Image VAE (16 ch, 8× down, 2×2 packed → 16×16 tokens × 64),
  `siglip2` = SigLIP2-so400m patch tokens with the Scale-RAE decoder (16×16 tokens × 1152). Encoder and decoder are
  frozen. Latents are standardised per channel with `stats/sq256_same/<arm>.pt` (included; recompute with
  `compute_latent_stats.py`).
* Data: the dataset zip `arith_sq256_1to1_1toMany_2026-10-17.zip` (sent separately) holds the one-to-one set
  `arithmetic_add_unordered_sq256_same` (train 9000 / test 1000) and the hard variants `…_ansfont_hard_k0..k7`
  plus the training index `…_anshard`. To regenerate: `make_square_gate_set.py` (base) and, for variant k,
  `make_square_gate_set.py --answer_fonts <8 ttf paths> --answer_rotate 30 --answer_jitter 10 --answer_variant k`,
  then `make_anshard_split.py 8`.

## 2. Environment

Two conda environments were used; the split exists only because the OCR scorer's `transformers` pins conflict with the
training stack.

| env | used by | key packages |
|---|---|---|
| `qwen-image-edit` | training, rendering, readers, plots | torch 2.7.0+cu128, diffusers 0.39.0.dev0 (needs `AutoencoderKLQwenImage`, `AutoencoderRAE`), transformers 5.8.0, accelerate 1.13.0, safetensors, numpy, pillow, matplotlib |
| `eval-ocr-b200` | `score_render_ocr_crop.py`, `om_fidelity_score.py` | torch 2.7.0, transformers 4.57.6 (Qwen3-VL), opencv-python 4.13 |

Weights: `bash download_weights.sh [vae] [siglip2] [ocr]` fetches everything from the Hugging Face Hub into the folder
names the code expects under `$SI_CKPTS` (Qwen-Image VAE for `--arm vae`; SigLIP2 so400m encoder + Scale-RAE decoder
for `--arm siglip2`; Qwen3-VL-2B-Instruct for OCR). `WEIGHTS.md` lists the repos, sizes and where each path is used.
The five non-system fonts of the hard variants are shipped in `fonts_extra/`.

`env.sh` is sourced by every shell script and by `train_t2i_arith_mse.py` (here a small shim that only provides the
path constants and checkpoint helpers the trainer imports). Set `SI_ROOT` (parent of `code/`, `data/`, `ckpts/`),
`SI_DATA`, `SI_CKPTS`, `CONDA_ROOT`, `SI_OCR_MODEL`.

## 3. Training (`train_i2i_sq_dit.py`)

Direct invocation (one GPU, batch 64, the settings used throughout):

```bash
cd projects/Qwen-Image-Arith-RAE-DiT
python train_i2i_sq_dit.py --arm vae --arch mmdit --out_dir runs/mm_s15_vae \
  --data_root $SI_DATA/arithmetic_add_unordered_sq256_same \
  --sigma_shift 15 --batch_size 64 --lr 2e-4 --warmup_steps 500 --depth 12 --hidden 1152 --heads 16 \
  --save_every 400 --max_steps 11200 --lr_horizon_epochs 37 --seed 42
```

Knobs that define the experiments in the write-up:

| experiment | flags |
|---|---|
| σ schedule | `--sigma_shift 1 / 15 / 40` (logit-normal with that shift) |
| single-σ training (dose response) | `--sigma_sampler fixed --sigma_fixed 0.8` |
| σ = 1 regressor | `--sigma_sampler fixed --sigma_fixed 1.0` |
| two-point mixture (E1) | `--sigma_sampler two_point --sigma_mix_frac 1/16 --sigma_mix_low 0.8 --sigma_mix_high 1.0` |
| null-source copier (A_DiT) | `--null_source --sigma_sampler uniform` |
| init from another run (E2, curriculum phase 2) | `--init_ckpt runs/<run>/step_N --warmup_steps 100` (weights only; fresh optimizer and LR) |
| hard one-to-many | `--data_root $SI_DATA/arithmetic_add_unordered_sq256_same_anshard` |
| LR horizon | `--lr_horizon_epochs N` (cosine horizon decoupled from `--max_steps`) |
| resume in place | `--resume` (same out_dir; weights, optimizer, step) |

Each snapshot is `runs/<run>/step_N/{dit/model.safetensors, meta.json}`; `final/` is written at the end, `DONE.json`
marks completion, a `STOP` file in the run dir stops the trainer.

Cluster wrappers (SLURM, our interactive holds): `script/sq_arm_step.sh` runs trainer + the eval watcher
(`sq_watch_eval.sh`: renders every new snapshot, OCR-scores it, writes `eval/ledger.tsv`, writes `STOP` at
`THRESH`); `script/run_sq_arm_hold.sh HOLD ARM RUN [trainer args]` launches that inside a running hold with env
`ARCH SHIFT MAX_STEPS SAVE_EVERY N_TEST THRESH WARMUP`; `script/mm_arm.sbatch` is the batch-job form. These contain
site-specific `srun --overlap` logic and are included for reference only.

## 4. Evaluation

**Render + OCR (the main accuracy number).** `render_eval_sq_dit.py --ckpt runs/<run>/step_N --out_dir OUT
--n_test 1000 --steps 24 [--save_latents] [--sigma_stop S]` samples K steps (`--steps 1` = the one-step read-out
x̂₀ = z − σ·v̂), decodes, writes `OUT/imgs/*_pred.png` and `OUT/report.json`. Then, in the OCR env,
`score_render_ocr_crop.py --manifest OUT/ocr_manifest.json --out_dir OUT/ocr --no_full` reads the answer-line crop and
writes `ocr/summary.json` (`ocr_acc`) plus per-item `*.crop.ocr.json` (`ocr_pred`, `ok`).

**One-step read-out at the model's own σ (dose response, counterfactual canvas).** `eval_x0_sigma_curve.py` and the
driver `own_sigma_sweep.py`; `--source wrong` writes a wrong answer into the canvas to separate copying from computing.

**Latent answer reader ("does the latent hold the answer").** `latent_answer_probe.py --arm vae --fit std|hard`
fits three ridge digit-slot heads on the answer-box tokens of ground-truth latents (never on model outputs);
`--check` reports transfer to held-out GT; `--apply OUT --probe cache/ansprobe/<arm>_<src>.pt` reads a render dir
saved with `--save_latents`. `answer_reader_variants.py` compares this reader with one-vs-rest and kNN readers and
with readers trained on the per-question mean latent.

**Hard one-to-many, all read-outs per snapshot.** `om_sched_eval.py --run <run> --every 800 --n_test 200 --ks 1 24`
renders K = 1 and 24, OCR-scores, applies the readers, writes `runs/<run>/eval_om/summary.jsonl` (`ocr_acc`,
`probe_std`, knows/renders cross-tabs). `script/om_eval_loop.py SPACE` keeps doing this for every `mm_om_*_<SPACE>`
run (`PREFIX`, `EVERY` env for other sets); `script/om100_phase2_poller.sh` launches the shift-15 → shift-1
continuations when a parent run reaches the switch snapshot.

**Glyph fidelity (correctness-independent).** `om_fidelity_codec_gt.py --arm vae` builds the calibration references
(GT variants and their pixel / latent means through the codec); `om_fidelity_score.py calib` sets τ and
`om_fidelity_score.py score --tau T` scores every `eval_om` dir (`fidelity.json`, `fidelity.jsonl`: `fid`,
`fid_and_ocr`, template read, chosen font / angle). `om_fidelity_examples.py RUN TAG` draws the four-cell contact sheet.

**Figures.** `om_three_lines.py --space vae|siglip2 --k 24` (accuracy / fidelity / fidelity-and-correct per schedule),
`om_acc_vs_fidelity.py`.

## 5. Things to know before running

* The watcher ledger (`eval/ledger.tsv`) is a multi-step read-out; single-σ and σ = 1-only models must be read at
  their own σ with the one-step scripts, or they look far slower than they are.
* `om_fidelity_score.py` needs OpenCV and must use uint8 images (cv2's float path of `TM_CCOEFF_NORMED` is numerically
  broken); it is CPU-only and uses a multiprocessing pool (`--procs`).
* `sq_codecs.py` also lists other tokenizers (MAE, WebSSL, DINOv2, Emu3.5, …); their codec modules are not included
  here, only `vae` and `siglip2` import cleanly from this package.
* Checkpoints saved by this trainer are plain diffusers `save_pretrained` folders (`dit/`), loadable with
  `MMDiT.from_pretrained(path)` from `mmdit_model.py`.

## 6. Public copies

* Code: https://github.com/SuhaoYu1020/Semantic-Image-Release
* Datasets (one-to-one base + hard one-to-many variants): https://huggingface.co/datasets/SuhaoYu1020/arith-sq256
* Trained checkpoints (15 final MMDiT snapshots used in the write-up): https://huggingface.co/SuhaoYu1020/latent-arith-mmdit
