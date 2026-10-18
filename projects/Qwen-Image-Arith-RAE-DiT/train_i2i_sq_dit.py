"""train_i2i_sq_dit.py -- the DiT-only arithmetic row on the 256x256 square render, with the tokenizer swapped among
the four first-round codecs (sq_codecs.SqCodec: vae | mae | webssl | siglip2), every one at its native 16x16 grid.

Design = Qwen-Image-Arith-VanillaDiT / -RAE-DiT, unchanged:
  condition = clean SOURCE tokens (question frame), channel-concatenated onto the
  input     = noised TARGET tokens (answered frame),           x_sigma = (1-sigma) z + sigma eps
  output    = rectified-flow velocity, trained with MSE to (eps - z); timestep-only adaLN-Zero; no text; no cross-attn.
  sampling  = Gaussian noise -> Euler on dx/dsigma = v -> codec.decode.

What is new here: 256 tokens instead of 2560, token dim D in {64, 768, 1024, 1152}, a ~0.2-0.3B DiT (default depth 12 x
hidden 1152 x 16 heads: hidden >= every arm's D, the RAE width rule), per-channel DATASET standardisation of every arm's
latent (sq_codecs), sigma shift by RAE's rule alpha = sqrt(256 D / 4096) unless overridden, step-cadence snapshots
(--save_every), and a STOP file (written by sq_watch_eval.sh once test OCR >= the threshold) that ends the run.

  python train_i2i_sq_dit.py --arm siglip2 --out_dir runs/sq_siglip2 --batch_size 64 --save_every 100
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EMIT_DIR = os.path.join(os.path.dirname(HERE), "Qwen-Image-Arith-Emit")
for d in (HERE, EMIT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from train_t2i_arith_mse import _find_latest_ckpt, prune_checkpoints, _full_state, _save_hf  # noqa: E402
from dit_model import VanillaDiT  # noqa: E402
from mmdit_model import MMDiT  # noqa: E402
from sigma_samplers import SigmaSampler  # noqa: E402
from sq_codecs import (SqCodec, ARMS, GRID, N_TOK, CANVAS, DEFAULT_DATA, DEFAULT_STATS_DIR,  # noqa: E402
                       to_pix01, answer_token_mask, rule_shift, sha256_file)

# conditioning architectures (same blocks, width, depth, heads; only how the clean source tokens enter):
#   chcat : single stream, [noised target ; source] channel-concat per position -> Linear(2D -> hidden)   (VanillaDiT)
#   mmdit : two streams with their own Linear(D -> hidden), QKV, MLP, adaLN; joint attention over both  (MMDiT, ~2x params)
ARCH_TAGS = {"chcat": "sq_dit_chcat", "mmdit": "sq_mmdit"}
ARCH = ARCH_TAGS["chcat"]            # overwritten from --arch in main()
DIT_SHA = sha256_file(os.path.join(HERE, "dit_model.py")) + "+" + sha256_file(os.path.join(HERE, "mmdit_model.py"))[:8]


class I2IDataset(Dataset):
    def __init__(self, root, jsonl):
        self.root = root
        self.items = [json.loads(l) for l in open(jsonl) if l.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return {"id": it["id"], "idx": i, "expr": str(it["expr"]), "answer": str(it["answer"]),
                "input_image": Image.open(os.path.join(self.root, it["input"])).convert("RGB"),
                "target_image": Image.open(os.path.join(self.root, it["target"])).convert("RGB")}


def collate(batch):
    return {k: [b[k] for b in batch] for k in ("id", "idx", "expr", "answer", "input_image", "target_image")}


# Arms whose frozen encoder is too slow to run inside every training step (Emu3.5 conv VQ fp32 0.9 s/step, FlexTok
# VAE + 18-layer resampler 2.3 s/step at batch 64 x 2 frames).  Their latents are deterministic (posterior mean / FSQ /
# VQ code vectors, no augmentation anywhere in this pipeline), so encoding the train split ONCE and indexing the cache is
# numerically the same training run, just without the encoder in the loop.  Round-1 arms keep encoding on the fly.
CACHE_ARMS = {"emu35", "tatitok128", "flextok256"}
CACHE_DIR = os.path.join(HERE, "cache", "sq256_same")


def build_latent_cache(codec, ds, data_root, train_jsonl, device, batch=128, log=print):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{codec.latent_id}__{os.path.basename(train_jsonl)}_{sha256_file(train_jsonl)}.pt")
    ids = [it["id"] for it in ds.items]
    if not os.path.isfile(path):
        t0 = time.time(); src_all, tgt_all = [], []
        for i0 in range(0, len(ds.items), batch):
            chunk = ds.items[i0:i0 + batch]
            with torch.no_grad():
                for key, acc in (("input", src_all), ("target", tgt_all)):
                    px = to_pix01([Image.open(os.path.join(data_root, it[key])).convert("RGB") for it in chunk]).to(device)
                    acc.append(codec.encode(px).float().cpu())
            if (i0 // batch) % 10 == 0:
                log(f"[cache] {i0 + len(chunk)}/{len(ds.items)} {time.time()-t0:.0f}s")
        blob = {"latent": codec.latent_id, "stats_sha": codec.stats_info.get("stats_sha"), "n": len(ids), "ids": ids,
                "src": torch.cat(src_all), "tgt": torch.cat(tgt_all), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
        tmp = f"{path}.tmp{os.getpid()}"; torch.save(blob, tmp); os.replace(tmp, path)
        log(f"[cache] built {path} src {tuple(blob['src'].shape)} tgt {tuple(blob['tgt'].shape)} in {time.time()-t0:.0f}s")
    blob = torch.load(path, map_location="cpu")
    assert blob["latent"] == codec.latent_id and blob["ids"] == ids and blob.get("stats_sha") == codec.stats_info.get("stats_sha"), path
    return path, blob["src"].to(device), blob["tgt"].to(device)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--stats", default=None, help=f"per-channel latent stats .pt (default {DEFAULT_STATS_DIR}/<arm>.pt)")
    p.add_argument("--no_stats", action="store_true", help="use the codec's inner latent as is (NOT unit variance)")
    p.add_argument("--data_root", default=DEFAULT_DATA)
    p.add_argument("--train_jsonl", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300, help="upper bound; the STOP file normally ends the run earlier")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--lr_horizon_epochs", type=int, default=0, help="cosine horizon in epochs (0 = the run)")
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--sigma_shift", default="rule", help="float, or 'rule' = sqrt(256*D/4096)")
    p.add_argument("--sigma_sampler", default="logit_normal", choices=["logit_normal", "reason_mix", "fixed", "uniform", "two_point"],
                   help="reason_mix: draw --sigma_reason_mix of the sigmas from p_shift(s)(1-A(s)) (noise levels where the answer "
                        "is NOT linearly readable from the noisy input; A from the arm's noise probe), the rest from p_shift")
    p.add_argument("--sigma_reason_mix", type=float, default=0.95)
    p.add_argument("--sigma_fixed", type=float, default=None, help="with --sigma_sampler fixed: train every sample at this single sigma")
    p.add_argument("--sigma_mix_frac", type=float, default=0.5, help="two_point: fraction of every batch trained at --sigma_mix_high")
    p.add_argument("--sigma_mix_low", type=float, default=0.3, help="two_point: sigma of the remaining samples (copy regime)")
    p.add_argument("--sigma_mix_high", type=float, default=1.0, help="two_point: sigma of the computing samples")
    p.add_argument("--init_ckpt", default=None, help="initialise the DiT weights from this snapshot dir (contains dit/); optimizer, LR schedule and step start fresh")
    p.add_argument("--null_source", action="store_true", help="zero the source conditioning: the model can only denoise the canvas (copier); with --sigma_sampler uniform this measures the DiT's own readability A_DiT(sigma)")
    p.add_argument("--sigma_reason_probe", default="auto", help="noise-probe results.json ('auto' = eval/result_gate4/probe_noise*/)")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "paged8bit"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--hidden", type=int, default=1152)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--save_every", type=int, default=100, help="snapshot every N optimizer steps (0 = off)")
    p.add_argument("--ckpt_every_epochs", type=int, default=0)
    p.add_argument("--keep_last_ckpts", type=int, default=0)
    p.add_argument("--stop_file", default=None, help="default <out_dir>/STOP; checked every step")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="no checkpoints")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="qwen-image-arith-sq-dit")
    p.add_argument("--arch", default="chcat", choices=sorted(ARCH_TAGS), help="conditioning architecture (see ARCH_TAGS)")
    p.add_argument("--latent_cache", default="auto", choices=["auto", "on", "off"],
                   help="encode the train split once and index it (auto = on for the slow-encoder arms %s)" % sorted(CACHE_ARMS))
    return p.parse_args()


def main():
    global ARCH
    args = parse_args()
    ARCH = ARCH_TAGS[args.arch]
    D = ARMS[args.arm]
    assert args.hidden % args.heads == 0
    assert args.hidden >= D, f"RAE width rule: hidden {args.hidden} < token dim {D}"
    stats_path = None if args.no_stats else (args.stats or os.path.join(DEFAULT_STATS_DIR, f"{args.arm}.pt"))
    if stats_path is not None:
        assert os.path.isfile(stats_path), f"missing latent stats {stats_path}; run compute_latent_stats.py --arm {args.arm}"
    train_jsonl = args.train_jsonl or os.path.join(args.data_root, "train.jsonl")
    stop_file = args.stop_file or os.path.join(args.out_dir, "STOP")
    os.makedirs(args.out_dir, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16",
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False, broadcast_buffers=False)])
    set_seed(args.seed)
    device = accelerator.device
    main_proc = accelerator.is_main_process
    use_wandb = args.wandb and main_proc
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=os.path.basename(args.out_dir), config=vars(args))

    in_dim, out_dim = (2 * D if args.arch == "chcat" else D), D
    t0 = time.time()
    codec = SqCodec(args.arm, stats_path=stats_path, with_decoder=False, dtype=torch.bfloat16)
    grid_h, grid_w = codec.grid            # 16x16 for every 2-D arm; a pseudo-grid for the 1-D arms (slots have no geometry)
    assert grid_h * grid_w == codec.n_tok, (codec.grid, codec.n_tok)
    shift = rule_shift(D, codec.n_tok) if str(args.sigma_shift) == "rule" else float(args.sigma_shift)
    assert shift > 0
    sampler = SigmaSampler(shift, args.sigma_sampler, args.sigma_reason_mix, arm=args.arm, probe=args.sigma_reason_probe,
                           device=device, fixed=args.sigma_fixed, two_point=(args.sigma_mix_frac, args.sigma_mix_low, args.sigma_mix_high)) if args.sigma_sampler != "logit_normal" else None
    ModelCls = VanillaDiT if args.arch == "chcat" else MMDiT
    model = ModelCls(in_dim=in_dim, out_dim=out_dim, grid_h=grid_h, grid_w=grid_w, hidden=args.hidden,
                       depth=args.depth, heads=args.heads, mlp_ratio=args.mlp_ratio)     # fp32 master weights
    n_params = model.n_params()
    if main_proc:
        rec = dict(vars(args)); rec.update({"arch": ARCH, "arm": args.arm, "grid_h": grid_h, "grid_w": grid_w,
                                            "in_dim": in_dim, "out_dim": out_dim, "sigma_shift_resolved": shift,
                                            "dit_model_sha256": DIT_SHA, "n_params": n_params, **codec.meta(),
                                            **(sampler.meta if sampler is not None else {}),
                                            "conditioning": "channel-concat clean source tokens; timestep adaLN-Zero; NO text"})
        json.dump(rec, open(os.path.join(args.out_dir, "args.json"), "w"), indent=2)
        print(f"[init] arm={args.arm} D={D} tokens={codec.n_tok} grid={grid_h}x{grid_w} geometric={codec.geometric} canvas={CANVAS}", flush=True)
        print(f"[codec] {json.dumps(codec.meta())}", flush=True)
        print(f"[dit] depth={args.depth} hidden={args.hidden} heads={args.heads} (head_dim {args.hidden//args.heads}) "
              f"params={n_params/1e6:.1f}M in_dim={in_dim} out_dim={out_dim}", flush=True)
        print(f"[sigma_shift] {shift:.3f} ({'RAE rule sqrt(N*D/4096)' if str(args.sigma_shift)=='rule' else 'override'}); "
              f"median sigma {shift/(1+shift):.3f}", flush=True)
        if sampler is not None:
            print(sampler.describe(), flush=True)

    resume_step, prev_meta = 0, None
    if args.resume:
        ck, resume_step = _find_latest_ckpt(args.out_dir)
        if ck is not None:
            model = ModelCls.from_pretrained(os.path.join(ck, "dit"), torch_dtype=torch.float32)
            prev_meta = json.load(open(os.path.join(ck, "meta.json")))
            for k, cur in (("sigma_shift", shift), ("latent", codec.latent_id), ("in_dim", in_dim), ("out_dim", out_dim),
                           ("arch", ARCH), ("depth", args.depth), ("hidden", args.hidden), ("heads", args.heads)):
                assert prev_meta.get(k, cur) == cur, f"refusing to resume: ckpt {k}={prev_meta.get(k)!r}, launch {cur!r}"
            if main_proc:
                print(f"[resume] {ck} step {resume_step}", flush=True)
        elif main_proc:
            print("[resume] nothing to resume -> fresh start", flush=True)
    if args.init_ckpt and not resume_step:
        model = ModelCls.from_pretrained(os.path.join(args.init_ckpt, "dit"), torch_dtype=torch.float32)
        if main_proc: print(f"[init_ckpt] weights from {args.init_ckpt} (fresh optimizer / schedule / step 0)", flush=True)
    model.requires_grad_(True).train()

    ds = I2IDataset(args.data_root, train_jsonl)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0)
    _mult = max(1, accelerator.num_processes)
    steps_per_epoch = math.ceil(len(loader) / (args.grad_accum * _mult))
    total_steps = args.max_steps or (args.epochs * steps_per_epoch)
    lr_total = (args.lr_horizon_epochs * steps_per_epoch) if args.lr_horizon_epochs else total_steps
    assert lr_total >= total_steps

    def lr_lambda(step):
        s = step / _mult
        if s < args.warmup_steps:
            return s / max(1, args.warmup_steps)
        prog = (s - args.warmup_steps) / max(1, lr_total - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    model = accelerator.prepare(model)
    codec.to(device)
    cache_src = cache_tgt = None
    use_cache = args.latent_cache == "on" or (args.latent_cache == "auto" and args.arm in CACHE_ARMS)
    if use_cache:
        cache_path, cache_src, cache_tgt = build_latent_cache(codec, ds, args.data_root, train_jsonl, device,
                                                              log=(lambda s: print(s, flush=True)) if main_proc else (lambda s: None))
        if main_proc:
            print(f"[cache] using {cache_path} ({cache_src.numel() * 4 / 1e9:.2f} GB per frame set on {device})", flush=True)
            rec_p = os.path.join(args.out_dir, "args.json")
            try:
                rec = json.load(open(rec_p)); rec["latent_cache"] = cache_path; json.dump(rec, open(rec_p, "w"), indent=1)
            except Exception as e:  # noqa: BLE001
                print(f"[cache] could not record cache path in args.json: {e}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=True)
    else:
        from bitsandbytes.optim import PagedAdamW8bit
        opt = PagedAdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    opt, loader, sched = accelerator.prepare(opt, loader, sched)
    for _ in range(resume_step):
        sched.step()
    if main_proc:
        print(f"[optim] {type(opt).__name__} lr={args.lr} bs={args.batch_size}x{args.grad_accum} steps/epoch={steps_per_epoch} "
              f"total_steps={total_steps} lr_horizon={lr_total} warmup={args.warmup_steps} (built {time.time()-t0:.1f}s)", flush=True)

    def write_meta(sd, global_step, epoch):
        json.dump({"global_step": global_step, "epoch": epoch, "arch": ARCH, "arm": args.arm, "sigma_shift": shift,
                   "depth": args.depth, "hidden": args.hidden, "heads": args.heads, "mlp_ratio": args.mlp_ratio,
                   "height": CANVAS, "width": CANVAS, "canvas": CANVAS, "in_dim": in_dim, "out_dim": out_dim,
                   "n_params": n_params, "dit_model_sha256": DIT_SHA, "total_steps": total_steps, "lr_horizon_steps": lr_total,
                   "data_root": args.data_root, **codec.meta(), "freeze_te": True, "no_text": True},
                  open(os.path.join(sd, "meta.json"), "w"), indent=1)        # written LAST

    def save_ckpt(tag, global_step, epoch):
        if args.smoke:
            return
        accelerator.wait_for_everyone()
        state = _full_state(accelerator, model)
        if main_proc:
            sd = os.path.join(args.out_dir, tag)
            os.makedirs(os.path.join(sd, "transformer"), exist_ok=True)   # completeness marker (_is_complete_ckpt)
            _save_hf(accelerator, model, os.path.join(sd, "dit"), state)
            write_meta(sd, global_step, epoch)
            print(f"[save] step {global_step} -> {sd}", flush=True)
            prune_checkpoints(args.out_dir, args.keep_last_ckpts)
        del state
        accelerator.wait_for_everyone()

    ANS_M = answer_token_mask(args.data_root, grid=codec.grid).to(device) if codec.geometric else None   # 1-D arms: no answer box
    logit_normal = torch.distributions.Normal(0.0, 1.0)
    global_step, checked, done, stopped = resume_step, False, False, False
    start_epoch = resume_step // steps_per_epoch if resume_step else 0
    t_train = time.time()
    log = {"mse": 0.0, "rel": 0.0, "rel_ans": 0.0, "gn": 0.0, "n": 0}
    clip_hits = clip_seen = 0
    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                tgt01 = to_pix01(batch["target_image"]).to(device)
                src01 = to_pix01(batch["input_image"]).to(device)
                with torch.no_grad():
                    if cache_tgt is not None:
                        idx = torch.as_tensor(batch["idx"], device=device)
                        tgt, src = cache_tgt[idx], cache_src[idx]         # (B,N,D) float32, standardised (cached)
                    else:
                        tgt = codec.encode(tgt01)                          # (B,N,D) float32, standardised
                        src = codec.encode(src01)
                if not checked:
                    ink_t, ink_s = (tgt01 < 0.5).float().mean().item(), (src01 < 0.5).float().mean().item()
                    assert ink_t > ink_s, (ink_t, ink_s)
                    zm, zs = float(tgt.mean()), float(tgt.std())
                    d_tok = (tgt - src).pow(2).sum(-1)[0]
                    hit = ANS_M[d_tok.topk(8).indices].float().mean().item() if ANS_M is not None else float("nan")
                    if main_proc:
                        print(f"[check] ink t={ink_t:.4f} s={ink_s:.4f} z(mean={zm:+.3f}, std={zs:.3f}) "
                              f"top8-diff-in-answer-box={hit:.2f} (chance {ANS_M.float().mean().item() if ANS_M is not None else float('nan'):.3f}; "
                              f"semantic encoders spread the change, so this is informative only for vae/mae; undefined for 1-D arms)", flush=True)
                    if stats_path is not None:
                        assert abs(zm) < 0.15 and 0.75 < zs < 1.3, f"standardised latent off: mean {zm:.3f} std {zs:.3f}"
                    if args.arm in ("vae", "mae"):
                        assert hit >= 0.5, f"token order / grid wrong for {args.arm}: {hit:.2f} of top-8 diff tokens in the answer box"
                B = tgt.shape[0]
                noise = torch.randn_like(tgt)
                if sampler is None:
                    u = logit_normal.sample((B,)).to(device) + math.log(shift)
                    sigma = torch.sigmoid(u)
                else:
                    sigma = sampler(B, device)
                sb = sigma.view(B, 1, 1)
                noisy = (1.0 - sb) * tgt + sb * noise
                flow_target = noise - tgt
                pred = model(noisy, sigma, torch.zeros_like(src) if args.null_source else src)
                loss = F.mse_loss(pred.float(), flow_target)
                accelerator.backward(loss)
                if not checked:
                    checked = True
                    if main_proc:
                        n_g = sum(1 for p in params if p.grad is not None)
                        assert n_g == len(params), f"{n_g}/{len(params)} params in graph"
                        print(f"[grad-check] {n_g}/{len(params)} params in graph", flush=True)
                with torch.no_grad():
                    ft = flow_target
                    se = (pred.detach().float() - ft).pow(2)
                    log["rel"] += (se.mean() / ft.pow(2).mean()).item()
                    if ANS_M is not None:
                        log["rel_ans"] += (se[:, ANS_M].mean() / ft[:, ANS_M].pow(2).mean()).item()
                gn = None
                if accelerator.sync_gradients and args.grad_clip > 0:
                    gn = accelerator.clip_grad_norm_(params, args.grad_clip)
                    if not torch.isfinite(gn):
                        raise RuntimeError(f"non-finite grad norm at step {global_step+1}")
                    clip_seen += 1; clip_hits += int(float(gn) > args.grad_clip)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            log["mse"] += loss.item(); log["n"] += 1
            if gn is not None:
                log["gn"] += float(gn)
            if accelerator.sync_gradients:
                global_step += 1
                if main_proc and global_step % args.log_every == 0:
                    n = max(1, log["n"]); el = time.time() - t_train
                    print(f"[step {global_step}/{total_steps}] ep {epoch} mse={log['mse']/n:.4f} rel={log['rel']/n:.4f} "
                          f"rel_ans={log['rel_ans']/n:.4f} gnorm={log['gn']/n:.3f} clip={clip_hits}/{max(1,clip_seen)} "
                          f"lr={sched.get_last_lr()[0]:.2e} {el/max(1,global_step-resume_step):.3f}s/step", flush=True)
                    if use_wandb:
                        wandb.log({"train/mse": log["mse"]/n, "train/rel_ans": log["rel_ans"]/n, "train/lr": sched.get_last_lr()[0]}, step=global_step)
                    log = {"mse": 0.0, "rel": 0.0, "rel_ans": 0.0, "gn": 0.0, "n": 0}
                if args.save_every and global_step % args.save_every == 0:
                    save_ckpt(f"step_{global_step}", global_step, epoch)
                if os.path.exists(stop_file):
                    stopped = True
                if stopped or (args.max_steps and global_step >= args.max_steps) or global_step >= total_steps:
                    done = True
                    break
        if not done and args.ckpt_every_epochs and (epoch + 1) % args.ckpt_every_epochs == 0:
            save_ckpt(f"step_{global_step}", global_step, epoch)
        if done:
            break

    if main_proc:
        print(f"[end] step {global_step} epoch {epoch} stopped_by_STOP={stopped} "
              f"clip={clip_hits}/{max(1,clip_seen)} elapsed {(time.time()-t_train)/60:.1f} min", flush=True)
    accelerator.wait_for_everyone()
    if not args.smoke:
        state = _full_state(accelerator, model)
        if main_proc:
            fd = os.path.join(args.out_dir, "final")
            os.makedirs(os.path.join(fd, "transformer"), exist_ok=True)
            _save_hf(accelerator, model, os.path.join(fd, "dit"), state)
            write_meta(fd, global_step, epoch)
            json.dump({"global_step": global_step, "epoch": epoch, "stopped_by_STOP": stopped,
                       "elapsed_min": (time.time() - t_train) / 60}, open(os.path.join(args.out_dir, "DONE.json"), "w"))
            print(f"[done] -> {fd}", flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
