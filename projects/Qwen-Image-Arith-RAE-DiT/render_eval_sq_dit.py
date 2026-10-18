"""render_eval_sq_dit.py -- sample the square-canvas DiT-only row (train_i2i_sq_dit.py) on the test split and decode.

Noise -> Euler on dx/dsigma = v over the SAME shifted schedule the checkpoint trained with (static SD3 shift
sigma' = s*sigma / (1 + (s-1)*sigma) on linspace(1, 1/steps, steps), then 0), batched, then codec.decode -> 256x256 PNGs
laid out exactly like every other renderer (imgs/<split>_<id>_pred|_gt|_in.png + report.json with test_rows/train_rows),
so score_render_ocr_crop.py reads it unchanged.  Codec and stats are resolved from the checkpoint's meta.json.

  python render_eval_sq_dit.py --ckpt runs/sq_siglip2/step_1200 --out_dir runs/sq_siglip2/eval/step_1200 --n_test 0
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EMIT_DIR = os.path.join(os.path.dirname(HERE), "Qwen-Image-Arith-Emit")
for d in (HERE, EMIT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch
from PIL import Image

from dit_model import VanillaDiT  # noqa: E402
from mmdit_model import MMDiT  # noqa: E402
from sq_codecs import SqCodec, DEFAULT_DATA, to_pix01, N_TOK  # noqa: E402

DEVICE = "cuda"
DT = torch.bfloat16


def build_sigmas(steps, shift):
    lin = np.linspace(1.0, 1.0 / steps, steps)
    s = shift * lin / (1 + (shift - 1) * lin)
    return [float(v) for v in s] + [0.0]


def select(jsonl, n=0, seed=0):
    if n < 0:
        return []
    items = [json.loads(l) for l in open(jsonl) if l.strip()]
    if n and n < len(items):
        items = random.Random(seed).sample(items, n)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data", default=None, help="default: the checkpoint's data_root")
    ap.add_argument("--n_test", type=int, default=0, help="0 = all test rows")
    ap.add_argument("--n_train", type=int, default=-1, help="-1 = skip, 0 = all, N = sample")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sampler_shift", default="train", help="'train' or a float")
    ap.add_argument("--save_latents", action="store_true", help="also save the final latents of every rendered item to <out_dir>/latents.pt (ids, x)")
    ap.add_argument("--sigma_stop", type=float, default=0.0, help="truncation ablation: once the next sigma would drop below this, decode x0 = x - sigma*v from the current sigma and stop (0 = full sampler)")
    ap.add_argument("--stats", default=None, help="override the stats file recorded in meta.json")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.ckpt, "meta.json")))
    assert meta.get("arch") in ("sq_dit_chcat", "sq_mmdit"), meta.get("arch")
    ModelCls = MMDiT if meta.get("arch") == "sq_mmdit" else VanillaDiT
    data = a.data or meta.get("data_root") or DEFAULT_DATA
    shift = float(meta["sigma_shift"]) if a.sampler_shift == "train" else float(a.sampler_shift)
    sigmas = build_sigmas(a.steps, shift)
    imgs_dir = os.path.join(a.out_dir, "imgs"); os.makedirs(imgs_dir, exist_ok=True)
    open(os.path.join(a.out_dir, "ckpt_used.txt"), "w").write(a.ckpt + "\n")
    t0 = time.time()
    stats = a.stats or meta.get("stats_path")
    codec = SqCodec(meta["arm"], stats_path=stats, with_decoder=True, dtype=DT).to(DEVICE).eval()
    assert codec.latent_id == meta["latent"], (codec.latent_id, meta["latent"])
    model = ModelCls.from_pretrained(os.path.join(a.ckpt, "dit"), torch_dtype=DT).to(DEVICE).eval()
    print(f"[render] {a.ckpt} arm={meta['arm']} step={meta.get('global_step')} shift={shift:.3f} steps={a.steps} "
          f"params={model.n_params()/1e6:.0f}M loaded {time.time()-t0:.1f}s", flush=True)
    lat_rmse = []; saved_lat = []; saved_ids = []

    @torch.no_grad()
    def gen_batch(src_pils, gt_pils, gen):
        src = codec.encode(to_pix01(src_pils).to(DEVICE))                    # (B,256,D) float32
        x = torch.randn(src.shape, generator=gen, device=DEVICE, dtype=torch.float32)
        for i in range(len(sigmas) - 1):
            s, s_next = sigmas[i], sigmas[i + 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(x, torch.full((x.shape[0],), s, device=DEVICE), src).float()
            if a.sigma_stop > 0.0 and s_next < a.sigma_stop:
                x = x + (0.0 - s) * v          # truncated: one-step x0 from the last sigma above --sigma_stop
                break
            x = x + (s_next - s) * v
        zgt = codec.encode(to_pix01(gt_pils).to(DEVICE))
        lat_rmse.extend((x - zgt).pow(2).mean(dim=(1, 2)).sqrt().tolist())
        if a.save_latents: saved_lat.append(x.detach().to("cpu", torch.float16))
        return codec.to_pil(codec.decode(x))

    def run_split(name, items):
        gen = torch.Generator(device=DEVICE).manual_seed(a.seed)
        rows = []
        for i0 in range(0, len(items), a.batch):
            chunk = items[i0:i0 + a.batch]
            src_pils = [Image.open(os.path.join(data, it["input"])).convert("RGB") for it in chunk]
            gt_pils = [Image.open(os.path.join(data, it["target"])).convert("RGB") for it in chunk]
            preds = gen_batch(src_pils, gt_pils, gen)
            if a.save_latents: saved_ids.extend([(name, int(it["id"])) for it in chunk])
            for it, im in zip(chunk, preds):
                base = f"{name}_{it['id']:06d}"
                im.save(os.path.join(imgs_dir, base + "_pred.png"))
                for suf, key in (("_gt.png", "target"), ("_in.png", "input")):
                    lp = os.path.join(imgs_dir, base + suf)
                    if not os.path.lexists(lp):
                        os.symlink(os.path.abspath(os.path.join(data, it[key])), lp)
                rows.append({"id": it["id"], "expr": str(it["expr"]), "gt": str(it["answer"]), "base": base})
            if (i0 // a.batch) % 4 == 0:
                print(f"[render] {name} {i0+len(chunk)}/{len(items)}", flush=True)
        return rows

    t = time.time()
    test_rows = run_split("test", select(os.path.join(data, "test.jsonl"), a.n_test, a.seed))
    train_rows = run_split("train", select(os.path.join(data, "train.jsonl"), a.n_train, a.seed)) if a.n_train >= 0 else []
    rep = {"ckpt": a.ckpt, "steps": a.steps, "sampler_shift": shift, "sigmas": sigmas, "meta": meta, "codec": codec.meta(),
           "latent_rmse_mean": float(np.mean(lat_rmse)) if lat_rmse else None, "n_test": len(test_rows), "n_train": len(train_rows),
           "test_rows": test_rows, "train_rows": train_rows, "render_sec": time.time() - t}
    json.dump(rep, open(os.path.join(a.out_dir, "report.json"), "w"), indent=1)
    if a.save_latents and saved_lat:
        torch.save({"ids": saved_ids, "x": torch.cat(saved_lat), "steps": a.steps, "sigma_stop": a.sigma_stop}, os.path.join(a.out_dir, "latents.pt"))
    print(f"[render] DONE {len(test_rows)} test + {len(train_rows)} train in {time.time()-t:.0f}s; "
          f"latent RMSE vs GT {rep['latent_rmse_mean']:.4f} -> {a.out_dir}/report.json", flush=True)


if __name__ == "__main__":
    main()
