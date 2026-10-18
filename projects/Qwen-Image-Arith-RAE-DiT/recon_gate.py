"""STAGE-0 GATE for the RAE-vs-VAE DiT-only arithmetic row.

Round-trips OUR OWN images through a frozen codec -- no DiT, no training -- and scores the
reconstructions with the EXACT metric of record (../Qwen-Image-Arith-ScratchTE/score_render_ocr_crop.py,
Qwen3-VL-2B, answer-line crop).  A round-trip upper-bounds anything a generative model in that
latent space can ever score, so this decides whether the 46-hour row is worth launching.

WHY THIS IS THE RIGHT GATE.  The risky component is the released ViT-XL decoder: it was trained
on a 16x16 SQUARE grid of ImageNet photographs and here is asked for a 32x80 grid of rendered
digits -- a 10x sequence-length extrapolation onto out-of-domain content.  The RAE authors'
own follow-up (Scale-RAE) reports that ImageNet-only decoders are their WORST case on text.

WHAT ELSE IT MEASURES, none of which anyone has measured before on this dataset:
  * the Qwen-VAE's own reconstruction ceiling on the identical rows (the baseline row has always
    been reported against an ASSUMED-perfect codec);
  * LATENT-ERROR TOLERANCE: OCR of decode(z + eps*randn) over an eps sweep.  Both codecs' z are
    ~unit variance so eps is directly comparable; the eps at which OCR crosses 90% is the latent
    RMSE a DiT must beat.  A decoder that only reads a PERFECT latent cannot serve a generative
    row -- this turns the on-manifold-ceiling objection into a number.
  * z statistics on OUR data.  The Qwen VAE uses PUBLISHED NATURAL-IMAGE constants
    (latents_std 1.125..3.269, latents_mean != 0), so on near-binary renders its z is probably
    NOT unit variance -- a first-order confound sitting inside the RUNNING baseline.

Output layout is a normal renderer eval dir per (codec, eps) so the shared scorer reads it with
zero changes, and its gt_ok column gives the OCR ceiling on the same batch for free.

Usage:
  python recon_gate.py --codec rae --rae_path $SI_CKPTS/RAE-dinov2-wReg-base-ViTXL-n08 \
      --pos_embed_mode interp --stat_mode released --out_dir eval/result/gate_rae_dinov2_interp
  python recon_gate.py --codec vae --out_dir eval/result/gate_vae
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EMIT_DIR = os.path.join(os.path.dirname(HERE), "Qwen-Image-Arith-Emit")
for d in (HERE, EMIT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

import numpy as np
import torch
from PIL import Image

from train_t2i_arith_mse import SI_CKPTS, SI_DATA, pack_latents  # noqa: E402
from rae_codec import RaeCodec, IMG_H, IMG_W, N_TOK  # noqa: E402
from scalerae_codec import ScaleRaeCodec  # noqa: E402  (Scale-RAE siglip2-so400m / webssl-dino300m decoders)

DEVICE = "cuda"
DT = torch.bfloat16

# MEASURED answer-line ink bbox on data/arithmetic_add_unordered (200 train pairs):
# y[281,340] x[81,365].  Half-open slices, +1 on the ends.
ANS_Y, ANS_X = (281, 341), (81, 366)


# --------------------------------------------------------------------------- VAE codec
class VaeCodec(torch.nn.Module):
    """Qwen-Image VAE behind the SAME interface as RaeCodec, so the gate scores both arms
    through one code path.  Deterministic (posterior MODE) to match the renderer's source
    encoding; the target's posterior std is reported separately, since the RAE arm has no
    posterior sampling at all and that is an unavoidable asymmetry worth bounding."""

    def __init__(self, model_path=None, dtype=DT, img_hw=(IMG_H, IMG_W)):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            model_path or f"{SI_CKPTS}/Qwen-Image", subfolder="vae", torch_dtype=dtype)
        self.vae.requires_grad_(False).eval()
        c = self.vae.config
        self.register_buffer("m", torch.tensor(c.latents_mean).view(1, -1, 1, 1, 1).float())
        self.register_buffer("s", torch.tensor(c.latents_std).view(1, -1, 1, 1, 1).float())
        self.dtype_ = dtype
        self.img_hw = (int(img_hw[0]), int(img_hw[1]))
        gh, gw = self.img_hw[0] // 16, self.img_hw[1] // 16
        self.latent_id = f"qwenvae_c16_d8_pack2x2_g{gh}x{gw}_d64_published"
        self.encoder_type, self.stat_mode, self.pos_embed_mode = "qwen_vae", "published", "n/a"

    to_pix01 = staticmethod(RaeCodec.to_pix01)
    to_pil = staticmethod(RaeCodec.to_pil)

    def encode(self, pix01, sample=False):
        x = (pix01.float() * 2 - 1).to(self.dtype_).unsqueeze(2)          # (B,3,1,H,W) in [-1,1]
        d = self.vae.encode(x).latent_dist
        z = d.sample() if sample else d.mode()
        return pack_latents((z.float() - self.m) / self.s)                # (B,2560,64)

    def posterior_std(self, pix01):
        x = (pix01.float() * 2 - 1).to(self.dtype_).unsqueeze(2)
        return float(self.vae.encode(x).latent_dist.std.float().mean())

    def decode(self, z):
        B, N, D = z.shape
        H, W = self.img_hw[0] // 8, self.img_hw[1] // 8
        C = D // 4
        u = z.float().view(B, 1, H // 2, W // 2, C, 2, 2).permute(0, 4, 1, 2, 5, 3, 6)
        u = u.reshape(B, C, 1, H, W) * self.s + self.m
        px = self.vae.decode(u.to(self.dtype_)).sample[:, :, 0]           # (B,3,H,W) in [-1,1]
        return ((px.float() + 1) / 2).clamp(0, 1)

    def meta(self):
        gh, gw = self.img_hw[0] // 16, self.img_hw[1] // 16
        return {"latent": self.latent_id, "encoder_type": self.encoder_type,
                "grid": [gh, gw], "tokens": gh * gw, "dim": 64, "img_hw": list(self.img_hw),
                "stat_mode": self.stat_mode, "pos_embed_mode": self.pos_embed_mode}


# --------------------------------------------------------------------------- RAE, official square path
class RaeNativeCodec(torch.nn.Module):
    """The RELEASED diffusers AutoencoderRAE used exactly as shipped: bicubic resize to its square
    encoder_input_size (256 for mae), 16x16 tokens, released latent stats, decoder at its own training
    geometry.  Zero sequence extrapolation -- the comparison point for the retargeted RaeCodec on the
    square re-rendered gate sets.  Latents are exposed flat as (B, 256, C) like every other codec."""

    def __init__(self, rae_path, dtype=DT, img_hw=(256, 256)):
        super().__init__()
        from diffusers import AutoencoderRAE
        from rae_codec import _patch_siglip_builder
        _patch_siglip_builder()
        rae = AutoencoderRAE.from_pretrained(rae_path, torch_dtype=dtype)
        meta_t = [n for n, t in list(rae.named_parameters()) + list(rae.named_buffers()) if getattr(t, "is_meta", False)]
        assert not meta_t, f"META tensors after loading {rae_path}: {meta_t[:5]}"
        assert bool(rae.config.reshape_to_2d) and float(rae.config.scaling_factor) == 1.0
        self.rae = rae.requires_grad_(False).eval()
        self.dtype_, self.img_hw = dtype, (int(img_hw[0]), int(img_hw[1]))
        s = int(rae.config.encoder_input_size); pe = int(rae.config.encoder_patch_size)
        self.side, self.dim = s // pe, int(rae.config.encoder_hidden_size)
        self.encoder_type, self.stat_mode, self.pos_embed_mode = rae.config.encoder_type, "released", "native"
        self.latent_id = f"rae_{self.encoder_type}_native{s}_g{self.side}x{self.side}_d{self.dim}_released"

    to_pix01 = staticmethod(RaeCodec.to_pix01)
    to_pil = staticmethod(RaeCodec.to_pil)

    def encode(self, pix01):
        z = self.rae.encode(pix01.to(self.dtype_)).latent                      # (B, C, 16, 16) normalized
        return z.float().flatten(2).transpose(1, 2).contiguous()               # (B, 256, C)

    def decode(self, z):
        B, N, C = z.shape
        z4 = z.float().transpose(1, 2).reshape(B, C, self.side, self.side).to(self.dtype_)
        px = self.rae.decode(z4).sample.float().clamp(0, 1)                    # (B, 3, 256, 256) in [0,1]
        if (px.shape[2], px.shape[3]) != self.img_hw:
            px = torch.nn.functional.interpolate(px, size=self.img_hw, mode="bicubic", align_corners=False).clamp_(0, 1)
        return px

    def meta(self):
        return {"latent": self.latent_id, "encoder_type": self.encoder_type, "grid": [self.side, self.side],
                "tokens": self.side * self.side, "dim": self.dim, "img_hw": list(self.img_hw),
                "stat_mode": self.stat_mode, "pos_embed_mode": self.pos_embed_mode,
                "encoder_input_size": int(self.rae.config.encoder_input_size)}


# --------------------------------------------------------------------------- metrics
def psnr(a, b, box=None):
    """a, b: (3,H,W) float in [0,1].  box = (y0,y1,x0,x1) or None for the full frame."""
    if box is not None:
        a = a[:, box[0]:box[1], box[2]:box[3]]
        b = b[:, box[0]:box[1], box[2]:box[3]]
    mse = float((a - b).pow(2).mean())
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def ink_iou(a, b, thr=0.5):
    """IoU of the ink masks (gray < thr).  Ink is only 2.02% of this canvas, so a pixel metric
    that is dominated by white is useless here; this one is not."""
    ga, gb = a.mean(0) < thr, b.mean(0) < thr
    u = float((ga | gb).sum())
    return 1.0 if u == 0 else float((ga & gb).sum()) / u


# --------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--codec", default="rae", choices=["rae", "vae", "scalerae", "rae_native"])
    p.add_argument("--img_hw", type=int, nargs=2, default=[IMG_H, IMG_W], metavar=("H", "W"),
                   help="canvas of the dataset images; 256 256 for the square re-rendered gate sets")
    p.add_argument("--rae_path", default=None, help="required for --codec rae")
    p.add_argument("--scalerae_encoder", default=None, choices=["siglip2_so400m", "webssl_dino300m"],
                   help="required for --codec scalerae (weights: ckpts/ScaleRAE-*_decoder + the encoder dirs)")
    p.add_argument("--model_path", default=f"{SI_CKPTS}/Qwen-Image", help="VAE arm only")
    p.add_argument("--stat_mode", default="released", choices=["released", "dataset", "none"])
    p.add_argument("--latent_stats", default=None)
    p.add_argument("--pos_embed_mode", default="interp", choices=["interp", "iso"])
    p.add_argument("--grid", type=int, nargs=2, default=None, metavar=("GH", "GW"),
                   help="scalerae only: token grid (image resized to 14*GH x 14*GW); default = img_hw/16 "
                        "(32x80 on the 512x1280 panel, 16x16 on a 256x256 square set)")
    p.add_argument("--decoder_ft", default=None, help="fine-tuned decoder safetensors (stage C)")
    p.add_argument("--data", default=f"{SI_DATA}/arithmetic_add_unordered")
    p.add_argument("--split", default="gate500", choices=["gate500", "test", "train"])
    p.add_argument("--n", type=int, default=500, help="rows at eps=0 (the headline)")
    p.add_argument("--n_eps", type=int, default=200, help="rows at each eps>0 (the curve)")
    p.add_argument("--eps", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20, 0.40])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--label", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    # no silent codec default: the VAE arm has no rae_path, but a rae arm without one would
    # otherwise have gated whichever encoder happened to be hardcoded here.
    assert args.codec not in ("rae", "rae_native") or args.rae_path, f"--codec {args.codec} requires --rae_path"
    assert args.codec != "scalerae" or args.scalerae_encoder, "--codec scalerae requires --scalerae_encoder"
    img_hw = (int(args.img_hw[0]), int(args.img_hw[1]))
    if args.codec == "rae":
        assert img_hw == (IMG_H, IMG_W), "--codec rae is the retargeted 32x80 codec; use --codec rae_native for square sets"
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- rows: gate500 is a FIXED Random(0) sample of TRAIN.  Every gate decision (encoder,
    # pos-embed mode, decoder variant, stats mode) is a SELECTION, so it must not be made on
    # the split the headline number lives on; the test split is opened only for the final table.
    src_jsonl = os.path.join(args.data, "train.jsonl" if args.split != "test" else "test.jsonl")
    items = [json.loads(l) for l in open(src_jsonl) if l.strip()]
    split_name = "train" if args.split != "test" else "test"
    if args.n and args.n < len(items):
        items = random.Random(args.seed).sample(items, args.n)

    # ---- codec
    if args.codec == "rae":
        codec = RaeCodec(args.rae_path, stat_mode=args.stat_mode, stats_path=args.latent_stats,
                         pos_embed_mode=args.pos_embed_mode, decoder_ft=args.decoder_ft,
                         with_decoder=True, dtype=DT).to(DEVICE)
    elif args.codec == "scalerae":
        # Scale-RAE ships no dataset statistics (both released DiT configs: diffusion_norm_stats_path=None), so
        # 'released' means 'none' here; 'dataset' + --latent_stats is still honoured for the DiT side.
        sm = "none" if args.stat_mode == "released" else args.stat_mode
        grid = tuple(args.grid) if args.grid else (img_hw[0] // 16, img_hw[1] // 16)
        codec = ScaleRaeCodec(args.scalerae_encoder, stat_mode=sm, stats_path=args.latent_stats,
                              pos_embed_mode=args.pos_embed_mode, with_decoder=True, dtype=DT,
                              grid=grid).to(DEVICE)
        codec.out_hw = img_hw                                        # decode() returns the dataset canvas
    elif args.codec == "rae_native":
        codec = RaeNativeCodec(args.rae_path, dtype=DT, img_hw=img_hw).to(DEVICE)
    else:
        codec = VaeCodec(args.model_path, dtype=DT, img_hw=img_hw).to(DEVICE)
    codec.eval()
    label = args.label or f"{args.codec}_{codec.encoder_type}_{codec.pos_embed_mode}"
    print(f"[gate] label={label} rows={len(items)} eps={args.eps}", flush=True)
    print(f"[codec] {json.dumps(codec.meta())}", flush=True)

    eps_list = sorted(set(float(e) for e in args.eps))
    dirs = {}
    for e in eps_list:
        d = os.path.join(args.out_dir, f"eps{e:.2f}")
        os.makedirs(os.path.join(d, "imgs"), exist_ok=True)
        dirs[e] = d

    rows_by_eps = {e: [] for e in eps_list}
    stats = {"z_mean": [], "z_std": [], "post_std": []}
    per_eps_px = {e: {"psnr_full": [], "psnr_ans": [], "iou": []} for e in eps_list}
    gen = torch.Generator(device=DEVICE).manual_seed(args.seed)
    z = None

    for i0 in range(0, len(items), args.batch):
        chunk = items[i0:i0 + args.batch]
        tgt_pils = [Image.open(os.path.join(args.data, it["target"])).convert("RGB") for it in chunk]
        # shape-agnostic PIL -> (B,3,H,W) in [0,1]; RaeCodec.to_pix01 asserts the 1280x512 panel canvas
        pix = torch.from_numpy(np.stack([np.asarray(p, dtype=np.uint8) for p in tgt_pils])
                               ).permute(0, 3, 1, 2).float().div_(255.0).to(DEVICE)
        assert pix.shape[2:] == img_hw, (pix.shape, img_hw)
        with torch.no_grad():
            z = codec.encode(pix).float()
            stats["z_mean"].append(float(z.mean()))
            stats["z_std"].append(float(z.std()))
            if args.codec == "vae" and i0 == 0:
                stats["post_std"].append(codec.posterior_std(pix))
            for e in eps_list:
                if e > 0 and i0 >= args.n_eps:
                    continue
                zz = z if e == 0 else z + e * torch.randn(z.shape, generator=gen, device=DEVICE)
                rec = codec.decode(zz.to(DT)).float()
                pils = codec.to_pil(rec)
                for j, it in enumerate(chunk):
                    base = f"{split_name}_{it['id']:06d}"
                    im_dir = os.path.join(dirs[e], "imgs")
                    pils[j].save(os.path.join(im_dir, base + "_pred.png"))
                    # gt/in are SYMLINKS to the dataset files: the scorer only reads them, and
                    # copying 3000 x 1280x512 PNGs would cost GB for nothing.
                    for suf, key in (("_gt.png", "target"), ("_in.png", "input")):
                        lp = os.path.join(im_dir, base + suf)
                        if not os.path.lexists(lp):
                            os.symlink(os.path.abspath(os.path.join(args.data, it[key])), lp)
                    rows_by_eps[e].append({"id": it["id"], "expr": str(it["expr"]),
                                           "gt": str(it["answer"]), "base": base})
                    g = pix[j].cpu()
                    r = rec[j].cpu()
                    bx = it.get("ans_box")                                   # square re-renders carry their own box
                    abox = (bx[1], bx[3], bx[0], bx[2]) if bx else (ANS_Y[0], ANS_Y[1], ANS_X[0], ANS_X[1])
                    per_eps_px[e]["psnr_full"].append(psnr(r, g))
                    per_eps_px[e]["psnr_ans"].append(psnr(r, g, abox))
                    per_eps_px[e]["iou"].append(ink_iou(r, g))
        if (i0 // args.batch) % 20 == 0:
            print(f"[gate] {i0 + len(chunk)}/{len(items)}", flush=True)

    # ---- per-channel z std, on the last batch's tokens (cheap, and enough to spot a collapse)
    with torch.no_grad():
        ch = z.reshape(-1, z.shape[-1]).std(0).cpu().numpy()
    zstat = {"z_mean": float(np.mean(stats["z_mean"])), "z_std": float(np.mean(stats["z_std"])),
             "per_channel_std_pct": {p: float(np.percentile(ch, p)) for p in (0, 5, 50, 95, 100)}}
    if stats["post_std"]:
        zstat["vae_posterior_std"] = float(np.mean(stats["post_std"]))

    for e in eps_list:
        d = dirs[e]
        px = per_eps_px[e]
        rep = {"gate": True, "codec": args.codec, "label": label, "eps": e,
               "meta": codec.meta(), "z_stats": zstat, "seed": args.seed,
               "n_rows": len(rows_by_eps[e]),
               "psnr_full": float(np.mean(px["psnr_full"])) if px["psnr_full"] else None,
               "psnr_answer": float(np.mean(px["psnr_ans"])) if px["psnr_ans"] else None,
               "ink_iou": float(np.mean(px["iou"])) if px["iou"] else None,
               "test_rows": rows_by_eps[e] if split_name == "test" else [],
               "train_rows": rows_by_eps[e] if split_name == "train" else []}
        with open(os.path.join(d, "report.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(f"[gate] eps={e:.2f} n={rep['n_rows']} psnr_full={rep['psnr_full']:.2f} "
              f"psnr_answer={rep['psnr_answer']:.2f} ink_iou={rep['ink_iou']:.4f} -> {d}", flush=True)
    print(f"[gate] z mean={zstat['z_mean']:+.4f} std={zstat['z_std']:.4f} "
          f"per-channel std p0/p50/p100={zstat['per_channel_std_pct'][0]:.3f}/"
          f"{zstat['per_channel_std_pct'][50]:.3f}/{zstat['per_channel_std_pct'][100]:.3f}", flush=True)
    print(f"[gate] DONE {label}", flush=True)


if __name__ == "__main__":
    main()
