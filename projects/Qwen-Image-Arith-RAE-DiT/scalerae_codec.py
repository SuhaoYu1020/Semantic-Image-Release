"""scalerae_codec.py -- the two Scale-RAE decoders (nyu-visionx/siglip2_decoder, nyu-visionx/webssl300m_decoder)
with their frozen encoders, retargeted to the 32x80 grid, exposing the SAME API as rae_codec.RaeCodec
(encode_raw / encode / decode_raw / decode / to_pil / meta).

Why not import third_party/Scale-RAE: its package __init__ pulls the whole LLM stack (ezcolorlog, torch_xla hooks)
and its decoder.py is a vendored transformers-4.42 ViTMAE.  Verified 2026-10-03 that the checkpoint's per-layer
parameter names are IDENTICAL to transformers 5.8.0's ViTMAELayer, so the decoder is rebuilt here from HF layers and
loaded with strict=True.

Feature convention (read off third_party/Scale-RAE, commit dab3800, and the two released DiT configs):
  siglip2_so400m : SiglipVisionModel(google/siglip2-so400m-patch14-224) -> hidden_states[-1]  (last encoder layer,
                   BEFORE post_layernorm; mm_vision_select_layer = -1), then parameter-free F.layer_norm over the 1152
                   channels, eps 1e-6  [multimodal_encoder/siglip_encoder.py interpolate()].  SigLIP has no CLS.
  webssl_dino300m: Dinov2Model(facebook/webssl-dino300m-full2b-224) -> last_hidden_state[:, 1:]  (post-LN, CLS dropped,
                   mm_vision_select_feature = 'patch'), NO further normalisation [dino_encoder.py].
  decoder        : caller prepends a zero CLS row which the decoder drops (drop_cls_token=True) and replaces by its own
                   trainable_cls_token; fixed 2-D sin-cos pos-embed; 28 ViTMAE layers (1152/16 heads/FFN 4096); LN;
                   Linear 1152->588 = 14*14*3; unpatchify p14; pixel = out * std + mean of the ENCODER's processor
                   [multimodal_decoder/__init__.py].  Both released DiT configs: diffusion_norm_stats_path = None, i.e.
                   NO dataset mean/std on the latent.  stat_mode='none' reproduces that; 'dataset' is available for the
                   DiT side (the decoder never sees it: decode() undoes it first).
Grid: patch 14, so the 1280x512 render is encoded at 1120x448 (0.875x bicubic, the same resize the dinov2 RAE arm
used; measured cost of that resize alone 39.39 dB / ink IoU 0.9963) -> 32x80 = 2560 tokens.  The decoder emits
448x1120; decode() resizes back to 512x1280 so downstream scorers see the usual geometry.

Self-test (CPU is fine, ~4 GB RSS, a few minutes):
  python scalerae_codec.py --encoder siglip2_so400m --n 3 --device cpu
  python scalerae_codec.py --encoder webssl_dino300m --n 3 --device cpu
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from transformers import AutoConfig, Dinov2Model, SiglipVisionModel, ViTMAEConfig
from transformers.models.vit_mae.modeling_vit_mae import ViTMAELayer

from rae_codec import GRID_H, GRID_W, IMG_H, IMG_W, N_TOK, ANSWER_ROWS, ANSWER_COLS, _sincos_grid

HERE = os.path.dirname(os.path.abspath(__file__))
SI_ROOT = os.environ.get("SI_ROOT") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SI_CKPTS = os.environ.get("SI_CKPTS") or os.path.join(SI_ROOT, "ckpts")
SI_DATA = os.environ.get("SI_DATA") or os.path.join(SI_ROOT, "data")

PATCH_E = 14
ENC_H, ENC_W = GRID_H * PATCH_E, GRID_W * PATCH_E          # 448 x 1120

SPECS = {
    "siglip2_so400m": dict(encoder_dir="siglip2-so400m-patch14-224", decoder_dir="ScaleRAE-siglip2_decoder",
                           dim=1152, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    "webssl_dino300m": dict(encoder_dir="webssl-dino300m-full2b-224", decoder_dir="ScaleRAE-webssl300m_decoder",
                            dim=1024, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}


class ScaleRaeDecoder(nn.Module):
    """GeneralDecoder (third_party/Scale-RAE/scale_rae/model/multimodal_decoder/decoder.py) rebuilt from HF layers.
    forward() takes the N patch tokens WITHOUT any CLS row (equivalent to the official 'prepend a zero CLS, embed,
    drop it' because a Linear of a zero vector is dropped anyway)."""

    def __init__(self, cfg: ViTMAEConfig, enc_dim: int, n_patches: int):
        super().__init__()
        D = cfg.decoder_hidden_size
        dc = copy.deepcopy(cfg)
        dc.hidden_size, dc.num_hidden_layers = D, cfg.decoder_num_hidden_layers
        dc.num_attention_heads, dc.intermediate_size = cfg.decoder_num_attention_heads, cfg.decoder_intermediate_size
        self.decoder_embed = nn.Linear(enc_dim, D, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, D), requires_grad=False)
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.decoder_layers = nn.ModuleList([ViTMAELayer(dc) for _ in range(cfg.decoder_num_hidden_layers)])
        self.decoder_norm = nn.LayerNorm(D, eps=cfg.layer_norm_eps)
        self.decoder_pred = nn.Linear(D, cfg.patch_size ** 2 * cfg.num_channels, bias=True)
        self.patch_size, self.num_channels, self.num_patches = cfg.patch_size, cfg.num_channels, n_patches

    def forward(self, tok):
        x = self.decoder_embed(tok)
        x = torch.cat([self.trainable_cls_token.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        x = x + self.decoder_pos_embed.to(x.dtype)
        for layer in self.decoder_layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        x = self.decoder_norm(x)
        return self.decoder_pred(x)[:, 1:, :]                 # (B, N, p*p*3)

    def unpatchify(self, logits, hw):
        p, c = self.patch_size, self.num_channels
        gh, gw = hw[0] // p, hw[1] // p
        assert gh * gw == logits.shape[1], (gh, gw, logits.shape)
        x = logits.reshape(logits.shape[0], gh, gw, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(logits.shape[0], c, gh * p, gw * p)


class ScaleRaeCodec(nn.Module):
    def __init__(self, encoder_type, stat_mode="none", stats_path=None, pos_embed_mode="interp",
                 with_decoder=True, dtype=torch.bfloat16, siglip_feature="official", resize_antialias=False,
                 grid=(GRID_H, GRID_W)):
        """grid=(gh, gw): token grid the codec runs at; the image is resized to (gh*14, gw*14).  Default 32x80 =
        the panel geometry (16 px/token on the 512x1280 render).  (16,40) keeps the decoder's trained row count and
        extrapolates only columns (2.5x); (16,16) is the decoder's exact training geometry (render squashed 2.5x)."""
        super().__init__()
        assert encoder_type in SPECS, encoder_type
        gh, gw = int(grid[0]), int(grid[1])
        s = SPECS[encoder_type]
        self.encoder_type, self.dim, self.dtype_ = encoder_type, s["dim"], dtype
        self.pos_embed_mode, self.siglip_feature, self.resize_antialias = pos_embed_mode, siglip_feature, resize_antialias
        self.enc_h, self.enc_w, self.patch_e = gh * PATCH_E, gw * PATCH_E, PATCH_E
        self.register_buffer("px_mean", torch.tensor(s["mean"]).view(1, 3, 1, 1))
        self.register_buffer("px_std", torch.tensor(s["std"]).view(1, 3, 1, 1))
        enc_dir, dec_dir = os.path.join(SI_CKPTS, s["encoder_dir"]), os.path.join(SI_CKPTS, s["decoder_dir"])

        # ---------------- encoder ----------------
        if encoder_type == "siglip2_so400m":
            # google/siglip2-so400m-patch14-224 is the FIXED-resolution SigLIP2 (SiglipConfig), so the vision tower
            # is SiglipVisionModel.  The safetensors holds the full SiglipModel ('vision_model.*' + 'text_model.*'):
            # load the vision subset by stripping the prefix (transformers-5.x SiglipVisionModel is flat).
            from safetensors import safe_open
            cfg = AutoConfig.from_pretrained(enc_dir)
            enc = SiglipVisionModel(cfg.vision_config)
            sd = {}
            with safe_open(os.path.join(enc_dir, "model.safetensors"), "pt") as f:
                for k in f.keys():
                    if k.startswith("vision_model."):
                        sd[k[len("vision_model."):]] = f.get_tensor(k)
            miss, unexp = enc.load_state_dict(sd, strict=False)
            assert not miss and not unexp, (miss[:5], unexp[:5])
            assert enc.config.hidden_size == self.dim and enc.config.patch_size == PATCH_E
        else:
            enc, info = Dinov2Model.from_pretrained(enc_dir, output_loading_info=True)
            assert not info["missing_keys"] and not info["unexpected_keys"], info
            assert enc.config.hidden_size == self.dim and enc.config.patch_size == PATCH_E
            assert not getattr(enc.config, "num_register_tokens", 0), enc.config
        meta = [n for n, t in list(enc.named_parameters()) + list(enc.named_buffers()) if getattr(t, "is_meta", False)]
        assert not meta, f"{len(meta)} META tensors after loading {enc_dir}: {meta[:5]}"
        self.encoder = enc.to(dtype).requires_grad_(False).eval()

        # ---------------- decoder ----------------
        if with_decoder:
            dcfg = ViTMAEConfig.from_pretrained(dec_dir)
            assert dcfg.patch_size == PATCH_E and dcfg.decoder_hidden_size == 1152 and dcfg.decoder_num_hidden_layers == 28
            n_native = (dcfg.image_size // dcfg.patch_size) ** 2                       # 256
            dec = ScaleRaeDecoder(dcfg, self.dim, n_native)
            sd = torch.load(os.path.join(dec_dir, "model.pt"), map_location="cpu", weights_only=True)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            dec.load_state_dict(sd, strict=True)                                        # all 456 tensors
            # the released pos-embed must be the sin-cos field our generator produces for 16x16 -- otherwise the
            # retargeted field below would be in a different coordinate system than the one the decoder was trained in
            chk = get_2d_sincos_pos_embed(1152, int(n_native ** 0.5), cls_token=True, extra_tokens=1, output_type="pt")
            assert torch.allclose(chk.float(), dec.decoder_pos_embed[0].float(), atol=1e-5), "pos-embed generator mismatch"
            if pos_embed_mode == "native":                 # 16x16 only; for the 224^2 sanity test
                self.grid = (int(n_native ** 0.5), int(n_native ** 0.5))
                new = dec.decoder_pos_embed.data.clone()
            else:
                self.grid = (gh, gw)
                if pos_embed_mode == "interp":             # both axes resampled into the trained [0,16) range
                    new = get_2d_sincos_pos_embed(1152, (gh, gw), cls_token=True, extra_tokens=1,
                                                  output_type="pt").unsqueeze(0)
                elif pos_embed_mode == "iso":              # isotropic spacing 16/gw on both axes
                    new = _sincos_grid(1152, 16.0 / max(gh, gw), gh=gh, gw=gw).unsqueeze(0)
                else:
                    raise ValueError(pos_embed_mode)
            del dec.decoder_pos_embed
            dec.register_buffer("decoder_pos_embed", new.float(), persistent=False)
            dec.num_patches = self.grid[0] * self.grid[1]
            self.decoder = dec.to(dtype).requires_grad_(False).eval()
        else:
            self.decoder, self.grid = None, (gh, gw)
        self.n_tok = self.grid[0] * self.grid[1]
        self.in_hw = (self.grid[0] * PATCH_E, self.grid[1] * PATCH_E)
        self.out_hw = (IMG_H, IMG_W)          # canvas decode() returns; callers on other canvases overwrite it

        # ---------------- latent statistics (DiT side only) ----------------
        if stat_mode == "none":
            mean, std = torch.zeros(self.dim), torch.ones(self.dim)
        elif stat_mode == "dataset":
            st = torch.load(stats_path, map_location="cpu")
            mean, std = st["mean"].float().view(-1), st["std"].float().view(-1)
            assert mean.numel() == self.dim and float(std.min()) > 1e-3
        else:
            raise ValueError(stat_mode)
        self.register_buffer("lat_mean", mean.view(1, 1, self.dim).float())
        self.register_buffer("lat_std", std.view(1, 1, self.dim).float())
        self.stat_mode = stat_mode
        self.latent_id = (f"scalerae_{encoder_type}_p{PATCH_E}_enc{self.in_hw[0]}x{self.in_hw[1]}"
                          f"_g{self.grid[0]}x{self.grid[1]}_d{self.dim}_{stat_mode}_{pos_embed_mode}")

    # ---------------- encode ----------------
    def _prep(self, pix01):
        x = pix01.float()
        if (x.shape[2], x.shape[3]) != self.in_hw:
            x = F.interpolate(x, size=self.in_hw, mode="bicubic", align_corners=False,
                              antialias=self.resize_antialias).clamp_(0, 1)
        return ((x - self.px_mean) / self.px_std).to(self.dtype_)

    def encode_raw(self, pix01):
        """(B,3,H,W) in [0,1] -> (B, n_tok, dim) tokens in the DECODER's input convention (no dataset norm)."""
        x = self._prep(pix01)
        if self.encoder_type == "siglip2_so400m":
            out = self.encoder(x, interpolate_pos_encoding=True, output_hidden_states=True)
            if self.siglip_feature == "official":          # last layer, pre post_layernorm, then param-free LN
                t = F.layer_norm(out.hidden_states[-1].float(), (self.dim,), eps=1e-6)
            elif self.siglip_feature == "postln":          # ablation: last_hidden_state (post_layernorm), no LN
                t = out.last_hidden_state.float()
            elif self.siglip_feature == "postln_ln":       # ablation: post_layernorm then param-free LN
                t = F.layer_norm(out.last_hidden_state.float(), (self.dim,), eps=1e-6)
            else:
                raise ValueError(self.siglip_feature)
        else:
            t = self.encoder(x).last_hidden_state[:, 1:].float()        # drop CLS; Dinov2 interpolates pos-embed itself
        assert t.shape[1:] == (self.n_tok, self.dim), (t.shape, self.n_tok, self.dim)
        return t.to(self.dtype_)

    def encode(self, pix01):
        return (self.encode_raw(pix01).float() - self.lat_mean) / (self.lat_std + 1e-5)

    # ---------------- decode ----------------
    def decode_raw(self, tok, out_hw="auto"):
        """(B, n_tok, dim) decoder-convention tokens -> (B,3,out_hw) float32 in [0,1].  out_hw='auto' -> self.out_hw
        (default 512x1280, the panel canvas; the gate sets it to the dataset canvas), None -> decoder-native in_hw."""
        logits = self.decoder(tok.to(self.dtype_))
        px = self.decoder.unpatchify(logits.float(), self.in_hw)
        px = (px * self.px_std + self.px_mean).clamp(0, 1)
        if out_hw == "auto":
            out_hw = self.out_hw
        if out_hw is not None and tuple(out_hw) != self.in_hw:
            px = F.interpolate(px, size=tuple(out_hw), mode="bicubic", align_corners=False).clamp_(0, 1)
        return px

    def decode(self, z, out_hw="auto"):
        return self.decode_raw(z.float() * (self.lat_std + 1e-5) + self.lat_mean, out_hw=out_hw)

    @staticmethod
    def to_pil(px01):
        from PIL import Image
        a = (px01.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(v) for v in a]

    @staticmethod
    def to_pix01(pils):
        """list of PIL RGB -> (B,3,H,W) float in [0,1]; same contract as RaeCodec.to_pix01 (recon_gate uses it)."""
        import numpy as np
        return torch.stack([torch.from_numpy(np.asarray(p.convert("RGB"))).permute(2, 0, 1).float() / 255.0
                            for p in pils])

    def meta(self):
        return {"latent": self.latent_id, "encoder_type": self.encoder_type, "patch_e": PATCH_E,
                "enc_h": self.in_hw[0], "enc_w": self.in_hw[1], "grid": list(self.grid), "tokens": self.n_tok,
                "dim": self.dim, "stat_mode": self.stat_mode, "pos_embed_mode": self.pos_embed_mode,
                "siglip_feature": self.siglip_feature, "n_skip": 0 if self.encoder_type == "siglip2_so400m" else 1,
                "decoder": SPECS[self.encoder_type]["decoder_dir"], "encoder": SPECS[self.encoder_type]["encoder_dir"]}


# ====================================================================================================================
# self-test: round-trip real training renders through encoder -> decoder, native 224^2 and the 32x80 grid
# ====================================================================================================================
def _psnr(a, b):
    mse = F.mse_loss(a.float(), b.float()).item()
    return 99.0 if mse <= 0 else 10 * math.log10(1.0 / mse)


def _answer_crop(px, hw):
    """pixel crop of the answer band using the token-row/col constants (token = hw/grid)."""
    th, tw = hw[0] / GRID_H, hw[1] / GRID_W
    r0, r1 = int(ANSWER_ROWS[0] * th), int(ANSWER_ROWS[1] * th)
    c0, c1 = int(ANSWER_COLS[0] * tw), int(ANSWER_COLS[1] * tw)
    return px[..., r0:r1, c0:c1]


def _ink_iou(a, b, thr=0.5):
    ia, ib = (a.mean(1) < thr), (b.mean(1) < thr)
    inter, union = (ia & ib).sum().item(), (ia | ib).sum().item()
    return inter / max(1, union)


def _load_images(data_root, n, which="target"):
    from PIL import Image
    import numpy as np
    rows = [json.loads(l) for l in open(os.path.join(data_root, "train.jsonl")) if l.strip()][:n]
    ims = []
    for r in rows:
        im = Image.open(os.path.join(data_root, r[which])).convert("RGB")
        ims.append(torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(ims), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=sorted(SPECS))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default=os.path.join(SI_DATA, "arithmetic_add_unordered"))
    ap.add_argument("--out", default=os.path.join(HERE, "eval", "scalerae_check"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip_grid", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    dt = torch.float32 if dev.type == "cpu" else torch.bfloat16
    os.makedirs(a.out, exist_ok=True)
    imgs, rows = _load_images(a.data, a.n)
    print(f"[data] {imgs.shape} from {a.data}; answers {[r['answer'] for r in rows]}", flush=True)
    report = {"encoder": a.encoder, "device": a.device, "n": a.n, "native224": {}, "grid": {}}

    # ---- A. native 224x224 (the decoder's own training geometry; squash-resized render) ----
    t0 = time.time()
    codec = ScaleRaeCodec(a.encoder, pos_embed_mode="native", dtype=dt).to(dev)
    print(f"[load] {codec.latent_id} in {time.time()-t0:.1f}s; enc {sum(p.numel() for p in codec.encoder.parameters())/1e6:.0f}M "
          f"dec {sum(p.numel() for p in codec.decoder.parameters())/1e6:.0f}M", flush=True)
    x224 = F.interpolate(imgs, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1).to(dev)
    feats = ["official", "postln", "postln_ln"] if a.encoder == "siglip2_so400m" else ["official"]
    for fm in feats:
        codec.siglip_feature = fm
        with torch.no_grad():
            t1 = time.time()
            tok = codec.encode_raw(x224)
            rec = codec.decode_raw(tok, out_hw=None)
        ps = [_psnr(rec[i], x224[i]) for i in range(len(rec))]
        report["native224"][fm] = {"psnr": ps, "tok_mean": float(tok.float().mean()), "tok_std": float(tok.float().std()),
                                   "sec": time.time() - t1}
        print(f"[224 {fm:10s}] PSNR {['%.2f' % p for p in ps]}  tok mean {float(tok.float().mean()):+.3f} "
              f"std {float(tok.float().std()):.3f}  ({time.time()-t1:.1f}s)", flush=True)
        ScaleRaeCodec.to_pil(torch.cat([x224, rec], dim=3))[0].save(os.path.join(a.out, f"{a.encoder}_224_{fm}.png"))
    codec.siglip_feature = "official"
    del codec

    # ---- B. the 32x80 grid at 448x1120, both pos-embed retargetings ----
    if not a.skip_grid:
        for pm in ("interp", "iso"):
            t0 = time.time()
            codec = ScaleRaeCodec(a.encoder, pos_embed_mode=pm, dtype=dt).to(dev)
            ref = F.interpolate(imgs, size=codec.in_hw, mode="bicubic", align_corners=False).clamp(0, 1).to(dev)
            with torch.no_grad():
                t1 = time.time()
                tok = codec.encode_raw(imgs.to(dev))
                rec_native = codec.decode_raw(tok, out_hw=None)          # 448x1120
                rec_full = codec.decode_raw(tok)                          # 512x1280
            ps = [_psnr(rec_native[i], ref[i]) for i in range(len(ref))]
            ps_ans = [_psnr(_answer_crop(rec_native[i], codec.in_hw), _answer_crop(ref[i], codec.in_hw)) for i in range(len(ref))]
            iou = [_ink_iou(_answer_crop(rec_full[i:i+1], (IMG_H, IMG_W)), _answer_crop(imgs[i:i+1].to(dev), (IMG_H, IMG_W)))
                   for i in range(len(ref))]
            report["grid"][pm] = {"psnr_full": ps, "psnr_answer": ps_ans, "ink_iou_answer": iou,
                                  "tok_mean": float(tok.float().mean()), "tok_std": float(tok.float().std()),
                                  "sec": time.time() - t1, "meta": codec.meta()}
            print(f"[grid {pm:6s}] PSNR full {['%.2f' % p for p in ps]}  answer {['%.2f' % p for p in ps_ans]}  "
                  f"ink IoU(ans) {['%.3f' % v for v in iou]}  ({time.time()-t1:.1f}s enc+dec, load {t1-t0:.1f}s)", flush=True)
            panel = torch.cat([imgs.to(dev), rec_full], dim=2)            # input on top, recon below
            for i in range(len(panel)):
                ScaleRaeCodec.to_pil(panel[i:i+1])[0].save(os.path.join(a.out, f"{a.encoder}_grid_{pm}_{i}.png"))
            del codec
    with open(os.path.join(a.out, f"{a.encoder}_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"[done] report -> {os.path.join(a.out, a.encoder + '_report.json')}")


if __name__ == "__main__":
    main()
