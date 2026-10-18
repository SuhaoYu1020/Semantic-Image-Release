"""sq_codecs.py -- the four first-round codecs on the 256x256 square render, every one at its decoder's NATIVE 16x16
token grid, behind ONE interface for the DiT-only row (condition = clean source tokens, input = noised target tokens,
output = target tokens):

    codec = SqCodec("siglip2", stats_path=".../stats/sq256_same/siglip2.pt")
    z   = codec.encode(pix01)      # (B,3,256,256) in [0,1] -> (B,256,D) float32, per-channel standardised
    img = codec.decode(z)          # (B,256,D) -> (B,3,256,256) float32 in [0,1]
    codec.dim, codec.n_tok (256), codec.grid (16,16), codec.latent_id, codec.meta()

arm      encoder -> decoder                                   inner latent / normalisation                     D
vae      Qwen-Image (Wan2.1) VAE, 256^2 -> 32x32x16 -> 2x2 pack  published latents_mean/std                      64
mae      RAE MAE-B/16 (released) -> ViT-XL decoder, OFFICIAL square path at 256  released per-dim stats           768
webssl   WebSSL-DINO300M/14 @224 -> Scale-RAE decoder            none (raw last_hidden_state, CLS dropped)        1024
siglip2  SigLIP2-So400M/14 @224 -> Scale-RAE decoder             param-free LayerNorm of hidden_states[-1]        1152

The OUTER per-channel standardisation uses statistics measured on OUR train renders (compute_latent_stats.py).  It is
applied on top of each codec's own inner normalisation so that every arm hands the DiT a unit-variance latent and the
rectified-flow sigma schedule means the same thing in every arm (WebSSL's raw tokens have std 1.33, the VAE's 0.85 on
these renders).  decode() undoes it before the decoder.  stats_path=None -> identity (raw inner latent; the gate ran
this way).
"""
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
SI_ROOT = os.environ.get("SI_ROOT") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SI_CKPTS = os.environ.get("SI_CKPTS") or os.path.join(SI_ROOT, "ckpts")
SI_DATA = os.environ.get("SI_DATA") or os.path.join(SI_ROOT, "data")

CANVAS = 256
GRID = (16, 16)
N_TOK = GRID[0] * GRID[1]
ARMS = {"vae": 64, "mae": 768, "webssl": 1024, "siglip2": 1152,
        # round 3 (2026-10-05): r3_codecs wrappers, same outer standardisation.  emu35 = Emu3.5 IBQ code vectors (16x16 grid,
        # decoded WITHOUT re-quantisation: the gate's eps-robust reading); tatitok128 = TA-TiTok KL posterior mean, 128 slots x 16,
        # empty caption, collapsed dims get seeded noise at decode; flextok256 = FlexTok-DFN FSQ quants, 256 registers x 6,
        # decode = round to the FSQ grid + 25-step RF decoder.  The 1-D arms have NO geometry: their slots are laid out on a
        # pseudo-grid (8x16 / 16x16) only so the DiT's 2-D sin-cos position embedding (identical module) gives each slot a
        # fixed distinct code; answer-box token masks are undefined for them (geometric=False).
        "emu35": 256, "tatitok128": 16, "flextok256": 6,
        # encoder-only arms for the probe-prediction test (2026-10-13): no MMDiT row, used by the transfer probes and the regressor E
        "clipL": 1024, "dinov2b": 768, "jepa": 1280, "llamagen": 8}
# intervention arms (intervene.py, 2026-10-11): a base arm whose answer-box tokens are rewritten by an invertible map that
# changes the answer-class geometry only (lin = scale the answer subspace, cls = move class means apart); same D / grid as base.
from intervene import INTERVENTIONS  # noqa: E402
for _a, (_b, _mode, _alpha) in INTERVENTIONS.items():
    ARMS[_a] = ARMS[_b]
ARM_GRID = {"tatitok128": (8, 16)}                       # everything else: GRID
ARM_GEOMETRIC = {"vae", "mae", "webssl", "siglip2", "emu35", "clipL", "dinov2b", "jepa", "llamagen"} | set(INTERVENTIONS)
DEFAULT_DATA = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same")
DEFAULT_STATS_DIR = os.path.join(HERE, "stats", "sq256_same")


def to_pix01(pils):
    """list of PIL RGB -> (B,3,H,W) float32 in [0,1] (shape-agnostic)."""
    a = np.stack([np.asarray(p.convert("RGB"), dtype=np.uint8) for p in pils])
    return torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(255.0)


def answer_token_mask(data_root, grid=GRID):
    """Boolean (N_TOK,) mask of the answer-DIGIT tokens from the dataset's _answer_token_box.json (inclusive ranges)."""
    import json
    b = json.load(open(os.path.join(data_root, "_answer_token_box.json")))
    assert list(b["grid"]) == list(grid), (b["grid"], grid)
    m = torch.zeros(grid, dtype=torch.bool)
    m[b["rows"][0]:b["rows"][1] + 1, b["cols"][0]:b["cols"][1] + 1] = True
    return m.reshape(-1)


def sha256_file(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:n]


class SqCodec(nn.Module):
    def __init__(self, arm, stats_path=None, with_decoder=True, dtype=torch.bfloat16):
        super().__init__()
        assert arm in ARMS, arm
        from recon_gate import VaeCodec, RaeNativeCodec            # sibling module: sets sys.path for rae_codec etc.
        from scalerae_codec import ScaleRaeCodec
        hw = (CANVAS, CANVAS)
        self.intervention = None
        req_arm = arm
        if arm in INTERVENTIONS:                                   # build the BASE codec, then wrap encode/decode
            from intervene import load_intervention
            self.intervention = load_intervention(arm)
            arm = INTERVENTIONS[req_arm][0]
        if arm == "vae":
            inner = VaeCodec(dtype=dtype, img_hw=hw)
        elif arm == "mae":
            inner = RaeNativeCodec(os.path.join(SI_CKPTS, "RAE-mae-base-p16-ViTXL-n08"), dtype=dtype, img_hw=hw)
        elif arm in ("webssl", "siglip2"):
            enc = {"webssl": "webssl_dino300m", "siglip2": "siglip2_so400m"}[arm]
            inner = ScaleRaeCodec(enc, stat_mode="none", pos_embed_mode="interp", with_decoder=with_decoder,
                                  dtype=dtype, grid=GRID)
            inner.out_hw = hw
        elif arm == "emu35":                                   # round-3 codecs: fp32 (quantiser / FSQ rounding), own decoders
            from r3_codecs import Emu35Codec
            inner = Emu35Codec(dtype=torch.float32, mode="cont")
        elif arm == "tatitok128":
            from r3_codecs import TatitokKlCodec
            inner = TatitokKlCodec(variant="sl128_vae", caption="", dtype=torch.float32)
        elif arm == "flextok256":
            from r3_codecs import FlexTokCodec
            inner = FlexTokCodec(budget=256, timesteps=25, dtype=torch.float32)
        elif arm == "clipL":
            from extra_codecs import ClipLCodec
            inner = ClipLCodec(dtype=dtype)
        elif arm == "dinov2b":
            inner = RaeNativeCodec(os.path.join(SI_CKPTS, "RAE-dinov2-wReg-base-ViTXL-n08"), dtype=dtype, img_hw=hw)
        elif arm == "jepa":
            from r2_codecs import build_r2_codec
            inner = build_r2_codec("jepa", dtype=dtype)
        elif arm == "llamagen":
            from r2_codecs import build_r2_codec
            inner = build_r2_codec("llamagen_cont")
        else:
            raise ValueError(arm)
        self.inner = inner.requires_grad_(False).eval()
        grid = ARM_GRID.get(arm, GRID)
        self.base_arm = arm; arm = req_arm
        self.arm, self.dim, self.n_tok, self.grid, self.dtype_ = arm, ARMS[arm], grid[0] * grid[1], grid, dtype
        self.geometric = arm in ARM_GEOMETRIC
        assert self.n_tok == getattr(inner, "n_tok", self.n_tok) and self.dim == getattr(inner, "dim", self.dim), (arm, inner.n_tok, inner.dim)
        if stats_path:
            st = torch.load(stats_path, map_location="cpu")
            assert st.get("arm") == self.base_arm and int(st["mean"].numel()) == self.dim, (stats_path, st.get("arm"), st["mean"].shape)
            mean, std = st["mean"].float().view(1, 1, -1), st["std"].float().clamp_min(1e-3).view(1, 1, -1)
            self.stats_info = {"stats_path": stats_path, "stats_sha": sha256_file(stats_path), "stats_n_tokens": int(st.get("n_tokens", 0)),
                               "std_min": float(std.min()), "std_med": float(std.median()), "std_max": float(std.max())}
        else:
            mean, std = torch.zeros(1, 1, self.dim), torch.ones(1, 1, self.dim)
            self.stats_info = {"stats_path": None}
        self.register_buffer("lat_mean", mean)
        self.register_buffer("lat_std", std)
        tag = "dstats" if stats_path else "raw"
        self.latent_id = f"sq{CANVAS}_{arm}_g{self.grid[0]}x{self.grid[1]}_d{self.dim}_{tag}"

    # inner latent (codec's own normalisation, no dataset stats) -- what compute_latent_stats.py measures
    @torch.no_grad()
    def encode_inner(self, pix01):
        z = self.inner.encode(pix01.to(self.lat_mean.device)).float()
        assert z.shape[1:] == (self.n_tok, self.dim), (z.shape, self.n_tok, self.dim)
        return z

    @torch.no_grad()
    def encode(self, pix01):
        z = (self.encode_inner(pix01) - self.lat_mean) / self.lat_std
        if self.intervention is not None:
            z = self.intervention.to(z.device).apply(z)
        return z

    @torch.no_grad()
    def decode(self, z):
        if self.intervention is not None:
            z = self.intervention.to(z.device).invert(z.float())
        return self.inner.decode(z.float() * self.lat_std + self.lat_mean).float().clamp(0, 1)

    to_pix01 = staticmethod(to_pix01)

    @staticmethod
    def to_pil(px01):
        from PIL import Image
        a = (px01.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(v) for v in a]

    def meta(self):
        m = {"latent": self.latent_id, "arm": self.arm, "base_arm": self.base_arm, "dim": self.dim, "tokens": self.n_tok, "grid": list(self.grid),
             "geometric": self.geometric, "canvas": CANVAS, "inner": self.inner.meta(), **self.stats_info}
        if self.intervention is not None:
            m["intervention"] = {"mode": self.intervention.mode, "alpha": self.intervention.alpha, "k": int(self.intervention.U.shape[0])}
        return m


def rule_shift(dim, n_tok=N_TOK):
    """RAE's dimension-dependent noise shift alpha = sqrt(m / 4096), m = total latent elements per image."""
    return float((n_tok * dim / 4096.0) ** 0.5)
