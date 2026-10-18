"""RAE latent contract for the 1280x512 arithmetic I2I task.

GEOMETRY.  Every supported encoder is configured so the token grid is EXACTLY 32 x 80 = 2560
tokens of 768 dims -- same token count, same row-major order, same 16-original-pixels-per-token
footprint as the packed Qwen-VAE baseline.  Decoder p_d=16 then emits exactly 512 x 1280.
    dinov2 : p_e=14, encode at 448 x 1120 (0.875x bicubic).  448/14=32, 1120/14=80.
    siglip2: p_e=16, encode at 512 x 1280 (NO resize).       512/16=32, 1280/16=80.
    mae    : p_e=16, encode at 512 x 1280 (NO resize).

WHY AutoencoderRAE.encode()/decode() ARE BYPASSED (all verified against diffusers 0.39.0.dev0):
  1. _resize_and_normalize() force-resizes to a SQUARE.  MEASURED on this dataset, the square-224
     path costs the answer region 16.27 dB PSNR (vs 27.90) and drops ink IoU to 0.851 (vs 0.996).
  2. reshape_to_2d needs a perfect-square token count; 2560 is not one.  We keep (B,N,C).
  3. RAEDecoder.interpolate_latent() resamples any grid back to sqrt(num_patches); setting
     decoder.num_patches = 2560 makes it an exact identity (early-returns on l == num_patches).
  4. unpatchify() defaults to a square and _decode() never passes original_image_size.
  5. The decoder pos-embed is generated for a square 16x16 grid; we regenerate it for (32,80).
     It is persistent=False AND listed in _keys_to_ignore_on_load_unexpected, so replacing it
     cannot disturb checkpoint loading.

LATENT NORM.  (a) the encoder's final LayerNorm has its affine stripped in the checkpoint, so
tokens are already zero-mean/unit-var ACROSS channels -- nothing to do.  (b) the dataset-side
[C,gh,gw] BatchNorm-like stat is tied to the ImageNet 16x16 grid.  MEASURED from the real
config.json: within-channel spatial variation is only 3.2% relative, so 'released' (collapse to
per-channel RMS: min 0.0819 / median 0.8540 / max 2.7376) is near-lossless AND has zero
dependence on our data -- symmetric with the VAE arm's published constants.
"""
import hashlib, torch, torch.nn.functional as F
from diffusers.models.embeddings import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid

IMG_H, IMG_W = 512, 1280
GRID_H, GRID_W = 32, 80
N_TOK, TOK_DIM = GRID_H * GRID_W, 768          # 2560 x 768
PATCH_D = 16
ENC_HW = {"dinov2": (448, 1120), "siglip2": (IMG_H, IMG_W), "mae": (IMG_H, IMG_W)}

# MEASURED on data/arithmetic_add_unordered (200 train pairs, today):
#   answer line ink bbox y[281,340] x[81,365] -> token rows 17..21, cols 5..22 (90 tok, 3.52%)
#   answer DIGITS only          x[159,365]    -> token rows 17..21, cols 9..22 (70 tok, 2.73%)
#   question line ink       y[171,240]        -> token rows 10..15
ANSWER_ROWS, ANSWER_COLS = (17, 22), (5, 23)
DIGIT_ROWS,  DIGIT_COLS  = (17, 22), (9, 23)
QUESTION_ROWS = (10, 16)

def token_mask(rows, cols):
    m = torch.zeros(GRID_H, GRID_W, dtype=torch.bool); m[rows[0]:rows[1], cols[0]:cols[1]] = True
    return m.reshape(-1)

def sha256_file(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:n]

def _sincos_grid(dim, spacing, gh=GRID_H, gw=GRID_W):
    """2-D sin-cos over gh x gw with the SAME spacing on both axes (geometrically isotropic), in
    diffusers' own layout: flat order row-major h-outer/w-inner, first half of the channels
    encodes W and second half H, one zero CLS row prepended.  The single-call API cannot express
    this -- its spacing is base/(L*iscale), which cannot be equal on two axes of different length
    -- so the grid is built by hand.  SELF-TESTED: at gh=gw=16, spacing 1.0, this reproduces
    get_2d_sincos_pos_embed(dim,16,cls_token=True,extra_tokens=1) exactly (allclose atol 1e-6)."""
    h = torch.arange(gh, dtype=torch.float32) * spacing
    w = torch.arange(gw, dtype=torch.float32) * spacing
    gw_m, gh_m = torch.meshgrid(w, h, indexing="xy")          # each (gh, gw)
    grid = torch.stack([gw_m, gh_m], dim=0)                   # [0]=W, [1]=H  (diffusers order)
    pe = get_2d_sincos_pos_embed_from_grid(dim, grid, output_type="pt")
    return torch.cat([torch.zeros(1, dim), pe], dim=0)

class _SiglipVisionWrapper(torch.nn.Module):
    """Re-nests a transformers-5.x flat SiglipVisionModel under `.vision_model`.

    WHY: the released RAE siglip2 checkpoint was saved with an OLDER transformers, where
    SiglipVisionModel contained a `.vision_model` (SiglipVisionTransformer) submodule, so EVERY
    encoder tensor in the file is named `encoder.vision_model.*` (verified: 206/206).
    transformers 5.8.0 flattened that away, so a freshly built model has `encoder.encoder.*`,
    `encoder.embeddings.*`, `encoder.head.*` -- NOT ONE key matches. from_pretrained then reports
    them all as "newly initialized", leaves them on the META device, and the first .to(cuda) dies
    with "Cannot copy out of meta tensor". Silent until that point: no key-mismatch exception.
    Re-nesting restores the checkpoint's own naming, which also makes diffusers' original
    `model.vision_model.post_layernorm` line work unchanged."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, *a, **kw):
        return self.vision_model(*a, **kw)

    @property
    def config(self):
        return self.vision_model.config


def _patch_siglip_builder():
    """Make diffusers build a siglip2 encoder whose parameter names match the released checkpoint.
    See _SiglipVisionWrapper for the root cause. No-op for dinov2/mae, whose names are unchanged."""
    import diffusers.models.autoencoders.autoencoder_rae as R
    if getattr(R, "_si_siglip_patched", False):
        return
    orig = R._build_encoder

    def build(encoder_type, hidden_size, patch_size, num_hidden_layers, head_dim=64):
        if encoder_type != "siglip2":
            return orig(encoder_type, hidden_size, patch_size, num_hidden_layers, head_dim)
        from transformers import SiglipVisionConfig, SiglipVisionModel
        m = SiglipVisionModel(SiglipVisionConfig(
            hidden_size=hidden_size, patch_size=patch_size, image_size=256,
            num_attention_heads=hidden_size // head_dim, num_hidden_layers=num_hidden_layers))
        if not hasattr(m, "vision_model"):          # transformers >= 5 flattened it; re-nest
            m = _SiglipVisionWrapper(m)
        # RAE strips the final LayerNorm affine (identity LN); the checkpoint has no
        # post_layernorm tensors at all (verified), so these must not stay as live params.
        ln = m.vision_model.post_layernorm
        ln.weight = None
        ln.bias = None
        m.requires_grad_(False)
        return m

    R._build_encoder = build
    R._si_siglip_patched = True


class RaeCodec(torch.nn.Module):
    """Frozen RAE encoder (+ decoder) retargeted to the 32x80 non-square grid."""

    def __init__(self, rae_path, stat_mode="released", stats_path=None, decoder_ft=None,
                 pos_embed_mode="interp", with_decoder=True, dtype=torch.bfloat16,
                 resize_antialias=False):
        super().__init__()
        from diffusers import AutoencoderRAE
        _patch_siglip_builder()
        rae = AutoencoderRAE.from_pretrained(rae_path, torch_dtype=dtype)
        # HARD GUARD against the silent failure the siglip2 checkpoint exposed: when the shipped
        # module tree and the checkpoint's key names disagree, from_pretrained does NOT raise -- it
        # reports "newly initialized" to the log and leaves those params on the META device. The
        # model then looks fine until the first .to(cuda), and a codec with a RANDOMLY INITIALISED
        # encoder would otherwise produce a perfectly plausible "this encoder does not work"
        # result. Fail loudly, here, with the offending names.
        meta = [n for n, t in list(rae.named_parameters()) + list(rae.named_buffers())
                if getattr(t, "is_meta", False)]
        assert not meta, (
            f"{len(meta)} tensors are on the META device after from_pretrained({rae_path!r}) -- the "
            f"checkpoint's key names do not match this transformers/diffusers version's module tree, "
            f"so those weights are RANDOM, not loaded. First few: {meta[:8]}")
        c = rae.config
        self.encoder_type = c.encoder_type
        assert self.encoder_type in ENC_HW, self.encoder_type
        assert int(c.encoder_hidden_size) == TOK_DIM and int(c.patch_size) == PATCH_D
        assert float(c.scaling_factor) == 1.0, c.scaling_factor
        self.enc_h, self.enc_w = ENC_HW[self.encoder_type]
        pe = int(c.encoder_patch_size)
        assert self.enc_h % pe == 0 and self.enc_w % pe == 0
        assert (self.enc_h // pe, self.enc_w // pe) == (GRID_H, GRID_W)
        self.n_skip = {"dinov2": 1 + int(getattr(rae.encoder.config, "num_register_tokens", 4)),
                       "siglip2": 0, "mae": 1}[self.encoder_type]
        self.patch_e, self.dtype_, self.resize_antialias = pe, dtype, bool(resize_antialias)
        self.latent_id = (f"rae_{self.encoder_type}_p{pe}_enc{self.enc_h}x{self.enc_w}"
                          f"_g{GRID_H}x{GRID_W}_d{TOK_DIM}_{stat_mode}_{pos_embed_mode}")

        if with_decoder:
            d = rae.decoder
            # NOTE this self-test only proves the generator is deterministic and unchanged by a
            # diffusers bump -- decoder_pos_embed is BUILT by this call and is persistent=False,
            # so it is never loaded from the checkpoint.  It is NOT evidence that the retargeted
            # field is usable by the trained decoder; only the stage-0 round-trip gate is.
            chk = get_2d_sincos_pos_embed(d.decoder_hidden_size, int(d.num_patches ** 0.5),
                                          cls_token=True, extra_tokens=1, output_type="pt")
            assert torch.allclose(chk.float(), d.decoder_pos_embed[0].float().cpu(), atol=1e-6)
            if pos_embed_mode == "interp":
                # base_size=16 divides positions by (grid/16): h,w BOTH in [0,16). An in-RANGE
                # resample of the trained coordinate field (spacing 0.5 on h, 0.2 on w).
                new = get_2d_sincos_pos_embed(d.decoder_hidden_size, (GRID_H, GRID_W),
                                              cls_token=True, extra_tokens=1, output_type="pt")
            elif pos_embed_mode == "iso":
                # ISOTROPIC spacing 16/GRID_W = 0.2 on BOTH axes so the pixel aspect is
                # undistorted: h in [0,6.40), w in [0,16.00). Both stay inside the trained range.
                new = _sincos_grid(d.decoder_hidden_size, 16.0 / GRID_W)
            else:
                raise ValueError(pos_embed_mode)
            d.register_buffer("decoder_pos_embed", new.unsqueeze(0).float(), persistent=False)
            d.num_patches = N_TOK                       # interpolate_latent() -> exact identity
            if decoder_ft:
                from safetensors.torch import load_file
                miss, unexp = d.load_state_dict(load_file(decoder_ft), strict=False)
                assert not [k for k in miss if "pos_embed" not in k] and not unexp, (miss, unexp)
        else:
            rae.decoder = None
        self.rae = rae.requires_grad_(False).eval()

        mean, std = self._load_stats(stat_mode, stats_path, rae)
        self.register_buffer("lat_mean", mean.view(1, 1, TOK_DIM).float())
        self.register_buffer("lat_std",  std.view(1, 1, TOK_DIM).float())
        assert float(self.lat_std.min()) > 1e-3, f"degenerate latent std {float(self.lat_std.min()):.2e}"
        self.stat_mode, self.pos_embed_mode = stat_mode, pos_embed_mode

    def _load_stats(self, mode, path, rae):
        if mode == "released":
            s, m = rae._latents_std.float(), rae._latents_mean.float()
            std  = torch.sqrt((s ** 2).mean(dim=(1, 2))) if s.ndim == 3 else s.expand(TOK_DIM).clone()
            mean = m.mean(dim=(1, 2))                  if m.ndim == 3 else m.expand(TOK_DIM).clone()
            return mean, std
        if mode == "dataset":
            d = torch.load(path, map_location="cpu")
            assert d["latent_id_geom"] == f"{self.encoder_type}_{self.enc_h}x{self.enc_w}_{GRID_H}x{GRID_W}", d
            return d["mean"].float(), d["std"].float()
        if mode == "none":
            return torch.zeros(TOK_DIM), torch.ones(TOK_DIM)
        raise ValueError(mode)

    # ---------------- pixels <-> tokens ----------------
    @staticmethod
    def to_pix01(pils):
        """PIL RGB 1280x512 -> (B,3,512,1280) float32 in [0,1].  NOT [-1,1]: each RAE encoder wants
        its OWN normalization (DINOv2/MAE = ImageNet mean/std; SigLIP2 = 0.5/0.5 -- read from the
        checkpoint's encoder_norm_mean/std, never hardcoded), and keeping a [0,1] copy is what makes
        the ink check principled (the baseline's `< 0` only meant 'dark' because VaeImageProcessor
        mapped to [-1,1])."""
        import numpy as np
        a = np.stack([np.asarray(p.convert("RGB"), dtype=np.uint8) for p in pils])
        x = torch.from_numpy(a).permute(0, 3, 1, 2).float().div_(255.0)
        assert x.shape[2:] == (IMG_H, IMG_W), x.shape
        return x

    def _prep(self, pix01):
        x = pix01.float()
        if (x.shape[2], x.shape[3]) != (self.enc_h, self.enc_w):
            # antialias=False on purpose: MEASURED on 24 of these images at 0.875x, the
            # round-trip is 39.39 dB / IoU 0.9963 WITHOUT antialias vs 38.44 dB / 0.9950 WITH.
            # At a downscale this mild the bicubic support already covers the kernel and
            # antialias only adds blur.  It also matches diffusers' own reference call.
            x = F.interpolate(x, size=(self.enc_h, self.enc_w), mode="bicubic",
                              align_corners=False, antialias=self.resize_antialias).clamp_(0, 1)
        return ((x - self.rae.encoder_mean.float()) / self.rae.encoder_std.float()).to(self.dtype_)

    def encode_raw(self, pix01):
        """(B,3,512,1280) in [0,1] -> (B,2560,768) RAW encoder tokens (pre dataset-norm)."""
        x = self._prep(pix01)
        assert x.shape[2:] == (self.enc_h, self.enc_w), x.shape
        e = self.rae.encoder
        if self.encoder_type == "mae":
            n = (self.enc_h // self.patch_e) * (self.enc_w // self.patch_e)
            noise = torch.arange(n, device=x.device, dtype=x.dtype)[None].expand(x.shape[0], -1)
            t = e(x, noise, interpolate_pos_encoding=True).last_hidden_state
        elif self.encoder_type == "siglip2":
            t = e(x, interpolate_pos_encoding=True).last_hidden_state
        else:   # dinov2: Dinov2WithRegistersEmbeddings.forward interpolates its [1,1370,768]
            t = e(x).last_hidden_state    # (37x37+1) pos-embed to (h/14,w/14) UNCONDITIONALLY,
                                          # in fp32 with antialias=True.  No flag needed.
        t = t[:, self.n_skip:]            # dinov2: [CLS, REG x4, patches] -> drop 5
        assert t.shape[1:] == (N_TOK, TOK_DIM), (t.shape, self.n_skip)
        return t

    def encode(self, pix01):
        """-> (B,2560,768) NORMALIZED float32 latents (the DiT's x-space). +1e-5 matches
        diffusers _normalize_latents; it is load-bearing (several channels have sd ~0.08)."""
        return (self.encode_raw(pix01).float() - self.lat_mean) / (self.lat_std + 1e-5)

    def decode_raw(self, tok):
        """(B,2560,768) RAW tokens -> (B,3,512,1280) float32 in [0,1].  No no_grad: the decoder
        fine-tuner calls this."""
        d = self.rae.decoder
        logits = d(tok.to(next(d.parameters()).dtype), return_dict=True).logits   # (B,2560,768)
        px = d.unpatchify(logits, original_image_size=(IMG_H, IMG_W))
        return (px.float() * self.rae.encoder_std.float() + self.rae.encoder_mean.float()).clamp(0, 1)

    def decode(self, z):
        return self.decode_raw(z.float() * (self.lat_std + 1e-5) + self.lat_mean)

    @staticmethod
    def to_pil(px01):
        from PIL import Image
        a = (px01.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(v) for v in a]

    def meta(self):
        return {"latent": self.latent_id, "encoder_type": self.encoder_type, "patch_e": self.patch_e,
                "enc_h": self.enc_h, "enc_w": self.enc_w, "grid": [GRID_H, GRID_W],
                "tokens": N_TOK, "dim": TOK_DIM, "stat_mode": self.stat_mode,
                "pos_embed_mode": self.pos_embed_mode, "n_skip": self.n_skip,
                "resize_antialias": self.resize_antialias,
                "lat_std_min": float(self.lat_std.min()), "lat_std_med": float(self.lat_std.median()),
                "lat_std_max": float(self.lat_std.max())}

