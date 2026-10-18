"""Minimal single-stream DiT (Peebles & Xie, "Scalable Diffusion Models with Transformers")
for latent-to-latent image editing with NO text pathway.

Faithful to the paper's block: pre-norm (LayerNorm without affine) -> adaLN-Zero modulation
(shift, scale, gate for attention and for the MLP, produced by SiLU -> Linear from the
conditioning vector) -> multi-head self-attention -> GELU(tanh) MLP; a final adaLN layer + linear
un-patchify; fixed 2-D sin-cos positional embedding; xavier init, adaLN and final layer zero-init.
The ONLY conditioning signal is the timestep (sigma). No class labels, no text, no cross-attention.

Tokens: the model works directly on the Qwen-Image VAE latents packed by the parent project's
pack_latents(): 16 channels x 2x2 patch = 64 numbers per token, 32 x 80 = 2560 tokens in row-major
order at 1280x512. Channel-concatenating the clean SOURCE latent to the noisy TARGET latent gives the
128-dim input per token (the "concat along channels" conditioning of inpainting / InstructPix2Pix);
the output is the 64-dim flow prediction for the target token. Patchify is therefore the paper's
p=2 linear patch embedding applied to the already-packed tokens.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sincos_2d(dim, grid_h, grid_w):
    """Fixed 2-D sin-cos positional embedding, (grid_h*grid_w, dim), ROW-MAJOR (h outer, w inner) --
    the same token order pack_latents() produces."""
    assert dim % 4 == 0, "dim must be divisible by 4"

    def one_d(d, pos):
        omega = torch.arange(d // 2, dtype=torch.float64) / (d / 2)
        omega = 1.0 / (10000.0 ** omega)
        out = pos.double()[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    hh, ww = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    emb = torch.cat([one_d(dim // 2, hh.reshape(-1)), one_d(dim // 2, ww.reshape(-1))], dim=1)
    return emb.float()


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])      # (B, heads, N, d)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_hidden, hidden))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, out_dim)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))

    def forward(self, x, c):
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm(x), shift, scale))


class VanillaDiT(nn.Module):
    CONFIG_KEYS = ("in_dim", "out_dim", "grid_h", "grid_w", "hidden", "depth", "heads", "mlp_ratio")

    def __init__(self, in_dim=128, out_dim=64, grid_h=32, grid_w=80, hidden=2048, depth=28, heads=32,
                 mlp_ratio=4.0):
        super().__init__()
        self.config = dict(in_dim=in_dim, out_dim=out_dim, grid_h=grid_h, grid_w=grid_w, hidden=hidden,
                           depth=depth, heads=heads, mlp_ratio=mlp_ratio)
        self.x_embed = nn.Linear(in_dim, hidden)                       # p=2 patch embedding on packed tokens
        self.register_buffer("pos_embed", sincos_2d(hidden, grid_h, grid_w)[None], persistent=False)
        self.t_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, mlp_ratio) for _ in range(depth)])
        self.final = FinalLayer(hidden, out_dim)
        self.grad_ckpt = False
        self.init_weights()

    def init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)
        for b in self.blocks:                                          # adaLN-Zero
            nn.init.zeros_(b.adaLN[-1].weight); nn.init.zeros_(b.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight); nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight); nn.init.zeros_(self.final.linear.bias)

    def forward(self, x, sigma, src):
        """x, src: (B, N, 64) packed latents (noisy target, clean source); sigma: (B,) in [0, 1].
        Returns the flow prediction (B, N, 64) for the target tokens."""
        assert x.shape == src.shape and x.shape[1] == self.pos_embed.shape[1], (x.shape, src.shape, self.pos_embed.shape)
        h = self.x_embed(torch.cat([x, src], dim=-1)) + self.pos_embed.to(x.dtype)
        c = self.t_embed(sigma.float() * 1000.0).to(h.dtype)
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                h = checkpoint(blk, h, c, use_reentrant=False)
            else:
                h = blk(h, c)
        return self.final(h, c)

    # ---- (de)serialisation, signature-compatible with the parent project's _save_hf() ----
    def save_pretrained(self, path, state_dict=None, safe_serialization=True):
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        sd = state_dict if state_dict is not None else self.state_dict()
        sd = {k: v.contiguous() for k, v in sd.items() if not k.endswith("pos_embed")}
        save_file(sd, os.path.join(path, "model.safetensors"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def from_pretrained(cls, path, torch_dtype=None):
        from safetensors.torch import load_file
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        m = cls(**{k: cfg[k] for k in cls.CONFIG_KEYS})
        missing, unexpected = m.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=False)
        missing = [k for k in missing if not k.endswith("pos_embed")]
        assert not missing and not unexpected, (missing, unexpected)
        return m.to(torch_dtype) if torch_dtype is not None else m

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
