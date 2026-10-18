"""Two-stream MMDiT twin of dit_model.VanillaDiT for the square-canvas, latent-to-latent, no-text rows (2026-10-05).

Same block internals as VanillaDiT (pre-norm LayerNorm without affine, adaLN-Zero shift/scale/gate from SiLU->Linear of
the timestep embedding, SDPA attention, GELU(tanh) MLP, xavier init, adaLN + final layer zero-init, fixed 2-D sin-cos
position embedding, timestep = sigma*1000), but the clean SOURCE tokens and the noised TARGET tokens are two STREAMS
(SD3 / Flux "double block" layout):

  * each stream has its own input projection Linear(D -> hidden) -- so no arm is compressed at the input (the channel-concat
    model's Linear(2D -> hidden) squeezes MAE/WebSSL/SigLIP2 by 1.3-2x) -- and its own LayerNorms, QKV, output projection,
    MLP and adaLN modulation in every block;
  * attention is JOINT: queries/keys/values of both streams are concatenated along the sequence, so every token attends to
    all 2N tokens, then the outputs are split back and each stream takes its own gated residual;
  * both streams are updated in every block except the LAST, whose source stream stops after contributing keys/values
    (SD3's "context_pre_only": its post-attention projection/MLP would never reach the loss and would be dead weights);
    the final adaLN + Linear(hidden -> D) reads the TARGET stream only.

Parameter count ~2x VanillaDiT at the same depth/width (two copies of every block weight); attention cost ~2x (2N tokens).
Interface identical to VanillaDiT: forward(x, sigma, src) -> flow prediction for x; save_pretrained / from_pretrained with
CONFIG_KEYS (in_dim here = the per-stream token dim D, not 2D).
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from dit_model import FinalLayer, TimestepEmbedder, modulate, sincos_2d


class StreamParams(nn.Module):
    """One stream's share of an MMDiT block: norms, qkv, out-proj, MLP, adaLN (same shapes as a DiTBlock)."""

    def __init__(self, hidden, heads, mlp_ratio=4.0, with_update=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.with_update = with_update
        if with_update:
            self.proj = nn.Linear(hidden, hidden)
            self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
            mlp_hidden = int(hidden * mlp_ratio)
            self.mlp = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden, hidden))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden, (6 if with_update else 2) * hidden))


class MMDiTBlock(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio=4.0, update_c=True):
        super().__init__()
        assert hidden % heads == 0
        self.heads, self.update_c = heads, update_c
        self.x = StreamParams(hidden, heads, mlp_ratio)                        # target stream (noised tokens)
        self.c = StreamParams(hidden, heads, mlp_ratio, with_update=update_c)  # source stream (clean condition tokens)

    def _qkv(self, sp, h, shift, scale):
        B, N, C = h.shape
        return sp.qkv(modulate(sp.norm1(h), shift, scale)).view(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)

    def forward(self, hx, hc, cond):
        B, Nx, C = hx.shape
        mx = self.x.adaLN(cond).chunk(6, dim=1)
        mc = self.c.adaLN(cond).chunk(6 if self.update_c else 2, dim=1)
        qkv_x = self._qkv(self.x, hx, mx[0], mx[1])
        qkv_c = self._qkv(self.c, hc, mc[0], mc[1])
        q = torch.cat([qkv_x[0], qkv_c[0]], dim=2)            # (B, heads, Nx+Nc, d): joint attention over both streams
        k = torch.cat([qkv_x[1], qkv_c[1]], dim=2)
        v = torch.cat([qkv_x[2], qkv_c[2]], dim=2)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, Nx + hc.shape[1], C)
        ox, oc = o[:, :Nx], o[:, Nx:]
        hx = hx + mx[2].unsqueeze(1) * self.x.proj(ox)
        hx = hx + mx[5].unsqueeze(1) * self.x.mlp(modulate(self.x.norm2(hx), mx[3], mx[4]))
        if self.update_c:
            hc = hc + mc[2].unsqueeze(1) * self.c.proj(oc)
            hc = hc + mc[5].unsqueeze(1) * self.c.mlp(modulate(self.c.norm2(hc), mc[3], mc[4]))
        return hx, hc


class MMDiT(nn.Module):
    CONFIG_KEYS = ("in_dim", "out_dim", "grid_h", "grid_w", "hidden", "depth", "heads", "mlp_ratio")

    def __init__(self, in_dim=64, out_dim=64, grid_h=16, grid_w=16, hidden=1152, depth=12, heads=16, mlp_ratio=4.0):
        super().__init__()
        self.config = dict(in_dim=in_dim, out_dim=out_dim, grid_h=grid_h, grid_w=grid_w, hidden=hidden,
                           depth=depth, heads=heads, mlp_ratio=mlp_ratio)
        self.x_embed = nn.Linear(in_dim, hidden)                       # target stream input projection (D -> hidden)
        self.c_embed = nn.Linear(in_dim, hidden)                       # source stream input projection (D -> hidden)
        self.register_buffer("pos_embed", sincos_2d(hidden, grid_h, grid_w)[None], persistent=False)
        self.t_embed = TimestepEmbedder(hidden)
        self.blocks = nn.ModuleList([MMDiTBlock(hidden, heads, mlp_ratio, update_c=(i < depth - 1)) for i in range(depth)])
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
        for b in self.blocks:                                          # adaLN-Zero on both streams
            for sp in (b.x, b.c):
                nn.init.zeros_(sp.adaLN[-1].weight); nn.init.zeros_(sp.adaLN[-1].bias)
        nn.init.zeros_(self.final.adaLN[-1].weight); nn.init.zeros_(self.final.adaLN[-1].bias)
        nn.init.zeros_(self.final.linear.weight); nn.init.zeros_(self.final.linear.bias)

    def forward(self, x, sigma, src):
        """x, src: (B, N, D) noised target / clean source tokens; sigma: (B,) in [0, 1].  Returns (B, N, D) flow for x."""
        assert x.shape == src.shape and x.shape[1] == self.pos_embed.shape[1], (x.shape, src.shape, self.pos_embed.shape)
        pe = self.pos_embed.to(x.dtype)
        hx = self.x_embed(x) + pe
        hc = self.c_embed(src) + pe
        cond = self.t_embed(sigma.float() * 1000.0).to(hx.dtype)
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                hx, hc = checkpoint(blk, hx, hc, cond, use_reentrant=False)
            else:
                hx, hc = blk(hx, hc, cond)
        return self.final(hx, cond)

    # ---- (de)serialisation, identical conventions to VanillaDiT ----
    def save_pretrained(self, path, state_dict=None, safe_serialization=True):
        from safetensors.torch import save_file
        os.makedirs(path, exist_ok=True)
        sd = state_dict if state_dict is not None else self.state_dict()
        sd = {k: v.contiguous() for k, v in sd.items() if not k.endswith("pos_embed")}
        save_file(sd, os.path.join(path, "model.safetensors"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(dict(self.config, arch="mmdit"), f, indent=2)

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


if __name__ == "__main__":
    m = MMDiT(in_dim=64, out_dim=64)
    x = torch.randn(2, 256, 64); s = torch.randn(2, 256, 64); t = torch.rand(2)
    y = m(x, t, s)
    print("out", tuple(y.shape), "params", f"{m.n_params()/1e6:.1f}M (VanillaDiT 12x1152 ~ 291M)")
