"""sigma_samplers.py -- training-time noise-level samplers for the square-canvas DiT rows.

logit_normal : the SD3 / RAE default, sigma = sigmoid(N(0,1) + log(shift)).   (train_i2i_sq_dit.py keeps its own copy of this
               path so that every earlier run stays bit-identical; this module is used for the other kinds only.)
reason_mix   : "reasoning-matched" sampler.  Let A(sigma) be the arm's noise-probe curve (probe_noise_curve.py: how often the
               answer is still linearly readable from x_sigma = (1-sigma) z + sigma eps).  A fraction `mix` of the sigmas is drawn
               from the density  q(sigma) ∝ p_shift(sigma) (1 - A(sigma))  -- i.e. only from noise levels where the answer has to be
               COMPUTED from the source rather than copied from the noisy input -- and the rest from p_shift itself (keeps a little
               low-noise coverage so the standard Euler sampler still works at inference).
               Effective reasoning-sample fraction  R_eff = mix + (1 - mix) R(shift);   mix = 0 recovers logit_normal exactly.
"""
import glob
import json
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = np.linspace(0.0005, 0.9995, 4000)


def logit_normal_density(shift, s):
    z = np.log(s / (1.0 - s)) - math.log(shift)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / (s * (1.0 - s))


def load_probe_curve(arm, path="auto"):
    paths = [path] if path != "auto" else sorted(glob.glob(os.path.join(HERE, "eval", "result_gate4", "probe_noise*", "results.json")))
    for p in paths:
        d = json.load(open(p))
        if arm in d:
            sg = np.asarray(d[arm]["sigmas"], dtype=np.float64); acc = np.asarray([c["exact"] for c in d[arm]["curve"]], dtype=np.float64)
            o = np.argsort(sg)
            return sg[o], acc[o], p
    raise FileNotFoundError(f"no noise-probe curve for arm {arm!r} in {paths}")


class SigmaSampler:
    def __init__(self, shift, kind="logit_normal", mix=0.0, arm=None, probe="auto", device="cpu", fixed=None, two_point=None):
        self.shift, self.kind, self.mix = float(shift), kind, float(mix)
        self.meta = {"sigma_sampler": kind, "sigma_shift": float(shift)}
        p = logit_normal_density(self.shift, GRID); p = p / np.trapezoid(p, GRID)
        self.table = None; self.fixed = None; self.uniform = None; self.two = None
        if kind == "two_point":                   # exactly round(f*B) samples of every batch at sigma_high, the rest at sigma_low
            f, lo, hi = two_point; assert 0.0 < f <= 1.0 and 0.0 < lo < hi <= 1.0, two_point
            self.two = (float(f), float(lo), float(hi)); self.meta.update({"sigma_two_point": list(self.two)}); return
        if kind == "uniform":                     # sigma ~ U(lo, 1): every noise level trained equally (null-source copier -> A_DiT(sigma))
            self.uniform = (0.02, 1.0); self.meta["sigma_uniform"] = list(self.uniform); return
        if kind == "fixed":                       # single noise level for every sample (sigma = 1 -> the MSE regressor at DiT scale)
            assert fixed is not None and 0.0 < float(fixed) <= 1.0, fixed
            self.fixed = float(fixed); self.meta["sigma_fixed"] = self.fixed; return
        if kind == "logit_normal":
            return
        assert kind == "reason_mix", kind
        assert 0.0 <= self.mix <= 1.0, self.mix
        sg, acc, src = load_probe_curve(arm, probe)
        A = np.clip(np.interp(GRID, sg, acc), 0.0, 1.0)
        R = float(np.trapezoid(p * (1.0 - A), GRID))
        q = (1.0 - self.mix) * p + self.mix * p * (1.0 - A) / R
        cdf = np.cumsum(q); cdf = cdf / cdf[-1]
        self.table = (torch.tensor(cdf, dtype=torch.float64, device=device), torch.tensor(GRID, dtype=torch.float32, device=device))
        qs = {f"sigma_q{int(100*t)}": float(np.interp(t, cdf, GRID)) for t in (0.05, 0.25, 0.5, 0.75, 0.95)}
        self.meta.update({"sigma_reason_mix": self.mix, "R_base": R, "R_eff": self.mix + (1.0 - self.mix) * R,
                          "probe_src": os.path.relpath(src, HERE), "sigma_mean": float(np.trapezoid(q * GRID, GRID) / np.trapezoid(q, GRID)),
                          "mass_below_0p5": float(np.trapezoid(q[GRID < 0.5], GRID[GRID < 0.5]) / np.trapezoid(q, GRID)), **qs})

    def __call__(self, B, device):
        if self.fixed is not None:
            return torch.full((B,), self.fixed, device=device)
        if self.uniform is not None:
            lo, hi = self.uniform; return lo + (hi - lo) * torch.rand(B, device=device)
        if self.two is not None:
            f, lo, hi = self.two; n_hi = int(round(f * B))
            return torch.cat([torch.full((n_hi,), hi, device=device), torch.full((B - n_hi,), lo, device=device)])
        if self.table is None:
            return torch.sigmoid(torch.randn(B, device=device) + math.log(self.shift))
        cdf, grid = self.table
        u = torch.rand(B, device=device, dtype=torch.float64)
        idx = torch.searchsorted(cdf, u).clamp_(0, grid.numel() - 1)
        return grid[idx]

    def describe(self):
        m = self.meta
        if self.fixed is not None:
            return f"[sigma_sampler] fixed sigma = {self.fixed:g} for every sample"
        if self.uniform is not None:
            return f"[sigma_sampler] uniform sigma in [{self.uniform[0]:g}, {self.uniform[1]:g}]"
        if self.two is not None:
            return f"[sigma_sampler] two_point: {100*self.two[0]:.0f}% of each batch at sigma {self.two[2]:g}, the rest at sigma {self.two[1]:g}"
        if self.table is None:
            return f"[sigma_sampler] logit_normal shift {self.shift:g}"
        return (f"[sigma_sampler] reason_mix mix={m['sigma_reason_mix']:.2f} shift {self.shift:g}: R_base={m['R_base']:.3f} -> R_eff={m['R_eff']:.3f}; "
                f"sigma mean {m['sigma_mean']:.3f} q05/q50/q95 {m['sigma_q5']:.3f}/{m['sigma_q50']:.3f}/{m['sigma_q95']:.3f}; "
                f"mass below 0.5 = {m['mass_below_0p5']:.4f}; A(sigma) from {m['probe_src']}")
