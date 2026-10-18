"""intervene.py -- causal test of "class separation d' paces the DiT": rewrite ONE latent so that only the answer-identity
structure of its answer-box tokens changes, train the same MMDiT on the rewritten latent, compare epochs-to-95.

Two interventions on the standardised tokens z (B, 256, D) of a base arm, both restricted to the 24 answer-box tokens
(flattened to a 24*D vector b) and both exactly invertible for decoding:

  lin  : b' = b + (alpha - 1) * U U^T (b - m)         U = orthonormal basis of the class-mean-difference directions of the
         three answer slots (fitted on clean train tokens).  A LINEAR map: between-class distances AND within-class spread
         along those directions both scale by alpha, so d' is unchanged while the answer subspace's share of the variance
         (and of the MSE) scales by alpha^2.  Inverse: b = b' + (1/alpha - 1) U U^T (b' - m).
  cls  : b' = b + (alpha - 1) * sum_s (m_{s,c_s(b)} - m)    c_s(b) = nearest class mean of slot s (read from b itself, so
         the map is label-free; nearest-centroid reads the clean answer at ~100%).  Class means move apart by alpha, the
         within-class spread does not: d' scales by alpha.  Inverse: read c_s from b' with the moved centroids, subtract.

Transform files: stats/intervene/<arm>.pt  {base, mode, alpha, box_idx, m, M (3,11,24D), U (k,24D), ...}
Build:  python intervene.py build --arm vae_cls2        (reads cache/probe/<base>_tgt_train3000_test1000.pt)
Check:  python intervene.py check --arm vae_cls2        (nearest-centroid accuracy, d' before/after, inverse error)
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

INTERVENE_DIR = os.path.join(HERE, "stats", "intervene")
PROBE_CACHE = os.path.join(HERE, "cache", "probe")
# arm name -> (base arm, mode, alpha)
INTERVENTIONS = {
    "vae_lin2": ("vae", "lin", 2.0),
    "vae_cls2": ("vae", "cls", 2.0),
    "vae_cls3": ("vae", "cls", 3.0),
    "siglip2_cls0p5": ("siglip2", "cls", 0.5),
    "mae_cls2": ("mae", "cls", 2.0),
    # 2026-10-12 whitening arms: per-token PCA whitening of ALL tokens (alpha = eps as a share of the mean eigenvalue); tests the
    # spectral-bias prediction of kernel_analysis.py (which eigen-directions of the token space carry the answer)
    "vae_whiten": ("vae", "whiten", 1e-3),
    "siglip2_whiten": ("siglip2", "whiten", 1e-3),
    "mae_whiten": ("mae", "whiten", 1e-3),
    "webssl_whiten": ("webssl", "whiten", 1e-3),
    "emu35_whiten": ("emu35", "whiten", 1e-3),
}
BLANK = 10


def answer_slots(ans):
    s = str(ans)
    return [int(s[i]) if i < len(s) else BLANK for i in range(3)]


def box_index(data_root, grid):
    b = json.load(open(os.path.join(data_root, "_answer_token_box.json")))
    assert list(b["grid"]) == list(grid), (b["grid"], grid)
    return torch.tensor([r * grid[1] + c for r in range(b["rows"][0], b["rows"][1] + 1) for c in range(b["cols"][0], b["cols"][1] + 1)])


class Intervention:
    """Holds the fitted transform; apply()/invert() act on standardised token tensors (B, N, D)."""

    def __init__(self, blob, device="cpu"):
        self.mode, self.alpha = blob["mode"], float(blob["alpha"])
        self.device = device
        if self.mode == "whiten":                                   # acts on every token: z' = (z - mu) W,  z = z' Winv + mu
            self.mu = blob["mu"].float().to(device); self.W = blob["W"].float().to(device); self.Winv = blob["Winv"].float().to(device)
            self.U = self.W; self.idx = blob["box_idx"].to(device)
            return
        self.idx = blob["box_idx"].to(device)
        self.m = blob["m"].float().to(device)                 # (24D,)
        self.M = blob["M"].float().to(device)                 # (3, 11, 24D) class means per slot (nan where class absent)
        self.U = blob["U"].float().to(device)                 # (k, 24D)
        self.valid = ~torch.isnan(self.M[:, :, 0])            # (3, 11)
        self.Mf = torch.nan_to_num(self.M)
        self.probe_Xtr = blob["probe_Xtr"].float().to(device)  # (n, 24D) centred train tokens (dual ridge support)
        self.probe_alpha = blob["probe_alpha"].float().to(device)  # (n, 33)
        self.probe_mu = blob["probe_mu"].float().to(device); self.probe_ym = blob["probe_ym"].float().to(device)
        self.device = device

    def to(self, device):
        if device != self.device:
            keys = ("idx", "mu", "W", "Winv", "U") if self.mode == "whiten" else ("idx", "m", "M", "U", "valid", "Mf", "probe_Xtr", "probe_alpha", "probe_mu", "probe_ym")
            for k in keys:
                setattr(self, k, getattr(self, k).to(device))
            self.device = device
        return self

    def _classes(self, b, means):
        """answer classes per slot read by the ridge probe fitted on clean train tokens (99.9% exact at sigma 0).
        Moving the class means by alpha only shifts each sample along its own class-mean offset, which raises the correct
        class logit, so the same probe reads clean and moved tokens; `means` is kept for the signature."""
        logits = ((b - self.probe_mu) @ self.probe_Xtr.t()) @ self.probe_alpha + self.probe_ym       # (B, 33)
        return logits.view(-1, 3, 11).argmax(-1)

    def _offset(self, b, means, classes):
        off = torch.zeros_like(b)
        for s in range(3):
            off += means[s][classes[:, s]] - self.m
        return off

    @torch.no_grad()
    def apply(self, z):
        if self.mode == "whiten":
            return ((z.float() - self.mu) @ self.W).to(z.dtype)
        B, N, D = z.shape
        b = z[:, self.idx].reshape(B, -1).float()
        if self.mode == "lin":
            b2 = b + (self.alpha - 1) * ((b - self.m) @ self.U.t()) @ self.U
        else:
            c = self._classes(b, self.Mf)
            b2 = b + (self.alpha - 1) * self._offset(b, self.Mf, c)
        z2 = z.clone(); z2[:, self.idx] = b2.view(B, len(self.idx), D).to(z.dtype)
        return z2

    @torch.no_grad()
    def invert(self, z2):
        if self.mode == "whiten":
            return (z2.float() @ self.Winv + self.mu).to(z2.dtype)
        B, N, D = z2.shape
        b2 = z2[:, self.idx].reshape(B, -1).float()
        if self.mode == "lin":
            b = b2 + (1.0 / self.alpha - 1) * ((b2 - self.m) @ self.U.t()) @ self.U
        else:
            moved = self.m + self.alpha * (self.Mf - self.m)   # class means after the move (3, 11, 24D)
            moved = torch.where(self.valid[..., None], moved, torch.full_like(moved, float("nan")))
            c = self._classes(b2, torch.nan_to_num(moved))
            b = b2 - (self.alpha - 1) * self._offset(b2, self.Mf, c)
        z = z2.clone(); z[:, self.idx] = b.view(B, len(self.idx), D).to(z2.dtype)
        return z

    def read_classes(self, z, moved=False):
        B = z.shape[0]; b = z[:, self.idx].reshape(B, -1).float()
        means = self.Mf if not moved else torch.nan_to_num(self.m + self.alpha * (self.Mf - self.m))
        return self._classes(b, means)


def load_intervention(arm, device="cpu"):
    blob = torch.load(os.path.join(INTERVENE_DIR, f"{arm}.pt"), map_location="cpu")
    return Intervention(blob, device)


def build(arm, data_root):
    base, mode, alpha = INTERVENTIONS[arm]
    cache = [f for f in os.listdir(PROBE_CACHE) if f.startswith(f"{base}_tgt_train")]
    assert cache, f"no probe token cache for {base} in {PROBE_CACHE}"
    blob = torch.load(os.path.join(PROBE_CACHE, sorted(cache)[-1]))
    grid = blob["grid"]; idx = box_index(data_root, grid)
    if mode == "whiten":
        X = blob["train"]["z"].reshape(-1, blob["train"]["z"].shape[-1]).float()      # pooled standardised tokens (n*256, D)
        mu = X.mean(0); C = torch.cov((X - mu).t()); evals, V = torch.linalg.eigh(C); eps = alpha * float(evals.mean())
        W = V @ torch.diag((evals.clamp_min(0) + eps).rsqrt()) @ V.t(); Winv = V @ torch.diag((evals.clamp_min(0) + eps).sqrt()) @ V.t()
        out = {"arm": arm, "base": base, "mode": mode, "alpha": alpha, "box_idx": idx, "mu": mu, "W": W, "Winv": Winv, "evals": evals.flip(0),
               "eps": eps, "latent": blob["latent"], "n_fit_tokens": int(X.shape[0]), "grid": grid, "k": int(W.shape[0])}
        os.makedirs(INTERVENE_DIR, exist_ok=True); torch.save(out, os.path.join(INTERVENE_DIR, f"{arm}.pt"))
        ev = evals.flip(0); print(f"[build] {arm}: PCA-whitening of {base} tokens (D={W.shape[0]}), eigenvalue share top1 {float(ev[0]/ev.sum()):.3f}, "
                                  f"top10 {float(ev[:10].sum()/ev.sum()):.3f}, min/max {float(ev.min()):.2e}/{float(ev.max()):.2e}, eps {eps:.2e}; fit on {X.shape[0]} tokens -> stats/intervene/{arm}.pt")
        return out
    Zh = blob["train"]["z"]; slots = torch.tensor([answer_slots(s) for s in blob["train"]["answer"]])
    b = Zh[:, idx].reshape(Zh.shape[0], -1).float(); del Zh
    m = b.mean(0)
    # ridge probe (dual form, linear kernel) on the clean box tokens: logits = (b - mu) Xtr^T alpha + ym
    Y = torch.nn.functional.one_hot(slots, 11).float().view(b.shape[0], -1); ym = Y.mean(0, keepdim=True)
    Xc = b - m; K = Xc @ Xc.t(); K.diagonal().add_(1e-2 * float(K.diagonal().mean()))
    alpha_probe = torch.linalg.solve(K, Y - ym)
    probe_acc = float(((((Xc @ Xc.t()) @ alpha_probe + ym).view(-1, 3, 11).argmax(-1)) == slots).all(1).float().mean())
    M = torch.full((3, 11, b.shape[1]), float("nan"))
    for s in range(3):
        for c in range(11):
            sel = slots[:, s] == c
            if sel.sum() >= 5:
                M[s, c] = b[sel].mean(0)
    diffs = torch.cat([(M[s][~torch.isnan(M[s, :, 0])] - m) for s in range(3)])        # all class-mean offsets
    Uu, S, Vh = torch.linalg.svd(diffs, full_matrices=False)
    k = int((S > 1e-3 * S[0]).sum())
    U = Vh[:k]                                                                          # orthonormal rows (k, 24D)
    out = {"arm": arm, "base": base, "mode": mode, "alpha": alpha, "box_idx": idx, "m": m, "M": M, "U": U, "k": k,
           "probe_Xtr": Xc.half(), "probe_alpha": alpha_probe, "probe_mu": m, "probe_ym": ym, "probe_train_exact": probe_acc,
           "latent": blob["latent"], "n_fit": int(b.shape[0]), "grid": grid, "singular_values": S[:k].tolist()}
    os.makedirs(INTERVENE_DIR, exist_ok=True)
    torch.save(out, os.path.join(INTERVENE_DIR, f"{arm}.pt"))
    print(f"[build] {arm}: base={base} mode={mode} alpha={alpha} k={k} dims (24D={b.shape[1]}), probe train exact {100*probe_acc:.1f}%, fit on {b.shape[0]} train rows -> {INTERVENE_DIR}/{arm}.pt")
    return out


def dprime(b, first_digit):
    labels = [c for c in range(10) if int((first_digit == c).sum()) >= 5]
    means = {c: b[first_digit == c].mean(0) for c in labels}
    pair, within = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            ci, cj = labels[i], labels[j]
            u = means[ci] - means[cj]; pair.append(float(u.norm())); u = u / (u.norm() + 1e-8)
            proj = torch.cat([(b[first_digit == ci] - means[ci]) @ u, (b[first_digit == cj] - means[cj]) @ u]); within.append(float(proj.std()))
    return float(np.mean(pair)) / float(np.mean(within)), float(np.mean(pair)), float(np.mean(within))


def check(arm, data_root):
    base, mode, alpha = INTERVENTIONS[arm]
    if mode == "whiten":
        it = load_intervention(arm); cache = [f for f in os.listdir(PROBE_CACHE) if f.startswith(f"{base}_tgt_train")]
        z = torch.load(os.path.join(PROBE_CACHE, sorted(cache)[-1]))["test"]["z"][:200].float()
        z2 = it.apply(z); back = it.invert(z2)
        print(f"[check] {arm}: whitened test tokens mean {float(z2.mean()):+.3f} std {float(z2.std()):.3f} (per-dim std min/max {float(z2.reshape(-1, z2.shape[-1]).std(0).min()):.2f}/{float(z2.reshape(-1, z2.shape[-1]).std(0).max()):.2f}); "
              f"round-trip max |err| {float((back - z).abs().max()):.2e}")
        return
    iv = load_intervention(arm)
    cache = sorted(f for f in os.listdir(PROBE_CACHE) if f.startswith(f"{base}_tgt_train"))[-1]
    blob = torch.load(os.path.join(PROBE_CACHE, cache))
    for split in ("test",):                                   # test only: 1000 rows keeps the full-token tensors small
        Z = blob[split]["z"].float(); slots = torch.tensor([answer_slots(s) for s in blob[split]["answer"]])
        c0 = iv.read_classes(Z)
        Z2 = iv.apply(Z); c2 = iv.read_classes(Z2, moved=True)
        Zb = iv.invert(Z2)
        b, b2 = Z[:, iv.idx].reshape(Z.shape[0], -1), Z2[:, iv.idx].reshape(Z.shape[0], -1)
        d0, p0, w0 = dprime(b, slots[:, 0]); d2, p2, w2 = dprime(b2, slots[:, 0])
        print(f"[check] {arm} {split}: nearest-centroid exact answer clean {100*float((c0 == slots).all(1).float().mean()):.1f}% | moved {100*float((c2 == slots).all(1).float().mean()):.1f}% | "
              f"inverse max abs err {float((Zb - Z).abs().max()):.2e} | d' {d0:.2f} -> {d2:.2f} (pair dist {p0:.1f} -> {p2:.1f}, within {w0:.2f} -> {w2:.2f}) | "
              f"box var share of answer subspace {float(((b - iv.m) @ iv.U.t()).pow(2).sum(1).mean() / (b - iv.m).pow(2).sum(1).mean()):.3f} -> "
              f"{float(((b2 - iv.m) @ iv.U.t()).pow(2).sum(1).mean() / (b2 - iv.m).pow(2).sum(1).mean()):.3f} | global token std {float(Z.std()):.3f} -> {float(Z2.std()):.3f}")


if __name__ == "__main__":
    from sq_codecs import DEFAULT_DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--arm", required=True, choices=sorted(INTERVENTIONS))
    ap.add_argument("--data", default=DEFAULT_DATA)
    a = ap.parse_args()
    (build if a.cmd == "build" else check)(a.arm, a.data)
