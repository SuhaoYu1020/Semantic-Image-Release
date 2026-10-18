"""Three read-outs per sigma schedule (hard one-to-many), one panel per schedule:
  1. accuracy in the latent : standard-font linear reader on the one-step x0 (summary.jsonl probe_std, K = 1)
  2. glyph fidelity         : the rendered answer is a clean glyph in one of the valid styles, ANY digits (fidelity.jsonl fid, K = 24)
  3. fidelity AND correct   : clean glyph AND OCR reads the right answer (fidelity.jsonl fid_and_ocr, K = 24)
Phase 1 of a switch arm is taken from its parent run (s15 -> s15to1_*, reg -> regto1_*).  Seeds: mean line, min-max band.
  python om_three_lines.py [--space siglip2] [--k 24]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SPE = 1125; BUDGET = 11200
PARENT = {"s15to1": "s15", "regto1": "reg"}
MAIN = [("s15", 0, "shift 15 throughout\n(high noise only)"), ("s15to1_75", 8400, "shift 15 → shift 1\nat 75% of the budget"), ("s15to1_50", 5600, "shift 15 → shift 1\nat 50%"),
        ("s15to1_25", 2800, "shift 15 → shift 1\nat 25%"), ("s1", 0, "shift 1 throughout\n(low noise only)")]
EXTRA = [("reg", 0, "σ = 1 regression\nthroughout"), ("regto1_50", 5600, "regression → shift 1\nat 50%")]
C_ACC, C_FID, C_BOTH = "#5b6fb0", "#c0392b", "#0f6f78"


def rows(run, fname):
    p = os.path.join(HERE, "runs", run, "eval_om", fname); return [json.loads(l) for l in open(p)] if os.path.isfile(p) else []


def series(run, fname, key, K):
    rs = sorted([r for r in rows(run, fname) if r["K"] == K and r.get(key) is not None and r.get("step") is not None], key=lambda r: r["step"])
    out = {}
    for r in rs: out[r["step"]] = 100 * r[key]          # later rows (re-scores) overwrite earlier ones
    return out


def curve(arm, space, fname, key, K, off):
    seeds = [f"mm_om_{arm}_{space}"] + [os.path.basename(p) for p in sorted(glob.glob(os.path.join(HERE, "runs", f"mm_om_{arm}_{space}_s4*")))]
    cs = []
    for run in seeds:
        c = {st + off: v for st, v in series(run, fname, key, K).items()}
        if not c: continue
        if off:
            par = PARENT[arm.split("_")[0]]; suf = run[len(f"mm_om_{arm}_{space}"):]; prun = f"mm_om_{par}_{space}{suf}"
            if not rows(prun, fname): prun = f"mm_om_{par}_{space}"
            for st, v in series(prun, fname, key, K).items():
                if st <= off: c[st] = v
        cs.append(c)
    if not cs: return None
    steps = sorted(set(s for c in cs for s in c)); M = np.array([[c.get(s, np.nan) for s in steps] for c in cs])
    return np.array(steps) / SPE, np.nanmean(M, 0), np.nanmin(M, 0), np.nanmax(M, 0), len(cs)


def draw(ax, arm, off, title, space, K):
    hi_col = "#d9efd9" if arm.startswith("reg") else "#f6d5d5"
    if arm == "s1": ax.axvspan(0, BUDGET / SPE, color="#d6e4f5", lw=0)
    elif off == 0: ax.axvspan(0, BUDGET / SPE, color=hi_col, lw=0)
    else: ax.axvspan(0, off / SPE, color=hi_col, lw=0); ax.axvspan(off / SPE, BUDGET / SPE, color="#d6e4f5", lw=0); ax.axvline(off / SPE, color="k", lw=0.9, ls="--")
    ends, n = [], 1
    for fname, key, Kk, col, mk, lab in [("summary.jsonl", "probe_std", 1, C_ACC, "^", "1  accuracy: answer in the one-step latent (reader)"),
                                          ("fidelity.jsonl", "fid", K, C_FID, "s", f"2  glyph fidelity: clean glyph in a valid style, any digits ({K} steps)"),
                                          ("fidelity.jsonl", "fid_and_ocr", K, C_BOTH, "o", f"3  fidelity AND correct: clean glyph AND OCR reads the answer ({K} steps)")]:
        c = curve(arm, space, fname, key, Kk, off)
        if c is None: ends.append("–"); continue
        x, m, lo, hi, n = c; ax.plot(x, m, color=col, lw=2.2, marker=mk, ms=3.5, label=lab)
        if n > 1: ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ends.append(f"{m[-1]:.0f}")
    ax.set_title(title + f"\nend: {ends[0]} / {ends[1]} / {ends[2]} %" + (f"  (n={n} seeds)" if n > 1 else ""), fontsize=9.5)
    ax.set_xlim(0, BUDGET / SPE); ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.set_xlabel("epochs (budget = 10)", fontsize=8.5); ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--space", default="siglip2"); ap.add_argument("--k", type=int, default=24); a = ap.parse_args()
    tau = None
    for p in glob.glob(os.path.join(HERE, "runs", "mm_om_*", "eval_om", "fidelity.jsonl")):
        for l in open(p): tau = json.loads(l).get("tau"); break
        if tau: break
    for arms, fname in [(MAIN, f"fig_om_three_lines_{a.space}_k{a.k}.png"), (EXTRA, f"fig_om_three_lines_{a.space}_k{a.k}_regression.png")]:
        fig, axes = plt.subplots(1, len(arms), figsize=(max(3.5 * len(arms) + 0.5, 11.5), 4.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (arm, off, title) in zip(axes, arms): draw(ax, arm, off, title, a.space, a.k)
        axes[0].set_ylabel("% of 200 test questions"); h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
        sp = "SigLIP2" if a.space == "siglip2" else a.space.upper()
        fig.suptitle(f"Hard one-to-many ({sp}), one budget of 10 epochs. Background: red = shift 15 (high noise), green = σ = 1 regression, blue = shift 1 (low noise); dashed = switch.\n"
                     f"Fidelity = best normalised correlation of the answer box with any valid rendering (any answer, 8 fonts, ±30°) ≥ τ = {tau}", fontsize=9.5 if len(arms) > 2 else 8)
        fig.tight_layout(rect=(0, 0.08, 1, 0.97)); out = os.path.join(HERE, "eval", "result_gate4", "viz", fname); fig.savefig(out, dpi=150); print("wrote", out)


if __name__ == "__main__":
    main()
