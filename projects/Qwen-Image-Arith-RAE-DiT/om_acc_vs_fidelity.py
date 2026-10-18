"""Accuracy vs fidelity growth curves per sigma schedule (hard one-to-many, SigLIP2).
accuracy  = answer present in the one-step latent (standard-font linear reader, K = 1)
fidelity  = answer legible AND correct in the rendered glyph (OCR; K = 24 solid, K = 1 dotted)
shaded    = knows-but-cannot-render gap."""
import glob, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); SPE = 1125; BUDGET = 11200
ARMS = [("s1", 0, "shift 1 throughout (low noise)"), ("s15", 0, "shift 15 throughout (high noise)"), ("reg", 0, "sigma = 1 regression throughout"), None,
        ("s15to1_25", 2800, "shift 15 -> 1 at 25%"), ("s15to1_50", 5600, "shift 15 -> 1 at 50%"), ("s15to1_75", 8400, "shift 15 -> 1 at 75%"), ("regto1_50", 5600, "regression -> shift 1 at 50%")]
def rows(run):
    p = os.path.join(HERE, "runs", run, "eval_om", "summary.jsonl"); return [json.loads(l) for l in open(p)] if os.path.isfile(p) else []
PARENT = {"s15to1": "s15", "regto1": "reg"}
def curve(arm, space, key, K, off):
    seeds = [f"mm_om_{arm}_{space}"] + [os.path.basename(p) for p in sorted(glob.glob(os.path.join(HERE, "runs", f"mm_om_{arm}_{space}_s4*")))]
    cs = []
    for run in seeds:
        rs = sorted([r for r in rows(run) if r["K"] == K and r.get(key) is not None], key=lambda r: r["step"])
        if not rs: continue
        c = {r["step"] + off: 100 * r[key] for r in rs}
        if off:  # phase 1 was trained inside the parent run; show it up to the switch
            par = PARENT[arm.split("_")[0]]; suf = run[len(f"mm_om_{arm}_{space}"):]; prun = f"mm_om_{par}_{space}{suf}"
            if not rows(prun): prun = f"mm_om_{par}_{space}"
            for r in rows(prun):
                if r["K"] == K and r.get(key) is not None and r["step"] <= off: c[r["step"]] = 100 * r[key]
        cs.append(c)
    if not cs: return None
    steps = sorted(set(s for c in cs for s in c)); M = np.array([[c.get(s, np.nan) for s in steps] for c in cs])
    return np.array(steps) / SPE, np.nanmean(M, 0), np.nanmin(M, 0), np.nanmax(M, 0), len(cs)
for space, fname in [("siglip2", "fig_om_acc_vs_fidelity.png"), ("vae", "fig_om_acc_vs_fidelity_vae.png")]:
    fig, axes2 = plt.subplots(2, 4, figsize=(15.5, 8.2), sharey=True); axes = axes2.ravel()
    for ax, spec in zip(axes, ARMS):
        if spec is None:
            ax.axis("off"); h, l = axes[0].get_legend_handles_labels(); ax.legend(h, l, fontsize=9, loc="center", title="curves", title_fontsize=9)
            ax.text(0.5, 0.08, "switch arms: weights carried over from the\nparent run at the dashed line, optimizer restarted", ha="center", fontsize=8, color="#444", transform=ax.transAxes); continue
        arm, off, title = spec
        acc = curve(arm, space, "probe_std", 1, off); fid24 = curve(arm, space, "ocr_acc", 24, off); fid1 = curve(arm, space, "ocr_acc", 1, off)
        # phase background: red = high noise / regression, blue = low noise
        hi_col = "#f6d5d5" if not arm.startswith("reg") else "#d9efd9"
        if arm == "s1": ax.axvspan(0, BUDGET / SPE, color="#d6e4f5", lw=0)
        elif off == 0: ax.axvspan(0, BUDGET / SPE, color=hi_col, lw=0)
        else: ax.axvspan(0, off / SPE, color=hi_col, lw=0); ax.axvspan(off / SPE, BUDGET / SPE, color="#d6e4f5", lw=0); ax.axvline(off / SPE, color="k", lw=0.8, ls="--")
        n = 1
        if acc is not None:
            x, m, lo, hi, n = acc; ax.plot(x, m, color="#5b6fb0", lw=2.2, marker="^", ms=3.5, label="accuracy: answer in the latent (reader, 1 step)")
            if n > 1: ax.fill_between(x, lo, hi, color="#5b6fb0", alpha=0.15, lw=0)
        if fid24 is not None:
            x2, m2, lo2, hi2, n2 = fid24; ax.plot(x2, m2, color="#0f6f78", lw=2.2, marker="o", ms=3.5, label="fidelity: legible & correct glyph (OCR, 24 steps)")
            if n2 > 1: ax.fill_between(x2, lo2, hi2, color="#0f6f78", alpha=0.15, lw=0)
            if acc is not None:
                common = sorted(set(np.round(x, 4)) & set(np.round(x2, 4)))
                a = np.interp(common, x, m); b = np.interp(common, x2, m2); ax.fill_between(common, b, a, where=a > b, color="#e0a040", alpha=0.25, lw=0, label="gap: knows, cannot render yet")
        if fid1 is not None:
            x3, m3, *_ = fid1; ax.plot(x3, m3, color="#b2641a", lw=1.3, ls=":", marker="s", ms=2.5, label="fidelity: legible & correct glyph (OCR, 1 step)")
        fin = lambda c: f"{c[1][-1]:.0f}" if c is not None else "-"
        ax.set_title(title + f"\nend of budget: accuracy {fin(acc)}% / fidelity {fin(fid24)}%" + (f"  (n={n} seeds)" if n > 1 else ""), fontsize=9)
        ax.set_xlim(0, BUDGET / SPE); ax.set_ylim(-3, 103); ax.grid(alpha=0.3); ax.set_xlabel("epochs (budget = 10)", fontsize=8); ax.tick_params(labelsize=8)
    axes[0].set_ylabel("% of 200 test questions"); axes[4].set_ylabel("% of 200 test questions")
    fig.suptitle(f"Hard one-to-many ({space.upper() if space=='vae' else 'SigLIP2'}), one budget of 10 epochs: accuracy (answer in the latent) vs fidelity (legible, correct glyph)\n"
                 "background: red = high noise (shift 15), green = sigma = 1 regression, blue = low noise (shift 1); dashed line = switch", fontsize=10.5)
    fig.tight_layout(); out = os.path.join(HERE, "eval", "result_gate4", "viz", fname); fig.savefig(out, dpi=140); print("wrote", out)
