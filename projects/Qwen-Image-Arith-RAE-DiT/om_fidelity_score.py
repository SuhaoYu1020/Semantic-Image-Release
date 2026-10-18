"""om_fidelity_score.py -- glyph fidelity for the hard one-to-many renders, INDEPENDENT of correctness.

F(image) = max over a bank of ALL valid answer renderings (199 answer strings x 8 answer fonts x 13 rotations, drawn exactly as
make_square_gate_set.py draws the hard variants) of the normalised cross-correlation (cv2 TM_CCOEFF_NORMED) between the bank glyph
and the answer-box crop of the output.  A committed, sharp glyph in one of the valid styles scores ~1; the pixel mean of several
styles, a ghost of two styles or a smear scores low.  legible := F >= TAU.  The arg-max also gives a template OCR read
(string) and the style the model committed to (font, angle).

  python om_fidelity_score.py calib                       -> eval/result_gate4/fidelity/calib.json + fig (GT variants, pixel means, preds)
  python om_fidelity_score.py score [--runs mm_om_*] [--tau T]  -> <evaldir>/fidelity.json per eval dir + runs/<run>/eval_om/fidelity.jsonl
Needs cv2: run with the eval-ocr-b200 env.  CPU only; uses a multiprocessing Pool (PROCS env, default 32).
"""
import glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
SI_ROOT = os.environ.get("SI_ROOT") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SI_DATA = os.environ.get("SI_DATA") or os.path.join(SI_ROOT, "data")
STD = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same")
HARD = [os.path.join(SI_DATA, f"arithmetic_add_unordered_sq256_same_ansfont_hard_k{k}") for k in range(8)]
CROP = (44, 123, 160, 180)                     # answer digits only: right of "A: " (digits start at x~59, jitter -10), below the question line (y<=122)
ANGLES = list(range(-30, 31, 5)); STRINGS = [str(s) for s in range(0, 199)]
OUT_ROOT = os.path.join(HERE, "eval", "result_gate4", "fidelity")
TAU_DEFAULT = float(os.environ.get("FID_TAU", "0"))   # 0 -> read from calib.json
TOPK = 48


def fonts():
    """The 8 answer fonts of the hard variants, as recorded in k0's stats.json (absolute paths of the rendering machine).
    A path that does not exist here is looked up by file name under $SI_DATA/fonts_extra and the system font dirs."""
    st = json.load(open(os.path.join(HARD[0], "stats.json"))); out = []
    for p, sz in st["answer_fonts"]:
        if not os.path.isfile(p):
            cands = [os.path.join(SI_DATA, "fonts_extra", os.path.basename(p))] + [os.path.join(d, os.path.basename(p)) for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation", "/usr/share/fonts/truetype")]
            p = next((c for c in cands if os.path.isfile(c)), p)
        out.append((p, int(sz)))
    return out


def glyph_layer(digits, afont, ang):
    """Exactly make_square_gate_set.render(): text on a padded L layer, rotate with expand, crop to ink bbox."""
    tb = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), digits, font=afont); tw, th = tb[2] - tb[0], tb[3] - tb[1]
    layer = Image.new("L", (tw + 48, th + 48), 0); ImageDraw.Draw(layer).text((24 - tb[0], 24 - tb[1]), digits, fill=255, font=afont)
    layer = layer.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=0)
    lb = layer.getbbox() or (0, 0, layer.width, layer.height); return np.asarray(layer.crop(lb), dtype=np.float32) / 255.0


_BANK = None
def bank():
    global _BANK
    if _BANK is None:
        meta, full, half = [], [], []
        for fi, (fp, fs) in enumerate(fonts()):
            af = ImageFont.truetype(fp, fs)
            for s in STRINGS:
                for ang in ANGLES:
                    t = glyph_layer(s, af, ang)
                    if t.shape[0] > CROP[3] - CROP[1] or t.shape[1] > CROP[2] - CROP[0]: continue
                    t8 = (t * 255).astype(np.uint8); meta.append((s, fi, ang)); full.append(t8); half.append(cv2.resize(t8, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        _BANK = (meta, full, half)
    return _BANK


def ink_crop(path):
    g = np.asarray(Image.open(path).convert("L").crop(CROP), dtype=np.float32) / 255.0; return 1.0 - g


def _integral(img8):
    return cv2.integral(img8.astype(np.float32))


def _place(img8, S, t8):
    """NCC max location of template t8 in img8 (uint8: cv2's float path is numerically broken for TM_CCOEFF_NORMED),
    plus the share of the window's ink that lies OUTSIDE the placed template box (a tiny template matching one digit of a
    two-digit output, or a smear wider than any glyph, leaves ink unexplained)."""
    r = cv2.matchTemplate(img8, t8, cv2.TM_CCOEFF_NORMED); y, x = np.unravel_index(int(r.argmax()), r.shape); ncc = float(r[y, x])
    th, tw = t8.shape; tot = S[-1, -1]
    inside = S[y + th, x + tw] - S[y, x + tw] - S[y + th, x] + S[y, x]
    outside = float(1.0 - inside / tot) if tot > 0 else 1.0
    return max(ncc, 0.0), outside


_K3 = np.ones((3, 3), np.uint8); SCALES = (0.85, 1.0, 1.15)
def _variants(t8):
    for sc in SCALES:
        ts = t8 if sc == 1.0 else cv2.resize(t8, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        yield ts; yield cv2.dilate(ts, _K3); yield cv2.erode(ts, _K3)


def score_ink(ink):
    """F = max over the bank of NCC x (1 - outside): coarse (half-res, every template) -> fine (full-res, top-K).
    Returns (F, string, font_idx, angle, ncc, outside)."""
    meta, full, half = bank()
    img8 = (np.clip(ink, 0, 1) * 255).astype(np.uint8); small = cv2.resize(img8, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    S_full, S_small = _integral(img8), _integral(small)
    if S_full[-1, -1] < 255 * 10: return 0.0, "", None, None, 0.0, 1.0     # (almost) no ink at all
    coarse = np.full(len(half), -1.0, dtype=np.float32)
    for i, t in enumerate(half):
        if t.shape[0] > small.shape[0] or t.shape[1] > small.shape[1]: continue
        ncc, out = _place(small, S_small, t); coarse[i] = ncc * (1.0 - out)
    best = (-1.0, None, 0.0, 1.0)
    for i in np.argsort(coarse)[-TOPK:]:
        if coarse[i] < 0: continue
        ncc, out = _place(img8, S_full, full[i]); v = ncc * (1.0 - out)
        if v > best[0]: best = (v, i, ncc, out)
    if best[1] is None: return 0.0, "", None, None, 0.0, 1.0
    s, fi, ang = meta[best[1]]; return best[0], s, fi, ang, best[2], best[3]


def score_path(p):
    try: return (p,) + score_ink(ink_crop(p))
    except Exception as e: return (p, float("nan"), None, None, None, float("nan"), float("nan"))


def score_arrays(name_arr):
    name, arr = name_arr; return (name,) + score_ink(arr)


def test_rows(eval_dir):
    rep = json.load(open(os.path.join(eval_dir, "report.json"))); return rep["test_rows"]


def ocr_items(eval_dir):
    out = {}
    od = os.path.join(eval_dir, "ocr")
    if not os.path.isdir(od): return out
    for fn in os.listdir(od):
        if fn.endswith(".crop.ocr.json"):
            d = json.load(open(os.path.join(od, fn)))
            for sp in d.get("splits", {}).values():
                for r in sp.get("rows", []): out[r["base"]] = (bool(r.get("ok")), r.get("ocr_pred") or r.get("ocr_raw"))
    return out


def calib(procs):
    os.makedirs(OUT_ROOT, exist_ok=True)
    ref = os.path.join(HERE, "runs", "mm_om_s15_siglip2", "eval_om", "final_k24"); rows = test_rows(ref)
    ids = [r["id"] for r in rows]
    groups = {}
    groups["gt_std_raw"] = [os.path.join(STD, "test", "targets", f"{i:06d}.png") for i in ids]
    for k in range(8): groups[f"gt_var_k{k}_raw"] = [os.path.join(HARD[k], "test", "targets", f"{i:06d}.png") for i in ids]
    groups["gt_codec_eval"] = [os.path.join(ref, "imgs", f"{r['base']}_gt.png") for r in rows]
    # codec round-trips (om_fidelity_codec_gt.py): the sharpest output a perfect model could produce, and the mean failure modes
    for arm in ("siglip2", "vae"):
        cg = os.path.join(OUT_ROOT, "codec_gt", arm)
        if not os.path.isdir(cg): continue
        for k in range(8): groups[f"gt_var_k{k}_codec_{arm}"] = [os.path.join(cg, f"k{k}", f"{i:06d}.png") for i in ids]
        groups[f"gt_std_codec_{arm}"] = [os.path.join(cg, "std", f"{i:06d}.png") for i in ids]
        for neg in ("mean8", "mean2", "latmean8"): groups[f"neg_{neg}_codec_{arm}"] = [os.path.join(cg, neg, f"{i:06d}.png") for i in ids]
    for run, tag in [("mm_om_s15_siglip2", "step_800_k24"), ("mm_om_s15_siglip2", "final_k24"), ("mm_om_s15_siglip2", "final_k1"), ("mm_om_s15to1_75_siglip2", "final_k24"),
                     ("mm_om_reg_siglip2", "final_k24"), ("mm_om_s1_siglip2", "final_k24"), ("mm_om_s15_vae", "final_k24")]:
        d = os.path.join(HERE, "runs", run, "eval_om", tag)
        if os.path.isdir(d): groups[f"pred_{run}_{tag}"] = [os.path.join(d, "imgs", f"{r['base']}_pred.png") for r in test_rows(d)]
    # synthetic negatives from the raw variants: pixel mean of 8, mean of 2 (ghost), mean of 4
    arrays = []
    rng = np.random.RandomState(0)
    for i in ids:
        v = [ink_crop(os.path.join(HARD[k], "test", "targets", f"{i:06d}.png")) for k in range(8)]
        arrays.append((f"neg_mean8|{i}", np.mean(v, 0)))
        a, b = rng.choice(8, 2, replace=False); arrays.append((f"neg_mean2|{i}", (v[a] + v[b]) / 2))
        q = rng.choice(8, 4, replace=False); arrays.append((f"neg_mean4|{i}", np.mean([v[j] for j in q], 0)))
    t0 = time.time(); res = {}
    with Pool(procs) as pool:
        for g, paths in groups.items():
            out = pool.map(score_path, paths, chunksize=4); res[g] = [{"path": p, "F": F, "read": s, "font": fi, "ang": a, "ncc": nc, "outside": ou} for p, F, s, fi, a, nc, ou in out]
            print(f"[calib] {g:40s} n={len(out)} F mean {np.nanmean([o[1] for o in out]):.3f}  p1 {np.nanpercentile([o[1] for o in out], 1):.3f}  p99 {np.nanpercentile([o[1] for o in out], 99):.3f}  ncc {np.nanmean([o[5] for o in out]):.3f} outside {np.nanmean([o[6] for o in out]):.3f}  ({time.time()-t0:.0f}s)", flush=True)
        out = pool.map(score_arrays, arrays, chunksize=4)
    for name, F, s, fi, a, nc, ou in out:
        g = name.split("|")[0]; res.setdefault(g, []).append({"path": name, "F": F, "read": s, "font": fi, "ang": a, "ncc": nc, "outside": ou})
    for g in ("neg_mean8", "neg_mean2", "neg_mean4"):
        Fs = [o["F"] for o in res[g]]; print(f"[calib] {g:40s} n={len(Fs)} F mean {np.mean(Fs):.3f}  p1 {np.percentile(Fs, 1):.3f}  p99 {np.percentile(Fs, 99):.3f}", flush=True)
    # tau: midpoint between the 1st percentile of the CODEC-decoded GT variants (both arms) and the 99th percentile of the codec-decoded
    # mean negatives (mean8 / mean4-raw / latmean8); falls back to the raw groups when no codec round-trips exist
    posg = [g for g in res if g.startswith("gt_var") and "_codec_" in g] or [g for g in res if g.startswith("gt_var")]
    negg = [g for g in res if g in ("neg_mean8_codec_siglip2", "neg_mean8_codec_vae", "neg_latmean8_codec_vae")] or ["neg_mean8"]
    lo_pos = min(float(np.percentile([o["F"] for o in res[g]], 1)) for g in posg); hi_neg = max(float(np.percentile([o["F"] for o in res[g]], 99)) for g in negg)
    tau = round((lo_pos + hi_neg) / 2, 3)
    per = {g: (float(np.percentile([o["F"] for o in v], 1)), float(np.percentile([o["F"] for o in v], 99))) for g, v in res.items()}
    print("[calib] per-group (p1, p99):", json.dumps({g: [round(x, 3) for x in p] for g, p in per.items()}, indent=0), flush=True)
    summ = {"tau": tau, "pos_p1": lo_pos, "neg_p99": hi_neg, "groups": {g: {"n": len(v), "F_mean": float(np.nanmean([o["F"] for o in v])), "legible_at_tau": float(np.mean([o["F"] >= tau for o in v])),
            "tmpl_read_correct": float(np.mean([o["read"] == str(rid) for o, rid in zip(v, ids)])) if len(v) == len(ids) else None} for g, v in res.items()}}
    # template-OCR correctness needs the answers, not ids
    ans = {r["id"]: r["gt"] for r in rows}
    for g, v in res.items():
        if len(v) == len(ids): summ["groups"][g]["tmpl_read_correct"] = float(np.mean([o["read"] == ans[i] for o, i in zip(v, ids)]))
    json.dump({"summary": summ, "items": res}, open(os.path.join(OUT_ROOT, "calib.json"), "w"))
    print(json.dumps(summ, indent=1), flush=True)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.2)); bins = np.linspace(0, 1, 51)
        for g, col, ls in [("gt_var_k0_raw", "#2a9d8f", ":"), ("gt_var_k0_codec_siglip2", "#2a9d8f", "-"), ("gt_var_k0_codec_vae", "#1d6f66", "--"), ("gt_std_codec_siglip2", "#264653", "-"),
                           ("neg_mean8_codec_siglip2", "#e76f51", "-"), ("neg_latmean8_codec_siglip2", "#b5362f", "-"), ("neg_mean2_codec_siglip2", "#f4a261", "-"), ("neg_mean8_codec_vae", "#e76f51", "--"),
                           ("pred_mm_om_s15_siglip2_step_800_k24", "#9d4edd", "-"), ("pred_mm_om_s15_siglip2_final_k24", "#3a86ff", "-"), ("pred_mm_om_reg_siglip2_final_k24", "#6c757d", "-"), ("pred_mm_om_s15_vae_final_k24", "#3a86ff", "--")]:
            if g in res: ax.hist([o["F"] for o in res[g]], bins=bins, histtype="step", lw=1.8, color=col, ls=ls, label=g)
        ax.axvline(tau, color="k", ls="--", lw=1); ax.text(tau, ax.get_ylim()[1] * 0.95, f" tau = {tau}", fontsize=9)
        ax.set_xlabel("F = best normalised correlation with any valid rendering (any answer, any font, any angle)"); ax.set_ylabel("count (200 test questions)"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
        ax.set_title("Glyph-fidelity calibration: raw GT variants vs pixel means of 2/4/8 variants vs model outputs", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(OUT_ROOT, "fig_fidelity_calib.png"), dpi=140); print("wrote fig", flush=True)
    except Exception as e: print("fig failed", e)


def score_runs(runs, tau, procs, force=False):
    jobs = []
    for run in runs:
        root = os.path.join(HERE, "runs", run, "eval_om")
        if not os.path.isdir(root): continue
        for d in sorted(glob.glob(os.path.join(root, "*_k*"))):
            if os.path.isfile(os.path.join(d, "report.json")) and (force or not os.path.isfile(os.path.join(d, "fidelity.json"))): jobs.append((run, d))
    print(f"[score] {len(jobs)} eval dirs, tau {tau}, procs {procs}", flush=True); t0 = time.time()
    with Pool(procs) as pool:
        for j, (run, d) in enumerate(jobs):
            rows = test_rows(d); paths = [os.path.join(d, "imgs", f"{r['base']}_pred.png") for r in rows]
            out = pool.map(score_path, paths, chunksize=4); oc = ocr_items(d)
            items = []
            for r, (p, F, s, fi, a, nc, ou) in zip(rows, out):
                ok = oc.get(r["base"], (False, None))
                items.append({"base": r["base"], "id": r["id"], "gt": r["gt"], "F": F, "ncc": nc, "outside": ou, "legible": bool(F >= tau), "tmpl_read": s, "font": fi, "ang": a, "ocr_ok": ok[0], "ocr_read": ok[1]})
            tag = os.path.basename(d); K = int(tag.rsplit("_k", 1)[1]); meta = os.path.join(HERE, "runs", run, tag.rsplit("_k", 1)[0], "meta.json")
            step = json.load(open(meta))["global_step"] if os.path.isfile(meta) else None
            n = len(items); rec = {"run": run, "tag": tag, "step": step, "K": K, "tau": tau, "n": n, "F_mean": float(np.nanmean([i["F"] for i in items])),
                   "fid": sum(i["legible"] for i in items) / n, "fid_and_ocr": sum(i["legible"] and i["ocr_ok"] for i in items) / n, "ocr": sum(i["ocr_ok"] for i in items) / n,
                   "tmpl_correct": sum(i["tmpl_read"] == i["gt"] for i in items) / n, "fid_and_tmpl": sum(i["legible"] and i["tmpl_read"] == i["gt"] for i in items) / n,
                   "font_hist": np.bincount([i["font"] for i in items if i["font"] is not None], minlength=8).tolist()}
            json.dump({"summary": rec, "items": items}, open(os.path.join(d, "fidelity.json"), "w"))
            with open(os.path.join(HERE, "runs", run, "eval_om", "fidelity.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
            print(f"[score] {j+1}/{len(jobs)} {run} {tag}: fid {100*rec['fid']:.1f}%  fid&ocr {100*rec['fid_and_ocr']:.1f}%  ocr {100*rec['ocr']:.1f}%  tmpl {100*rec['tmpl_correct']:.1f}%  F {rec['F_mean']:.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["calib", "score", "bench"]); ap.add_argument("--runs", nargs="*", default=None); ap.add_argument("--tau", type=float, default=TAU_DEFAULT)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("PROCS", "32"))); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    t0 = time.time(); meta, full, half = bank(); print(f"[bank] {len(meta)} templates in {time.time()-t0:.0f}s", flush=True)
    if a.mode == "bench":
        p = os.path.join(HERE, "runs", "mm_om_s15_siglip2", "eval_om", "final_k24", "imgs"); ps = sorted(glob.glob(os.path.join(p, "*_pred.png")))[:5]
        t0 = time.time(); [print(score_path(x)) for x in ps]; print(f"{(time.time()-t0)/len(ps):.2f} s/image single core")
    elif a.mode == "calib": calib(a.procs)
    else:
        tau = a.tau or json.load(open(os.path.join(OUT_ROOT, "calib.json")))["summary"]["tau"]
        runs = a.runs or sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "runs", "mm_om_*")))
        if a.force:
            for r in runs:
                fj = os.path.join(HERE, "runs", r, "eval_om", "fidelity.jsonl")
                if os.path.isfile(fj): os.remove(fj)
        score_runs(runs, tau, a.procs, a.force)
