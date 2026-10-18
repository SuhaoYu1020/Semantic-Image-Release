"""One-to-many schedule curriculum evaluation (hard anshard set).  For each snapshot of a run and each K in --ks:
  render_eval (K steps, --save_latents) -> OCR crop accuracy (answer written correctly), latent answer-reader accuracy
  (std + hard readers: representation holds the answer), fidelity on the answer region (RMSE to the nearest of the 8 hard
  variants / to their pixel mean, ratio < 1 = committed to one mode; RMSE to the standard-font target), latent RMSE
  outside / inside the answer box vs the standard target.  Appends rows to runs/<run>/eval_om/summary.jsonl (idempotent)."""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from PIL import Image
from sq_codecs import SqCodec, DEFAULT_DATA, to_pix01, SI_DATA
from intervene import box_index
DEVICE = "cuda"; PY = sys.executable
CONDA_ROOT = os.environ.get("CONDA_ROOT", os.path.expanduser("~/miniconda3")); SI_REPO = os.environ.get("SI_REPO", os.path.dirname(os.path.dirname(HERE)))
PY_OCR = os.path.join(CONDA_ROOT, "envs", "eval-ocr-b200", "bin", "python"); SCORER = os.path.join(SI_REPO, "projects", "Qwen-Image-Arith-ScratchTE", "score_render_ocr_crop.py")
OCR_MODEL = os.environ.get("SI_OCR_MODEL"); CROP = (34, 86, 160, 180)   # answer-box pixel box [58,110,136,156] padded 24 px (rotated / jittered variants spill over)

def gray(p, crop=CROP):
    return np.asarray(Image.open(p).convert("L").crop(crop), dtype=np.float32) / 255.0

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--snapshots", nargs="*", default=None)
    ap.add_argument("--every", type=int, default=800); ap.add_argument("--n_test", type=int, default=200); ap.add_argument("--ks", nargs="+", type=int, default=[1, 24])
    ap.add_argument("--probes", nargs="+", default=["std", "hard"]); a = ap.parse_args()
    rdir = os.path.join(HERE, "runs", a.run); ra = json.load(open(os.path.join(rdir, "args.json"))); arm = ra["arm"]; data = ra["data_root"]
    tags = a.snapshots or sorted([d for d in os.listdir(rdir) if d.startswith("step_") and int(d.split("_")[1]) % a.every == 0], key=lambda t: int(t.split("_")[1]))
    if os.path.isdir(os.path.join(rdir, "final")) and "final" not in tags: tags.append("final")
    out_root = os.path.join(rdir, "eval_om"); os.makedirs(out_root, exist_ok=True); summ = os.path.join(out_root, "summary.jsonl")
    done = {(r["tag"], r["K"]) for r in (json.loads(l) for l in open(summ))} if os.path.isfile(summ) else set()
    codec = SqCodec(arm, stats_path=ra.get("stats_path"), with_decoder=False, dtype=torch.bfloat16).to(DEVICE).eval(); box = box_index(DEFAULT_DATA, codec.grid)
    mask = torch.zeros(codec.n_tok, dtype=torch.bool); mask[box] = True
    for tag in tags:
        ck = os.path.join(rdir, tag)
        if not os.path.isfile(os.path.join(ck, "meta.json")): continue
        step = json.load(open(os.path.join(ck, "meta.json")))["global_step"]
        for K in a.ks:
            if (tag, K) in done: continue
            out = os.path.join(out_root, f"{tag}_k{K}"); os.makedirs(out, exist_ok=True)
            if not os.path.isfile(os.path.join(out, "latents.pt")):
                subprocess.run([PY, os.path.join(HERE, "render_eval_sq_dit.py"), "--ckpt", ck, "--out_dir", out, "--n_test", str(a.n_test), "--n_train", "-1", "--steps", str(K), "--save_latents"], check=True, cwd=HERE)
            if not os.path.isfile(os.path.join(out, "ocr", "summary.json")):
                man = os.path.join(out, "ocr_manifest.json"); json.dump({"root": rdir, "configs": [{"n": tag, "group": "sq", "label": tag, "dir": f"eval_om/{tag}_k{K}"}]}, open(man, "w"))
                cmd = [PY_OCR, SCORER, "--manifest", man, "--out_dir", os.path.join(out, "ocr"), "--no_full"] + (["--model", OCR_MODEL] if OCR_MODEL else [])
                env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}; env["PYTHONPATH"] = SI_REPO   # the OCR env must not see the qwen-image-edit extras
                subprocess.run(cmd, check=True, cwd=HERE, env=env)
            s = json.load(open(os.path.join(out, "ocr", "summary.json"))); row = [r for r in s["rows"] if r["n"] == tag][0]; t = row.get("test") or {}
            rec = {"run": a.run, "arm": arm, "tag": tag, "step": step, "K": K, "ocr_acc": t.get("ocr_acc"), "gt_ocr_acc": t.get("gt_ocr_acc"), "n": t.get("n"), "unreadable": t.get("unreadable"), "within2": t.get("within2")}
            # per-item OCR correctness (crop read) keyed by image base -> cross-tab with the latent readers
            ocr_items = {}
            for fn in os.listdir(os.path.join(out, "ocr")):
                if fn.endswith(".crop.ocr.json"):
                    d = json.load(open(os.path.join(out, "ocr", fn)))
                    for sp in d.get("splits", {}).values():
                        for r in sp.get("rows", []): ocr_items[r["base"]] = bool(r.get("ok"))
            for src in a.probes:
                pp = os.path.join(HERE, "cache", "ansprobe", f"{arm}_{src}.pt")
                if os.path.isfile(pp):
                    subprocess.run([PY, os.path.join(HERE, "latent_answer_probe.py"), "--arm", arm, "--apply", out, "--probe", pp], check=True, cwd=HERE)
                    pj = json.load(open(os.path.join(out, f"probe_{src}.json"))); rec[f"probe_{src}"] = pj["acc"]
                    Lids = torch.load(os.path.join(out, "latents.pt"))["ids"]; gtmap = {}
                    rep0 = json.load(open(os.path.join(out, "report.json")))
                    for r in rep0["test_rows"] + rep0.get("train_rows", []): gtmap[(r["base"].split("_")[0], r["id"])] = (r["base"], r["gt"])
                    kk = kn = nk = 0
                    for pred, (sp, i) in zip(pj["preds"], Lids):
                        base, g = gtmap[(sp, int(i))]; knows = (pred == g); renders = ocr_items.get(base, False)
                        kk += knows and renders; kn += knows and not renders; nk += (not knows) and renders
                    n_ = len(Lids); rec[f"knows_renders_{src}"] = kk / n_; rec[f"knows_not_renders_{src}"] = kn / n_; rec[f"renders_not_knows_{src}"] = nk / n_
            # fidelity on the answer region + latent RMSE in/out of the box
            rep = json.load(open(os.path.join(out, "report.json"))); L = torch.load(os.path.join(out, "latents.pt")); X = L["x"].float()
            dn, dm, dstd, lin, lout, ratios, greys, inks = [], [], [], [], [], [], [], []
            for i, r in enumerate(rep["test_rows"]):
                pid = r["id"]; pred = gray(os.path.join(out, "imgs", f"{r['base']}_pred.png"))
                vars_ = [gray(os.path.join(SI_DATA, f"arithmetic_add_unordered_sq256_same_ansfont_hard_k{k}", "test", "targets", f"{pid:06d}.png")) for k in range(8)]
                std_p = os.path.join(DEFAULT_DATA, "test", "targets", f"{pid:06d}.png"); std = gray(std_p)
                d_each = [rmse(pred, v) for v in vars_]; d_near = min(d_each); d_mean = rmse(pred, np.mean(vars_, 0))
                dn.append(d_near); dm.append(d_mean); ratios.append(d_near / max(d_mean, 1e-6)); dstd.append(rmse(pred, std))
                ink = pred < 0.8; inks.append(float(ink.mean())); greys.append(float(((pred > 0.25) & ink).sum() / max(ink.sum(), 1)))   # grey share of the ink: blurred means are grey, committed glyphs are black
                with torch.no_grad(): zs = codec.encode(to_pix01([Image.open(std_p).convert("RGB")]).to(DEVICE)).float().cpu()[0]
                x = X[i]; lin.append(float((x[mask] - zs[mask]).pow(2).mean().sqrt())); lout.append(float((x[~mask] - zs[~mask]).pow(2).mean().sqrt()))
            rec.update({"d_nearest": float(np.mean(dn)), "d_mean8": float(np.mean(dm)), "ratio_near_over_mean": float(np.mean(ratios)), "frac_committed": float(np.mean([r < 1 for r in ratios])),
                        "d_std": float(np.mean(dstd)), "lat_rmse_box": float(np.mean(lin)), "lat_rmse_nonbox": float(np.mean(lout)), "grey_share": float(np.mean(greys)), "ink_frac": float(np.mean(inks))})
            with open(summ, "a") as f: f.write(json.dumps(rec) + "\n")
            print(f"[om_eval] {a.run} {tag} K={K}: ocr {100*(rec['ocr_acc'] or 0):.1f}% probe_std {100*rec.get('probe_std', float('nan')):.1f}% probe_hard {100*rec.get('probe_hard', float('nan')):.1f}% "
                  f"knows&renders/knows-only (std) {100*rec.get('knows_renders_std', float('nan')):.0f}/{100*rec.get('knows_not_renders_std', float('nan')):.0f}% unreadable {100*(rec['unreadable'] or 0):.0f}% grey {100*rec['grey_share']:.0f}% ratio {rec['ratio_near_over_mean']:.2f} d_std {rec['d_std']:.3f} lat_nonbox {rec['lat_rmse_nonbox']:.3f}", flush=True)

if __name__ == "__main__":
    main()
