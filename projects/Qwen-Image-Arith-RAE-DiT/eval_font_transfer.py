"""eval_font_transfer.py -- zero-shot glyph transfer: does an MMDiT trained on DejaVuSansMono-Bold canvases still write the
right answer when the SOURCE canvas is rendered in another font?  Separates "the diffusion model computes on digit identities
that the representation makes font-invariant" from "it matches pixel templates".

For each run (runs/<run>, checkpoint <ckpt_tag>) and each font test set data/arithmetic_add_unordered_sq256_same_font_<tag>:
  gen   : generate the answer canvas from the font-F source (render_eval_sq_dit.py machinery, 24 Euler steps, train shift)
  recon : encoder->decoder round trip of the font-F TARGET canvas (the reconstruction ceiling for that font)
then ONE batched crop-OCR pass over every (run, font, kind) and a JSON + figure.
  python eval_font_transfer.py --runs mm_siglip2 mm_vae mm_webssl mm_emu35 mm_mae --fonts liberation_regular dejavu_oblique press_start_2p orbitron caveat permanent_marker --n_test 500
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
from PIL import Image

from dit_model import VanillaDiT
from mmdit_model import MMDiT
from render_eval_sq_dit import build_sigmas
from sq_codecs import SqCodec, SI_DATA, to_pix01

DEVICE = "cuda"; DT = torch.bfloat16
NAMES = {"siglip2": "SigLIP2", "vae": "Qwen VAE", "webssl": "WebSSL-DINO", "emu35": "Emu3.5 IBQ", "mae": "MAE-L"}
COLS = {"siglip2": "#2a78d6", "webssl": "#eb6834", "mae": "#1baf7a", "vae": "#eda100", "emu35": "#e87ba4"}


@torch.no_grad()
def generate(model, codec, sigmas, src_pils, gen):
    src = codec.encode(to_pix01(src_pils).to(DEVICE))
    x = torch.randn(src.shape, generator=gen, device=DEVICE, dtype=torch.float32)
    for i in range(len(sigmas) - 1):
        s, s_next = sigmas[i], sigmas[i + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(x, torch.full((x.shape[0],), s, device=DEVICE), src).float()
        x = x + (s_next - s) * v
    return x


def write_set(out, items, pils, data):
    os.makedirs(os.path.join(out, "imgs"), exist_ok=True); rows = []
    for it, im in zip(items, pils):
        base = f"test_{it['id']:06d}"; im.save(os.path.join(out, "imgs", base + "_pred.png"))
        for suf, key in (("_gt.png", "target"), ("_in.png", "input")):
            lp = os.path.join(out, "imgs", base + suf)
            if not os.path.lexists(lp):
                os.symlink(os.path.abspath(os.path.join(data, it[key])), lp)
        rows.append({"id": it["id"], "expr": str(it["expr"]), "gt": str(it["answer"]), "base": base})
    json.dump({"test_rows": rows, "train_rows": []}, open(os.path.join(out, "report.json"), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ckpt_tag", default="final")
    ap.add_argument("--fonts", nargs="+", required=True, help="font tags; 'base' = the training font set")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default=os.path.join(HERE, "eval", "result_gate4", "font_transfer"))
    ap.add_argument("--ocr_python", default=os.path.join(os.environ.get("CONDA_ROOT", ""), "envs", "eval-ocr-b200", "bin", "python"))
    ap.add_argument("--skip_ocr", action="store_true")
    ap.add_argument("--ocr_only", action="store_true", help="re-run only the OCR stage on the images already written")
    a = ap.parse_args()
    os.makedirs(a.out_root, exist_ok=True)
    manifest = {"root": a.out_root, "configs": []}; done = {}
    for run in ([] if a.ocr_only else a.runs):
        rdir = os.path.join(HERE, "runs", run); meta0 = json.load(open(os.path.join(rdir, "args.json"))); arm = meta0["arm"]
        ck = os.path.join(rdir, a.ckpt_tag); meta = json.load(open(os.path.join(ck, "meta.json")))
        codec = SqCodec(arm, stats_path=meta0.get("stats_path"), with_decoder=True, dtype=DT).to(DEVICE).eval()
        ModelCls = MMDiT if meta.get("arch") == "sq_mmdit" else VanillaDiT
        model = ModelCls.from_pretrained(os.path.join(ck, "dit"), torch_dtype=DT).to(DEVICE).eval()
        sigmas = build_sigmas(a.steps, float(meta["sigma_shift"]))
        print(f"[font] {run}: arm={arm} ckpt={a.ckpt_tag} step={meta.get('global_step')} shift={meta['sigma_shift']}", flush=True)
        for font in a.fonts:
            data = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same" + ("" if font == "base" else f"_font_{font}"))
            items = [json.loads(l) for l in open(os.path.join(data, "test.jsonl")) if l.strip()][: a.n_test]
            t0 = time.time()
            for kind in ("gen", "recon"):
                tag = f"{run}__{font}__{kind}"; out = os.path.join(a.out_root, tag)
                if os.path.isfile(os.path.join(out, "report.json")):
                    manifest["configs"].append({"n": tag, "group": "font", "label": tag, "dir": tag}); continue
                pils_out = []
                gen = torch.Generator(device=DEVICE).manual_seed(a.seed)
                for i in range(0, len(items), a.batch):
                    chunk = items[i:i + a.batch]
                    if kind == "gen":
                        src = [Image.open(os.path.join(data, it["input"])).convert("RGB") for it in chunk]
                        x = generate(model, codec, sigmas, src, gen)
                    else:
                        tgt = [Image.open(os.path.join(data, it["target"])).convert("RGB") for it in chunk]
                        with torch.no_grad():
                            x = codec.encode(to_pix01(tgt).to(DEVICE))
                    with torch.no_grad():
                        pils_out += codec.to_pil(codec.decode(x))
                write_set(out, items, pils_out, data)
                manifest["configs"].append({"n": tag, "group": "font", "label": tag, "dir": tag})
            print(f"[font] {run} {font}: gen + recon written ({time.time()-t0:.0f}s)", flush=True)
        del model, codec; torch.cuda.empty_cache()
    man_p = os.path.join(a.out_root, "ocr_manifest.json")
    if a.ocr_only:
        manifest = json.load(open(man_p))
    else:
        json.dump(manifest, open(man_p, "w"))
    if a.skip_ocr:
        return
    ocr_out = os.path.join(a.out_root, "ocr"); os.makedirs(ocr_out, exist_ok=True)
    scorer = os.path.join(HERE, "..", "Qwen-Image-Arith-ScratchTE", "score_render_ocr_crop_batched.py")
    cmd = [a.ocr_python, scorer, "--model", os.environ["SI_OCR_MODEL"], "--manifest", man_p, "--out_dir", ocr_out, "--no_full", "--batch", "48"]
    env = dict(os.environ); env["PYTHONPATH"] = os.path.abspath(os.path.join(HERE, ".."))     # the OCR env must NOT see third_party/pyextra_qie
    print("[font] OCR:", " ".join(cmd), flush=True); subprocess.run(cmd, check=True, env=env)
    summ = json.load(open(os.path.join(ocr_out, "summary.json")))
    res = {}
    for r in summ["rows"]:
        run, font, kind = r["n"].split("__"); b = r.get("test") or {}
        res.setdefault(run, {}).setdefault(font, {})[kind] = {"ocr_acc": b.get("ocr_acc"), "gt_ocr_acc": b.get("gt_ocr_acc"), "unreadable": b.get("unreadable"), "n": b.get("n")}
    json.dump(res, open(os.path.join(a.out_root, "font_transfer_results.json"), "w"), indent=1)
    print("\nrun            " + " ".join(f"{f[:12]:>14s}" for f in a.fonts))
    for run in res:
        print(f"{run:14s} " + " ".join(f"{100*(res[run].get(f, {}).get('gen', {}).get('ocr_acc') or 0):5.1f}/{100*(res[run].get(f, {}).get('recon', {}).get('ocr_acc') or 0):5.1f}   " for f in a.fonts) + "   (gen / recon ceiling)")
    make_figure(res, a)


def make_figure(res, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    runs = list(res); fonts = [f for f in a.fonts]
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=SURF); ax.set_facecolor(SURF)
    w = 0.8 / len(runs); x = np.arange(len(fonts))
    for i, run in enumerate(runs):
        arm = json.load(open(os.path.join(HERE, "runs", run, "args.json"))).get("base_arm") or json.load(open(os.path.join(HERE, "runs", run, "args.json")))["arm"]
        g = [100 * (res[run].get(f, {}).get("gen", {}).get("ocr_acc") or 0) for f in fonts]
        c = [100 * (res[run].get(f, {}).get("recon", {}).get("ocr_acc") or 0) for f in fonts]
        ax.bar(x + (i - len(runs) / 2 + 0.5) * w, g, w * 0.9, color=COLS.get(arm, "#777"), label=f"{NAMES.get(arm, run)}: generated")
        ax.scatter(x + (i - len(runs) / 2 + 0.5) * w, c, marker="_", s=260, color=INK, linewidths=1.6, zorder=4, label="reconstruction ceiling" if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels(fonts, fontsize=9); ax.set_ylim(0, 105); ax.set_ylabel("exact answer OCR (%)")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(length=0)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_color(GRID)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title(f"Zero-shot glyph transfer: source canvas in a new font, MMDiT trained on DejaVuSansMono-Bold ({a.n_test} test problems); dash = encoder round trip of that font", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_root, "fig_font_transfer.png"), dpi=150, facecolor=SURF)
    print("[font] figure ->", os.path.join(a.out_root, "fig_font_transfer.png"), flush=True)


if __name__ == "__main__":
    main()
