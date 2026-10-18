"""eval_x0_sigma_curve.py -- one-step read-out of a trained MMDiT at every noise level.

For sigma on a grid the input is z = (1 - sigma) x_src + sigma eps, one step gives x0 = z - sigma v(z, sigma, src), which is
decoded and OCR-scored.  --source true: x_src is the answered canvas (below the readability threshold a copier scores);
--source blank: x_src is the question canvas (nothing to copy, the answer has to be computed).  Also records the cosine
between v and the ideal velocity eps - x_true inside and outside the answer box.
  python eval_x0_sigma_curve.py --run mm_s1_vae --ckpt final --source true
"""
import argparse, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from PIL import Image
from mmdit_model import MMDiT
from dit_model import VanillaDiT
from eval_font_transfer import write_set
from sq_codecs import SI_DATA, SqCodec, to_pix01
from intervene import box_index
import make_square_gate_set as MS

DEVICE = "cuda"; DT = torch.bfloat16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="final")
    ap.add_argument("--source", default="true", choices=["true", "blank", "wrong"], help="wrong: the canvas carries a WRONG answer (counterfactual): a copier follows it, a computer overwrites it")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--n_test", type=int, default=300); ap.add_argument("--batch", type=int, default=50); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_root", default=os.path.join(HERE, "eval", "result_gate4", "x0_sigma_curve"))
    ap.add_argument("--ocr_python", default=os.path.join(os.environ.get("CONDA_ROOT", ""), "envs", "eval-ocr-b200", "bin", "python"))
    a = ap.parse_args()
    rdir = os.path.join(HERE, "runs", a.run); ra = json.load(open(os.path.join(rdir, "args.json"))); arm = ra["arm"]
    ck = os.path.join(rdir, a.ckpt); meta = json.load(open(os.path.join(ck, "meta.json")))
    data = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same")
    items = [json.loads(l) for l in open(os.path.join(data, "test.jsonl")) if l.strip()][: a.n_test]
    def wrong_answer(ans):
        ans = int(ans)
        for w in (ans + 7, ans - 7, ans + 1, ans - 1):
            if w >= 0 and w != ans and len(str(w)) == len(str(ans)): return w
        return ans + 1
    wrongs = {int(it["id"]): wrong_answer(it["answer"]) for it in items}
    fs0 = items[0].get("font_size", 28); ls0 = max(4, fs0 // 2)
    out_root = os.path.join(a.out_root, f"{a.run}@{a.ckpt}__{a.source}"); os.makedirs(out_root, exist_ok=True)
    codec = SqCodec(arm, stats_path=ra.get("stats_path"), with_decoder=True, dtype=DT).to(DEVICE).eval()
    ModelCls = MMDiT if meta.get("arch") == "sq_mmdit" else VanillaDiT
    model = ModelCls.from_pretrained(os.path.join(ck, "dit"), torch_dtype=DT).to(DEVICE).eval()
    box = box_index(data, codec.grid).to(DEVICE); mask = torch.zeros(codec.n_tok, dtype=torch.bool, device=DEVICE); mask[box] = True
    print(f"[x0curve] {a.run}@{a.ckpt} arm={arm} shift={meta['sigma_shift']} source={a.source} n={len(items)}", flush=True)
    manifest = {"root": out_root, "configs": []}; stats = {}
    for sigma in a.sigmas:
        tag = f"s{sigma:.2f}"; out = os.path.join(out_root, tag); pils = []; cos_in, cos_out, cos_ans = [], [], []; R_hat, R_star = [], []
        gen = torch.Generator(device=DEVICE).manual_seed(a.seed)
        for i in range(0, len(items), a.batch):
            chunk = items[i:i + a.batch]
            src_p = [Image.open(os.path.join(data, it["input"])).convert("RGB") for it in chunk]
            tgt_p = [Image.open(os.path.join(data, it["target"])).convert("RGB") for it in chunk]
            with torch.no_grad():
                src = codec.encode(to_pix01(src_p).to(DEVICE)); x_true = codec.encode(to_pix01(tgt_p).to(DEVICE))
                if a.source == "wrong":
                    wr_p = [MS.render(MS.LAYOUTS["same"](it["expr"]), f"A: {wrongs[int(it['id'])]}", fs0, 256, 8, ls0)[0] for it in chunk]
                    x_src = codec.encode(to_pix01(wr_p).to(DEVICE))
                else:
                    x_src = x_true if a.source == "true" else src
                eps = torch.randn(x_true.shape, generator=gen, device=DEVICE, dtype=torch.float32)
                z = (1.0 - sigma) * x_src + sigma * eps
                with torch.autocast("cuda", dtype=DT):
                    v = model(z, torch.full((z.shape[0],), sigma, device=DEVICE), torch.zeros_like(src) if ra.get("null_source") else src).float()   # null-source copier sees no question
                x0 = z - sigma * v; v_star = eps - x_true
                cin = torch.nn.functional.cosine_similarity(v[:, mask].reshape(v.shape[0], -1), v_star[:, mask].reshape(v.shape[0], -1), dim=1)
                cout = torch.nn.functional.cosine_similarity(v[:, ~mask].reshape(v.shape[0], -1), v_star[:, ~mask].reshape(v.shape[0], -1), dim=1)
                # answer-specific cosine: compare what the model WROTE in the box (x0 - question canvas) with what should be there (x_true - question canvas);
                # unlike cos(v, v*) this ignores the noise term and the shared background, so it separates copying/computing from predicting epsilon
                r_hat = (x0 - src)[:, mask].reshape(x0.shape[0], -1); r_star = (x_true - src)[:, mask].reshape(x0.shape[0], -1)
                cos_ans += torch.nn.functional.cosine_similarity(r_hat, r_star, dim=1).tolist()
                R_hat.append(r_hat.cpu()); R_star.append(r_star.cpu())
                cos_in += cin.tolist(); cos_out += cout.tolist()
                pils += codec.to_pil(codec.decode(x0))
        write_set(out, items, pils, data)
        manifest["configs"].append({"n": tag, "group": "x0", "label": tag, "dir": tag})
        Rh, Rs = torch.cat(R_hat), torch.cat(R_star)   # centred over the test items: removes the "some digits are written here" component shared by every answer
        cos_c = torch.nn.functional.cosine_similarity(Rh - Rh.mean(0, keepdim=True), Rs - Rs.mean(0, keepdim=True), dim=1)
        stats[tag] = {"sigma": sigma, "cos_v_box": float(np.mean(cos_in)), "cos_v_outside": float(np.mean(cos_out)), "cos_answer_box": float(np.mean(cos_ans)), "cos_answer_box_centred": float(cos_c.mean())}
        print(f"[x0curve] sigma {sigma:.2f}: cos(v, v*) box {np.mean(cos_in):.3f} outside {np.mean(cos_out):.3f}  answer-specific cos {np.mean(cos_ans):.3f} centred {float(cos_c.mean()):.3f}", flush=True)
    del model, codec; torch.cuda.empty_cache()
    man_p = os.path.join(out_root, "ocr_manifest.json"); json.dump(manifest, open(man_p, "w"))
    ocr_out = os.path.join(out_root, "ocr"); os.makedirs(ocr_out, exist_ok=True)
    scorer = os.path.join(HERE, "..", "Qwen-Image-Arith-ScratchTE", "score_render_ocr_crop_batched.py")
    cmd = [a.ocr_python, scorer, "--model", os.environ["SI_OCR_MODEL"], "--manifest", man_p, "--out_dir", ocr_out, "--no_full", "--batch", "48"]
    env = dict(os.environ); env["PYTHONPATH"] = os.path.abspath(os.path.join(HERE, ".."))
    subprocess.run(cmd, check=True, env=env)
    summ = json.load(open(os.path.join(ocr_out, "summary.json")))
    for r in summ["rows"]:
        b = r.get("test") or {}
        if r["n"] in stats: stats[r["n"]].update({"ocr_acc": b.get("ocr_acc"), "unreadable": b.get("unreadable"), "n": b.get("n")})
    if a.source == "wrong":   # per item: did the output follow the (wrong) answer written in the input, or compute the correct one?
        import glob as _glob
        for jp in _glob.glob(os.path.join(ocr_out, "*.crop.ocr.json")):
            d = json.load(open(jp)); tag = os.path.basename(jp).split("_")[0]
            rows = d.get("splits", {}).get("test", {}).get("rows", [])
            if not rows or tag not in stats: continue
            fol = cor = 0
            for r in rows:
                iid = int(r["base"].split("_")[-1]); pred = str(r.get("ocr_pred"))
                if pred == str(wrongs.get(iid)): fol += 1
                elif pred == str(r["gt"]): cor += 1
            stats[tag].update({"frac_follows_input": fol / len(rows), "frac_correct": cor / len(rows), "frac_other": 1 - (fol + cor) / len(rows)})
    res = {"run": a.run, "ckpt": a.ckpt, "arm": arm, "shift": meta["sigma_shift"], "source": a.source, "curve": [stats[f"s{s:.2f}"] for s in a.sigmas]}
    json.dump(res, open(os.path.join(out_root, "results.json"), "w"), indent=1)
    print(f"[x0curve] {a.run}@{a.ckpt} {a.source}: " + "  ".join(f"{c['sigma']:.1f}:{100*(c.get('ocr_acc') or 0):.0f}" for c in res["curve"]), flush=True)
    if a.source == "wrong": print(f"[x0curve] {a.run}@{a.ckpt} wrong-answer canvas: " + "  ".join(f"{c['sigma']:.1f}: follows {100*c.get('frac_follows_input', 0):.0f}% / correct {100*c.get('frac_correct', 0):.0f}% / other {100*c.get('frac_other', 0):.0f}%" for c in res["curve"]), flush=True)


if __name__ == "__main__":
    main()
