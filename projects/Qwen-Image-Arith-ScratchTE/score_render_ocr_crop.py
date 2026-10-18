"""OCR re-scoring with the ANSWER-LINE CROP (Qwen3-VL-2B), alongside the full-image read.

Why: on 1280x512 full frames Qwen3-VL-2B drops/zeros the last digit of perfectly crisp
monospace numbers (43->4, 106->100; GT frames themselves read only 0.087 whole-image),
while a crop to the answer line (lower half, ink bbox, 24 px pad) read 16/16 (probe
2026-09-02, vista_setup/ocr_probe.py). Whole-image scores are therefore a biased LOWER
bound. This scorer reads every rendered image twice -- crop (metric of record) and full
frame (for comparison with the historical tables) -- and also reads the GT frame's crop,
which measures the OCR ceiling on the very same batch.

Reuses score_render_ocr.py (manifest format, row discovery, number parsing, HTML page).
Manifest / output layout identical; per-row fields:
  ocr_raw / ocr_pred / ok           crop read (drives the HTML + summary 'acc')
  ocr_raw_full / ocr_pred_full / ok_full   whole-frame read (legacy metric)
  within2                            |crop read - gt| <= 2
  gt_ocr_raw / gt_ok                 crop read of the GT frame (OCR ceiling)
Usage:
  python score_render_ocr_crop.py --manifest M.json --out_dir OUT [--model ...] [--limit N]
                                  [--no_full] [--no_gt]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EMIT_DIR = os.path.join(os.path.dirname(HERE), "Qwen-Image-Arith-Emit")
if EMIT_DIR not in sys.path:
    sys.path.insert(0, EMIT_DIR)

import torch
from PIL import Image

import score_render_ocr as base  # noqa: E402  (model loader, parse_int, num_eq, find_rows_json, html)

P_CROP = "Read the number in this image. Output digits only."


def crop_answer_line(im, pad=24, ink_thresh=200):
    """Lower half of the frame, tight bbox around the ink, generous padding. Falls back
    to the whole image when the lower half is blank."""
    g = im.convert("L")
    low = g.crop((0, g.height // 2, g.width, g.height))
    bb = low.point(lambda v: 255 if v < ink_thresh else 0).getbbox()
    if not bb:
        return im
    x0, y0, x1, y1 = bb
    return im.crop((max(0, x0 - pad), g.height // 2 + max(0, y0 - pad),
                    min(g.width, x1 + pad), g.height // 2 + min(low.height, y1 + pad)))


@torch.no_grad()
def ask(model, proc, img, prompt, max_new_tokens=16):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _within(pred, gt, k=2):
    try:
        return abs(int(pred) - int(str(gt))) <= k
    except (TypeError, ValueError):
        return False


def score_dir(model, proc, eval_dir, img_suffix, limit, do_full, do_gt):
    test_rows, train_rows, src = base.find_rows_json(eval_dir)
    imgs = os.path.join(eval_dir, "imgs")
    out = {"src_json": src, "img_suffix": img_suffix, "method": "answer-line crop", "splits": {}}
    for split, rows in (("test", test_rows), ("train", train_rows)):
        if limit:
            rows = rows[:limit]
        scored = []
        for r in rows:
            pp = os.path.join(imgs, r["base"] + img_suffix)
            gt = str(r.get("gt", ""))
            row = base._row_core(r)
            if not os.path.isfile(pp):
                row.update(ocr_raw="(missing image)", ocr_pred=None, ok=False, within2=False)
                scored.append(row)
                continue
            im = Image.open(pp).convert("RGB")
            raw_c = ask(model, proc, crop_answer_line(im), P_CROP)
            pred_c = base.parse_int(raw_c)
            row.update(ocr_raw=raw_c, ocr_pred=pred_c, ok=base.num_eq(pred_c, gt),
                       within2=_within(pred_c, gt))
            if do_full:
                raw_f = base.ocr_one(model, proc, im, 16)
                pred_f = base.parse_int(raw_f)
                row.update(ocr_raw_full=raw_f, ocr_pred_full=pred_f, ok_full=base.num_eq(pred_f, gt))
            if do_gt:
                gp = os.path.join(imgs, r["base"] + "_gt.png")
                if os.path.isfile(gp):
                    raw_g = ask(model, proc, crop_answer_line(Image.open(gp).convert("RGB")), P_CROP)
                    row.update(gt_ocr_raw=raw_g, gt_ok=base.num_eq(base.parse_int(raw_g), gt))
            scored.append(row)
        n = len(scored)
        def frac(key):
            vals = [x.get(key) for x in scored if key in x]
            return (sum(bool(v) for v in vals) / len(vals)) if vals else None
        out["splits"][split] = {
            "n": n,
            "ocr_acc": frac("ok"),                 # crop exact  (metric of record)
            "within2": frac("within2"),
            "unreadable": (sum(x["ocr_pred"] is None for x in scored) / n) if n else None,
            "ocr_acc_full": frac("ok_full"),       # whole-frame exact (legacy)
            "gt_ocr_acc": frac("gt_ok"),           # OCR ceiling on this batch
            "rows": scored,
        }
    allrows = out["splits"]["test"]["rows"] + out["splits"]["train"]["rows"]
    na = len(allrows)
    out["ocr_acc_all"] = (sum(x["ok"] for x in allrows) / na) if na else None
    out["n_all"] = na
    return out


def _p(x):
    return "  n/a" if x is None else f"{x*100:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--img_width", type=int, default=460)
    ap.add_argument("--no_full", action="store_true", help="skip the whole-frame read")
    ap.add_argument("--no_gt", action="store_true", help="skip the GT-frame crop read")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    root = os.path.expanduser(os.path.expandvars(man["root"]))
    only = set(s.strip() for s in args.only.split(",") if s.strip())
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[ocr-crop] loading {args.model} ...", flush=True)
    t0 = time.time()
    model, proc = base.load_model(args.model, args.attn)
    print(f"[ocr-crop] model ready in {time.time()-t0:.0f}s", flush=True)

    summary = []
    lines = [f"{'config':40s} split    n  crop%  ±2%  unread%  full%  gtOCR%"]
    for cfg in man["configs"]:
        if only and cfg["n"] not in only:
            continue
        eval_dir = os.path.join(root, cfg["dir"])
        img_suffix = cfg.get("img", "_pred.png")
        cfg["_model"], cfg["_root"] = args.model + " (answer-line crop)", root
        ts = time.time()
        try:
            res = score_dir(model, proc, eval_dir, img_suffix, args.limit,
                            not args.no_full, not args.no_gt)
        except Exception as e:           # a missing/partial render dir must not abort the other configs
            print(f"[ocr-crop] {cfg['n']} ERROR {repr(e)[:160]} -- skipped", flush=True)
            lines.append(f"{cfg['n'][:40]:40s} ERROR {repr(e)[:60]}")
            continue
        stem = f"{cfg['n']}_{cfg['label']}"[:150].replace("/", "_")
        json.dump({"config": {k: cfg[k] for k in cfg if not k.startswith("_")}, **res},
                  open(os.path.join(args.out_dir, stem + ".crop.ocr.json"), "w"), indent=2)
        open(os.path.join(args.out_dir, stem + ".html"), "w").write(
            base.per_config_html(cfg, res, eval_dir, img_suffix, args.img_width))
        srow = {"n": cfg["n"], "label": cfg["label"]}
        for split in ("test", "train"):
            s = res["splits"][split]
            if not s["n"]:
                continue
            srow[split] = {k: s[k] for k in ("n", "ocr_acc", "within2", "unreadable",
                                            "ocr_acc_full", "gt_ocr_acc")}
            lines.append(f"{cfg['n'][:40]:40s} {split:5s} {s['n']:4d} {_p(s['ocr_acc'])} "
                         f"{_p(s['within2'])} {_p(s['unreadable'])} {_p(s['ocr_acc_full'])} "
                         f"{_p(s['gt_ocr_acc'])}")
        summary.append(srow)
        print(f"[ocr-crop] {cfg['n']} done in {time.time()-ts:.0f}s", flush=True)
        print("\n".join(lines[-2:]), flush=True)
    json.dump({"model": args.model, "method": "answer-line crop (lower half, ink bbox, pad 24)",
               "prompt_crop": P_CROP, "prompt_full": base.PROMPT, "rows": summary},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    open(os.path.join(args.out_dir, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"[ocr-crop] -> {args.out_dir}/summary.txt", flush=True)


if __name__ == "__main__":
    main()
