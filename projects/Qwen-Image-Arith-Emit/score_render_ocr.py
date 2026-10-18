"""OCR-based re-scoring of rendered arithmetic answers with a VLM (Qwen3-VL-2B-Instruct).

WHY: the existing scorers measure answer-correctness only via the TEXT PROBE (the
emitted answer token) or via an OCR-FREE delta-cosine render proxy. Neither reads the
actual NUMBER the diffusion model drew. This script closes that gap: it runs a real OCR
model on each rendered image, parses the number after "A:", and checks it numerically
against the ground-truth answer -> a TRUE image-numeric accuracy that works for single-
and multi-digit alike.

INPUT: a manifest.json listing eval dirs. Each eval dir holds:
    - one of {report.json, render_emit_ablate.json, render_randvocab.json} with
      test_rows / train_rows, each row = {base, expr, gt, ...}
    - imgs/<base>_{in,gt,pred[,pred0]}.png
GT is read from that per-dir json (the 'id' field is NOT the dataset line id, so we never
touch the dataset jsonl). The image OCR'd is <base> + cfg['img'] (default '_pred.png';
'_pred0.png' for the emit-token=0 ablations).

OUTPUT (into --out_dir):
    <n>_<label>.ocr.json   per-config rows + accuracies
    <n>_<label>.html       per-config page: accuracy + every example (in|gt|pred + OCR read)
    summary.json / summary.html   accuracy table across all configs (requested vs variants)

Env: eval-ocr (transformers>=4.57 with Qwen3VL, torch, pillow). No accelerate / qwen_vl_utils
needed. Single GPU. The model is loaded ONCE and reused across all manifest dirs.

Usage:
  python score_render_ocr.py --manifest eval/script/ocr_rescore_manifest.json \
      --out_dir eval/result/ocr_rescore_qwen3vl2b \
      [--model Qwen/Qwen3-VL-2B-Instruct] [--limit 0] [--only 07] [--img_width 460]
"""
import argparse
import base64
import html as _html
import io
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

PROMPT = (
    'This image shows two lines of text. The first line is a question "Q: ... = ?". '
    'The second line starts with "A:". Read the number written after "A:" on the answer '
    "line and output ONLY that number (digits only, no spaces, no other text). "
    "If there is no readable number after \"A:\", output NONE."
)
# --crop: the OCR model sees ONLY the answer line (the question line is cut away), so it
# cannot "solve" the question itself instead of reading the drawn digits. Default box =
# bottom half of the 1280x512 canvas (the "A:" line sits at y~270-350; "Q:" at ~160-240).
PROMPT_CROP = (
    'This image shows one line of text starting with "A:". Read the number written after '
    '"A:" and output ONLY that number (digits only, no spaces, no other text). '
    "If there is no readable number after \"A:\", output NONE."
)
CROP = None  # None = full image; else (x0, y0, x1, y1) in pixels, or "bottom" = bottom half


def crop_for_ocr(img):
    if CROP is None:
        return img
    if CROP == "bottom":
        return img.crop((0, img.height // 2, img.width, img.height))
    return img.crop(tuple(CROP))

# ----------------------------------------------------------------------------- model


def load_model(model_id, attn):
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation=attn,
    ).to("cuda").eval()
    return model, proc


@torch.no_grad()
def ocr_one(model, proc, img, max_new_tokens):
    img = crop_for_ocr(img)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT_CROP if CROP is not None else PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True).strip()


def parse_int(s):
    """First signed integer in the OCR reply, or None."""
    m = re.search(r"-?\d+", s.replace(" ", ""))
    return m.group(0) if m else None


def num_eq(pred_str, gt_str):
    if pred_str is None:
        return False
    try:
        return int(pred_str) == int(str(gt_str))
    except ValueError:
        return str(pred_str).strip() == str(gt_str).strip()


# ----------------------------------------------------------------------------- io


def find_rows_json(eval_dir):
    """Return (test_rows, train_rows) from whichever report-like json the dir has."""
    for fn in ("report.json", "render_emit_ablate.json", "render_randvocab.json",
               "render_score.json"):
        p = os.path.join(eval_dir, fn)
        if os.path.isfile(p):
            j = json.load(open(p))
            if isinstance(j, dict) and ("test_rows" in j or "train_rows" in j):
                return j.get("test_rows", []), j.get("train_rows", []), fn
    # last resort: any json with test_rows/train_rows
    for fn in os.listdir(eval_dir):
        if fn.endswith(".json"):
            try:
                j = json.load(open(os.path.join(eval_dir, fn)))
            except Exception:
                continue
            if isinstance(j, dict) and ("test_rows" in j or "train_rows" in j):
                return j.get("test_rows", []), j.get("train_rows", []), fn
    raise FileNotFoundError(f"no report-like json with test_rows/train_rows in {eval_dir}")


def datauri(path, width):
    im = Image.open(path).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def esc(s):
    return _html.escape(str(s))


# ----------------------------------------------------------------------------- scoring


def score_dir(model, proc, eval_dir, img_suffix, limit, max_new_tokens):
    test_rows, train_rows, src = find_rows_json(eval_dir)
    imgs = os.path.join(eval_dir, "imgs")
    out = {"src_json": src, "img_suffix": img_suffix, "splits": {}}
    for split, rows in (("test", test_rows), ("train", train_rows)):
        if limit:
            rows = rows[:limit]
        scored = []
        for r in rows:
            base = r["base"]
            pp = os.path.join(imgs, base + img_suffix)
            gt = str(r.get("gt", ""))
            if not os.path.isfile(pp):
                scored.append({**_row_core(r), "ocr_raw": "(missing image)",
                               "ocr_pred": None, "ok": False})
                continue
            img = Image.open(pp).convert("RGB")
            raw = ocr_one(model, proc, img, max_new_tokens)
            pred = parse_int(raw)
            scored.append({**_row_core(r), "ocr_raw": raw, "ocr_pred": pred,
                           "ok": num_eq(pred, gt)})
        n = len(scored)
        out["splits"][split] = {
            "n": n,
            "ocr_acc": (sum(x["ok"] for x in scored) / n) if n else None,
            "rows": scored,
        }
    # overall (test+train pooled)
    allrows = out["splits"]["test"]["rows"] + out["splits"]["train"]["rows"]
    na = len(allrows)
    out["ocr_acc_all"] = (sum(x["ok"] for x in allrows) / na) if na else None
    out["n_all"] = na
    return out


def _row_core(r):
    return {"base": r["base"], "expr": r.get("expr", ""), "gt": str(r.get("gt", "")),
            "probe": str(r.get("probe", r.get("pred", "")))}


# ----------------------------------------------------------------------------- html


def per_config_html(cfg, res, eval_dir, img_suffix, img_width):
    imgs = os.path.join(eval_dir, "imgs")

    def acc_pct(x):
        return "n/a" if x is None else f"{x*100:.1f}%"

    def section(split):
        s = res["splits"][split]
        if not s["rows"]:
            return ""
        cards = ""
        for r in s["rows"]:
            ok = "ok" if r["ok"] else "bad"
            # per-image, INDEPENDENT: T2I runs never write _in.png (there is no input
            # image), and one shared try/except used to blank ALL THREE panes on that
            # first miss -> the review page rendered with zero images, killing the only
            # visual check that catches a silent pipeline failure.
            def _uri(suffix):
                try:
                    return datauri(os.path.join(imgs, r["base"] + suffix), img_width)
                except Exception:
                    return ""
            in_u, gt_u, pr_u = _uri("_in.png"), _uri("_gt.png"), _uri(img_suffix)
            cards += f"""
      <div class="card">
        <div class="meta"><span class="mono">{esc(r['expr'])} = {esc(r['gt'])}</span>
          <span class="rd">OCR read <b class="{ok}">{esc(r['ocr_pred'] if r['ocr_pred'] is not None else r['ocr_raw'])}</b>
          · GT {esc(r['gt'])} · probe {esc(r['probe'])}</span></div>
        <div class="trip">
          <figure><img src="{in_u}"><figcaption>input</figcaption></figure>
          <figure><img src="{gt_u}"><figcaption>GT target</figcaption></figure>
          <figure><img src="{pr_u}" class="pred"><figcaption>render ({esc(img_suffix)})</figcaption></figure>
        </div>
      </div>"""
        return (f"<h2>{esc(split).upper()} <span class='sub'>(n={s['n']} · "
                f"OCR acc {acc_pct(s['ocr_acc'])})</span></h2><div class='grid'>{cards}</div>")

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>OCR re-score · {esc(cfg['n'])} {esc(cfg['label'])}</title><style>
:root{{--bg:#0b0d12;--panel:#141821;--line:#2a3142;--txt:#e6e9ef;--muted:#8a93a6;--accent:#7cc7ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,"Segoe UI","PingFang SC",sans-serif}}
header{{padding:16px 26px;border-bottom:1px solid var(--line);background:#0e1119}}
header h1{{margin:0 0 4px;font-size:17px}}header .m{{color:var(--muted);font-size:13px;line-height:1.6}}
.wrap{{padding:18px 26px;max-width:1400px;margin:0 auto}}
h2{{font-size:16px;margin:22px 0 12px}}.sub{{color:var(--muted);font-size:13px;font-weight:400}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}}
.card{{border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:10px 12px}}
.meta{{display:flex;justify-content:space-between;align-items:center;font-size:13px;margin-bottom:8px;gap:10px}}
.mono{{font-family:monospace}}.rd{{color:var(--muted);font-size:12px}}
.ok{{color:#84e4a8}}.bad{{color:#ff6b9d}}
.trip{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}}
figure{{margin:0}}figure img{{width:100%;border-radius:6px;border:1px solid var(--line);background:#fff}}
figure img.pred{{border-color:var(--accent)}}figcaption{{color:var(--muted);font-size:11px;text-align:center;margin-top:3px}}
.banner{{background:#0f1a26;border:1px solid #28527a;border-radius:8px;padding:12px 16px;margin:12px 0;font-size:13px;line-height:1.7}}
</style></head><body>
<header><h1>OCR re-score [{esc(cfg['n'])}] {esc(cfg['label'])}</h1>
<div class="m">model: {esc(cfg.get('_model'))} · image OCR'd: <span class="mono">&lt;base&gt;{esc(img_suffix)}</span>
· source dir: <span class="mono">{esc(os.path.relpath(eval_dir, cfg['_root']))}</span><br>
OCR accuracy = parsed number after "A:" equals the ground-truth answer (numeric). probe = the
text-probe answer recorded at eval time (for reference).</div></header>
<div class="wrap">
<div class="banner"><b>OCR image-numeric accuracy</b> — overall {acc_pct(res['ocr_acc_all'])} (n={res['n_all']}) ·
test {acc_pct(res['splits']['test']['ocr_acc'])} (n={res['splits']['test']['n']}) ·
train {acc_pct(res['splits']['train']['ocr_acc'])} (n={res['splits']['train']['n']}).</div>
{section("test")}
{section("train")}
</div></body></html>"""
    return page


def summary_html(rows, model_id):
    def acc(x):
        return "—" if x is None else f"{x*100:.0f}%"

    def trow(r):
        cls = "miss" if r.get("missing") else ""
        link = (f"<a href='{esc(r['html'])}'>{esc(r['n'])}</a>" if r.get("html")
                else esc(r["n"]))
        note = esc(r.get("note", ""))
        return (f"<tr class='{cls}'><td class='mono'>{link}</td><td>{esc(r['label'])}</td>"
                f"<td class='r'>{acc(r.get('ocr_all'))}</td>"
                f"<td class='r'>{acc(r.get('ocr_test'))}</td>"
                f"<td class='r'>{acc(r.get('ocr_train'))}</td>"
                f"<td class='r dim'>{r.get('n_all','—')}</td>"
                f"<td class='mono dim'>{esc(r.get('img','_pred.png'))}</td>"
                f"<td class='dim'>{note}</td></tr>")

    req = [r for r in rows if r["group"] == "requested"]
    var = [r for r in rows if r["group"] == "variant"]
    head = ("<tr><th>#</th><th>config</th><th>OCR acc<br>overall</th><th>test</th>"
            "<th>train</th><th>n</th><th>img</th><th>note</th></tr>")
    body_req = "".join(trow(r) for r in req)
    body_var = "".join(trow(r) for r in var)
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>OCR re-score summary · Qwen3-VL-2B</title><style>
body{{margin:0;background:#0b0d12;color:#e6e9ef;font-family:-apple-system,"Segoe UI","PingFang SC",sans-serif}}
header{{padding:16px 26px;border-bottom:1px solid #2a3142;background:#0e1119}}
h1{{margin:0 0 4px;font-size:18px}}.m{{color:#8a93a6;font-size:13px;line-height:1.6}}
.wrap{{padding:18px 26px;max-width:1500px;margin:0 auto}}
h2{{font-size:15px;margin:22px 0 10px;color:#cdd3df}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{border:1px solid #2a3142;padding:6px 10px;text-align:left}}
th{{background:#141821;color:#8a93a6;font-weight:600}}
td.r{{text-align:right;font-variant-numeric:tabular-nums}}
.mono{{font-family:monospace}}.dim{{color:#8a93a6}}
tr.miss td{{color:#ff6b9d}}
a{{color:#7cc7ff}}
</style></head><body>
<header><h1>OCR re-score summary — Qwen3-VL-2B-Instruct</h1>
<div class="m">Independent image-numeric accuracy: the OCR model reads the number drawn after
"A:" in each rendered image and compares it (numerically) to the ground-truth answer.
This verifies what the model actually DREW, unlike the text-probe / delta-cosine metrics.
model: {esc(model_id)}</div></header>
<div class="wrap">
<h2>Requested configs (1–20)</h2>
<table>{head}{body_req}</table>
<h2>Weighting variants (scored for free)</h2>
<table>{head}{body_var}</table>
</div></body></html>"""


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--limit", type=int, default=0, help="rows per split (0=all); for smoke")
    ap.add_argument("--only", default="", help="comma list of config n's to run (smoke)")
    ap.add_argument("--img_width", type=int, default=460)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--crop", default="none",
                    help="'none' (full image, original), 'bottom' (bottom half = the answer "
                         "line only), or 'x0,y0,x1,y1' pixel box. Cropping removes the "
                         "question so the VLM must READ the drawn digits, not solve.")
    args = ap.parse_args()
    global CROP
    if args.crop == "bottom":
        CROP = "bottom"
    elif args.crop != "none":
        CROP = tuple(int(v) for v in args.crop.split(","))
        assert len(CROP) == 4, args.crop
    print(f"[ocr] crop={CROP if CROP is not None else 'none (full image)'}", flush=True)

    man = json.load(open(args.manifest))
    # manifests store $SI_REPO-relative roots so they survive a move (MIGRATION.md)
    root = os.path.expanduser(os.path.expandvars(man["root"]))
    only = set(s.strip() for s in args.only.split(",") if s.strip())
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[ocr] loading {args.model} (attn={args.attn}) ...", flush=True)
    t0 = time.time()
    model, proc = load_model(args.model, args.attn)
    print(f"[ocr] model ready in {time.time()-t0:.0f}s", flush=True)

    summary_rows = []
    for cfg in man["configs"]:
        if only and cfg["n"] not in only:
            continue
        srow = {"n": cfg["n"], "label": cfg["label"], "group": cfg["group"],
                "img": cfg.get("img", "_pred.png")}
        if not cfg.get("dir"):
            srow.update(missing=True, note=cfg.get("note", "no eval dir"))
            summary_rows.append(srow)
            print(f"[ocr] {cfg['n']:>3} SKIP (no dir): {cfg.get('note','')}", flush=True)
            continue
        eval_dir = os.path.join(root, cfg["dir"])
        img_suffix = cfg.get("img", "_pred.png")
        cfg["_model"], cfg["_root"] = args.model, root
        ts = time.time()
        try:
            res = score_dir(model, proc, eval_dir, img_suffix, args.limit, args.max_new_tokens)
        except Exception as e:
            srow.update(missing=True, note=f"ERROR: {repr(e)[:120]}")
            summary_rows.append(srow)
            print(f"[ocr] {cfg['n']:>3} ERROR {repr(e)[:120]}", flush=True)
            continue
        stem = f"{cfg['n']}_{cfg['label']}"
        json.dump({"config": {k: cfg[k] for k in cfg if not k.startswith("_")},
                   "crop": args.crop, **res},
                  open(os.path.join(args.out_dir, stem + ".ocr.json"), "w"), indent=2)
        html_name = stem + ".html"
        open(os.path.join(args.out_dir, html_name), "w").write(
            per_config_html(cfg, res, eval_dir, img_suffix, args.img_width))
        srow.update(ocr_all=res["ocr_acc_all"], n_all=res["n_all"],
                    ocr_test=res["splits"]["test"]["ocr_acc"],
                    ocr_train=res["splits"]["train"]["ocr_acc"], html=html_name)
        summary_rows.append(srow)
        print(f"[ocr] {cfg['n']:>3} done in {time.time()-ts:.0f}s  "
              f"OCRacc all={_p(res['ocr_acc_all'])} test={_p(res['splits']['test']['ocr_acc'])} "
              f"train={_p(res['splits']['train']['ocr_acc'])}  ({stem})", flush=True)

    summary_rows.sort(key=lambda r: (0 if r["group"] == "requested" else 1, r["n"]))
    json.dump({"model": args.model, "crop": args.crop,
               "prompt": PROMPT_CROP if CROP is not None else PROMPT, "rows": summary_rows},
              open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    open(os.path.join(args.out_dir, "summary.html"), "w").write(
        summary_html(summary_rows, args.model))
    print(f"[ocr] -> {os.path.join(args.out_dir, 'summary.html')}", flush=True)


def _p(x):
    return "n/a" if x is None else f"{x*100:.0f}%"


if __name__ == "__main__":
    main()
