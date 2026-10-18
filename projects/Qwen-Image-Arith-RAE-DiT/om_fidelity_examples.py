"""Contact sheet explaining the glyph-fidelity score: for one eval dir, show per item the output crop, the best-matching bank
glyph (answer string / font / angle the search picked), and the numbers (F = ncc x ink explained, OCR read, GT), grouped into the
four cells of legible x OCR-correct.
  python om_fidelity_examples.py RUN TAG [--per 5] -> eval/result_gate4/fidelity/examples_<RUN>_<TAG>.png
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import om_fidelity_score as M
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("tag"); ap.add_argument("--per", type=int, default=5); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
d = os.path.join(HERE, "runs", a.run, "eval_om", a.tag); fid = json.load(open(os.path.join(d, "fidelity.json"))); tau = fid["summary"]["tau"]; items = fid["items"]
FONTS = M.fonts(); CROP = M.CROP; S = 2; W, H = (CROP[2] - CROP[0]) * S, (CROP[3] - CROP[1]) * S
cells = [("legible AND OCR correct", lambda i: i["legible"] and i["ocr_ok"]), ("legible, OCR WRONG (clean glyph, wrong number)", lambda i: i["legible"] and not i["ocr_ok"]),
         ("NOT legible, OCR correct (right number, not a clean glyph)", lambda i: (not i["legible"]) and i["ocr_ok"]), ("NOT legible, OCR wrong", lambda i: (not i["legible"]) and not i["ocr_ok"])]
rng = np.random.RandomState(a.seed)
try: FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13); FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: FONT = FONT_B = ImageFont.load_default()
colw = W + 8 + W // 2 + 8; rowh = H + 44
sheet = Image.new("RGB", (a.per * colw + 20, len(cells) * (rowh + 30) + 60), "white"); dr = ImageDraw.Draw(sheet)
dr.text((10, 8), f"{a.run} / {a.tag}   fidelity = legible (F >= {tau});  F = best NCC against any valid rendering x share of ink explained;  n = {len(items)}", fill="black", font=FONT_B)
dr.text((10, 30), "each item: output crop (left) | the bank glyph the search matched best, drawn in that font/angle (right, half size) | F (ncc, outside) | template read / OCR read / GT", fill="#444", font=FONT)
y0 = 60
for ci, (name, pred) in enumerate(cells):
    sel = [i for i in items if pred(i)]; dr.text((10, y0), f"{name}: {len(sel)}/{len(items)} = {100 * len(sel) / len(items):.0f}%", fill="black", font=FONT_B)
    pick = [sel[j] for j in rng.choice(len(sel), min(a.per, len(sel)), replace=False)] if sel else []
    for k, it in enumerate(pick):
        x = 10 + k * colw; y = y0 + 22
        crop = Image.open(os.path.join(d, "imgs", f"{it['base']}_pred.png")).convert("RGB").crop(CROP).resize((W, H), Image.NEAREST); sheet.paste(crop, (x, y))
        if it["font"] is not None:
            fp, fs = FONTS[it["font"]]; g = M.glyph_layer(it["tmpl_read"], ImageFont.truetype(fp, fs), it["ang"])
            gim = Image.fromarray((255 - g * 255).astype(np.uint8)).convert("RGB"); gw, gh = gim.size; sc = min((W // 2) / gw, H / gh, 2.0)
            gim = gim.resize((max(1, int(gw * sc)), max(1, int(gh * sc))), Image.NEAREST); sheet.paste(gim, (x + W + 8, y + (H - gim.size[1]) // 2))
        col = "#1a7f37" if it["legible"] else "#b42318"
        dr.text((x, y + H + 2), f"F={it['F']:.2f} (ncc {it['ncc']:.2f}, out {it['outside']:.2f})", fill=col, font=FONT)
        dr.text((x, y + H + 18), f"tmpl {it['tmpl_read']} / OCR {it['ocr_read']} / GT {it['gt']}", fill=("#1a7f37" if it["ocr_ok"] else "#b42318"), font=FONT)
    y0 += rowh + 30
out = os.path.join(HERE, "eval", "result_gate4", "fidelity", f"examples_{a.run}_{a.tag}.png"); sheet.save(out); print("wrote", out, sheet.size)
