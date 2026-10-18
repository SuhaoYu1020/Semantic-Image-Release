"""make_square_gate_set.py -- RE-RENDER arithmetic_add_unordered on a SQUARE canvas, so every codec runs at its decoder's
native geometry (16x16 tokens at 256^2, zero sequence extrapolation) instead of squashing the 1280x512 render.

Same problems, ids and fields as the source jsonl; same font (DejaVuSansMono-Bold, black on white); same two-line structure
with the answer line in the LOWER half (the OCR scorer crops the lower-half ink).

Layouts (canvas 256x256, 8 px side margin; font size = the largest that fits the longest line over the whole dataset):
  same     "Q: 41 + 12 = ?" / "A: 53"   -> 14 chars -> 28 px font, digit cell 17x20 px (~1.3 tokens tall)   [user's pick]
  compact  "41 + 12"        / "A: 53"   ->  7 chars -> 56 px font, digit cell 34x43 px (~2.7 tokens tall)

Splits: --split all (default: train + test, every row) | train | test | gate500 (the 500-row Random(0) train sample
recon_gate uses, written as train.jsonl -- the 2026-10-03 gate sets were made this way).

Output: data/arithmetic_add_unordered_sq<canvas>_<layout>/{train.jsonl, test.jsonl, <split>/inputs|targets/<id>.png,
stats.json, _answer_token_box.json}.  Rows keep every source field and add "ans_box"/"q_box" [x0,y0,x1,y1] (text bboxes),
"layout", "canvas", "font_size".  _answer_token_box.json mirrors the source file's keys for the 16x16 token grid
(answer DIGITS only, union over the test split, margin 1 token, inclusive ranges).

  python make_square_gate_set.py --layout same                 # full train + test
  python make_square_gate_set.py --layout same --split gate500 # just the gate subset (for recon_gate)
"""
import argparse
import json
import math
import os
import random
import shutil

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SI_ROOT = os.environ.get("SI_ROOT") or os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SI_DATA = os.environ.get("SI_DATA") or os.path.join(SI_ROOT, "data")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
ADVANCE = None          # --advance N: draw character by character with a fixed N px advance (phase-aligned glyphs on the token grid)
ANSWER_FONT = None      # --answer_font TTF [--answer_font_size S]: the answer DIGITS are drawn in this font ("A: " and the question stay in FONT_PATH)
ANSWER_FONT_SIZE = None
ANSWER_FONTS = None     # --answer_fonts a.ttf,b.ttf,...: per sample a RANDOM one of these (sizes auto-fit to --answer_height px digit height)
ANSWER_HEIGHT = 20
ANSWER_ROT = 0.0        # --answer_rotate D: random rotation of the digits in [-D, D] degrees
ANSWER_JITTER = 0       # --answer_jitter P: random horizontal shift in [-P, P] px (vertical half of it)
ANSWER_VARIANT = 0      # --answer_variant k: seeds the random draw, so the same problem gets a different draw per variant
LAYOUTS = {"same": lambda expr: f"Q: {expr} = ?", "compact": lambda expr: f"{expr}", "reverse": lambda expr: f"Q: {expr}"}   # reverse: expr already holds "? + ? = c"


def fit_font(lines, canvas, margin):
    """largest font size whose longest line fits canvas - 2*margin."""
    longest = max(lines, key=len)
    size = 8
    while True:
        f = ImageFont.truetype(FONT_PATH, size + 1)
        if f.getlength(longest) > canvas - 2 * margin:
            return size
        size += 1


def _draw_line(d, x, y, text, font):
    """draw text at (x, y); with ADVANCE set, place every character at x + ADVANCE * i instead of the font's own advance"""
    if ADVANCE is None:
        d.text((x, y), text, fill="black", font=font); return
    for i, ch in enumerate(text):
        d.text((x + ADVANCE * i, y), ch, fill="black", font=font)


def _line_width(text, font):
    return font.getlength(text) if ADVANCE is None else ADVANCE * len(text)


def render(q, a, size, canvas, margin, ls):
    """Returns (image, answer_line_box, question_line_box, answer_digits_box); boxes are [x0,y0,x1,y1] text bboxes."""
    font = ImageFont.truetype(FONT_PATH, size)
    img = Image.new("RGB", (canvas, canvas), "white")
    d = ImageDraw.Draw(img)
    qb = d.textbbox((0, 0), q, font=font); ab = d.textbbox((0, 0), a, font=font)
    qh, ah = qb[3] - qb[1], ab[3] - ab[1]
    y_top = (canvas - (qh + ls + ah)) // 2
    _draw_line(d, margin, y_top - qb[1], q, font)
    ay = y_top + qh + ls
    _draw_line(d, margin, ay - ab[1], a, font)
    a_box = [margin + ab[0], ay, margin + (ab[2] if ADVANCE is None else _line_width(a, font)), ay + ah]
    q_box = [margin + qb[0], y_top, margin + (qb[2] if ADVANCE is None else _line_width(q, font)), y_top + qh]
    digits = a[len("A: "):] if a.startswith("A: ") and len(a) > 3 else ""
    if digits and ANSWER_FONTS is not None:                              # HARD one-to-many: random font + rotation + jitter per (problem, variant)
        import random as _r
        rng = _r.Random(f"{q}|{a}|{ANSWER_VARIANT}")
        x_d = margin + _line_width("A: ", font)
        afp, asz = rng.choice(ANSWER_FONTS); afont = ImageFont.truetype(afp, asz)
        ang = rng.uniform(-ANSWER_ROT, ANSWER_ROT); dx = rng.randint(-ANSWER_JITTER, ANSWER_JITTER); dy = rng.randint(-ANSWER_JITTER // 2, ANSWER_JITTER // 2)
        img = Image.new("RGB", (canvas, canvas), "white"); d = ImageDraw.Draw(img)
        _draw_line(d, margin, y_top - qb[1], q, font)
        _draw_line(d, margin, ay - ab[1], "A: ", font)
        db_s = d.textbbox((x_d, ay - ab[1]), digits, font=font)            # where the standard-font digits would sit
        cx, cy = (db_s[0] + db_s[2]) / 2 + dx, (db_s[1] + db_s[3]) / 2 + dy
        tb = ImageDraw.Draw(Image.new("L", (8, 8))).textbbox((0, 0), digits, font=afont); tw, th = tb[2] - tb[0], tb[3] - tb[1]
        layer = Image.new("L", (tw + 48, th + 48), 0); ImageDraw.Draw(layer).text((24 - tb[0], 24 - tb[1]), digits, fill=255, font=afont)
        layer = layer.rotate(ang, resample=Image.BICUBIC, expand=True, fillcolor=0)
        lb = layer.getbbox() or (0, 0, layer.width, layer.height); layer = layer.crop(lb)
        px, py = int(round(cx - layer.width / 2)), int(round(cy - layer.height / 2))
        img.paste((0, 0, 0), (px, py, px + layer.width, py + layer.height), mask=layer)
        d_box = [px, py, px + layer.width, py + layer.height]
        a_box = [a_box[0], min(a_box[1], d_box[1]), max(a_box[2], d_box[2]), max(a_box[3], d_box[3])]
    elif digits and ANSWER_FONT is not None:                             # one-to-many targets: digits in another font, same place
        x_d = margin + _line_width("A: ", font)
        afont = ImageFont.truetype(ANSWER_FONT, ANSWER_FONT_SIZE or size)
        img = Image.new("RGB", (canvas, canvas), "white"); d = ImageDraw.Draw(img)
        _draw_line(d, margin, y_top - qb[1], q, font)
        _draw_line(d, margin, ay - ab[1], "A: ", font)
        db_s = d.textbbox((x_d, ay - ab[1]), digits, font=font)            # where the standard-font digits would sit
        db_f = d.textbbox((0, 0), digits, font=afont)
        y_f = (db_s[1] + db_s[3]) / 2 - (db_f[1] + db_f[3]) / 2              # vertically centre the answer-font digits on them
        d.text((x_d, y_f), digits, fill="black", font=afont)
        db = d.textbbox((x_d, y_f), digits, font=afont); d_box = [db[0], db[1], db[2], db[3]]
        a_box = [a_box[0], min(a_box[1], d_box[1]), max(a_box[2], d_box[2]), max(a_box[3], d_box[3])]
    elif digits:
        x_d = margin + _line_width("A: ", font)
        if ADVANCE is None:
            db = d.textbbox((x_d, ay - ab[1]), digits, font=font); d_box = [db[0], db[1], db[2], db[3]]
        else:
            db = d.textbbox((0, ay - ab[1]), digits[0], font=font)
            d_box = [x_d + db[0], db[1], x_d + ADVANCE * (len(digits) - 1) + db[2], db[3]]
    else:
        d_box = None
    return img, a_box, q_box, d_box


def _work(args):
    r, sp, qf_name, size, canvas, margin, ls, out = args
    q = LAYOUTS[qf_name](r["expr"])
    im_in, _, q_box_in, _ = render(q, "A:", size, canvas, margin, ls)
    im_tg, a_box, q_box, d_box = render(q, f"A: {r['answer']}", size, canvas, margin, ls)
    rel_in, rel_tg = f"{sp}/inputs/{r['id']:06d}.png", f"{sp}/targets/{r['id']:06d}.png"
    im_in.save(os.path.join(out, rel_in)); im_tg.save(os.path.join(out, rel_tg))
    row = dict(r)
    row.update({"split": sp, "input": rel_in, "target": rel_tg, "ans_box": a_box, "q_box": q_box,
                "digits_box": d_box, "layout": qf_name, "canvas": canvas, "font_size": size})
    bad = not (q_box[3] <= canvas // 2 <= a_box[1])
    return row, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="same", choices=sorted(LAYOUTS))
    ap.add_argument("--canvas", type=int, default=256)
    ap.add_argument("--margin", type=int, default=8)
    ap.add_argument("--split", default="all", choices=["all", "train", "test", "gate500"])
    ap.add_argument("--n", type=int, default=0, help="rows per split (0 = all); gate500 forces 500")
    ap.add_argument("--font_size", type=int, default=0, help="0 = auto-fit over the whole dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=os.path.join(SI_DATA, "arithmetic_add_unordered"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--font", default=None, help="TTF path (default DejaVuSansMono-Bold); with --font_tag the output dir gets a _font_<tag> suffix")
    ap.add_argument("--font_tag", default=None)
    ap.add_argument("--advance", type=int, default=None, help="fixed per-character advance in px (e.g. 16 = one token: every glyph at the same sub-token phase)")
    ap.add_argument("--answer_font", default=None, help="TTF for the answer DIGITS only (question and 'A: ' stay in --font); output dir gets _ansfont_<tag>")
    ap.add_argument("--answer_font_size", type=int, default=None, help="size for --answer_font (default: the main font size)")
    ap.add_argument("--ansfont_tag", default=None)
    ap.add_argument("--answer_fonts", default=None, help="comma-separated TTFs: each sample's answer digits use a random one (sizes fit to --answer_height)")
    ap.add_argument("--answer_height", type=int, default=20)
    ap.add_argument("--answer_rotate", type=float, default=0.0)
    ap.add_argument("--answer_jitter", type=int, default=0)
    ap.add_argument("--answer_variant", type=int, default=0)
    a = ap.parse_args()
    global FONT_PATH, ADVANCE, ANSWER_FONT, ANSWER_FONT_SIZE, ANSWER_FONTS, ANSWER_HEIGHT, ANSWER_ROT, ANSWER_JITTER, ANSWER_VARIANT
    if a.answer_fonts:
        def _fit_h(path, h):
            dd = ImageDraw.Draw(Image.new("RGB", (8, 8))); best = 8
            for sz in range(8, 64):
                f_ = ImageFont.truetype(path, sz); bb = [dd.textbbox((0, 0), c, font=f_) for c in "0123456789"]
                if max(b[3] - b[1] for b in bb) <= h: best = sz
                else: break
            return best
        ANSWER_FONTS = [(p_, _fit_h(p_, a.answer_height)) for p_ in a.answer_fonts.split(",")]
        ANSWER_HEIGHT, ANSWER_ROT, ANSWER_JITTER, ANSWER_VARIANT = a.answer_height, a.answer_rotate, a.answer_jitter, a.answer_variant
        print("[answer_fonts]", ANSWER_FONTS, "rot", ANSWER_ROT, "jitter", ANSWER_JITTER, "variant", ANSWER_VARIANT, flush=True)
    if a.font:
        FONT_PATH = a.font
    if a.advance:
        ADVANCE = a.advance
    if a.answer_font:
        ANSWER_FONT, ANSWER_FONT_SIZE = a.answer_font, a.answer_font_size
    out = a.out or os.path.join(SI_DATA, f"arithmetic_add_unordered_sq{a.canvas}_{a.layout}" + (f"_font_{a.font_tag}" if a.font_tag else "") + (f"_ansfont_{a.ansfont_tag}" if a.ansfont_tag else ""))
    qf = LAYOUTS[a.layout]

    # ---- rows per split
    splits = {}
    if a.split == "gate500":
        items = [json.loads(l) for l in open(os.path.join(a.src, "train.jsonl")) if l.strip()]
        splits["train"] = random.Random(a.seed).sample(items, 500)       # EXACTLY recon_gate's gate500 selection
    else:
        for sp in (["train", "test"] if a.split == "all" else [a.split]):
            items = [json.loads(l) for l in open(os.path.join(a.src, f"{sp}.jsonl")) if l.strip()]
            splits[sp] = items[:a.n] if a.n else items

    # ---- font: fit over EVERY line of the whole dataset (both splits) so train/test/gate renders are identical
    all_src = {}
    for sp in ("train", "test"):
        p = os.path.join(a.src, f"{sp}.jsonl")
        if os.path.isfile(p):
            all_src[sp] = [json.loads(l) for l in open(p) if l.strip()]
    lines = [qf(r["expr"]) for rs in all_src.values() for r in rs] + [f"A: {r['answer']}" for rs in all_src.values() for r in rs]
    size = a.font_size or fit_font(lines, a.canvas, a.margin)
    ls = max(4, size // 2)
    half = a.canvas // 2
    font = ImageFont.truetype(FONT_PATH, size)
    db0 = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), "0", font=font)

    n_bad, n_done, dig_union = 0, 0, None
    for sp, rows in splits.items():
        os.makedirs(os.path.join(out, sp, "inputs"), exist_ok=True)
        os.makedirs(os.path.join(out, sp, "targets"), exist_ok=True)
        from multiprocessing import Pool
        jobs = [(r, sp, a.layout, size, a.canvas, a.margin, ls, out) for r in rows]
        with Pool(a.workers) as pool, open(os.path.join(out, f"{sp}.jsonl"), "w") as fh:
            for row, bad in pool.imap(_work, jobs, chunksize=32):        # imap keeps the source order
                n_bad += int(bad)
                fh.write(json.dumps(row) + "\n")
                n_done += 1
                d_box = row["digits_box"]
                if sp == "test" and d_box:
                    dig_union = d_box if dig_union is None else [min(dig_union[0], d_box[0]), min(dig_union[1], d_box[1]),
                                                                 max(dig_union[2], d_box[2]), max(dig_union[3], d_box[3])]
        print(f"[{sp}] {len(rows)} rows -> {out}/{sp}", flush=True)

    # ---- metadata
    src_stats = {}
    p = os.path.join(a.src, "stats.json")
    if os.path.isfile(p):
        src_stats = json.load(open(p))
    meta = {"layout": a.layout, "canvas": a.canvas, "margin": a.margin, "font": FONT_PATH, "font_size": size, "advance": ADVANCE,
            "answer_font": ANSWER_FONT, "answer_font_size": ANSWER_FONT_SIZE, "answer_fonts": ANSWER_FONTS, "answer_rotate": ANSWER_ROT, "answer_jitter": ANSWER_JITTER, "answer_variant": ANSWER_VARIANT,
            "line_spacing": ls, "digit_cell_wh": [round(font.getlength("0"), 1), db0[3] - db0[1]],
            "token_px": a.canvas // 16, "grid": [16, 16], "split_arg": a.split,
            "n_rendered": {sp: len(rows) for sp, rows in splits.items()}, "seed": a.seed, "src": a.src,
            "answer_line_outside_lower_half": n_bad, "source_stats": src_stats}
    json.dump(meta, open(os.path.join(out, "stats.json"), "w"), indent=1)
    if dig_union:                                                        # same keys as the source file, 16x16 grid
        tp = a.canvas // 16
        x0, y0, x1, y1 = [int(math.floor(dig_union[0])), int(dig_union[1]), int(math.ceil(dig_union[2])), int(dig_union[3])]
        cols = [int(max(0, x0 // tp - 1)), int(min(15, (x1 - 1) // tp + 1))]
        rows_ = [int(max(0, y0 // tp - 1)), int(min(15, (y1 - 1) // tp + 1))]
        box = {"pixel_box": [x0, x1, y0, y1], "cols": cols, "rows": rows_, "grid": [16, 16], "margin": 1,
               "n_probe": len(splits.get("test", [])), "jsonl": os.path.join(out, "test.jsonl"),
               "n_tokens": (rows_[1] - rows_[0] + 1) * (cols[1] - cols[0] + 1), "note": "answer DIGITS only, inclusive ranges"}
        json.dump(box, open(os.path.join(out, "_answer_token_box.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "source_stats"}))
    print("wrote", out, f"({n_done} rows)")


if __name__ == "__main__":
    main()
