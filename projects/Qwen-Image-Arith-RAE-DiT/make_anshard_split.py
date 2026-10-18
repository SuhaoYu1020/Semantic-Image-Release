"""make_anshard_split.py -- HARD one-to-many targets: every training problem x K variants whose answer digits differ in font
(8 fonts), rotation (+-30 deg) and position (+-10 px), rendered by make_square_gate_set.py --answer_fonts ... --answer_variant k.
Question image identical across variants (standard font).  Their pixel mean is an illegible smear.  test.jsonl = standard test."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sq_codecs import SI_DATA
K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same")
VAR = {f"k{k}": os.path.join(SI_DATA, f"arithmetic_add_unordered_sq256_same_ansfont_hard_k{k}") for k in range(K)}
OUT = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same_anshard"); os.makedirs(OUT, exist_ok=True)
rows = {t: [json.loads(l) for l in open(os.path.join(d, "train.jsonl")) if l.strip()] for t, d in VAR.items()}
base = [json.loads(l) for l in open(os.path.join(BASE, "train.jsonl")) if l.strip()]
n = len(base); assert all(len(r) == n for r in rows.values()), {t: len(r) for t, r in rows.items()}
with open(os.path.join(OUT, "train.jsonl"), "w") as f:
    for i in range(n):
        for t in VAR:
            r = dict(rows[t][i]); assert r["expr"] == base[i]["expr"] and r["answer"] == base[i]["answer"]
            r["input"] = os.path.join(BASE, base[i]["input"]); r["target"] = os.path.join(VAR[t], r["target"]); r["variant"] = t
            f.write(json.dumps(r) + "\n")
with open(os.path.join(OUT, "test.jsonl"), "w") as f:
    for l in open(os.path.join(BASE, "test.jsonl")):
        if l.strip():
            r = json.loads(l); r["input"] = os.path.join(BASE, r["input"]); r["target"] = os.path.join(BASE, r["target"]); f.write(json.dumps(r) + "\n")
boxes = [json.load(open(os.path.join(d, "_answer_token_box.json"))) for d in VAR.values()] + [json.load(open(os.path.join(BASE, "_answer_token_box.json")))]
pb = [min(b["pixel_box"][0] for b in boxes), max(b["pixel_box"][1] for b in boxes), min(b["pixel_box"][2] for b in boxes), max(b["pixel_box"][3] for b in boxes)]
cols = [min(b["cols"][0] for b in boxes), max(b["cols"][1] for b in boxes)]; rws = [min(b["rows"][0] for b in boxes), max(b["rows"][1] for b in boxes)]
box = dict(boxes[0]); box.update({"pixel_box": pb, "cols": cols, "rows": rws, "n_tokens": (rws[1] - rws[0] + 1) * (cols[1] - cols[0] + 1), "jsonl": os.path.join(OUT, "test.jsonl"), "note": "UNION over the hard variants and the standard render"})
json.dump(box, open(os.path.join(OUT, "_answer_token_box.json"), "w"), indent=1)
st = json.load(open(os.path.join(BASE, "stats.json"))); st.update({"anshard": VAR, "k": K, "note": "hard one-to-many: K variants per problem (random font / rotation / jitter of the answer digits); test = standard"})
json.dump(st, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
print(OUT, "train rows", n * K, "box cols", cols, "rows", rws, box["n_tokens"], "test", sum(1 for l in open(os.path.join(OUT, "test.jsonl")) if l.strip()))
