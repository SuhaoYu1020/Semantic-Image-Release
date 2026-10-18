"""Codec round-trip references for the glyph-fidelity calibration (GPU).
For the 200 test questions of the curriculum evals: every hard variant GT (k0..k7) and the standard GT through encode->decode of the
arm's codec (= the sharpest output a perfect model could produce), plus negatives through the same codec: the pixel mean of the 8
variants, the pixel mean of 2, and the decoded LATENT mean of the 8 (what a one-step regressor converges to).
  python om_fidelity_codec_gt.py --arm siglip2   -> eval/result_gate4/fidelity/codec_gt/<arm>/{k0..k7,std,mean8,mean2,latmean8}/<id>.png
"""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sq_codecs import SqCodec, SI_DATA, DEFAULT_DATA, to_pix01  # noqa: E402
HARD = [os.path.join(SI_DATA, f"arithmetic_add_unordered_sq256_same_ansfont_hard_k{k}") for k in range(8)]
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--ref_run", default=None); ap.add_argument("--bs", type=int, default=16); a = ap.parse_args()
ref = a.ref_run or f"mm_om_s15_{a.arm}"; ra = json.load(open(os.path.join(HERE, "runs", ref, "args.json")))
rows = json.load(open(os.path.join(HERE, "runs", ref, "eval_om", "final_k24", "report.json")))["test_rows"]; ids = [r["id"] for r in rows]
out_root = os.path.join(HERE, "eval", "result_gate4", "fidelity", "codec_gt", a.arm)
codec = SqCodec(a.arm, stats_path=ra.get("stats_path"), with_decoder=True, dtype=torch.bfloat16).to("cuda").eval()
rng = np.random.RandomState(0)


def load(p): return Image.open(p).convert("RGB")


@torch.no_grad()
def roundtrip(pils):
    outs = []
    for i in range(0, len(pils), a.bs):
        z = codec.encode(to_pix01(pils[i:i + a.bs]).to("cuda")); outs += codec.to_pil(codec.decode(z))
    return outs


@torch.no_grad()
def decode_latent_mean(groups):
    outs = []
    for i in range(0, len(groups), a.bs):
        zs = [codec.encode(to_pix01(g).to("cuda")).float().mean(0, keepdim=True) for g in groups[i:i + a.bs]]
        outs += codec.to_pil(codec.decode(torch.cat(zs, 0)))
    return outs


def save(tag, pils):
    d = os.path.join(out_root, tag); os.makedirs(d, exist_ok=True)
    for i, im in zip(ids, pils): im.save(os.path.join(d, f"{i:06d}.png"))
    print(f"[codec_gt] {a.arm} {tag}: {len(pils)} saved", flush=True)


var = [[load(os.path.join(HARD[k], "test", "targets", f"{i:06d}.png")) for i in ids] for k in range(8)]
for k in range(8): save(f"k{k}", roundtrip(var[k]))
save("std", roundtrip([load(os.path.join(DEFAULT_DATA, "test", "targets", f"{i:06d}.png")) for i in ids]))
mean8 = [Image.fromarray(np.mean([np.asarray(var[k][j], np.float32) for k in range(8)], 0).astype(np.uint8)) for j in range(len(ids))]
save("mean8", roundtrip(mean8))
mean2 = []
for j in range(len(ids)):
    p, q = rng.choice(8, 2, replace=False); mean2.append(Image.fromarray(((np.asarray(var[p][j], np.float32) + np.asarray(var[q][j], np.float32)) / 2).astype(np.uint8)))
save("mean2", roundtrip(mean2))
save("latmean8", decode_latent_mean([[var[k][j] for k in range(8)] for j in range(len(ids))]))
print("done", out_root, flush=True)
