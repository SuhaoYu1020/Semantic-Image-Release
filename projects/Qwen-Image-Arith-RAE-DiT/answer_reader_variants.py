"""Can a better reader recover the answer from VAE answer-box latents in the hard one-to-many task?
Readers (all on the flattened 24 answer-box tokens, standardised):
  slot_ridge     3 digit-slot ridge heads (the existing reader)
  ovr199_ridge   one ridge detector PER ANSWER (199 classes), arg-max  (= "one extractor per answer, take the max")
  knn1 / knn5    nearest exemplar(s) by cosine in the bank of training latents, majority vote
Training sets:
  var   the 8 rendered variants of each training problem (random font / rotation / jitter)
  mean  the per-problem MEAN latent of the 8 variants (= what a converged one-step model at sigma 1 outputs)
Evaluated on: held-out GT variants, held-out GT means, the standard-font GT, and the one-step (K = 1) outputs of trained models.
  python answer_reader_variants.py --arm vae --n_prob 2000 --apply runs/mm_om_s15_vae/eval_om/final_k1 runs/mm_om_s15to1_50_vae/eval_om/final_k1
-> eval/result_gate4/fidelity/reader_variants_<arm>.json
"""
import argparse, json, os, sys, random, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from PIL import Image
from sq_codecs import SqCodec, DEFAULT_DATA, to_pix01, SI_DATA
from intervene import box_index, answer_slots, BLANK
from latent_answer_probe import fit_ridge, predict, slots_to_str, encode_paths
DEVICE = "cuda"; HARD = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same_anshard")
VAR = [os.path.join(SI_DATA, f"arithmetic_add_unordered_sq256_same_ansfont_hard_k{k}") for k in range(8)]
ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--stats_path", default=None); ap.add_argument("--n_prob", type=int, default=2000)
ap.add_argument("--n_test", type=int, default=500); ap.add_argument("--lam", type=float, default=10.0); ap.add_argument("--apply", nargs="*", default=[]); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
t0 = time.time()
codec = SqCodec(a.arm, stats_path=a.stats_path, with_decoder=False, dtype=torch.bfloat16).to(DEVICE).eval(); box = box_index(DEFAULT_DATA, codec.grid).to(DEVICE)
# ---- training problems: id -> 8 variant target paths
rows = [json.loads(l) for l in open(os.path.join(VAR[0], "train.jsonl")) if l.strip()]; random.Random(a.seed).shuffle(rows); rows = rows[:a.n_prob]
ans_tr = [str(r["answer"]) for r in rows]
Xv = [encode_paths(codec, [os.path.join(VAR[k], r["target"]) for r in rows], box) for k in range(8)]          # 8 x (n_prob, d)
X_var = torch.cat(Xv); y_var = ans_tr * 8; X_mean = torch.stack(Xv).mean(0); y_mean = ans_tr
print(f"[enc] train {a.arm}: {X_var.shape[0]} variant latents, {X_mean.shape[0]} means, d={X_var.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
# ---- held-out test: standard GT, 8 variants, their mean
trows = [json.loads(l) for l in open(os.path.join(DEFAULT_DATA, "test.jsonl")) if l.strip()][:a.n_test]; ans_te = [str(r["answer"]) for r in trows]
Tv = [encode_paths(codec, [os.path.join(VAR[k], "test", "targets", f"{r['id']:06d}.png") for r in trows], box) for k in range(8)]
T_std = encode_paths(codec, [r["target"] if os.path.isabs(r["target"]) else os.path.join(DEFAULT_DATA, r["target"]) for r in trows], box)
T_var = torch.cat(Tv); yT_var = ans_te * 8; T_mean = torch.stack(Tv).mean(0)
print(f"[enc] test: {T_var.shape[0]} variants, {T_mean.shape[0]} means ({time.time()-t0:.0f}s)", flush=True)
ANS = sorted(set(ans_tr) | set(ans_te), key=int); aidx = {s: i for i, s in enumerate(ANS)}


def make_readers(X, y, tag):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd; R = {}
    S = torch.tensor([answer_slots(s) for s in y]); Ws = [fit_ridge(Xs, torch.nn.functional.one_hot(S[:, k], 11).float(), a.lam) for k in range(3)]
    R[f"slot_ridge@{tag}"] = lambda Z, Ws=Ws: [slots_to_str(p) for p in torch.stack([predict(W, (Z - mu) / sd) for W in Ws], 1)]
    Y = torch.nn.functional.one_hot(torch.tensor([aidx[s] for s in y]), len(ANS)).float(); W199 = fit_ridge(Xs, Y, a.lam)
    R[f"ovr199_ridge@{tag}"] = lambda Z, W=W199: [ANS[i] for i in predict(W, (Z - mu) / sd)]
    B = torch.nn.functional.normalize(Xs, dim=1).to(DEVICE); yb = torch.tensor([aidx[s] for s in y], device=DEVICE)
    def knn(Z, k):
        Q = torch.nn.functional.normalize((Z - mu) / sd, dim=1).to(DEVICE); out = []
        for i in range(0, Q.shape[0], 256):
            sim = Q[i:i + 256] @ B.t(); nn_ = sim.topk(k, dim=1).indices; lab = yb[nn_]
            if k == 1: out += [ANS[int(v)] for v in lab[:, 0]]
            else:
                for row in lab: out.append(ANS[int(torch.mode(row).values)])
        return out
    R[f"knn1@{tag}"] = lambda Z: knn(Z, 1); R[f"knn5@{tag}"] = lambda Z: knn(Z, 5)
    return R


readers = {}; readers.update(make_readers(X_var, y_var, "var")); readers.update(make_readers(X_mean, y_mean, "mean")); print(f"[fit] {len(readers)} readers ({time.time()-t0:.0f}s)", flush=True)
evals = {"gt_variants": (T_var, yT_var), "gt_mean_of_8": (T_mean, ans_te), "gt_standard": (T_std, ans_te)}
for d in a.apply:
    L = torch.load(os.path.join(d, "latents.pt")); rep = json.load(open(os.path.join(d, "report.json"))); gt = {("test", r["id"]): r["gt"] for r in rep["test_rows"]}
    Z = L["x"].float()[:, box.cpu()].reshape(L["x"].shape[0], -1); evals[f"model:{d.split('runs/')[-1]}"] = (Z, [gt[tuple(i)] for i in L["ids"]])
res = {}
for ename, (Z, y) in evals.items():
    res[ename] = {rname: float(np.mean([p == g for p, g in zip(f(Z), y)])) for rname, f in readers.items()}
    print(f"[{ename}] " + "  ".join(f"{r}:{100*v:.1f}" for r, v in res[ename].items()), flush=True)
out = os.path.join(HERE, "eval", "result_gate4", "fidelity", f"reader_variants_{a.arm}.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump({"arm": a.arm, "n_prob": a.n_prob, "n_test": a.n_test, "results": res}, open(out, "w"), indent=1); print("wrote", out, f"({time.time()-t0:.0f}s)")
