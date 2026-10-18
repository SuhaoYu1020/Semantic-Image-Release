"""Linear answer reader on answer-box latents ("does the representation already hold the answer?").
Fit: ridge classifiers for the three answer-digit slots (0-9 + BLANK) on encoder latents of rendered TARGET images:
  --fit std   standard-font targets (data/arithmetic_add_unordered_sq256_same train)
  --fit hard  the 8 hard one-to-many variants (anshard train: random font / rotation / jitter)
Apply: --apply <render_eval out_dir with latents.pt> --probe cache/ansprobe/<arm>_<std|hard>.pt -> accuracy (all slots right)
Also --check: read the probe on ground-truth test latents (standard targets and each hard variant) = transfer baseline."""
import argparse, json, os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
from PIL import Image
from sq_codecs import SqCodec, DEFAULT_DATA, to_pix01, SI_DATA
from intervene import box_index, answer_slots, BLANK
DEVICE = "cuda"; HARD = os.path.join(SI_DATA, "arithmetic_add_unordered_sq256_same_anshard"); OUT = os.path.join(HERE, "cache", "ansprobe")

def load_codec(arm, stats_path=None):
    return SqCodec(arm, stats_path=stats_path, with_decoder=False, dtype=torch.bfloat16).to(DEVICE).eval()

@torch.no_grad()
def encode_paths(codec, paths, box, bs=64):
    feats = []
    for i in range(0, len(paths), bs):
        pils = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        z = codec.encode(to_pix01(pils).to(DEVICE)).float()
        feats.append(z[:, box].reshape(z.shape[0], -1).cpu())
    return torch.cat(feats)

def fit_ridge(X, Y, lam):
    """X (n,d) standardised, Y (n,k) one-hot; primal if d<=n else dual."""
    n, d = X.shape; Xb = torch.cat([X, torch.ones(n, 1)], 1)
    if d + 1 <= n:
        A = Xb.t() @ Xb + lam * torch.eye(d + 1); return torch.linalg.solve(A, Xb.t() @ Y)
    K = Xb @ Xb.t() + lam * torch.eye(n); return Xb.t() @ torch.linalg.solve(K, Y)

def predict(W, X):
    return (torch.cat([X, torch.ones(X.shape[0], 1)], 1) @ W).argmax(1)

def slots_to_str(s):
    out = ""
    for v in s:
        if v == BLANK: break
        out += str(int(v))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--stats_path", default=None)
    ap.add_argument("--fit", choices=["std", "hard"]); ap.add_argument("--n_fit", type=int, default=6000); ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--apply", default=None, help="render_eval out_dir containing latents.pt + report.json"); ap.add_argument("--probe", default=None)
    ap.add_argument("--check", action="store_true", help="read the probe on ground-truth test latents (std + 8 hard variants)")
    ap.add_argument("--n_check", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp", action="store_true", help="2-layer MLP reader (hidden 1024, 40 epochs Adam) instead of ridge; saved as <arm>_<src>_mlp.pt")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); codec = box = None
    if a.fit or a.check:
        codec = load_codec(a.arm, a.stats_path); box = box_index(DEFAULT_DATA, codec.grid).to(DEVICE)
    if a.fit and os.path.isfile(os.path.join(OUT, f"{a.arm}_{a.fit}" + ("_mlp" if a.mlp else "") + ".pt")) and not os.environ.get("REFIT"):
        print(f"[probe fit] {a.arm} {a.fit}: cached", flush=True)
    elif a.fit:
        root = DEFAULT_DATA if a.fit == "std" else HARD
        rows = [json.loads(l) for l in open(os.path.join(root, "train.jsonl")) if l.strip()]; random.Random(a.seed).shuffle(rows); rows = rows[:a.n_fit]
        X = encode_paths(codec, [r["target"] if os.path.isabs(r["target"]) else os.path.join(root, r["target"]) for r in rows], box)
        S = torch.tensor([answer_slots(r["answer"]) for r in rows]); mu, sd = X.mean(0), X.std(0) + 1e-6; Xs = (X - mu) / sd
        n = len(rows); ntr = int(0.9 * n); Ws = []; accs = []
        if a.mlp:
            torch.manual_seed(a.seed); d = Xs.shape[1]
            net = torch.nn.Sequential(torch.nn.Linear(d, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(1024, 33)).to(DEVICE)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2); Xg, Sg = Xs.to(DEVICE), S.to(DEVICE)
            for ep in range(40):
                net.train(); perm = torch.randperm(ntr, device=DEVICE)
                for i in range(0, ntr, 128):
                    idx = perm[i:i + 128]; out = net(Xg[idx]).view(-1, 3, 11); loss = sum(torch.nn.functional.cross_entropy(out[:, k], Sg[idx, k]) for k in range(3))
                    opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): P = net(Xg[ntr:]).view(-1, 3, 11).argmax(-1).cpu()
            accs = [float((P[:, k] == S[ntr:, k]).float().mean()) for k in range(3)]; full = float((P == S[ntr:]).all(1).float().mean())
            torch.save({"arm": a.arm, "source": a.fit, "mlp": {k: v.cpu() for k, v in net.state_dict().items()}, "mu": mu, "sd": sd, "box": box.cpu(), "n_fit": n, "holdout_slot_acc": accs, "holdout_acc": full}, os.path.join(OUT, f"{a.arm}_{a.fit}_mlp.pt"))
            print(f"[probe fit MLP] {a.arm} {a.fit}: n={n} d={d} holdout slot acc {['%.3f' % v for v in accs]} all-slots {full:.3f} -> cache/ansprobe/{a.arm}_{a.fit}_mlp.pt", flush=True)
        else:
            for k in range(3):
                Y = torch.nn.functional.one_hot(S[:, k], 11).float(); W = fit_ridge(Xs[:ntr], Y[:ntr], a.lam); Ws.append(W)
                accs.append(float((predict(W, Xs[ntr:]) == S[ntr:, k]).float().mean()))
            P = torch.stack([predict(W, Xs[ntr:]) for W in Ws], 1); full = float((P == S[ntr:]).all(1).float().mean())
            torch.save({"arm": a.arm, "source": a.fit, "W": Ws, "mu": mu, "sd": sd, "box": box.cpu(), "lam": a.lam, "n_fit": n, "holdout_slot_acc": accs, "holdout_acc": full},
                       os.path.join(OUT, f"{a.arm}_{a.fit}.pt"))
            print(f"[probe fit] {a.arm} {a.fit}: n={n} d={X.shape[1]} holdout slot acc {['%.3f' % v for v in accs]} all-slots {full:.3f} -> cache/ansprobe/{a.arm}_{a.fit}.pt", flush=True)
    if a.check or a.apply:
        pr = torch.load(a.probe or os.path.join(OUT, f"{a.arm}_{a.fit or 'std'}" + ("_mlp" if a.mlp else "") + ".pt")); mu, sd = pr["mu"], pr["sd"]
        if "mlp" in pr:
            d = pr["mu"].numel(); net = torch.nn.Sequential(torch.nn.Linear(d, 1024), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(1024, 33)); net.load_state_dict(pr["mlp"]); net.eval()
            def read(X):
                with torch.no_grad(): P = net(((X - mu) / sd)).view(-1, 3, 11).argmax(-1)
                return [slots_to_str(p) for p in P]
        else:
            Ws = pr["W"]
            def read(X):
                Xs = (X - mu) / sd; P = torch.stack([predict(W, Xs) for W in Ws], 1); return [slots_to_str(p) for p in P]
    if a.check:
        rows = [json.loads(l) for l in open(os.path.join(DEFAULT_DATA, "test.jsonl")) if l.strip()][:a.n_check]; gts = [str(r["answer"]) for r in rows]
        res = {}
        X = encode_paths(codec, [r["target"] if os.path.isabs(r["target"]) else os.path.join(DEFAULT_DATA, r["target"]) for r in rows], box); res["standard"] = float(np.mean([p == g for p, g in zip(read(X), gts)]))
        for k in range(8):
            vd = os.path.join(SI_DATA, f"arithmetic_add_unordered_sq256_same_ansfont_hard_k{k}", "test", "targets")
            paths = [os.path.join(vd, f"{r['id']:06d}.png") for r in rows]
            if all(os.path.isfile(p) for p in paths[:3]):
                X = encode_paths(codec, paths, box); res[f"hard_k{k}"] = float(np.mean([p == g for p, g in zip(read(X), gts)]))
        print(f"[probe check] {a.arm} probe={pr['source']}: " + " ".join(f"{k}:{100*v:.1f}" for k, v in res.items()), flush=True)
        json.dump(res, open(os.path.join(OUT, f"check_{a.arm}_{pr['source']}" + ("_mlp" if "mlp" in pr else "") + ".json"), "w"), indent=1)
    if a.apply:
        L = torch.load(os.path.join(a.apply, "latents.pt")); rep = json.load(open(os.path.join(a.apply, "report.json")))
        gt = {("test", r["id"]): r["gt"] for r in rep["test_rows"]}; gt.update({("train", r["id"]): r["gt"] for r in rep.get("train_rows", [])})
        X = L["x"].float()[:, pr["box"]].reshape(L["x"].shape[0], -1); preds = read(X)
        ok = [p == gt[tuple(i)] for p, i in zip(preds, L["ids"])]; acc = float(np.mean(ok))
        out = {"probe": pr["source"], "arm": a.arm, "n": len(ok), "acc": acc, "steps": L.get("steps"), "preds": preds}
        json.dump(out, open(os.path.join(a.apply, f"probe_{pr['source']}.json"), "w")); print(f"[probe apply] {a.apply} probe={pr['source']} K={L.get('steps')}: {100*acc:.1f}% (n={len(ok)})", flush=True)

if __name__ == "__main__":
    main()
