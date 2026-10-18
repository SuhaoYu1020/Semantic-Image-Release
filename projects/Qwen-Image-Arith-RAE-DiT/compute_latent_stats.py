"""compute_latent_stats.py -- per-channel mean/std of each codec's INNER latent on OUR square train renders, so
SqCodec can hand the DiT a unit-variance latent in every arm.

  python compute_latent_stats.py --arm siglip2 [--n 2000] [--data .../arithmetic_add_unordered_sq256_same]
  -> stats/sq256_same/<arm>.pt  {"mean": (D,), "std": (D,), "n_tokens", "arm", "inner_latent", "data", "global_std"}

n problems x 2 frames (input + target) x 256 tokens; n=2000 -> ~1M tokens per channel.  Also prints the global std
before/after and the per-channel std quantiles, which is the number that tells you whether the eps-tolerance curve of
the gate (measured in RAW/inner units) translates 1:1 to the DiT's standardised space.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
from PIL import Image

from sq_codecs import SqCodec, DEFAULT_DATA, DEFAULT_STATS_DIR, to_pix01, ARMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=2000, help="train problems (both frames each); 0 = all")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out_dir", default=DEFAULT_STATS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16 if dev.type == "cuda" else torch.float32
    os.makedirs(a.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(a.data, "train.jsonl")) if l.strip()]
    if a.n and a.n < len(rows):
        import random
        rows = random.Random(a.seed).sample(rows, a.n)
    paths = [os.path.join(a.data, r[k]) for r in rows for k in ("input", "target")]
    t0 = time.time()
    codec = SqCodec(a.arm, stats_path=None, with_decoder=False, dtype=dt).to(dev)
    print(f"[stats] {a.arm} codec {codec.latent_id} loaded in {time.time()-t0:.1f}s; {len(paths)} frames", flush=True)
    D = codec.dim
    s1 = torch.zeros(D, dtype=torch.float64, device=dev)
    s2 = torch.zeros(D, dtype=torch.float64, device=dev)
    n = 0
    tok_norm_sq = 0.0
    for i in range(0, len(paths), a.batch):
        pix = to_pix01([Image.open(p) for p in paths[i:i + a.batch]]).to(dev)
        z = codec.encode_inner(pix).double()                          # (B,256,D)
        z2 = z.reshape(-1, D)
        s1 += z2.sum(0); s2 += (z2 * z2).sum(0); n += z2.shape[0]
        tok_norm_sq += float((z2 * z2).sum())
        if (i // a.batch) % 20 == 0:
            print(f"[stats] {i + pix.shape[0]}/{len(paths)}", flush=True)
    mean = s1 / n
    var = (s2 / n - mean * mean).clamp_min(0)
    std = var.sqrt()
    g_std = float((var.mean()) ** 0.5)                                # global (pooled) std of the inner latent
    out = {"mean": mean.float().cpu(), "std": std.float().cpu(), "n_tokens": n, "arm": a.arm,
           "inner_latent": codec.inner.latent_id, "data": a.data, "n_problems": len(rows), "global_std": g_std,
           "global_mean": float(mean.mean()), "rms": float((tok_norm_sq / (n * D)) ** 0.5)}
    p = os.path.join(a.out_dir, f"{a.arm}.pt")
    torch.save(out, p)
    q = torch.quantile(std.float().cpu(), torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
    print(f"[stats] {a.arm}: D={D} tokens={n} global_std={g_std:.4f} global_mean={float(mean.mean()):+.4f} "
          f"channel std p0/p5/p50/p95/p100 = {' / '.join(f'{v:.3f}' for v in q.tolist())}; "
          f"channels with std<0.05: {int((std < 0.05).sum())} -> {p}", flush=True)


if __name__ == "__main__":
    main()
