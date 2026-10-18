"""Speed of learning to COMPUTE at the model's own training sigma: blank-canvas one-step accuracy at every k-th checkpoint."""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); run, sigma = sys.argv[1], sys.argv[2]; every = int(sys.argv[3]) if len(sys.argv) > 3 else 400
source = sys.argv[4] if len(sys.argv) > 4 else "blank"; ROOT = os.path.join(HERE, "eval", "result_gate4", "own_sigma_sweep" + ("" if source == "blank" else f"_{source}"))   # 4th arg: blank | wrong (counterfactual canvas)
steps = sorted(int(d.split("_")[1]) for d in os.listdir(os.path.join(HERE, "runs", run)) if d.startswith("step_"))
for st in [s for s in steps if s % every == 0] + ["final"]:
    tag = f"step_{st}" if st != "final" else "final"
    if os.path.isfile(os.path.join(ROOT, f"{run}@{tag}__{source}", "results.json")): continue   # already swept (resumable after a crash)
    subprocess.run([sys.executable, os.path.join(HERE, "eval_x0_sigma_curve.py"), "--run", run, "--ckpt", tag, "--source", source, "--sigmas", sigma, "1.0", "--n_test", "200", "--out_root", ROOT], check=False, cwd=HERE)
