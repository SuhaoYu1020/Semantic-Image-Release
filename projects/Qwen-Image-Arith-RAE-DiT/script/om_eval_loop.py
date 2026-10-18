"""om_eval_loop.py SPACE -- keep evaluating every runs/mm_om_*_<SPACE> run (snapshots every 800 steps, K = 1 and 24,
200 test questions) until all of them are DONE and evaluated; idempotent through eval_om/summary.jsonl.  Launch inside a
hold with script/run_py_hold.sh (which sources env.sh)."""
import glob, os, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); space = sys.argv[1]; reverse = "--reverse" in sys.argv   # a second loop in reverse order shares the work
PREFIX = os.environ.get("PREFIX", "mm_om"); EVERY = os.environ.get("EVERY", "800")    # PREFIX=mm_om100 EVERY=2250 for the 100-epoch set
while True:
    runs = sorted(glob.glob(os.path.join(HERE, "runs", f"{PREFIX}_*_{space}")), reverse=reverse); done = 0
    for d in runs:
        if os.path.isfile(os.path.join(d, "DONE.json")): done += 1
        subprocess.run([sys.executable, os.path.join(HERE, "om_sched_eval.py"), "--run", os.path.basename(d), "--every", EVERY, "--n_test", "200", "--ks", "1", "24"], cwd=HERE, check=False)
    print(f"[loop {time.strftime('%H:%M:%S')}] {space}: {done}/{len(runs)} runs done", flush=True)
    if len(runs) >= 7 and done == len(runs):
        print("[loop] all arms done and evaluated", flush=True); break
    time.sleep(900)
