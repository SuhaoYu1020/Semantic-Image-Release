"""Fit the latent answer readers (std + hard) for the given arms and run their transfer checks (compute node: no /tmp)."""
import subprocess, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for arm in (sys.argv[1:] or ["vae", "siglip2"]):
    for src in ("std", "hard"):
        subprocess.run([sys.executable, os.path.join(HERE, "latent_answer_probe.py"), "--arm", arm, "--fit", src, "--check"], cwd=HERE, check=False)
