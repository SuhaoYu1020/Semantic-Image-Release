"""Shim for the export: the path constants and checkpoint helpers that train_i2i_sq_dit.py / recon_gate.py import from
the original Qwen-Image text-to-image trainer (train_t2i_arith_mse.py).  Everything below is copied verbatim from it."""
import glob, json, os, re
import torch
import os as _os
def _si_root():
    d = _os.environ.get("SI_ROOT")
    if d:
        return d
    d = _os.path.dirname(_os.path.abspath(__file__))
    while d != "/" and not _os.path.exists(_os.path.join(d, "env.sh")):
        d = _os.path.dirname(d)
    return _os.path.dirname(d) if d != "/" else _os.getcwd()
def _si_repo():
    d = _os.path.dirname(_os.path.abspath(__file__))
    while d != "/" and not _os.path.exists(_os.path.join(d, "env.sh")):
        d = _os.path.dirname(d)
    return d if d != "/" else _os.getcwd()
SI_ROOT  = _si_root()
SI_REPO  = _os.environ.get("SI_REPO")  or _si_repo()
SI_DATA  = _os.environ.get("SI_DATA")  or _os.path.join(SI_ROOT, "data")
SI_CKPTS = _os.environ.get("SI_CKPTS") or _os.path.join(SI_ROOT, "ckpts")
SI_RUNS  = _os.environ.get("SI_RUNS")  or _os.path.join(SI_ROOT, "runs")
# --- end portable paths ------------------------------------------------------


def pack_latents(latents):
    """5D (B, C, T=1, H, W) -> packed (B, (H/2)*(W/2), C*4)."""
    B, C, T, H, W = latents.shape
    assert T == 1, f"expected temporal=1, got {T}"
    latents = latents.view(B, C, T, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 3, 5, 1, 4, 6)
    latents = latents.reshape(B, T * (H // 2) * (W // 2), C * 4)
    return latents



def _is_complete_ckpt(d):
    """A step_* dir is loadable iff it holds transformer/ + meta.json -- and, unless
    it is a --freeze_te (DiT-only) ckpt, text_encoder/. meta.json is written LAST."""
    meta_path = os.path.join(d, "meta.json")
    if not (os.path.isdir(os.path.join(d, "transformer")) and os.path.exists(meta_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    if meta.get("adapter") and not os.path.isfile(os.path.join(d, "adapter.pt")):
        return False
    if meta.get("freeze_te"):
        return True
    return os.path.isdir(os.path.join(d, "text_encoder"))


def _complete_ckpts_sorted(out_dir):
    out = []
    for d in glob.glob(os.path.join(out_dir, "step_*")):
        m = re.search(r"step_(\d+)$", d)
        if m and os.path.isdir(d) and _is_complete_ckpt(d):
            out.append((int(m.group(1)), d))
    out.sort()
    return out


def _find_latest_ckpt(out_dir):
    cks = _complete_ckpts_sorted(out_dir)
    return (cks[-1][1], cks[-1][0]) if cks else (None, 0)


def prune_checkpoints(out_dir, keep_last):
    """Keep only the keep_last newest COMPLETE step_* ckpts. keep_last<=0 keeps all."""
    if keep_last <= 0:
        return
    import shutil
    for _, d in _complete_ckpts_sorted(out_dir)[:-keep_last]:
        shutil.rmtree(d, ignore_errors=True)
        print(f"[prune] removed old checkpoint {d}", flush=True)


# ----------------------------- args -----------------------------------------



from diffusers.models.modeling_utils import ModelMixin as _DiffusersModelMixin
_orig_mm_getattr = _DiffusersModelMixin.__getattr__


def _mm_getattr_no_dunder(self, name):
    if name.startswith("__"):
        return torch.nn.Module.__getattr__(self, name)
    return _orig_mm_getattr(self, name)


_DiffusersModelMixin.__getattr__ = _mm_getattr_no_dunder


# ---------------------------------------------------------------------------
# FSDP-safe checkpoint helpers (no-ops in behaviour under DDP / single process)
# ---------------------------------------------------------------------------
def _full_state(accelerator, model):
    """Full state dict of a prepared model. Under FSDP this is a COLLECTIVE: every rank
    must call it; rank 0 receives the gathered, CPU-offloaded dict, other ranks an empty
    one. Under DDP / single process it is simply the unwrapped model's state_dict()."""
    state = accelerator.get_state_dict(model)
    # Under FSDP the gathered dict is the fp32 master copy, on rank 0's CPU. Cast it to
    # the run's compute dtype RIGHT HERE (before any other gather) so the host-RAM peak
    # is one model's fp32 copy, not the sum of all of them (DiT 82 GB + TE 30 GB killed
    # a 116 GB node). Non-floating entries and non-FSDP runs pass through unchanged.
    want = torch.bfloat16 if accelerator.mixed_precision == "bf16" else None
    if want is not None and state and accelerator.is_main_process:
        for k in list(state.keys()):
            v = state[k]
            if torch.is_tensor(v) and v.is_floating_point() and v.dtype != want:
                state[k] = v.to(want)
                del v
    return state


def _save_hf(accelerator, model, path, state):
    """save_pretrained() on rank 0 from a pre-gathered full state dict. diffusers'
    ModelMixin.save_pretrained has no state_dict= argument, so the unwrapped module's
    state_dict() is redirected to the gathered dict for the duration of the call
    (identical output under DDP). transformers' PreTrainedModel takes state_dict=."""
    if not accelerator.is_main_process:
        return
    m = accelerator.unwrap_model(model)
    # FSDP's gathered FULL_STATE_DICT comes back in fp32; write in the model's own dtype
    # (bf16 here) so checkpoints match the single-process ones in size and precision.
    # (under FSDP accelerate keeps fp32 MASTER weights for bf16 mixed precision, so the
    #  live param dtype would say fp32; the run's precision setting is the right guide)
    want = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(str(accelerator.mixed_precision), None)
    if want is not None and any(v.is_floating_point() and v.dtype != want for v in state.values()):
        state = {k: (v.to(want) if v.is_floating_point() and v.dtype != want else v)
                 for k, v in state.items()}
    if hasattr(m, "save_config"):  # diffusers ModelMixin
        m.state_dict = lambda *a, **k: state
        try:
            m.save_pretrained(path, safe_serialization=True)
        finally:
            del m.state_dict
    else:  # transformers PreTrainedModel
        m.save_pretrained(path, state_dict=state, safe_serialization=True)


