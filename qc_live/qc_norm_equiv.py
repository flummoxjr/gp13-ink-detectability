"""QC test 1: normalization equivalence.

Runs the SAME checkpoint locally on 256^3 tiles that lie fully inside the CLI
reference region (smoke_1203_prob.npy, origin (7281,7007,11615), 672^3), with
BOTH normalizations:
  A) screen_band.py's: percentile [1,99] of NONZERO voxels
  B) villa CLI's percentile_minmax: percentile [1,99] of ALL voxels
then compares each single-tile output against the CLI's blended reference over
the same coordinates (avg-pooled 4x to the screen's probL2 grid):
Pearson r + mean abs diff + threshold stats.
"""
import json
import os
import sys

import numpy as np

STUBS = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\b3441997-0118-49b2-8364-cbdf28fc6397\scratchpad\stubs"
VILLA = r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src"
sys.path.insert(0, STUBS)
sys.path.insert(0, VILLA)

import torch  # noqa: E402
import zarr  # noqa: E402
import fsspec  # noqa: E402
from vesuvius.models.build.build_network_from_config import NetworkFromConfig  # noqa: E402


# --- vendored from vesuvius/models/run/inference.py (importing it pulls nnunetv2) ---
def _tuple_if_sequence(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _normalize_train_py_model_config(checkpoint_data):
    model_config = checkpoint_data.get('model_config')
    if model_config:
        return dict(model_config)
    legacy_config = checkpoint_data.get('config')
    if not isinstance(legacy_config, dict):
        return {}
    model_config = dict(legacy_config.get('model_config') or {})
    if 'patch_size' in legacy_config:
        patch_size = _tuple_if_sequence(legacy_config['patch_size'])
        model_config.setdefault('patch_size', patch_size)
        model_config.setdefault('train_patch_size', patch_size)
    if 'batch_size' in legacy_config:
        model_config.setdefault('batch_size', legacy_config['batch_size'])
        model_config.setdefault('train_batch_size', legacy_config['batch_size'])
    if 'in_channels' in legacy_config:
        model_config.setdefault('in_channels', legacy_config['in_channels'])
    targets = legacy_config.get('targets')
    if targets:
        model_config.setdefault('targets', targets)
    model_name = (legacy_config.get('wandb_run_name')
                  or legacy_config.get('model_name')
                  or legacy_config.get('out_dir'))
    if model_name:
        model_config.setdefault('model_name', str(model_name))
    if 'enable_deep_supervision' in legacy_config:
        model_config.setdefault('enable_deep_supervision', legacy_config['enable_deep_supervision'])
    else:
        model_config.setdefault('enable_deep_supervision', False)
    return model_config


def _legacy_checkpoint_uses_ema_for_inference(checkpoint_data):
    legacy_config = checkpoint_data.get('config')
    if not isinstance(legacy_config, dict):
        return False
    ema_config = legacy_config.get('ema')
    if not isinstance(ema_config, dict):
        return False
    return bool(ema_config.get('validate', False)) and isinstance(checkpoint_data.get('ema_model'), dict)


def _select_train_py_state_dict(checkpoint_data):
    if _legacy_checkpoint_uses_ema_for_inference(checkpoint_data):
        return checkpoint_data['ema_model'], 'ema_model'
    return checkpoint_data.get('model', checkpoint_data), 'model'
# --- end vendored ---

CKPT = r"D:\vesuvius-data\trackD\models\ink3d\ckpt_78k_fullsup.pth"
VOL = "vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
REF = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\smoke_1203_prob.npy"
REF_ORIGIN = (7281, 7007, 11615)
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\qc_live\qc_norm_equiv_result.json"
TILES = [(7424, 7168, 11776), (7680, 7168, 11776)]  # fully inside ref region
TILE = 256


def load_ink3d(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = _normalize_train_py_model_config(ck)

    class Mgr:
        model_config = mc
        targets = mc.get("targets", {})
        train_patch_size = mc.get("train_patch_size", mc.get("patch_size", (256, 256, 256)))
        train_batch_size = mc.get("train_batch_size", mc.get("batch_size", 1))
        in_channels = mc.get("in_channels", 1)
        autoconfigure = mc.get("autoconfigure", False)
        enable_deep_supervision = bool(mc.get("enable_deep_supervision", False))
        model_name = mc.get("model_name", "Model")
        spacing = [1, 1, 1]

    model = NetworkFromConfig(Mgr())
    sd, src = _select_train_py_state_dict(ck)
    prefixes = ("module.", "_orig_mod.")

    def strip(k):
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        return k

    model.load_state_dict({strip(k): v for k, v in sd.items()}, strict=True)
    print("loaded state dict from", src, flush=True)
    return model


def norm_screen(x):
    """screen_band.py: [1,99] of nonzero."""
    if not (x > 0).any():
        return None
    lo, hi = np.percentile(x[x > 0], [1, 99])
    if hi <= lo:
        return None
    return np.clip((x.astype(np.float32) - lo) / (hi - lo), 0, 1)


def norm_villa(x):
    """villa data/volume.py percentile_minmax: [1,99] of ALL voxels."""
    lo, hi = np.percentile(x, (1.0, 99.0))
    d = float(hi - lo)
    if d <= 1e-8:
        return None
    return (np.clip(x.astype(np.float32), lo, hi) - lo) / d


def pool4(a):
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    return torch.nn.functional.avg_pool3d(t, 4)[0, 0].numpy()


def stats(p):
    return {"pmax": round(float(p.max()), 4),
            "pmean": round(float(p.mean()), 5),
            "f05": round(float((p > 0.5).mean()), 5),
            "f08": round(float((p > 0.8).mean()), 5)}


def main():
    fs = fsspec.filesystem("s3", anon=True)
    store = zarr.open(fs.get_mapper(VOL), mode="r")["0"]
    ref = np.load(REF).astype(np.float32)
    print("ref loaded", ref.shape, ref.dtype, flush=True)

    model = load_ink3d(CKPT).cuda().eval().half()
    results = {"ref_global": stats(ref), "tiles": []}

    with torch.no_grad():
        for tz, ty, tx in TILES:
            print("tile", (tz, ty, tx), flush=True)
            ct = np.asarray(store[tz:tz + TILE, ty:ty + TILE, tx:tx + TILE])
            rz, ry, rx = tz - REF_ORIGIN[0], ty - REF_ORIGIN[1], tx - REF_ORIGIN[2]
            ref_sub = ref[rz:rz + TILE, ry:ry + TILE, rx:rx + TILE]
            row = {"tile": [tz, ty, tx],
                   "ct_nonzero_frac": round(float((ct > 0).mean()), 4),
                   "ref": stats(ref_sub)}
            refL2 = pool4(ref_sub)
            for name, fn in (("screen_norm", norm_screen), ("villa_norm", norm_villa)):
                nrm = fn(ct)
                if nrm is None:
                    row[name] = {"error": "norm degenerate"}
                    continue
                x = torch.from_numpy(nrm).half().cuda()[None, None]
                logits = model(x)
                if isinstance(logits, dict):
                    logits = next(iter(logits.values()))
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
                pL2 = pool4(prob)
                # compare to CLI ref at L2, and trim 8 L2-voxels (32 L0) of border
                # to exclude sliding-window blending edge effects in the reference
                c = 8
                a = pL2[c:-c, c:-c, c:-c].ravel()
                b = refL2[c:-c, c:-c, c:-c].ravel()
                r_full = float(np.corrcoef(pL2.ravel(), refL2.ravel())[0, 1])
                r_core = float(np.corrcoef(a, b)[0, 1])
                row[name] = {**stats(prob),
                             "corr_L2_full": round(r_full, 4),
                             "corr_L2_core": round(r_core, 4),
                             "mad_L2_full": round(float(np.abs(pL2 - refL2).mean()), 5),
                             "mad_L2_core": round(float(np.abs(a - b).mean()), 5),
                             "agree05_core": round(float(((a > 0.5) == (b > 0.5)).mean()), 4)}
                # norm intensity diagnostics
                row[name]["norm_mean"] = round(float(nrm.mean()), 4)
                del x, logits
                torch.cuda.empty_cache()
            # also cross-compare the two normalizations' percentile cuts
            nz = ct[ct > 0]
            row["pcts"] = {
                "nonzero_1_99": [float(v) for v in np.percentile(nz, [1, 99])] if nz.size else None,
                "all_1_99": [float(v) for v in np.percentile(ct, [1, 99])]}
            results["tiles"].append(row)
            print(json.dumps(row, indent=1), flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
