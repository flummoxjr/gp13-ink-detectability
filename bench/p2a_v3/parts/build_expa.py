"""EXPERIMENT A rung builder -- all degraded win1 volumes, from the native
2.215um window (iso depth throughout). Rungs (pre-registered): pitch
{3.24,4.32,5.5,6.5,8.0,12.0}um via anti-aliased resample 2.215->P then
regrid P->9.36 (the 9.36 and 2.215 rows reuse baseline); noise k in
{1,2,4,8} x sigma_plate on the 9.36 iso grid; bit4; blur sigma {1.0,2.0} px."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

NZ, S = cl.NZ_WIN, cl.S_WIN
NZ9, S9 = cl.ISO_NZ, cl.S9
A = cl.EXPA_ANCHOR

def main():
    base = np.asarray(cl.read_zarr0(os.path.join(cl.DATA, f"{A}_iso.zarr")))
    assert base.shape == (NZ9, S9, S9) and base.dtype == np.uint8, \
        (base.shape, base.dtype)
    m9 = np.load(os.path.join(cl.DATA, f"{A}_mask936.npy"))
    assert m9.shape == (S9, S9), m9.shape
    slab = base.astype(np.float32)                 # all 15 iso layers
    d = (slab[:-1] - slab[1:]) / np.sqrt(2.0)
    dm = np.broadcast_to(m9, d.shape)
    sigma_plate = 1.4826 * float(np.median(np.abs(d[dm])))
    json.dump(dict(sigma_plate=sigma_plate, dtype="uint8", n_layers=int(NZ9)),
              open(os.path.join(cl.RESULTS, "expA_sigma.json"), "w"), indent=1)
    cl.say(f"RUNGS sigma_plate={sigma_plate:.3f} DN (uint8, in-mask, all "
           f"{NZ9} iso layers, adjacent-slice MAD estimator)")
    for k in cl.NOISE_KS:
        p = os.path.join(cl.DATA, f"rung_n{k}.zarr")
        if os.path.exists(p):
            continue
        rng = np.random.default_rng([cl.SEED, k])
        noisy = base.astype(np.float32) + rng.normal(
            0.0, k * sigma_plate, size=base.shape).astype(np.float32)
        cl.write_group_zarr(p, np.clip(np.rint(noisy), 0, 255).astype(np.uint8))
        cl.say(f"RUNGS built noise k={k}")
    p = os.path.join(cl.DATA, "rung_bit4.zarr")
    if not os.path.exists(p):
        cl.write_group_zarr(p, cl.quant4(base))
        cl.say("RUNGS built bit4 (uint8 -> 16 levels via rint(v/17)*17)")
    from scipy import ndimage
    for sg in cl.BLUR_SIGMAS:
        p = os.path.join(cl.DATA, f"rung_blur{sg}.zarr")
        if os.path.exists(p):
            continue
        b = ndimage.gaussian_filter(base.astype(np.float32), sigma=sg,
                                    mode="nearest")
        cl.write_group_zarr(p, np.clip(np.rint(b), 0, 255).astype(np.uint8))
        cl.say(f"RUNGS built blur sigma={sg}")
    vol = None
    for P in cl.PITCHES:
        if abs(P - cl.MODEL_PITCH) < 1e-9:
            continue
        p = os.path.join(cl.DATA, f"rung_p{P}.zarr")
        if os.path.exists(p):
            continue
        if vol is None:
            vol = np.asarray(cl.read_zarr0(
                os.path.join(cl.DATA, f"{A}_native.zarr")))
            assert vol.shape == (NZ, S, S) and vol.dtype == np.uint8, \
                (vol.shape, vol.dtype)
        nzP = cl.rint_shape(NZ, cl.P2A_PITCH, P)
        SP = cl.rint_shape(S, cl.P2A_PITCH, P)
        cl.say(f"RUNGS pitch {P}um stage A -> ({nzP},{SP},{SP})")
        tmpA = os.path.join(cl.DATA, "tmp", f"p{P}_A.f32")
        mmA = cl.resample_stack(lambda z: vol[z], NZ, S, S,
                                (nzP, SP, SP), tmpA, tag=f"p{P}A")
        tmpB = os.path.join(cl.DATA, "tmp", f"p{P}_B.f32")
        mmB = cl.resample_stack(lambda z: np.asarray(mmA[z]), nzP, SP, SP,
                                (NZ9, S9, S9), tmpB, tag=f"p{P}B")
        del mmA; os.remove(tmpA)
        v = np.clip(np.rint(np.asarray(mmB)), 0, 255).astype(np.uint8)
        del mmB; os.remove(tmpB)
        cl.write_group_zarr(p, v)
        cl.say(f"RUNGS built pitch {P}um")

if __name__ == "__main__":
    main()
