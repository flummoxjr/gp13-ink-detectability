"""Re-run the flag's nulls keeping the raw prominence arrays (for the figure)."""
import os, sys, numpy as np
from multiprocessing import Pool
HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
sys.path.insert(0, HERE)
import vf_common as C
import vf_validated as V

THETAS = list(range(0, 180, 15))
_G = {}


def _init(ds, block, px):
    a = C.load(C.FLAG, "forward", ds=ds); m = a > 0
    _G.update(a=a, m=m, cache=C.rot_cache(m, THETAS), px=px, block=block)


def _one(i):
    if i < 0:
        return C.ruling_score(_G["a"], _G["cache"], THETAS, _G["px"])[0]
    rng = np.random.default_rng(1000 + i)
    pa, _ = C.block_permute(_G["a"], _G["m"], rng, _G["block"])
    return C.ruling_score(pa, _G["cache"], THETAS, _G["px"])[0] if pa is not None else np.nan


if __name__ == "__main__":
    out = {}
    for ds, block, tag in ((2, 32, "screen_ds8"), (1, 64, "screen_ds4")):
        px = C.PX_UM_DS4["PHerc1447"] * ds
        with Pool(16, initializer=_init, initargs=(ds, block, px)) as p:
            r = p.map(_one, [-1] + list(range(400)), chunksize=2)
        out[tag + "_obs"] = np.array([r[0]]); out[tag + "_null"] = np.array(r[1:], float)
        print(tag, "obs", r[0], "null mean", np.nanmean(r[1:]), "sd", np.nanstd(r[1:], ddof=1))
    px = C.PX_UM_DS4["PHerc1447"]
    with Pool(16, initializer=V._init, initargs=(px,)) as p:
        r = p.map(V._one, [-1] + list(range(400)), chunksize=2)
    out["validated_obs"] = np.array([r[0][0]])
    out["validated_null"] = np.array([x[0] for x in r[1:]], float)
    print("validated obs", r[0][0], "null mean", np.nanmean(out["validated_null"]))
    np.savez_compressed(os.path.join(HERE, "vf_nulls.npz"), **out)
    print("wrote vf_nulls.npz")
