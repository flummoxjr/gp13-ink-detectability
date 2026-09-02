"""Build P = sum_z I(z) for Frag2/Frag6 bands + per-layer in-mask stats.
I_air rule identical to m1_run.build_P: pooled median inside mask over the
deepest 10 layers.  Adds per-layer in-mask median/mean (for per-plate BULK)."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from g1v2_lib import MET
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def run(tag, voldir, maskpng, Hp, W, R0, NR, iair_v1):
    t0 = time.time()
    M = np.array(Image.open(os.path.join(MET, maskpng)))
    M = (M[..., 0] if M.ndim == 3 else M) > 0
    assert M.shape == (Hp, W)
    Mb = M[R0:R0 + NR]
    P = np.zeros((NR, W), np.int64)
    hist10 = np.zeros(65536, np.int64)
    stats = []
    for z in range(65):
        a = np.load(os.path.join(MET, voldir, "L%02d.npy" % z))
        assert a.shape == (NR, W), (z, a.shape)
        P += a
        h = np.bincount(a[Mb], minlength=65536).astype(np.int64)
        if z >= 55: hist10 += h
        c = np.cumsum(h); n = int(c[-1])
        med = float(np.searchsorted(c, (n + 1) // 2)) if n % 2 else 0.5 * (
            np.searchsorted(c, n // 2) + np.searchsorted(c, n // 2 + 1))
        mean = float((h * np.arange(65536, dtype=np.float64)).sum() / n)
        stats.append(dict(layer=z, inmask_median=med, inmask_mean=round(mean, 2)))
        if z % 16 == 0: print("  %s L%02d  med %.0f  %.0fs" % (tag, z, med, time.time() - t0), flush=True)
    n = int(hist10.sum()); c = np.cumsum(hist10)
    I_air = float(np.searchsorted(c, (n + 1) // 2)) if n % 2 else 0.5 * (
        np.searchsorted(c, n // 2) + np.searchsorted(c, n // 2 + 1))
    np.save(os.path.join(HERE, "%s_P.npy" % tag), P)
    json.dump(dict(tag=tag, band=[R0, R0 + NR - 1], W=W, I_air=I_air,
                   I_air_pool_n=n, I_air_v1=iair_v1, I_air_ok=bool(I_air == iair_v1),
                   per_layer=stats),
              open(os.path.join(HERE, "%s_layers.json" % tag), "w"), indent=1)
    print("%s: I_air %.1f (v1 %.1f, match %s)  pool_n %d  %.0fs" % (
        tag, I_air, iair_v1, I_air == iair_v1, n, time.time() - t0), flush=True)

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "f2":
        run("f2", "f2vol", "f2/f2_mask.png", 14830, 9506, 7392, 5024, 24674.0)
    else:
        run("f6", "f6vol", "f2/f6_mask.png", 8853, 6205, 640, 7712, 27115.0)
