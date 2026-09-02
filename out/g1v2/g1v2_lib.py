"""G1 v2 shared library (PREREG_G1_V2.md, git a2e0001).

Estimator: imported UNCHANGED from metrology/m1_lib.py (the committed PREREG_G1
harness: T=32, stride 32, origin (0,0), MINPX=64, weight min(n_ink,n_blank),
positive=label&mask, negative=mask&~label, rigid shifts seed 20260820 |dx|>=128).

block_boot_sd: copied VERBATIM from ship/null_scaling/ns3_corrected.py (the
implementation whose Frag1 whole-plate output 1227.6 / z +1.56 is the prereg's
known-before anchor).  verify_verbatim() proves the copy matches the source.
"""
import os, sys
import numpy as np

MET = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\35d67aba-4ea6-4b13-a3c0-2f3fc87bbe13\scratchpad\metrology"
NS  = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\35d67aba-4ea6-4b13-a3c0-2f3fc87bbe13\scratchpad\ship\null_scaling"
sys.path.insert(0, MET)
import m1_lib as G          # committed v1 estimator, unchanged

NREP = 4000
BY, BX = 2, 8               # prereg-fixed block

# ---- BEGIN VERBATIM COPY from ns3_corrected.py ----
def block_boot_sd(d, w, nyg, nxg, by, bx, nrep=NREP, seed=0, anchors=4):
    """cluster bootstrap sd of T = sum(w d)/sum(w) over (by,bx) tile blocks."""
    rng = np.random.default_rng(seed)
    sds = []
    offs = [(0, 0), (by // 2, bx // 2), (0, bx // 2), (by // 2, 0)][:anchors]
    D2 = d.reshape(nyg, nxg); W2 = w.reshape(nyg, nxg)
    for oy, ox in offs:
        # pad so the offset partition covers the grid
        py = (-(nyg + oy)) % by; px = (-(nxg + ox)) % bx
        Dp = np.pad(D2 * W2, ((oy, py), (ox, px)))
        Wp = np.pad(W2, ((oy, py), (ox, px)))
        YB, XB = Dp.shape[0] // by, Dp.shape[1] // bx
        Sw = Dp.reshape(YB, by, XB, bx).sum((1, 3)).ravel()
        Wb = Wp.reshape(YB, by, XB, bx).sum((1, 3)).ravel()
        keep = Wb > 0
        Sw, Wb = Sw[keep], Wb[keep]
        nb = len(Sw)
        if nb < 8:
            continue
        idx = rng.integers(0, nb, size=(nrep, nb))
        Ts = Sw[idx].sum(1) / Wb[idx].sum(1)
        sds.append(float(Ts.std(ddof=1)))
    return float(np.mean(sds)), int(nb), sds
# ---- END VERBATIM COPY ----

def verify_verbatim():
    """assert the copied function text equals the ns3_corrected.py source."""
    src = open(os.path.join(NS, "ns3_corrected.py")).read()
    here = open(os.path.abspath(__file__)).read()
    def body(txt):
        a = txt.index("def block_boot_sd(")
        b = txt.index("return float(np.mean(sds)), int(nb), sds", a)
        return txt[a:b + len("return float(np.mean(sds)), int(nb), sds")]
    assert body(src) == body(here), "block_boot_sd drifted from ns3_corrected.py"
    return True

def boot_pack(d, w, nyg, nxg, tag=""):
    """prereg v2 null for one fragment: headline seed=2 (the anchor-producing
    seed), 10-seed cv (seeds 100..109, as ns3), block-size sensitivity."""
    sd, nb, _ = block_boot_sd(d, w, nyg, nxg, BY, BX, seed=2)
    seeds = [block_boot_sd(d, w, nyg, nxg, BY, BX, seed=100 + s)[0] for s in range(10)]
    sens = {}
    for by, bx in [(1, 1), (1, 4), (2, 4), (2, 8), (3, 8), (4, 12)]:
        s_, n_, _ = block_boot_sd(d, w, nyg, nxg, by, bx, seed=1)
        sens["block_%dx%d" % (by, bx)] = dict(sd=round(s_, 1), n_blocks=n_)
    import numpy as _np
    return dict(block=[BY, BX], nrep=NREP, boot_sd=round(sd, 1), n_blocks=nb,
                seed_headline=2,
                boot_sd_10seeds=dict(mean=round(float(_np.mean(seeds)), 1),
                                     sd=round(float(_np.std(seeds, ddof=1)), 2),
                                     cv_pct=round(100 * float(_np.std(seeds, ddof=1) / _np.mean(seeds)), 2)),
                sensitivity=sens)

def obs_field(P, L, M, H, W, T=32):
    """observed per-tile field under the committed rule; float32 round-trip to
    match ns2_obsfield.npz storage convention exactly."""
    ny, nx = H // T, W // T
    Mv = G.tileview(M, ny, nx, T); Pv = G.tileview(P, ny, nx, T)
    d, w, ok = G.excess_full(Pv, L, Mv, ny, nx, T)
    d32 = np.nan_to_num(d).astype(np.float32)
    w32 = w.astype(np.float32)
    obs = float((d32.astype(np.float64) * w32.astype(np.float64)).sum()
                / w32.astype(np.float64).sum())
    return d32, w32, int(ok.sum()), obs, ny, nx, Pv, Mv
