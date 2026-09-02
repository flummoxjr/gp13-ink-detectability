"""G1 v2 measurement for Frag2/Frag6: obs field, G0 AUC, block-bootstrap null
(prereg), rigid 400-draw diagnostic (committed generator; draws 0..39 must
equal v1's 40 committed null values).  Rigid loop checkpoints to npz and can
be re-invoked until 400 draws are done."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1v2_lib as V
from g1v2_lib import G, MET
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

CFG = dict(
 f2=dict(name="Frag2", plate="PHercParis2Fr143 54keV_exposed_surface",
         lab="f2/f2_inklabels.png", msk="f2/f2_mask.png", ir="f2/f2_ir.png",
         Hp=14830, W=9506, R0=7392, NR=5024,
         v1=dict(obs=1678.14, n=4014, auc=0.9461, sd40=933.97, mean40=-69.86)),
 f6=dict(name="Frag6", plate="PHerc51Cr4Fr8 PHerc0051Cr04Fr08_53keV_3.24um surface_processing",
         lab="f2/f6_inklabels.png", msk="f2/f6_mask.png", ir="f2/f6_ir.png",
         Hp=8853, W=6205, R0=640, NR=7712,
         v1=dict(obs=1113.53, n=1857, auc=0.9616, sd40=1839.09, mean40=767.08)))

def load(tag):
    c = CFG[tag]
    def img(p):
        a = np.array(Image.open(os.path.join(MET, p)))
        return a[..., 0] if a.ndim == 3 else a
    L = img(c["lab"]) > 0; M = img(c["msk"]) > 0; IR = img(c["ir"])
    for A in (L, M, IR): assert A.shape == (c["Hp"], c["W"]), A.shape
    R0, NR = c["R0"], c["NR"]
    Lb, Mb, Ib = L[R0:R0 + NR], M[R0:R0 + NR], IR[R0:R0 + NR]
    P = np.load(os.path.join(HERE, "%s_P.npy" % tag))
    assert P.shape == (NR, c["W"])
    return c, P, Lb, Mb, Ib

def main(tag, budget_s):
    t0 = time.time()
    V.verify_verbatim()
    c, P, Lb, Mb, Ib = load(tag)
    NR, W = c["NR"], c["W"]
    d32, w32, n, obs, ny, nx, Pv, Mv = V.obs_field(P, Lb, Mb, NR, W)
    Tv = G.tileview((255 - Ib).astype(np.uint8), ny, nx, 32)
    auc, na = G.paired_auc(Tv, Lb, Mv, ny, nx, 32); assert na == n
    print("%s obs %+9.2f n %d  G0 %.4f  (v1 %+9.2f / %d / %.4f)  %.0fs" % (
        c["name"], obs, n, auc, c["v1"]["obs"], c["v1"]["n"], c["v1"]["auc"],
        time.time() - t0), flush=True)

    ck = os.path.join(HERE, "%s_rigid400.npz" % tag)
    if os.path.exists(ck):
        z = np.load(ck); Tt = z["Tt"].copy(); Nt = z["Nt"].copy(); done = int(z["done"])
    else:
        Tt = np.full(400, np.nan); Nt = np.zeros(400, np.int64); done = 0
    sh = G.shifts_for(NR, W, n=400)
    while done < 400 and time.time() - t0 < budget_s:
        dy, dx = sh[done]
        e, nn, _, _ = G.excess(Pv, G.roll(Lb, dy, dx), Mv, ny, nx, 32)
        Tt[done] = e; Nt[done] = nn; done += 1
        if done % 25 == 0:
            np.savez(ck, Tt=Tt, Nt=Nt, done=done)
            print("  rigid draw %d/400  %.0fs" % (done, time.time() - t0), flush=True)
    np.savez(ck, Tt=Tt, Nt=Nt, done=done)
    if done < 400:
        print("CHECKPOINT %d/400 -- re-invoke" % done, flush=True); return

    ok = np.isfinite(Tt)
    assert ok.all(), "non-finite rigid draws"
    first40 = np.round(Tt[:40], 2)
    v1_null = json.load(open(os.path.join(MET, "m1_%s.json" % ("frag2" if tag == "f2" else "frag6"))))
    key = [k for k in v1_null["runs"] if k.startswith("fetched")][0]
    nv = np.array(v1_null["runs"][key]["null_values"])
    first40_match = bool(np.allclose(first40, nv, atol=0.011))
    sd400 = float(Tt.std(ddof=1)); mean400 = float(Tt.mean())
    sd40 = float(Tt[:40].std(ddof=1)); mean40 = float(Tt[:40].mean())

    d64 = d32.astype(np.float64); w64 = w32.astype(np.float64)
    bp = V.boot_pack(d64, w64, ny, nx)
    z_block = obs / bp["boot_sd"]
    print("%s block 2x8 sd %.1f z %+0.3f cv %.2f%% | rigid400 sd %.1f mean %+.1f (40: %.1f/%.1f) first40 %s" % (
        c["name"], bp["boot_sd"], z_block, bp["boot_sd_10seeds"]["cv_pct"],
        sd400, mean400, sd40, mean40, first40_match), flush=True)

    out = dict(
        fragment=c["name"], plate=c["plate"],
        analysis_band="fetched band, plate rows %d..%d (%d x %d)" % (c["R0"], c["R0"] + NR - 1, NR, W),
        grid=[ny, nx], n_tiles=n, obs=round(obs, 2), sum_weight=float(w32.sum()),
        G0=dict(paired_auc=round(float(auc), 4), v1_value=c["v1"]["auc"],
                reproduces_v1=bool(abs(auc - c["v1"]["auc"]) < 5e-5), passes=bool(auc >= 0.90)),
        block_null=bp, z_block=round(z_block, 4),
        rigid_diag=dict(n_draws=400, sd=round(sd400, 1), mean=round(mean400, 1),
                        sd_first40=round(sd40, 1), mean_first40=round(mean40, 1),
                        first40_match_v1=first40_match,
                        v1_sd40=c["v1"]["sd40"], v1_mean40=c["v1"]["mean40"],
                        z_vs_400=round((obs - mean400) / sd400, 4),
                        null_tiles=dict(min=int(Nt.min()), median=int(np.median(Nt)), max=int(Nt.max()))),
        reproduction=dict(obs_v1=c["v1"]["obs"], n_v1=c["v1"]["n"],
                          obs_ok=bool(abs(obs - c["v1"]["obs"]) < 0.011), n_ok=bool(n == c["v1"]["n"])),
        elapsed_s=round(time.time() - t0, 1))
    np.savez_compressed(os.path.join(HERE, "%s_obsfield_v2.npz" % tag), d=d32, w=w32)
    json.dump(out, open(os.path.join(HERE, "g1v2_%s.json" % tag), "w"), indent=1)
    print("done %.1fs" % out["elapsed_s"], flush=True)

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 480)
