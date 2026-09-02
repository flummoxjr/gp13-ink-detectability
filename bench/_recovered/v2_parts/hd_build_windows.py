"""Assemble requested windows (argv names; default all three) from fetched
chunks, assert every embedded fingerprint, write native + 9.36um zarrs, the
scoring label crops, and the 9.36um mask (win1's feeds sigma_plate)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

CH = 128
CHUNKDIR = os.path.join(cl.DATA, "p2a_chunks")

def main(names):
    import numcodecs
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    codec = numcodecs.Blosc()
    ink_full = np.array(Image.open(
        os.path.join(cl.DATA, "500p2a_inklabels.tif"))) > 0
    msk_full = np.array(Image.open(
        os.path.join(cl.DATA, "500p2a_supervision_mask.tif"))) > 0
    assert ink_full.shape == cl.P2A_SHAPE[1:], ink_full.shape
    r = dict(ink=int(ink_full.sum()), mask=int(msk_full.sum()),
             ink_and_mask=int((ink_full & msk_full).sum()),
             ink_outside_mask=int((ink_full & ~msk_full).sum()),
             annot_blank=int((msk_full & ~ink_full).sum()))
    assert r == cl.P2A_RASTER, f"raster label counts drifted: {r}"
    cl.say("P2A_BUILD raster label counts EXACT "
           f"(ink_outside_mask={r['ink_outside_mask']})")
    S = 4438
    NZ, S9 = 65, cl.rint_shape(4438, cl.P2A_PITCH, cl.MODEL_PITCH)   # 2048
    NZ9 = cl.rint_shape(65, cl.P2A_PITCH, cl.MODEL_PITCH)            # 30
    for name in names:
        w = cl.WINDOWS[name]
        y0, x0 = w["y0"], w["x0"]
        ink = ink_full[y0:y0 + S, x0:x0 + S]
        msk = msk_full[y0:y0 + S, x0:x0 + S]
        got = dict(ink=int(ink.sum()), ink_and_mask=int((ink & msk).sum()),
                   ink_outside_mask=int((ink & ~msk).sum()),
                   annot_blank=int((msk & ~ink).sum()), mask=int(msk.sum()))
        exp = {k: w[k] for k in got}
        assert got == exp, f"{name} label counts drifted: {got} != {exp}"
        np.save(os.path.join(cl.DATA, f"{name}_ink.npy"), ink)
        np.save(os.path.join(cl.DATA, f"{name}_mask.npy"), msk)
        m9 = cl.zoom_to(msk.astype(np.float32), (S9, S9),
                        prefilter=False) >= 0.5
        np.save(os.path.join(cl.DATA, f"{name}_mask936.npy"), m9)
        npy = os.path.join(cl.DATA, "tmp", f"{name}.u8")
        vol = np.memmap(npy, dtype=np.uint8, mode="w+", shape=(NZ, S, S))
        cy0, cx0 = y0 // CH, x0 // CH
        for cy in range(cy0, (y0 + S - 1) // CH + 1):
            for cx in range(cx0, (x0 + S - 1) // CH + 1):
                raw = open(os.path.join(CHUNKDIR, f"0.{cy}.{cx}"), "rb").read()
                arr = np.frombuffer(codec.decode(raw), np.uint8)
                arr = arr.reshape(NZ, CH, CH)
                ys, xs = cy * CH - y0, cx * CH - x0
                ty0, ty1 = max(0, ys), min(S, ys + CH)
                tx0, tx1 = max(0, xs), min(S, xs + CH)
                vol[:, ty0:ty1, tx0:tx1] = \
                    arr[:, ty0 - ys:ty1 - ys, tx0 - xs:tx1 - xs]
        sub = np.asarray(vol[::8, ::4, ::4]).astype(np.float32)
        stats = (float(sub.mean()), float(sub.std()),
                 float((sub == 0).mean()), int(sub.max()))
        assert abs(stats[0] - w["vol_mean"]) < 0.5, (name, stats)
        assert abs(stats[1] - w["vol_std"]) < 0.5, (name, stats)
        assert abs(stats[2] - w["vol_zero_frac"]) < 0.002, (name, stats)
        assert stats[3] == 255, (name, stats)
        cl.say(f"P2A_BUILD {name} volume stats match recorded "
               f"(mean={stats[0]:.3f} sd={stats[1]:.3f} zf={stats[2]:.5f})")
        zn = os.path.join(cl.DATA, f"{name}_native.zarr")
        if not os.path.exists(zn):
            cl.write_group_zarr(zn, np.asarray(vol))
        z9 = os.path.join(cl.DATA, f"{name}_936.zarr")
        if not os.path.exists(z9):
            tmp = os.path.join(cl.DATA, "tmp", f"{name}_936.f32")
            mm = cl.resample_stack(lambda z: np.asarray(vol[z]), NZ, S, S,
                                   (NZ9, S9, S9), tmp, tag=f"{name}936")
            v = np.clip(np.rint(np.asarray(mm)), 0, 255).astype(np.uint8)
            del mm; os.remove(tmp)
            cl.write_group_zarr(z9, v)
        cl.save_preview(np.asarray(vol[NZ // 2]),
                        os.path.join(cl.OUT, "previews", f"{name}_midslice.png"))
        del vol; os.remove(npy)
        cl.say(f"P2A_BUILD {name} zarrs ready (native 65x{S}x{S}, "
               f"9.36um {NZ9}x{S9}x{S9})")

if __name__ == "__main__":
    names = sys.argv[1:] or list(cl.WINDOWS)
    for n in names:
        assert n in cl.WINDOWS, f"unknown window {n}"
    main(names)
