"""SECONDARY arm only: build the Frag1 9.36um uint16 baseline zarr from the
verified per-layer manifest. No planb, no probe, no rungs -- the Frag1 curve
is CUT (pre-registered): with baseline ~0.69 the DETECTABLE floor (0.75) sits
above baseline, so every rung would be undetectable by construction."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

TIFDIR = os.path.join(cl.DATA, "frag1")
MAN = os.path.join(cl.DATA, "frag1_manifest.json")
NZ9 = cl.rint_shape(cl.FRAG1_NZ, cl.NATIVE_PITCH, cl.MODEL_PITCH)   # 23 (22.5 half-up)
H9 = cl.rint_shape(cl.FRAG1_H, cl.NATIVE_PITCH, cl.MODEL_PITCH)     # 2832
W9 = cl.rint_shape(cl.FRAG1_W, cl.NATIVE_PITCH, cl.MODEL_PITCH)     # 2191

def layer_getter():
    man = json.load(open(MAN))["layers"]
    def get(z):
        m = man[f"{z:02d}"]
        return np.memmap(os.path.join(TIFDIR, f"{z:02d}.tif"),
                         dtype=(m["endian"] + "u2"), mode="r", offset=m["so"],
                         shape=(cl.FRAG1_H, cl.FRAG1_W))
    return get

def main():
    path = os.path.join(cl.DATA, "frag1_base_u16.zarr")
    if not os.path.exists(path):
        cl.say(f"SEC_BUILD frag1 baseline: 3.24 -> 9.36um, target ({NZ9},{H9},{W9}) uint16")
        tmp = os.path.join(cl.DATA, "tmp", "frag1_base.f32")
        mm = cl.resample_stack(layer_getter(), cl.FRAG1_NZ, cl.FRAG1_H,
                               cl.FRAG1_W, (NZ9, H9, W9), tmp, tag="frag1base")
        v16 = np.clip(np.rint(np.asarray(mm)), 0, 65535).astype(np.uint16)
        del mm; os.remove(tmp)
        cl.write_group_zarr(path, v16)
        cl.save_preview(v16[NZ9 // 2], os.path.join(cl.OUT, "previews",
                                                    "sec_frag1_base_midslice.png"))
    cl.say(f"SEC_BUILD frag1 baseline done: {path}")

if __name__ == "__main__":
    main()
