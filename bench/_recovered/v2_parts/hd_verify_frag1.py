"""Parse every Frag1 layer's OWN IFD (StripOffsets is PER LAYER: 260 for 00-09,
262 for 10-64 -- but we PARSE each file, never assume), cross-check tifffile,
fingerprint the labels, and write data/frag1_manifest.json."""
import json, os, struct, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

TIFDIR = os.path.join(cl.DATA, "frag1")

def parse_ifd(path):
    """Hand-rolled TIFF IFD parse: both byte orders, front or end-of-file IFD."""
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:2] == b"II":
            end = "<"
        elif head[:2] == b"MM":
            end = ">"
        else:
            raise AssertionError(f"{path}: not a TIFF")
        ifd_off = struct.unpack(end + "I", head[4:8])[0]
        f.seek(ifd_off)
        n = struct.unpack(end + "H", f.read(2))[0]
        blob = f.read(n * 12)
        tags = {}
        for i in range(n):
            tag, typ, cnt = struct.unpack(end + "HHI", blob[i*12:i*12+8])
            raw = blob[i*12+8:i*12+12]
            if typ == 3 and cnt == 1:
                tags[tag] = struct.unpack(end + "H", raw[:2])[0]
            elif typ == 4 and cnt == 1:
                tags[tag] = struct.unpack(end + "I", raw)[0]
        return end, ifd_off, tags

def main():
    import tifffile
    man = {}
    so_hist = {}
    for L in range(cl.FRAG1_NZ):
        p = os.path.join(TIFDIR, f"{L:02d}.tif")
        end, ifd_off, t = parse_ifd(p)
        W, H = t.get(256), t.get(257)
        bits, comp = t.get(258), t.get(259)
        so, sbc = t.get(273), t.get(279)
        assert (H, W) == (cl.FRAG1_H, cl.FRAG1_W), f"L{L:02d} dims {H}x{W}"
        assert bits == 16 and comp == 1, f"L{L:02d} bits={bits} comp={comp}"
        assert sbc == cl.FRAG1_SBC, f"L{L:02d} StripByteCounts={sbc}"
        size = os.path.getsize(p)
        assert size == so + sbc, f"L{L:02d} size {size} != SO {so} + SBC {sbc}"
        with tifffile.TiffFile(p) as tf:
            page = tf.pages[0]
            assert page.dataoffsets[0] == so, \
                f"L{L:02d} tifffile offset {page.dataoffsets[0]} != parsed {so}"
            assert page.shape == (cl.FRAG1_H, cl.FRAG1_W)
            arr_rows = page.asarray()[[0, cl.FRAG1_H - 1], :]
        mm = np.memmap(p, dtype=(end + "u2"), mode="r", offset=so,
                       shape=(cl.FRAG1_H, cl.FRAG1_W))
        assert np.array_equal(np.asarray(mm[[0, cl.FRAG1_H - 1], :]), arr_rows), \
            f"L{L:02d} memmap rows != tifffile rows"
        del mm
        man[f"{L:02d}"] = dict(so=int(so), endian=end, size=int(size))
        so_hist[so] = so_hist.get(so, 0) + 1
        if L % 10 == 0:
            cl.say(f"FRAG1_VERIFY layer {L:02d} SO={so} endian="
                   f"{'MM' if end == '>' else 'II'} ok")
    for name, sha, exp in (("inklabels.png", cl.FRAG1_LABEL_SHA, None),
                           ("mask.png", cl.FRAG1_MASK_SHA, None)):
        got = cl.sha256_file(os.path.join(TIFDIR, name))
        assert got == sha, f"{name} sha256 {got} != laptop ground truth {sha}"
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    ink = np.array(Image.open(os.path.join(TIFDIR, "inklabels.png"))) > 0
    msk = np.array(Image.open(os.path.join(TIFDIR, "mask.png"))) > 0
    counts = dict(ink=int(ink.sum()), mask=int(msk.sum()),
                  ink_and_mask=int((ink & msk).sum()),
                  blank=int((msk & ~ink).sum()),
                  ink_outside_mask=int((ink & ~msk).sum()))
    assert counts["ink_and_mask"] == cl.FRAG1_INK_PX, counts
    assert counts["mask"] == cl.FRAG1_MASK_PX, counts
    assert counts["blank"] == cl.FRAG1_BLANK_PX, counts
    assert counts["ink_outside_mask"] == cl.FRAG1_INK_OUTSIDE_MASK, counts
    with open(os.path.join(cl.DATA, "frag1_manifest.json"), "w") as f:
        json.dump(dict(layers=man, labels=counts), f, indent=1)
    cl.say(f"FRAG1_VERIFY all 65 layers parsed per-file; observed StripOffsets "
           f"histogram {so_hist} (expected {{260:10, 262:55}} -- parsed, not "
           f"assumed); labels sha256 + class counts EXACT match")

if __name__ == "__main__":
    main()
