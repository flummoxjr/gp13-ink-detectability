
# ------------------------------------------------------- v3 additions -------
SCRIPTS_DIR = os.environ.get("SCRIPTS", os.path.dirname(os.path.abspath(__file__)))

def resample_pred(pred, out_hw):
    """Prediction map -> an exact target grid; anti-aliased iff downsampling.
    (upsample_pred is kept for the native-grid upsampling path.)"""
    pred = np.asarray(pred, dtype=np.float32)
    if tuple(pred.shape) == tuple(out_hw):
        return pred
    down = any(i > o for i, o in zip(pred.shape, out_hw))
    return zoom_to(pred, out_hw, prefilter=down)

def load_ctl_labels():
    """The embedded w035 crop labels (packbits -> zlib -> base64), verified by
    sha256 of the raw packed bits and by exact class counts."""
    import base64, zlib
    b64 = open(os.path.join(SCRIPTS_DIR, "ctl_labels.b64")).read().strip()
    raw = zlib.decompress(base64.b64decode(b64))
    got = hashlib.sha256(raw).hexdigest()
    assert got == CTL_LABELS_SHA, f"ctl labels sha256 {got} != {CTL_LABELS_SHA}"
    y0, y1, x0, x1 = CTL_CROP
    H, W = y1 - y0, x1 - x0
    a = np.unpackbits(np.frombuffer(raw, np.uint8))[:2 * H * W]
    a = a.reshape(2, H, W).astype(bool)
    ink, sup = a[0], a[1]
    pos, neg = ink & sup, sup & ~ink
    assert int(pos.sum()) == CTL_N_POS, int(pos.sum())
    assert int(neg.sum()) == CTL_N_NEG, int(neg.sum())
    return pos, neg

def ctl_arm_shape(factor):
    """Output (nz, H, W) of a CTL resample by `factor` (half-up rounding)."""
    y0, y1, x0, x1 = CTL_CROP
    nz = CTL_SHAPE[0]
    def r(n):
        return max(1, int(np.floor(n * factor + 0.5 + 1e-9)))
    return (r(nz), r(y1 - y0), r(x1 - x0))

# consistency asserts (fail at import time, i.e. before any stage runs)
assert rint_shape(NZ_WIN, P2A_PITCH, MODEL_PITCH) == ISO_NZ, \
    rint_shape(NZ_WIN, P2A_PITCH, MODEL_PITCH)
assert rint_shape(S_WIN, P2A_PITCH, MODEL_PITCH) == S9, \
    rint_shape(S_WIN, P2A_PITCH, MODEL_PITCH)
assert ctl_arm_shape(CTL_FAULT_FACTOR) == (55, 4743, 5243), \
    ctl_arm_shape(CTL_FAULT_FACTOR)
assert ctl_arm_shape(CTL_HALF_FACTOR) == (14, 1216, 1344), \
    ctl_arm_shape(CTL_HALF_FACTOR)
