# ---------------------------------------------------------------- expected --
# Every number below was measured against the live servers and the laptop
# ground-truth copies BEFORE this script was written. Parse, then assert --
# never assume.
MODEL_PITCH = 9.36

# 500p2a -- PITCH CORRECTED 2026-09-01. v1/v2 used 4.32 um, read off the
# meta.json "volume" string. Three independent measurements say 2.215 um:
#   (1) mesh bbox x<=16037, z<=27616 cannot fit the 4.317um volume
#       (9423 x 15838) and fits only the 2.215um one (18209 x 28096);
#   (2) the surface-volume canvas 26239 x 16182 is 1:1 with those extents;
#   (3) label geometry: median component height 2.49 mm, stroke width 0.61 mm
#       at 2.215um (Herculaneum-typical) vs 4.86 / 1.19 mm at 4.32um.
# See trackD/bench/P2A_PITCH_RESOLUTION.md.
BUCKET = ("https://huggingface.co/buckets/scrollprize/datasets/resolve/"
          "ink/unused/500p2a")
P2A_SHAPE = (65, 26239, 16182)             # .zarray, verified live
P2A_CHUNK = (65, 128, 128)
P2A_PITCH = 2.215                           # CORRECTED (was 4.32)
P2A_PITCH_WRONG = 4.32                      # the Aug-25 value; fault arithmetic only
P2A_LABEL_BYTES = 3009475
P2A_MASK_BYTES = 2983168
P2A_RASTER = dict(ink=12856732, mask=34871346, ink_and_mask=12230762,
                  ink_outside_mask=625970, annot_blank=22640584)
WINDOWS = {
    "win1": dict(y0=12416, x0=6912, size=4438, ink=4389424, ink_and_mask=4388955,
                 ink_outside_mask=469, annot_blank=8576280, mask=12965235,
                 vol_mean=76.114, vol_std=32.313, vol_zero_frac=0.0),
    "win2": dict(y0=18432, x0=7424, size=4438, ink=2862504, ink_and_mask=2862504,
                 ink_outside_mask=0, annot_blank=4778034, mask=7640538,
                 vol_mean=72.612, vol_std=32.042, vol_zero_frac=0.0),
    "win3": dict(y0=12160, x0=2432, size=4438, ink=2158076, ink_and_mask=1536773,
                 ink_outside_mask=621303, annot_blank=2237639, mask=3774412,
                 vol_mean=74.129, vol_std=31.922, vol_zero_frac=0.00023),
}
NZ_WIN, S_WIN = 65, 4438
# depth modes (asserted against rint_shape below, after it is defined)
ISO_NZ = 15          # 65 * 2.215 / 9.36 = 15.38 -> 15 (infer.py zero-pads to 17)
FIT17_NZ = 17        # 65 layers -> 17 at 8.47 um
S9 = 1050            # 4438 * 2.215 / 9.36 = 1050.2 -> 1050
DEPTH_MODES = {"iso": ISO_NZ, "fit17": FIT17_NZ}
PRIMARY_DEPTH = "iso"

# CTL -- PHerc0139 w035 native 9.362um surface volume (public S3, zarr v2,
# level 0 [28,5820,5240] uint8, chunks [28,128,128], compressor null,
# dimension_separator "/"; verified live 2026-09-01).
CTL_SV = ("https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/"
          "segments/20260317000000-w035_2026031718/surface-volumes/"
          "9.362um-1.2m-113keV-volume-20250728140407.zarr")
CTL_SHAPE = (28, 5820, 5240)
CTL_CHUNK = (28, 128, 128)
CTL_CHUNK_BYTES = 28 * 128 * 128           # 458752, raw (compressor null)
CTL_PITCH = 9.362
CTL_CROP = (512, 2944, 384, 3072)          # y0, y1, x0, x1 (128-aligned)
CTL_N_POS = 334035                         # ink & sup inside the crop
CTL_N_NEG = 737086                         # sup & ~ink inside the crop
CTL_LABELS_SHA = "23ad57aed651ca8ff81e13bd4b829d359af7d6711fb904b68168c850e84aa4cf"
CTL_FAULT_FACTOR = P2A_PITCH_WRONG / P2A_PITCH   # 1.9504 -- exactly v2's error
CTL_HALF_FACTOR = 0.5
CTL_HARNESS_MIN_FWD = 0.95
CTL_DEPTHREV_MAX = 0.80
CTL_FAULT_REPRODUCED_MAX = 0.75
CTL_FAULT_NOT_REPRODUCED_MIN = 0.85

# ------------------------------------------------------------------ prereg --
GATE_BASELINE_AUC = 0.85
EXPA_ANCHOR = "win1"
DETECT_AUC_MIN = 0.75
DETECT_RETAIN_MIN = 0.50
PITCHES = [3.24, 4.32, 5.5, 6.5, 8.0, 9.36, 12.0]
NOISE_KS = [1, 2, 4, 8]
BLUR_SIGMAS = [1.0, 2.0]
N_TRANSLATIONS = 40
MIN_SHIFT_MM = 3.75           # 1.5 x the measured 2.49 mm median letter height
MAX_SHIFT_FRAC = 0.60         # v2's 0.40 leaves an EMPTY annulus at 2.215um
MIN_PSEUDO_POS = 50000
CONFOUND_MEDIAN_AUC = 0.60
CONFOUND_FRAC = 0.20          # OR-clause: >= 20% of nulls at/above 0.60
GAP_MIN = 0.15

