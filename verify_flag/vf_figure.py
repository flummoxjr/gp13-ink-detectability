"""flag_verification.png - the whole case on one sheet."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

HERE = r"C:\Users\benbl\Desktop\Vsuvious\trackD\verify_flag"
sys.path.insert(0, HERE)
import vf_common as C

UM_F, UM_C = 34.56, 37.448
P = json.load(open(os.path.join(HERE, "vf_perm.json")))
T = json.load(open(os.path.join(HERE, "vf_targeted.json")))
B = json.load(open(os.path.join(HERE, "vf_battery.json")))
K = json.load(open(os.path.join(HERE, "vf_tiles.json")))
V = json.load(open(os.path.join(HERE, "vf_validated.json")))
FAM = json.load(open(os.path.join(HERE, "vf_family.json")))
NZ = np.load(os.path.join(HERE, "vf_nulls.npz"))

flag = C.load(C.FLAG); flagr = C.load(C.FLAG, "reverse"); mask = flag > 0
cz = np.load(os.path.join(HERE, "vf_ctrl_ds4.npz"))
ctrl, cmask, clab = cz["ctrl"].astype(float), cz["cmask"], cz["clab"]

thetas = list(range(0, 180, 15))
cache = C.rot_cache(mask, thetas)
prof = C.profile(flag, 15, cache); profr = C.profile(flagr, 15, cache)
cov = cache[15][1][cache[15][2]]

# control's own ruling profile (the validated positive), for panel G
cth = np.arange(170, 185, 1.0)
ccache = C.rot_cache(cmask, cth)
cbest = (0, None, None, None)
for t in cth:
    pr = C.profile(ctrl * cmask, t, ccache)
    pm = C.band_prom(pr, UM_C, 90)
    if pm[0] > cbest[0]:
        cbest = (pm[0], t, pm[2], pr)
cprof = cbest[3]
print(f"control best: theta={cbest[1]} period={cbest[2]:.2f}mm prom={cbest[0]:.1f}")

RED, BLU, GRN, GRY, ORG = "#A33A2E", "#3A6B8C", "#4E7A3A", "#808080", "#C8791E"
fig = plt.figure(figsize=(19, 14.5))
gs = fig.add_gridspec(3, 4, height_ratios=[1.05, .95, .95], hspace=.42, wspace=.26)


def ac(p):
    q = p - p.mean(); a = np.correlate(q, q, "full")[len(q) - 1:]
    return a / a[0]


# ---------------- row 1 ---------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.imshow(flag, cmap="magma", vmin=0, vmax=220,
          extent=[0, flag.shape[1] * UM_F / 1000, flag.shape[0] * UM_F / 1000, 0])
ax.set_title(f"A   THE FLAG  PHerc1447 / {C.FLAG}\nforward prediction, ds4 = 34.56 um/px",
             fontsize=9.5)
ax.set_xlabel("mm"); ax.set_ylabel("mm")
for c in K["components"]:
    r0, r1, c0, c1 = c["bbox_ds4"]
    ax.add_patch(plt.Rectangle((c0 * UM_F / 1000, r0 * UM_F / 1000), (c1 - c0) * UM_F / 1000,
                               (r1 - r0) * UM_F / 1000, fill=False, ec="cyan", lw=.9, ls="--"))
    ax.text(c1 * UM_F / 1000 + .4, (r0 + r1) / 2 * UM_F / 1000, f"mean\n{c['mean']:.0f} DN",
            color="cyan", fontsize=7.5, va="center")
ax.text(.02, .02, "three disjoint inference-patch regions\nwith different mean response",
        transform=ax.transAxes, color="w", fontsize=7.5, va="bottom")

ax = fig.add_subplot(gs[0, 1])
ax.imshow(mask, cmap="gray", extent=[0, flag.shape[1] * UM_F / 1000,
                                     flag.shape[0] * UM_F / 1000, 0])
ax.set_title("B   valid mask: 44.6% of the canvas\nevery edge on the 64-px inference stride",
             fontsize=9.5)
ax.set_xlabel("mm")
ax.text(.02, .02, "ink_9um patch 128 px, overlap 0.5 -> stride 64 px\n"
                  "gcd(mask edge coords) = 64 px EXACTLY, rows and cols\n"
                  "mesh is 100% inpainted but 30% of its grid\n"
                  "points are degenerate -> whole patches skipped",
        transform=ax.transAxes, color="#FFD37F", fontsize=7.2, va="bottom",
        bbox=dict(fc="black", ec="none", alpha=.82, pad=2.5))

ax = fig.add_subplot(gs[0, 2])
ys, xs = np.nonzero(clab)
r0 = max(0, int(np.median(ys)) - 200); c0 = max(0, int(np.median(xs)) - 200)
ax.imshow(ctrl[r0:r0 + 400, c0:c0 + 400], cmap="magma", vmin=0, vmax=220,
          extent=[0, 400 * UM_C / 1000, 400 * UM_C / 1000, 0])
ax.set_title("C   POSITIVE CONTROL  PHerc0139 w035\nsame checkpoint, human-verified Greek letters",
             fontsize=9.5)
ax.set_xlabel("mm")
ax.text(.02, .02, "strokes 0.62 mm wide, 3.7 mm^2\n69.4% of letter px > 195 DN",
        transform=ax.transAxes, color="w", fontsize=7.5, va="bottom")

ax = fig.add_subplot(gs[0, 3])
lab, n = ndi.label(mask)
big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
ys, xs = np.nonzero(lab == big)
r0 = int(np.median(ys)) - 216; c0 = int(np.median(xs)) - 216
ax.imshow(flag[r0:r0 + 433, c0:c0 + 433], cmap="magma", vmin=0, vmax=220,
          extent=[0, 433 * UM_F / 1000, 433 * UM_F / 1000, 0])
ax.set_title("D   THE FLAG at the same mm scale as C\n(interior of the largest patch)", fontsize=9.5)
ax.set_xlabel("mm")
ax.text(.02, .02, "components 0.12 mm^2, 0.22 mm wide (31x smaller\nthan letters); 0.67% of px > 195 DN",
        transform=ax.transAxes, color="w", fontsize=7.5, va="bottom")

# ---------------- row 2 ---------------------------------------------------
ax = fig.add_subplot(gs[1, :2])
x = np.arange(len(prof)) * UM_F / 1000
for lo, hi, cc in ((0, 7.4, BLU), (7.4, 14.4, ORG), (14.4, 29, GRN)):
    ax.axvspan(lo, hi, color=cc, alpha=.07)
ax.plot(x, prof, lw=1.0, color=BLU, label="forward - the flagged projection (theta=15)")
ax.plot(x, profr, lw=1.0, color=ORG, alpha=.8, label="reverse render (map-scale r = 0.753)")
for k in range(5):
    ax.axvline(k * 7.24, color=RED, ls=":", lw=1.1)
a2 = ax.twinx(); a2.plot(x, cov / cov.max(), color=GRY, lw=1.1, ls="--")
a2.set_ylabel("mask coverage", fontsize=8, color=GRY)
ax.set_xlabel("mm along the theta=15 projection axis")
ax.set_ylabel("masked row mean (DN)")
ax.set_title("E   What the screen actually scored: the profile is a 3-step function tracking the three "
             "inference patches,\nnot a comb.  Red = the claimed 7.24 mm 'ruling'.  "
             "The reverse render reproduces it.", fontsize=9.5)
ax.legend(fontsize=7.5, loc="lower right")

ax = fig.add_subplot(gs[1, 2])
prom, _, per_mm, periods, band = C.band_prom(prof, UM_F, 0.0, return_spec=True)
ax.semilogy(periods * UM_F / 1000, band / np.median(band), "o-", ms=4, lw=.9, color=BLU)
ax.axvline(7.24, color=RED, ls=":", lw=1.3)
for e in (1.7, 8.4):
    ax.axvline(e, color="k", lw=1)
ax.axhline(1, color=GRY, lw=.6)
bb = T["band_bins"]
ax.set_xlabel("period (mm)"); ax.set_ylabel("power / band median (= the score)")
ax.set_title(f"F   the 1.7-8.4 mm band holds only {bb['n_integer_bins']} Fourier bins here;\n"
             f"the 'peak' is k=4 - the LOWEST bin in the band", fontsize=9.5)

ax = fig.add_subplot(gs[1, 3])
af, acc = ac(prof), ac(cprof)
ax.plot(np.arange(len(af)) * UM_F / 1000, af, lw=1.2, color=BLU, label="FLAG (claimed 7.24 mm)")
ax.plot(np.arange(len(acc)) * UM_C / 1000, acc, lw=1.2, color=RED,
        label=f"CONTROL letters ({cbest[2]:.2f} mm)")
for k in range(1, 4):
    ax.axvline(k * 7.24, color=BLU, ls=":", lw=.9)
    ax.axvline(k * cbest[2], color=RED, ls=":", lw=.9)
ax.axhline(0, color=GRY, lw=.6); ax.set_xlim(0, 24); ax.set_ylim(-.6, 1)
ax.set_xlabel("lag (mm)"); ax.set_ylabel("autocorrelation")
ax.set_title("G   a real ruling repeats at 1P, 2P, 3P.\nFLAG: -0.27, +0.01, -0.03  ->  not periodic",
             fontsize=9.5)
ax.legend(fontsize=7.5)

# ---------------- row 3 ---------------------------------------------------
ax = fig.add_subplot(gs[2, 0])
ns = NZ["screen_ds8_null"]; nv = NZ["validated_null"]
ax.hist(ns, bins=40, color=BLU, alpha=.55, label=f"screen null, 400 perms\nsd={ns.std(ddof=1):.2f}")
ax.hist(nv, bins=40, color=GRN, alpha=.55, label=f"validated-protocol null\n(erode 40, detrend)")
ax.axvline(NZ["screen_ds8_obs"][0], color=RED, lw=1.8, label="observed (ds8) 35.4")
m16 = ns[:16]
ax.axvspan(m16.mean() - 3.92, m16.mean() + 3.92, color=RED, alpha=.12)
ax.annotate(f"the screen's 16-draw sd was 3.92;\nthe true sd is {ns.std(ddof=1):.2f}  (2.2x)\n"
            f"null max = {ns.max():.1f}  >  observed",
            xy=(.98, .55), xycoords="axes fraction", ha="right", fontsize=7.2, color=RED)
ax.set_xlabel("band prominence"); ax.set_ylabel("count")
ax.set_title("H   the null is heavy-tailed; 16 draws\nseverely underestimate its width", fontsize=9.5)
ax.legend(fontsize=6.8, loc="upper right")

ax = fig.add_subplot(gs[2, 1])
rng = np.random.default_rng(1)
z16 = []
for d in P["others"].values():
    nn = np.array(d["nulls"])
    idx = np.array([rng.permutation(len(nn))[:17] for _ in range(20000)])
    s = nn[idx]; z16.append((s[:, 0] - s[:, 1:].mean(1)) / (s[:, 1:].std(1) + 1e-9))
z16 = np.concatenate(z16)
ax.hist(np.clip(z16, -4, 12), bins=90, color=GRY, alpha=.7, log=True)
ax.axvline(5.94, color=RED, lw=1.8, label="the flag, z = +5.94")
ax.axvline(np.percentile(z16, 100 * (1 - 1 / 80)), color="k", ls="--", lw=1.2,
           label=f"expected max of 80 tests = {np.percentile(z16,100*(1-1/80)):.2f}")
ax.set_xlabel("z under the screen's OWN null (16-permutation estimator)")
ax.set_ylabel("count (log)")
ax.set_title(f"I   multiple comparisons: P(max of 80 >= 5.94) = "
             f"{FAM['P_max80_ge_5.94_screen16']:.2f}\n"
             f"E[# segments with z>=5 among 80] = {FAM['expected_n_ge5_of_80']:.2f}   (observed: 1)",
             fontsize=9.5)
ax.legend(fontsize=7.2)

ax = fig.add_subplot(gs[2, 2])
m = B["morphology"]["p80"]; I = B["intensity"]
groups = ["comp area\n(mm^2)", "stroke width\n(mm)", "frac px\n> 195 DN", "fwd/rev\ncorrelation"]
letters = [m["letter_area_mm2_p50"], m["letter_width_mm_p50"], I["letter_frac_gt_blankp99"], 0.055]
blanks = [m["blank_area_mm2_p50"], m["blank_width_mm_p50"], I["blank_frac_gt_blankp99"], np.nan]
flags = [m["flag_area_mm2_p50"], m["flag_width_mm_p50"], I["flag_frac_gt_blankp99"],
         B["symmetry"]["flag_fwd_rev_r"]]
xx = np.arange(4); w = .26
ax.bar(xx - w, letters, w, color=RED, label="control LETTERS (proven ink)")
ax.bar(xx, blanks, w, color=GRY, label="control blank papyrus")
ax.bar(xx + w, flags, w, color=BLU, label="THE FLAG")
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(groups, fontsize=7.5)
ax.set_title(f"J   the other three battery tests.  KS(flag vs letters):\n"
             f"area D={m['KS_vs_letter_area_mm2'][0]:.2f}, width D={m['KS_vs_letter_width_mm'][0]:.2f} "
             f"- complete separation", fontsize=9.5)
ax.legend(fontsize=7, loc="lower left")

ax = fig.add_subplot(gs[2, 3]); ax.axis("off")
d8 = P["flag"]["ds2"]; d4 = P["flag"]["ds1"]; w_ = T["within_component_null"]
txt = (
    "V E R D I C T :   R E F U T E D\n"
    "--------------------------------------------\n"
    "line-ruling z, same statistic, more nulls\n"
    f"  screen  (16 perms, ds8)     +5.94\n"
    f"  400 perms, ds8              {d8['z']:+.2f}  [{d8['z_ci95'][0]:+.2f},{d8['z_ci95'][1]:+.2f}]"
    f"  p={d8['empirical_p']:.3f}\n"
    f"  400 perms, ds4 (full)       {d4['z']:+.2f}  [{d4['z_ci95'][0]:+.2f},{d4['z_ci95'][1]:+.2f}]"
    f"  p={d4['empirical_p']:.3f}\n"
    f"  VALIDATED protocol, 400     {V['z']:+.2f}  [{V['z_ci95'][0]:+.2f},{V['z_ci95'][1]:+.2f}]"
    f"  p={V['empirical_p']:.2f}\n"
    f"  control w035, same protocol +25.6 .. +34.1\n"
    f"  PHerc1203, same protocol     -2.7 .. +1.0\n\n"
    "familywise (screen's own 16-perm null)\n"
    f"  P(max of 80 >= 5.94) = {FAM['P_max80_ge_5.94_screen16']:.2f}\n"
    f"  expected max of 80   = {FAM['expected_max_z16_80']:.2f}\n"
    f"  E[# with z>=5 of 80] = {FAM['expected_n_ge5_of_80']:.2f}, observed 1\n\n"
    "what produced 5.94\n"
    "  1. 16-draw sd = 3.92 vs true 8.43 (2.2x low)\n"
    "     -> +5.94 becomes +2.43 on its own data\n"
    "  2. residual +2.43 is a 4-cycle step function:\n"
    f"     3 patch means remove {100*T['patch_demeaned']['frac_prom_removed']:.0f}% of the score;\n"
    "     eroding the 40-px tile rim: 47.3 -> 13.3\n"
    "  3. no detrend + 14-bin band -> the 'period'\n"
    "     is the band's lowest bin, k=4\n\n"
    "the period is not line spacing\n"
    f"  autocorr(1P) = -0.27 (a comb gives > 0)\n"
    "  halves disagree; erosion moves it 6.8-7.9 mm\n"
    "  control ruling 4.68 mm, 5x more cycles\n\n"
    "and it is not ink by any other test\n"
    f"  fwd/rev r {B['symmetry']['flag_fwd_rev_r']:.3f} (letters 0.055) - z-symmetric\n"
    f"  frac>195 DN {I['flag_frac_gt_blankp99']:.4f} (letters 0.694)\n"
    f"  W1 to blank papyrus {I['W1_flag_to_blank']:.0f}, to letters {I['W1_flag_to_letter']:.0f}")
ax.text(-.08, 1.02, txt, transform=ax.transAxes, va="top", fontsize=7.9, family="monospace")

fig.suptitle("Verification of the corpus periodicity flag:  PHerc1447 / z_dbg_gen_00166_inp_hr   "
             "(screen reported ruling_z = +5.94 at 7.26 mm)", fontsize=13, y=.985)
fig.savefig(os.path.join(HERE, "flag_verification.png"), dpi=100, bbox_inches="tight")
print("wrote flag_verification.png")
