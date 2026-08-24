"""K3 sensitivity bound calculations - Pb detection floor for the Paris 4 74/110 keV screen."""
import json
import math

OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\out\k3_sensitivity_bound.json"

# ---------------- NIST/XCOM mass attenuation (cm^2/g), log-log interpolation ----------------
C_TAB = [(50, 0.1871), (60, 0.1753), (80, 0.1610), (100, 0.1514), (150, 0.1347)]
H_TAB = [(50, 0.3355), (60, 0.3260), (80, 0.3091), (100, 0.2944), (150, 0.2651)]
O_TAB = [(50, 0.2132), (60, 0.1907), (80, 0.1678), (100, 0.1553), (150, 0.1361)]
PB_BELOW = [(50, 8.041), (60, 5.021), (80, 2.419), (88.004, 1.910)]
PB_ABOVE = [(88.006, 7.683), (100, 5.549), (150, 2.014)]
K_EDGE = 88.005


def loglog(E, tab):
    for (e0, m0), (e1, m1) in zip(tab, tab[1:]):
        if e0 <= E <= e1:
            t = (math.log(E) - math.log(e0)) / (math.log(e1) - math.log(e0))
            return math.exp(math.log(m0) + t * (math.log(m1) - math.log(m0)))
    raise ValueError(E)


def pb(E):
    return loglog(E, PB_BELOW if E < K_EDGE else PB_ABOVE)


def cellulose(E):  # C6H10O5 mass fractions
    return 0.4445 * loglog(E, C_TAB) + 0.0622 * loglog(E, H_TAB) + 0.4934 * loglog(E, O_TAB)


mu = {
    "C_74": loglog(74, C_TAB), "C_110": loglog(110, C_TAB),
    "cell_74": cellulose(74), "cell_110": cellulose(110),
    "Pb_74": pb(74), "Pb_110": pb(110),
    "Pb_53": pb(53), "Pb_70": pb(70),
    "C_53": loglog(53, C_TAB), "C_70": loglog(70, C_TAB),
    "cell_53": cellulose(53), "cell_70": cellulose(70),
}
a74, a110 = mu["Pb_74"], mu["Pb_110"]
R_PB = a74 / a110

# ---------------- screen constants ----------------
SIGMA = 0.12591
VOX_CM = 91.064e-4           # L1 voxel depth in cm
VOX_L0_CM = 45.532e-4
RHO_PB = 11.35
# frames: (mu74_papyrus, mu110_papyrus, ceil74, ceil110)  [linear atten, cm^-1]
FRAMES = {
    # stage-1 medians, no offsets (windows as exported)
    "raw": dict(m74=0.063, m110=0.0505, c74=0.270, c110=0.200),
    # stage-2 as-implemented (air offsets -0.0310/-0.0268 subtracted -> +0.0310/+0.0268)
    # referee fix: m110 = 0.0505 + 0.0268 = 0.0773 (was 0.0768 = 0.050 + 0.0268, inconsistent
    # with the raw frame's 0.0505, which is pinned by the measured ratio median 1.248)
    "corrected": dict(m74=0.094, m110=0.0773, c74=0.301, c110=0.2268),
    # hypothetical perfectly-calibrated Compton frame
    "physical": dict(m74=0.0565, m110=0.050, c74=None, c110=None),
}


def x_for_shift(fr, dR):
    """Pb effective density x (g/cm^3) giving aggregated-ratio drop dR, ignoring clipping.
    Returns None if unreachable (target below pure-Pb asymptote)."""
    R0 = fr["m74"] / fr["m110"]
    Rt = R0 - dR
    if Rt <= R_PB:
        return None
    return (fr["m74"] - Rt * fr["m110"]) / (Rt * a110 - a74)


def sat(fr):
    """(x at 110-channel clip, ratio at clip, max ratio drop)."""
    xs = (fr["c110"] - fr["m110"]) / a110
    Rmin = (fr["m74"] + a74 * xs) / fr["c110"]
    return xs, Rmin, fr["m74"] / fr["m110"] - Rmin


def ratio_at(fr, x, clip=True):
    f74 = fr["m74"] + a74 * x
    f110 = fr["m110"] + a110 * x
    if clip and fr["c110"]:
        f74 = min(f74, fr["c74"])
        f110 = min(f110, fr["c110"])
    return f74 / f110


def areal_ug(x):  # Pb areal density over one L1 voxel depth, ug/cm^2
    return x * VOX_CM * 1e6


def t_pb_um(x):  # metallic-Pb-equivalent thickness, um
    return x * VOX_CM / RHO_PB * 1e4


res = {"inputs": {
    "sigma_detrended_aggregated_ratio": SIGMA,
    "threshold_sigma": 4, "required_shift_4sigma": 4 * SIGMA,
    "voxel_um_L1": 91.064, "aggregation": "3x3x3, den>=14 papyrus voxels",
    "low_channel_voxels": 2, "n_tested_approx": 268e6,
    "air_offsets": [-0.030988, -0.026824],
    "cobright_gate_p75": [0.117051, 0.099765],
    "windows_raw": {"74": [-0.058, 0.270], "110": [-0.040, 0.200]},
}, "mass_attenuation_cm2_g": {k: round(v, 4) for k, v in mu.items()},
    "ratios": {
        "Pb_74_110": round(R_PB, 4),
        "C_74_110": round(mu["C_74"] / mu["C_110"], 4),
        "cellulose_74_110": round(mu["cell_74"] / mu["cell_110"], 4),
        "Pb_53_70": round(mu["Pb_53"] / mu["Pb_70"], 4),
        "C_53_70": round(mu["C_53"] / mu["C_70"], 4),
        "cellulose_53_70": round(mu["cell_53"] / mu["cell_70"], 4),
    }, "frames": {}}

for name, fr in FRAMES.items():
    R0 = fr["m74"] / fr["m110"]
    d = {"R0": round(R0, 4),
         "slope_dR_dx_linearized": round((a74 - R0 * a110) / fr["m110"], 2)}
    x4 = x_for_shift(fr, 4 * SIGMA)
    d["x_4sigma_unclipped_g_cm3"] = None if x4 is None else round(x4, 5)
    d["x_4sigma_areal_ug_cm2"] = None if x4 is None else round(areal_ug(x4), 1)
    if fr["c110"]:
        xs, Rmin, dRmax = sat(fr)
        d.update(x_sat_g_cm3=round(xs, 5), areal_sat_ug_cm2=round(areal_ug(xs), 1),
                 t_pb_equiv_sat_um=round(t_pb_um(xs), 3),
                 ratio_at_sat=round(Rmin, 4), max_ratio_drop=round(dRmax, 4),
                 max_sigma_level_d1=round(dRmax / SIGMA, 2),
                 reachable_4sigma=(x4 is not None and fr["m110"] + a110 * x4 <= fr["c110"]),
                 # post-clip behavior: ratio re-crosses the papyrus baseline here, then pegs
                 # at c74/c110 once BOTH channels clip
                 areal_recross_baseline_ug_cm2=round(
                     areal_ug((R0 * fr["c110"] - fr["m74"]) / a74), 1),
                 areal_both_clip_ug_cm2=round(areal_ug((fr["c74"] - fr["m74"]) / a74), 1))
        # detection windows at lower thresholds (full-neighborhood ink, d=1)
        for k in (3.0, 2.5, 2.0):
            dR = k * SIGMA
            xlo = x_for_shift(fr, dR)
            if xlo is None or xlo > xs:
                d[f"window_{k}sigma"] = None
                continue
            xhi = ((R0 - dR) * fr["c110"] - fr["m74"]) / a74  # post-clip re-rise bound
            d[f"window_{k}sigma"] = {
                "x_g_cm3": [round(xlo, 5), round(xhi, 5)],
                "areal_ug_cm2": [round(areal_ug(xlo), 1), round(areal_ug(xhi), 1)],
                "min_neighborhood_ink_fraction": round(dR / dRmax, 2)}
    else:
        d["reachable_4sigma"] = x4 is not None
        d["max_ratio_drop_unclipped"] = round(R0 - R_PB, 4)
        d["max_sigma_level_unclipped"] = round((R0 - R_PB) / SIGMA, 2)
    res["frames"][name] = d

# ---------------- scenario table (RAW frame per-voxel response, then diluted) ----------------
fr = FRAMES["raw"]
R0 = fr["m74"] / fr["m110"]
scen = {}
for label, areal in [("Tack_large_84ug", 84.0), ("Tack_small_16ug", 16.0),
                     ("dense_300ug", 300.0), ("saturating_311ug", areal_ug(sat(fr)[0])),
                     ("extreme_1000ug", 1000.0)]:
    x = areal / 1e6 / VOX_CM
    dRv = R0 - ratio_at(fr, x)
    scen[label] = {"areal_ug_cm2": round(areal, 1), "x_g_cm3": round(x, 5),
                   "per_voxel_shift": round(dRv, 4),
                   "sigma_d1.0": round(dRv / SIGMA, 2),
                   "sigma_d0.5": round(0.5 * dRv / SIGMA, 2),
                   "sigma_d0.33": round(0.33 * dRv / SIGMA, 2)}
res["scenarios_raw_frame"] = scen

# ---------------- co-bright gate floor ----------------
gate = {}
for label, b74, b110 in [("median_papyrus", 0.094, 0.0773), ("pv_dimmed_surface", 0.061, 0.050)]:
    xg = max((0.117051 - b74) / a74, (0.099765 - b110) / a110)
    gate[label] = {"x_gate_g_cm3": round(xg, 5), "areal_gate_ug_cm2": round(areal_ug(xg), 1)}
res["cobright_gate_floor"] = gate

# ---------------- L0 clipping tightens ceiling (surface-layer geometry) ----------------
xs0 = (fr["c110"] - fr["m110"]) / a110  # same intensive threshold at L0
f74_L1 = 0.5 * ((fr["m74"] + a74 * xs0) + fr["m74"])
f110_L1 = 0.5 * (fr["c110"] + fr["m110"])
res["L0_clip_surface_layer"] = {
    "areal_at_L0_clip_ug_cm2": round(xs0 * VOX_L0_CM * 1e6, 1),
    "L1_ratio_after_binning": round(f74_L1 / f110_L1, 4),
    "max_sigma_level": round((R0 - f74_L1 / f110_L1) / SIGMA, 2)}

# ---------------- f*w translations ----------------
fw = {}
xs, _, dRmax = sat(fr)
x3 = x_for_shift(fr, 3 * SIGMA)
for label, x in [("3sigma_floor_d1", x3), ("saturation_ceiling", xs)]:
    e = {"x_g_cm3": round(x, 5), "fw_product_rho1.5": round(x / 1.5, 4)}
    for t_um in (10, 15, 20):
        f = t_um / 91.064
        e[f"w_pct_layer{t_um}um_rho1.5"] = round(100 * x / (f * 1.5), 1)
    fw[label] = e
res["fw_translation"] = fw

# Gaussian-tail bookkeeping
res["fpr"] = {"gaussian_expected_at_4sigma": round(268e6 * 0.5 * math.erfc(4 / math.sqrt(2))),
              "observed": 2}

# Tack/Brun anchors (web-verified against PMC4745103 / PMC4833268, 2026-08-16)
res["literature"] = {
    "tack2016_srep20763": {"Pb_areal_letters_ug_cm2": [84, 16], "err": 5,
                           "Pb_wt_pct_in_probed_matrix": [0.78, 0.15],
                           "mc_model": "homogeneous papyrus+ink layer, 300 um thick, rho 0.36 "
                                       "g/cm3, 3.5 keV beam (ESRF ID21); wt% is Pb in the probed "
                                       "papyrus+ink mixture, NOT an ink-layer concentration "
                                       "(0.0078*0.36*300um = 84 ug/cm2, consistent)",
                           "speciation": "Pb-L3 XANES closest to lead(II)acetate-like carboxylate; "
                                         "galena disfavored by S-K XANES; minium absent"},
    "brun2016_pnas": "same two Institut-de-France fragments, XRF at ESRF ID21 (2.48/3 keV, Pb "
                     "M-edges) + ID11 XRD (50 keV); reports the same 84+/-5 and 16+/-5 ug/cm2; "
                     "Pb intentional (pigment or drier) argued"}

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
