"""Training configs.
  python cfggen.py synthetic          -> out/cfg/syn.json (pherc0814 real labels + random volume, 30 it)
  python cfggen.py loso <seed> <steps> <save_every> -> out/cfg/loso_s<seed>.json via make_holdout_config.py
"""
import json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.environ["SCRIPTS"])
import curvelib as cl

S = os.environ["SCRIPTS"]; OUT = cl.OUT; LABELS = os.environ["LABELS"]; VOLS = os.environ["VOLS"]; RUNS = os.environ["RUNS"]
RECIPE = os.path.join(S, "aligned21_hybrid_3d2d.json")
CONTRACT = os.path.join(S, "aligned21_fixed_scroll_prior.json")
CFG = os.path.join(OUT, "cfg"); os.makedirs(CFG, exist_ok=True)
NPROC = os.cpu_count() or 8
ARM = int(os.environ.get("ARM", "0"))


def arm_keys():
    """Bet A arm config keys (trackD PREREG_BET_A): 0 -> none (recipe byte-identical),
    1 -> input_degradation from the k2b index, 2 -> input_whitening."""
    if ARM == 1:
        idx = json.load(open(os.path.join(S, "k2b_index.json")))
        return {"input_degradation": {"enabled": True, "index": idx, "probability": 1.0,
                                      "apply_blur": True, "apply_noise": True, "apply_headroom": True, "seed": 1}}
    if ARM == 2:
        return {"input_whitening": {"enabled": True, "n_samples": 64, "sample_size": 128, "q_ref": 0.02, "max_gain": 8.0, "seed": 2}}
    return {}


def synthetic():
    import zarr
    seg = "pherc0814-46527"
    fam = "aligned-scrollprizeorg-21slices"
    lab = zarr.open(os.path.join(LABELS, fam, seg, f"{seg}_inklabels.zarr"), mode="r")["0"]
    shape = tuple(int(x) for x in lab.shape)
    vol_dir = os.path.join(cl.DATA, "syn", "volumes", "aligned9"); os.makedirs(vol_dir, exist_ok=True)
    vp = os.path.join(vol_dir, f"{seg}.zarr")
    if not os.path.exists(vp):
        rng = np.random.default_rng(1)
        v = rng.integers(40, 200, size=shape, dtype=np.uint8)
        cl.write_group_zarr(vp, v)
    r = json.load(open(RECIPE))
    r.pop("fixed_scroll_prior", None)
    r.update(sampling_strategy="uniform", num_iterations=30, save_every=30, val_every=30, val_steps=2,
             batch_size=8, dataloader_workers=min(2, max(0, NPROC - 2)), warmup_steps=5, log_every=5,
             out_dir=os.path.join(RUNS, "syn"), seed=42,
             datasets=[{"segments_path": os.path.join(LABELS, fam), "segments": [seg],
                        "surface_volume_paths": {seg: vp}, "volume_scale": 0,
                        "sampling_scroll": "0814",
                        "sampling_physical_segment_keys": {seg: "0814:46527"},
                        "sampling_representation_keys": {seg: f"public_2p4_level2_zmean4:{seg}"}}])
    r.update(arm_keys())
    p = os.path.join(CFG, "syn.json"); json.dump(r, open(p, "w"), indent=1)
    cl.say(f"CFG synthetic: {seg} labels ({shape}) + random volume, 30 iterations, batch 8, arm {ARM} -> {p}")


def loso(seed, steps, save_every):
    out = os.path.join(CFG, f"loso_s{seed}.json")
    run_dir = os.path.join(RUNS, f"s{seed}")
    cmd = [sys.executable, os.path.join(S, "make_holdout_config.py"), "--labels-root", os.path.join(LABELS, "aligned-scrollprizeorg-21slices"),
           "--volumes-root", os.path.join(VOLS, "aligned9"), "--exclude-scroll", "0139", "--seed", str(seed),
           "--recipe", RECIPE, "--contract", CONTRACT, "--out", out, "--run-dir", run_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    cl.say("CFG generator: " + " | ".join(l.strip() for l in (r.stdout + r.stderr).strip().splitlines()[-5:]))
    assert r.returncode == 0, "make_holdout_config failed"
    c = json.load(open(out))
    q = c["fixed_scroll_prior"]["target_batch_counts"]
    assert q == {"1667": 40, "Paris4": 20, "0814": 4}, q
    kept = sorted(s for d in c["datasets"] for s in d["segments"])
    assert len(kept) == 15 and len(c["datasets"]) == 3, (len(kept), len(c["datasets"]))
    for d in c["datasets"]:
        for s, vp in d["surface_volume_paths"].items():
            assert os.path.exists(vp), f"volume missing for {s}: {vp}"
    c.update(num_iterations=int(steps), save_every=int(save_every), val_every=int(save_every),
             dataloader_workers=min(12, max(2, NPROC - 2)), out_dir=run_dir, seed=int(seed))
    c["fixed_scroll_prior"]["seed"] = int(seed)
    c.update(arm_keys())
    json.dump(c, open(out, "w"), indent=1)
    cl.say(f"CFG loso seed {seed}: 15 kept, quotas {q}, {steps} iterations, save every {save_every}, "
           f"workers {c['dataloader_workers']}, ARM {ARM} ({', '.join(arm_keys().keys()) or 'baseline'}) -> {out}")


if __name__ == "__main__":
    if sys.argv[1] == "synthetic":
        synthetic()
    else:
        loso(sys.argv[2], sys.argv[3], sys.argv[4])
