import json
import statistics

q = json.load(open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\pherc0813_mesh_qc.json"))
print("=== PHerc0813 new meshes ===")
for r in q:
    if "profile" in r:
        p = r["profile"]
        mod = (max(p) - min(p)) / max(statistics.mean(p), 1e-9)
        print(f"{r['name'][-6:]}: {r['area_cm2']:5} cm2 | surf {r['surface_mean_DN']:6} DN | "
              f"empty {r['surface_zero_frac']:.3f} | 404s {r.get('chunks_missing_404','?')} | "
              f"modulation {mod:.3f}")
    else:
        print(f"{r['name'][-6:]}: {str(r.get('error'))[:70]}")

try:
    c = json.load(open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\control_profile.json"))
    print("\n=== control (PHerc0139 w035, letters proven) ===")
    for k, v in c.items():
        mod = (max(v) - min(v)) / max(statistics.mean(v), 1e-9)
        print(f"{k}: min {min(v)} max {max(v)} modulation {mod:.3f}")
except Exception as e:
    print("control:", e)
