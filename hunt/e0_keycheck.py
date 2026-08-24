"""Compare villa-built state_dict keys+shapes against the checkpoint manifest read over HTTP range."""
import io, json, pickle, sys, urllib.request, zipfile
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt")
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone

URL = ("https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500/resolve/main/"
       "dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt")

exec(open(r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\peek_pt.py").read().split("f = HTTPRangeFile")[0]
     .replace("URL = sys.argv[1]", ""))

f = HTTPRangeFile(URL)
z = zipfile.ZipFile(f)
pkl = [n for n in z.namelist() if n.endswith("data.pkl")][0]
obj = Unp(io.BytesIO(z.read(pkl))).load()
ck_teacher = obj["teacher"]
ck = {k.replace("backbone.", "", 1): tuple(v["shape"])
      for k, v in ck_teacher.items() if k.startswith("backbone.") and isinstance(v, dict) and v.get("__tensor__")}
print(f"checkpoint teacher.backbone tensors: {len(ck)}  (HTTP {f.nbytes/1e6:.2f} MB)")

mc = dict(obj["config"]["model"]); mc["input_channels"] = 1
bb = build_dinovol_2_backbone(mc)
built = {k: tuple(v.shape) for k, v in bb.state_dict().items()}
print(f"villa-built state_dict tensors:      {len(built)}")

only_ck = sorted(set(ck) - set(built))
only_bb = sorted(set(built) - set(ck))
shape_mismatch = sorted(k for k in set(ck) & set(built) if ck[k] != built[k])
print(f"\nin checkpoint but NOT in built model : {len(only_ck)}  {only_ck[:8]}")
print(f"in built model but NOT in checkpoint : {len(only_bb)}  {only_bb[:8]}")
print(f"name matches but SHAPE differs       : {len(shape_mismatch)}  "
      f"{[(k, ck[k], built[k]) for k in shape_mismatch[:5]]}")
verdict = "PASS" if not (only_ck or only_bb or shape_mismatch) else "FAIL"
print(f"\nSTRICT-LOAD PREDICTION: {verdict}")
