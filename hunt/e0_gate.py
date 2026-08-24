"""Gate E0 — the real one, on the released weights.

Loads scrollprize/dinovol_v2_ps8_with_paris4_352500 teacher backbone with villa's builder,
checks strict key agreement, runs a 128^3 forward, and records the NOISE NULL for
cos(token, avg_ref_embedding) -- the number that tells you whether tau=0.5 means anything.

Download the weights first (see embedding_prospecting.md section 6 / E0):
  curl -L -C - --retry 5 -o D:/vesuvius-data/trackD/models/dinovol/teacher_backbone_ps8_paris4_352500.pt \
    https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500/resolve/main/dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
from vesuvius.models.build.pretrained_backbones import dinovol_2_eva as EVA
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.build.pretrained_backbones.dinov2 import _load_teacher_backbone_state
from vesuvius.models.build.pretrained_backbones.dinovol_2_eva import apply_rotary_embedding

CKPT = r"D:\vesuvius-data\trackD\models\dinovol\teacher_backbone_ps8_paris4_352500.pt"
REF = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\assets\avg_ref_embedding.npy"
TOKENS = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\assets\recorded_embeddings.npy"
OUT = r"C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\out\e0_gate.json"

EXPECT = 863606099


def patched_attn_forward(self, x, rope=None, attn_mask=None):
    """Exact: zero-pad head_dim to a multiple of 8 so fused SDPA kernels are eligible.

    q.k is unchanged by zero padding, softmax is unchanged, and zero-padded v contributes only
    zero output columns which are sliced off. scale must be passed explicitly because SDPA would
    otherwise use 1/sqrt(padded_head_dim). Verified fp32 rel err ~3e-06 (e0_sdpa_pad_exactness.py).
    """
    B, N, C = x.shape
    if self.qkv is not None:
        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv = F.linear(x, weight=self.qkv.weight,
                           bias=torch.cat((self.q_bias, self.k_bias, self.v_bias)))
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
        q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)
    if rope is not None:
        q = apply_rotary_embedding(q, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
        k = apply_rotary_embedding(k, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
    hd = q.shape[-1]
    pad = (-hd) % 8
    if pad:
        q, k, v = F.pad(q, (0, pad)), F.pad(k, (0, pad)), F.pad(v, (0, pad))
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask,
        dropout_p=self.attn_drop.p if self.training else 0.,
        scale=hd ** -0.5,
    )
    if pad:
        x = x[..., :hd]
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(self.norm(x)))


res = {}
size = os.path.getsize(CKPT)
print(f"checkpoint: {size/1e6:.1f} MB (expect {EXPECT/1e6:.1f})")
if size != EXPECT:
    raise SystemExit(f"INCOMPLETE DOWNLOAD: {size} != {EXPECT}. Re-run curl -C - to resume.")

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("top-level keys:", list(ck.keys()), "| step:", ck.get("step"))
cfg = ck["config"]
assert isinstance(cfg.get("model"), dict), "embedded config.model missing"

mc = dict(cfg["model"])
mc["input_channels"] = 1
backbone = build_dinovol_2_backbone(mc)
state = _load_teacher_backbone_state(ck)
missing, unexpected = backbone.load_state_dict(state, strict=False)
print(f"STRICT: teacher keys {len(state)} | missing {len(missing)} | unexpected {len(unexpected)}")
if missing:
    print("  missing:", missing[:10])
if unexpected:
    print("  unexpected:", unexpected[:10])
res.update(n_teacher_keys=len(state), n_missing=len(missing), n_unexpected=len(unexpected),
           params_M=sum(p.numel() for p in backbone.parameters()) / 1e6, step=int(ck.get("step", -1)))
print(f"params: {res['params_M']:.1f} M")

dev = "cuda" if torch.cuda.is_available() else "cpu"
backbone = backbone.to(dev).eval()

ref = np.load(REF).astype(np.float32)
toks = np.load(TOKENS).astype(np.float32)


def token_grid(vol_t):
    with torch.inference_mode():
        out = backbone.forward_features(vol_t, masks=None, view_kind="global")
    t = out["x_norm_patchtokens"][0]
    return t


orig_attn = EVA.EvaAttention.forward
stock_tn = None
for mode, fn in (("stock", None), ("padded", patched_attn_forward)):
    EVA.EvaAttention.forward = orig_attn if fn is None else fn
    torch.manual_seed(0)
    x = torch.randn(1, 1, 128, 128, 128, device=dev, dtype=torch.float32)
    t0 = time.time()
    tk = token_grid(x)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    raw_norm = tk.norm(dim=-1)
    tn = F.normalize(tk, dim=-1).float().cpu().numpy()
    cs = tn @ ref
    nn_cs = (tn @ toks.T).max(axis=1)
    print(f"[{mode}] tokens {tuple(tk.shape)} in {dt:.1f}s | raw token norm "
          f"median {raw_norm.median().item():.3f} (NOT 1 -> you must L2-normalise)")
    print(f"[{mode}] NOISE NULL cos(token, avg_ref): min {cs.min():.3f} p50 {np.median(cs):.3f} "
          f"p95 {np.percentile(cs,95):.3f} max {cs.max():.3f} | frac>0.5 {(cs>0.5).mean():.4f}")
    print(f"[{mode}] NOISE NULL max-cos over the 128 expert tokens: p50 {np.median(nn_cs):.3f} "
          f"max {nn_cs.max():.3f}")
    res[mode] = {
        "seconds": dt,
        "raw_token_norm_median": float(raw_norm.median()),
        "noise_cos_avgref": {"min": float(cs.min()), "p50": float(np.median(cs)),
                             "p95": float(np.percentile(cs, 95)), "max": float(cs.max()),
                             "frac_gt_0.5": float((cs > 0.5).mean())},
        "noise_maxcos_128": {"p50": float(np.median(nn_cs)), "max": float(nn_cs.max())},
    }
    if stock_tn is None:
        stock_tn = tn
    else:
        agree = (stock_tn * tn).sum(axis=-1)
        res["agreement_cos_stock_vs_padded"] = {
            "min": float(agree.min()), "median": float(np.median(agree))}
        print(f"[agreement] cosine(stock token, padded token): min {agree.min():.6f} "
              f"median {np.median(agree):.6f}")
EVA.EvaAttention.forward = orig_attn

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(res, f, indent=1)
print("\nwrote", OUT)
print(json.dumps(res, indent=1))

verdict = "PASS" if res["n_missing"] == 0 and res["n_unexpected"] == 0 else "FAIL"
print(f"\nGATE E0: {verdict}")
