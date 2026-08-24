"""Measure dinovol_v2_ps8 forward throughput with the exact released config (random weights)."""
import json
import sys
import time

import torch

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone

MODEL_CFG = {
    "input_channels": 1,
    "global_crops_size": [128, 128, 128],
    "local_crops_size": [64, 64, 64],
    "embed_dim": 864,
    "patch_size": [8, 8, 8],
    "embedding_type": "default",
    "deeper_embed_patch_chunk_size": None,
    "deeper_embed_batch_chunk_size": 1,
    "depth": 24,
    "num_heads": 16,
    "qkv_bias": True,
    "qkv_fused": True,
    "mlp_ratio": 8.0 / 3.0,
    "swiglu_mlp": True,
    "scale_mlp": True,
    "scale_attn_inner": False,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.2,
    "drop_path_uniform": False,
    "init_values": None,
    "use_abs_pos_emb": False,
    "use_rot_pos_emb": True,
    "num_reg_tokens": 4,
    "grad_checkpointing": False,
    "block_chunks": 0,
    "rope_type": "mixed",
    "rope_kwargs": {"base": 100.0, "normalize_coords": "separate",
                    "shift_coords": 0.05, "jitter_coords": 1.05, "rescale_coords": 2.0},
    "model_type": "v2",
}

bb = build_dinovol_2_backbone(MODEL_CFG)
n = sum(p.numel() for p in bb.parameters())
print(f"built OK: {n/1e6:.1f} M params  (card claims 215.9 M)")
print("state_dict keys:", len(bb.state_dict()))
dev = "cuda"
bb = bb.to(dev).eval()
print(torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB VRAM")

res = {"params_M": n / 1e6, "n_state_keys": len(bb.state_dict())}
for name, shape in [("128x128x128", (128, 128, 128)), ("32x128x128", (32, 128, 128)),
                    ("64x128x128", (64, 128, 128))]:
    for dt_name, dt in [("bf16", torch.bfloat16), ("fp32", torch.float32)]:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            m = bb.to(dt)
            x = torch.randn(1, 1, *shape, device=dev, dtype=dt)
            with torch.inference_mode():
                o = m.forward_features(x, masks=None, view_kind="global")
                tok = o["x_norm_patchtokens"]
                torch.cuda.synchronize()
                reps = 5
                t = time.time()
                for _ in range(reps):
                    m.forward_features(x, masks=None, view_kind="global")
                torch.cuda.synchronize()
                dtsec = (time.time() - t) / reps
            vox = shape[0] * shape[1] * shape[2]
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{name:>12} {dt_name}: tokens {tuple(tok.shape)}  {dtsec*1000:7.1f} ms  "
                  f"{vox/dtsec/1e6:7.1f} Mvox/s  peak {peak:.2f} GB")
            res[f"{name}_{dt_name}"] = {"ms": dtsec * 1000, "mvox_s": vox / dtsec / 1e6,
                                        "peak_gb": peak, "tokens": list(tok.shape)}
        except Exception as e:
            print(f"{name:>12} {dt_name}: FAILED {type(e).__name__}: {str(e)[:160]}")
            res[f"{name}_{dt_name}"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

print(json.dumps(res, indent=1))
