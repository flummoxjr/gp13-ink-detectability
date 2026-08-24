"""Patch dinovol Eva attention with exact head_dim zero-padding; measure end-to-end gain."""
import sys, time, torch
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\benbl\Desktop\Vsuvious\villa\vesuvius\src")
from vesuvius.models.build.pretrained_backbones import dinovol_2_eva as EVA
from vesuvius.models.build.pretrained_backbones.dinovol_2_builder import build_dinovol_2_backbone
from vesuvius.models.build.pretrained_backbones.dinovol_2_eva import apply_rotary_embedding

CFG = {"input_channels":1,"global_crops_size":[128,128,128],"local_crops_size":[64,64,64],
 "embed_dim":864,"patch_size":[8,8,8],"embedding_type":"default","depth":24,"num_heads":16,
 "qkv_bias":True,"qkv_fused":True,"mlp_ratio":8/3,"swiglu_mlp":True,"scale_mlp":True,
 "scale_attn_inner":False,"proj_drop_rate":0.0,"attn_drop_rate":0.0,"drop_path_rate":0.2,
 "drop_path_uniform":False,"init_values":None,"use_abs_pos_emb":False,"use_rot_pos_emb":True,
 "num_reg_tokens":4,"grad_checkpointing":False,"block_chunks":0,"rope_type":"mixed",
 "rope_kwargs":{"base":100.0,"normalize_coords":"separate","shift_coords":0.05,
                "jitter_coords":1.05,"rescale_coords":2.0},"model_type":"v2"}

_orig_forward = EVA.EvaAttention.forward if hasattr(EVA, "EvaAttention") else None
ATTN_CLS = None
for nm in dir(EVA):
    o = getattr(EVA, nm)
    if isinstance(o, type) and issubclass(o, torch.nn.Module) and "ttention" in nm:
        ATTN_CLS = o; print("attention class:", nm)
if ATTN_CLS is None:
    # find it via an instantiated model
    m = build_dinovol_2_backbone(CFG)
    ATTN_CLS = type(m.blocks[0].attn); print("attention class (via model):", ATTN_CLS.__name__)

_orig = ATTN_CLS.forward


def patched(self, x, rope=None, attn_mask=None):
    B, N, C = x.shape
    if self.qkv is not None:
        if self.q_bias is None:
            qkv = self.qkv(x)
        else:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
            qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
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
        q = F.pad(q, (0, pad)); k = F.pad(k, (0, pad)); v = F.pad(v, (0, pad))
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                       dropout_p=0., scale=hd ** -0.5)
    if pad:
        x = x[..., :hd]
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.norm(x); x = self.proj(x); x = self.proj_drop(x)
    return x


torch.manual_seed(0)
bb = build_dinovol_2_backbone(CFG).cuda().eval()

for dt in (torch.bfloat16,):
    m = bb.to(dt)
    x = torch.randn(1, 1, 128, 128, 128, device="cuda", dtype=dt)
    with torch.inference_mode():
        ATTN_CLS.forward = _orig
        ref = m.forward_features(x, masks=None, view_kind="global")["x_norm_patchtokens"].clone()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(3): m.forward_features(x, masks=None, view_kind="global")
        torch.cuda.synchronize(); ms_a = (time.time() - t) / 3 * 1000

        ATTN_CLS.forward = patched
        new = m.forward_features(x, masks=None, view_kind="global")["x_norm_patchtokens"].clone()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(3): m.forward_features(x, masks=None, view_kind="global")
        torch.cuda.synchronize(); ms_b = (time.time() - t) / 3 * 1000

    a = F.normalize(ref.float(), dim=-1)[0]
    b = F.normalize(new.float(), dim=-1)[0]
    cos = (a * b).sum(-1)
    print(f"\n128^3 {dt}: stock {ms_a:.0f} ms -> padded-SDPA {ms_b:.0f} ms  "
          f"({ms_a/ms_b:.1f}x faster)")
    print(f"  token-wise cosine(stock, patched): min {cos.min():.6f} "
          f"median {cos.median():.6f} mean {cos.mean():.6f}")

# 32x128x128 window too (the surface-volume shape)
x = torch.randn(1, 1, 32, 128, 128, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    ATTN_CLS.forward = _orig
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3): bb.forward_features(x, masks=None, view_kind="global")
    torch.cuda.synchronize(); ms_a = (time.time() - t) / 3 * 1000
    ATTN_CLS.forward = patched
    torch.cuda.synchronize(); t = time.time()
    for _ in range(3): bb.forward_features(x, masks=None, view_kind="global")
    torch.cuda.synchronize(); ms_b = (time.time() - t) / 3 * 1000
print(f"32x128x128 bf16: stock {ms_a:.0f} ms -> padded {ms_b:.0f} ms ({ms_a/ms_b:.1f}x)")
