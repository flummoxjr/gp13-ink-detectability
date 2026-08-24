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

def patched(self, x, rope=None, attn_mask=None):
    B, N, C = x.shape
    qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
    qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if rope is not None:
        q = apply_rotary_embedding(q, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
        k = apply_rotary_embedding(k, rope, prefix_tokens=self.num_prefix_tokens).type_as(v)
    hd = q.shape[-1]; pad = (-hd) % 8
    if pad: q, k, v = F.pad(q,(0,pad)), F.pad(k,(0,pad)), F.pad(v,(0,pad))
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0., scale=hd**-0.5)
    if pad: x = x[..., :hd]
    x = x.transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(self.norm(x)))

EVA.EvaAttention.forward = patched
bb = build_dinovol_2_backbone(CFG).cuda().eval().to(torch.bfloat16)

for shape, batches in [((32,128,128), (1,2,4,8,16)), ((128,128,128), (1,2,4))]:
    for B in batches:
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(B,1,*shape, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                bb.forward_features(x, masks=None, view_kind="global"); torch.cuda.synchronize()
                t=time.time()
                for _ in range(3): bb.forward_features(x, masks=None, view_kind="global")
                torch.cuda.synchronize(); ms=(time.time()-t)/3*1000
            vox=B*shape[0]*shape[1]*shape[2]
            print(f"{str(shape):>16} B={B:2d}: {ms:8.0f} ms  {ms/B:7.0f} ms/window  "
                  f"{vox/(ms/1000)/1e6:6.2f} Mvox/s  peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        except Exception as e:
            print(f"{str(shape):>16} B={B:2d}: FAILED {type(e).__name__}: {str(e)[:90]}")
            break
