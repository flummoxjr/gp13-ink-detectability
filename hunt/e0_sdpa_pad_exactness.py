"""Is zero-padding head_dim 54->56 numerically exact AND fast? Test both."""
import time, torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
HD, PAD = 54, 56


def padded_sdpa(q, k, v, scale):
    pq = F.pad(q, (0, PAD - HD))
    pk = F.pad(k, (0, PAD - HD))
    pv = F.pad(v, (0, PAD - HD))
    o = F.scaled_dot_product_attention(pq, pk, pv, scale=scale)
    return o[..., :HD]


for N in (1024, 2048, 4096):
    for dt in (torch.bfloat16, torch.float32):
        q = torch.randn(1, 16, N, HD, device="cuda", dtype=dt)
        k = torch.randn_like(q); v = torch.randn_like(q)
        scale = HD ** -0.5
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(5): F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize(); ms_math = (time.time() - t) / 5 * 1000
        best = None
        for name, be in (("CUDNN", SDPBackend.CUDNN_ATTENTION),
                         ("MEM_EFF", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel(be):
                    out = padded_sdpa(q, k, v, scale)
                    torch.cuda.synchronize(); t = time.time()
                    for _ in range(5): padded_sdpa(q, k, v, scale)
                    torch.cuda.synchronize(); ms = (time.time() - t) / 5 * 1000
                err = (out.float() - ref.float()).abs().max().item()
                rel = err / ref.float().abs().max().item()
                print(f"N={N:5d} {str(dt).split('.')[-1]:>9}  MATH {ms_math:8.2f} ms  "
                      f"{name}+pad {ms:7.2f} ms  speedup {ms_math/ms:6.1f}x  "
                      f"max|abs err| {err:.3e}  rel {rel:.2e}")
                best = ms if best is None else min(best, ms)
            except Exception as e:
                print(f"N={N:5d} {str(dt).split('.')[-1]:>9}  {name}+pad NOTSUP: {type(e).__name__}")
