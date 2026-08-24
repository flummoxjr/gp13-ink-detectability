import time, torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

print("torch", torch.__version__, torch.cuda.get_device_name(0))
for N, hd in [(4096, 54), (4096, 64), (1024, 54), (1024, 64)]:
    q = torch.randn(1, 16, N, hd, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    row = [f"N={N:5d} head_dim={hd:3d}"]
    for name, backend in [("FLASH", SDPBackend.FLASH_ATTENTION),
                          ("MEM_EFF", SDPBackend.EFFICIENT_ATTENTION),
                          ("CUDNN", SDPBackend.CUDNN_ATTENTION),
                          ("MATH", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                o = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(10):
                    F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                ms = (time.time() - t) / 10 * 1000
            row.append(f"{name}={ms:7.2f}ms")
        except Exception as e:
            row.append(f"{name}=NOTSUP({type(e).__name__})")
    print("  ".join(row))
