"""Investigation D step 3: measure ink_canonical_2um's real input contract + cost.

Downloads scrollprize/ink_canonical_2um, loads it with villa's own inference wrapper
(model_resnet3d_3d_decoder.load_model), and measures VRAM + throughput at the documented
contract (62 layers, 256x256 tiles) on the local RTX 4090 Laptop.
"""
import json
import os
import sys
import time

import torch

VILLA = r"C:\Users\benbl\Desktop\Vsuvious\villa\ink-detection\optimized_inference"
sys.path.insert(0, VILLA)
CKPT_DIR = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\b3441997-0118-49b2-8364-cbdf28fc6397\scratchpad\ink2um"


def main():
    from huggingface_hub import hf_hub_download
    os.makedirs(CKPT_DIR, exist_ok=True)
    t = time.time()
    ck = hf_hub_download("scrollprize/ink_canonical_2um", "r152_3ddec_v2_l5_epoch13.ckpt",
                         local_dir=CKPT_DIR)
    print(f"checkpoint {os.path.getsize(ck)/2**30:.2f} GiB in {time.time()-t:.0f}s -> {ck}")

    raw = torch.load(ck, map_location="cpu", weights_only=False)
    hp = raw.get("hyper_parameters", {})
    print("hyper_parameters:", json.dumps({k: v for k, v in hp.items()
                                           if isinstance(v, (int, float, str, bool, list))})[:800])
    print("epoch", raw.get("epoch"), "global_step", raw.get("global_step"))
    sd = raw["state_dict"]
    print("state_dict tensors:", len(sd))
    # infer the depth the encoder expects from the stem conv
    for k, v in sd.items():
        if v.ndim == 5:
            print("first 5-D weight:", k, tuple(v.shape))
            break

    from model_resnet3d_3d_decoder import load_model
    dev = torch.device("cuda")
    for frames in (62,):
        m = load_model(ck, dev, num_frames=frames)
        for bs in (1, 2, 4, 8):
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                x = torch.randn(bs, 1, frames, 256, 256, device=dev)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    y = m.forward(x)              # warm
                    torch.cuda.synchronize()
                    n, t0 = 6, time.time()
                    for _ in range(n):
                        y = m.forward(x)
                    torch.cuda.synchronize()
                    dt = (time.time() - t0) / n
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"frames={frames} batch={bs}: out {tuple(y.shape)}  "
                      f"{dt*1e3:7.1f} ms/step = {bs/dt:6.2f} tiles/s  peak VRAM {peak:.2f} GiB")
                del x, y
            except torch.cuda.OutOfMemoryError:
                print(f"frames={frames} batch={bs}: OOM on 16GB")
                torch.cuda.empty_cache()
                break


if __name__ == "__main__":
    main()
