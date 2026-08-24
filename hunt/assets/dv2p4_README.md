---
library_name: pytorch
pipeline_tag: feature-extraction
license: mit
tags:
  - vesuvius
  - herculaneum
  - dinov2
  - dinov3
  - 3d
  - self-supervised
  - representation-learning
  - scrolls
  - volumetric
---

# dinovol — v2 backbone, patch size 8 (`paris4` run, step 352500)

A 3D, DINOv2/DINOv3-style **self-supervised representation model** trained on
volumetric micro-CT scans of carbonized Herculaneum scrolls. This repository
publishes the **EMA teacher backbone** from the `paris4` training run at
training step **352500**.

Training code: [`dinovol`](https://github.com/ScrollPrize/dinovol). This is a
*representation/feature-extraction* model — there is no task-specific head; you
take its dense patch embeddings and use them downstream.

## Model details

| | |
|---|---|
| Backbone family | DINOv2/EVA ViT, 3D, with 3D RoPE (DINOv3-style) |
| `model_type` | `v2` |
| Embedding dim | 864 |
| Depth | 24 blocks |
| Attention heads | 16 |
| MLP | SwiGLU, `mlp_ratio` 8/3 |
| Register tokens | 4 |
| Patch size | 8 × 8 × 8 |
| Global crop size (train) | 128 × 128 × 128 |
| Input channels | 1 (grayscale CT) |
| Positional encoding | RoPE `mixed` (base 100, `normalize_coords=separate`, `rescale=2.0`, `shift=0.05`, `jitter=1.05`); no absolute pos-emb |
| Backbone parameters | **215.9 M** |
| Training step | 352500 |
| W&B run | `model_v2__shift005_jitter105__r342500__paris4__20260416` |

Pretraining objective: DINO + iBOT + KoLeo, with late dense-feature refinement
via **Gram anchoring** (DINOv3-style), trained with AMP.

## Training data

Self-supervised on 11 open-data Herculaneum volumes (scale 0) from
`s3://vesuvius-challenge-open-data/`:

`PHerc0009B`, `PHerc0500P2`, `PHerc0814`, `PHerc1299`, `PHerc0343P`,
`PHerc0332`, `PHerc0139`, `PHercMAN5`, `PHerc1451`, `PHercMANB`, `PHercParis4`.

## Files

| File | Size | Use |
|---|---|---|
| `dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt` | ~0.86 GB | **Recommended for inference.** Slim file: `{step, config, teacher backbone weights}`. Loads directly with the repo's loader. |
| `checkpoint_step_352500_paris4.pt` | ~5.0 GB | Full training checkpoint (`student`, EMA `teacher`, `optimizer`, `scaler`, loss buffers, Gram teacher, RNG state). Use to **resume training** or for full reproducibility. |
| `config.json` | — | The `model` config block, for quick inspection. |

Both `.pt` files embed the full training `config`, so the architecture is rebuilt
automatically — no separate config needed at load time.

## Usage

Clone the training repo and use its loader (the config travels inside the weights):

```python
import torch
from huggingface_hub import hf_hub_download
from dinovol_2.eval import embedding_utils as eu

path = hf_hub_download(
    "scrollprize/dinovol_v2_ps8_with_paris4_352500",
    "dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt",
)

loaded = eu.load_backbone_from_checkpoint(path, device="cuda")
backbone = loaded.backbone.eval()

# dense patch embeddings for a 1-channel volume, dims multiples of patch_size (8)
vol = torch.randn(1, 1, 128, 128, 128, device="cuda")
with torch.no_grad():
    out = backbone.forward_features(vol, masks=None, view_kind="global")
patch_tokens = out["x_norm_patchtokens"]   # (B, num_patches, 864)
```

For a normalized, windowed embedding grid over a real OME-Zarr volume, use
`eu.compute_patch_embedding_grid(...)`, which applies the checkpoint's
normalization scheme and tiles the volume. The repo also ships a napari
inspector at `dinovol_2/eval/napari_visualizer.py`.

## License

MIT. See `LICENSE`.

## Caveats

- Trained on single-channel Herculaneum micro-CT; behavior on other modalities is
  untested.
- Pretraining only — no finetuning / segmentation head is included.
