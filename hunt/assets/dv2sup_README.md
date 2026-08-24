---
license: mit
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - vesuvius
  - herculaneum
  - dinov2
  - dinov3
  - 3d
  - self-supervised
  - contrastive-learning
  - representation-learning
  - scrolls
  - volumetric
---

# dinovol — v2 backbone, patch size 8, supervised-contrastive fiber fine-tune (step 362500)

A further fine-tuned checkpoint of
[`scrollprize/dinovol_v2_ps8_with_paris4_352500`](https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500),
continuing training from that backbone's step 352500 with an added
**supervised-contrastive ("supcon") objective** on top of the ongoing DINO/iBOT
losses. This is a *representation/feature-extraction* model — there is no
task-specific head for direct inference; downstream models consume its dense
patch embeddings. It is the frozen encoder used by
[`scrollprize/fiber_dinoguided_2class_step010000`](https://huggingface.co/scrollprize/fiber_dinoguided_2class_step010000).

Training code: [`dinovol`](https://github.com/ScrollPrize/dinovol) (base pretraining).
This fine-tuning stage's own training script was not located in the available
repository snapshot at the time of writing, so the description below is
reconstructed from the checkpoint's embedded config and its Weights & Biases
run, not re-verified against source code.

## Model details

Same backbone architecture as the base checkpoint (unchanged by this
fine-tuning stage, confirmed by inspecting both checkpoints directly):

| | |
|---|---|
| Backbone family | DINOv2/EVA ViT, 3D, with 3D RoPE (DINOv3-style) |
| `model_type` | `v2` |
| Embedding dim | 864 |
| Depth | 24 blocks |
| Attention heads | 16 |
| MLP | SwiGLU, `mlp_ratio` ≈ 2.667 |
| Register tokens | 4 |
| Patch size | 8 × 8 × 8 |
| Global crop size (train) | 128 × 128 × 128 |
| Input channels | 1 (grayscale CT) |
| Positional encoding | RoPE `mixed` (base 100, `normalize_coords=separate`, `rescale=2.0`, `shift=0.05`, `jitter=1.05`) |
| Continues from | step 352500 of `dinovol_v2_ps8_with_paris4_352500` |
| This checkpoint's step | 362500 |
| W&B run (this fine-tune) | [`fiber_supcon3__paris4_0332__from352500__t0p1__w1p0__warm2k__20260522`](https://wandb.ai/vesuvius-challenge/dinov2_pretrain/runs/58oj0suf) (`58oj0suf`, project `dinov2_pretrain`, state **finished**) |

Verified directly from the checkpoint file: `student` and `teacher` sub-payloads
each carry `backbone` (463 tensors) + `dino_head` (8) + `ibot_head` (8) + a
**new `contrastive_head`** (4 tensors: a 2-layer MLP projecting 864 → 512 → 128)
not present in the base checkpoint. The field `contrastive_warmup_start_step`
is recorded as 352500 inside this checkpoint, matching the base backbone's step
exactly, so this is a genuine continuation rather than a restart.

## What changed vs. the base backbone: supervised-contrastive fine-tuning

On top of the continuing DINO + iBOT + KoLeo objectives, this stage adds a
**supervised contrastive loss** over patch tokens, driven by a per-anchor
label with `num_labeled_classes: 3`. The checkpoint's own config logs per-batch
label counts under the names `contrastive_labels_air`, `contrastive_labels_fiber`,
and `contrastive_labels_ignore` — this strongly suggests the 3 label IDs are
**{air, fiber, ignore/unlabeled}** rather than a fiber-orientation split, but we
did not locate the exact patch-labeling source code, so treat this as a
well-supported inference rather than a confirmed fact.

Other contrastive settings read from the checkpoint's config: temperature 0.1,
projection dim 128, 2-layer head (hidden dim 512), loss weight 1.0, a 2000-step
warmup starting at step 352500, teacher targets used for positives/negatives,
cross-rank gathering enabled, plus a variance-regularization term
(`variance_loss_weight=0.5`, `variance_target_similarity=-0.5`). Patch/anchor
labeling used heuristic thresholds including `air_threshold_hu=70.0`,
`patch_air_threshold=0.7`, `patch_fiber_threshold=0.0`,
`patch_surface_threshold=0.1`, and `tube_thickness_voxels=1.5`. The recurrence
of a "70" raw-intensity threshold here echoes the same convention seen in
`otsu_light_threshold=70` in the downstream `fiber_dinoguided_2class` training
config, and in the upstream teacher UNet's `dark70` filename fragment (see
[`scrollprize/fiber_selftrain_teacher_epoch30`](https://huggingface.co/scrollprize/fiber_selftrain_teacher_epoch30)) —
though we can't confirm all three uses are the literal same threshold serving
an identical purpose, just that the value recurs across this lineage.

Training data for this stage: crops from **PHercParis4** and **PHerc0332**
(2.399 µm scan), drawn 75% / 25% from a "fiber-aware" sampler vs. a generic one.
The fiber-aware sampler reads from a `fiber_manifest` / `fiber_cache_dir` under
an `autoreg_fiber` path — plausibly connected to (not proven identical to)
villa PR [#825](https://github.com/ScrollPrize/villa/pull/825)'s cross-frame
fiber-registration infrastructure, which added matching functionality under
similarly-named `autoreg-fiber-*` branches in the same repository. Gram
anchoring was disabled for this stage (`gram.enabled: false`).

### Metrics (W&B run `58oj0suf`, near end of training, train / val)

No accuracy metric applies to self-supervised pretraining — these are training
losses only, logged at a point where the run's internal step counter had
reached ~364821 (state: finished; the last **saved** checkpoint remained step
362500, since the next scheduled save at 365000 was never reached):

| metric | train | val |
|---|---|---|
| total loss | 24.26 | 23.94 |
| contrastive loss (supcon + variance) | 8.55 | 8.59 |
| — supcon component | 8.51 | 8.54 |
| — variance component | 0.087 | 0.106 |
| DINO global loss | 1.56 | 1.53 |
| DINO local loss | 9.46 | 9.26 |
| iBOT loss | 4.76 | 4.65 |
| KoLeo loss | -0.76 | -0.87 |
| Gram loss | 0 (disabled) | 0 (disabled) |
| learning rate | 1.87e-05 | — |

## Files

| File | Size | Use |
|---|---|---|
| `frozen_dino_backbone__3class_supcon__step_362500.pt` | ~4.7 GB | Full training-checkpoint payload: `student`, `teacher` (each with `backbone` + `dino_head` + `ibot_head` + `contrastive_head`), `optimizer`, `scaler`, loss-centering buffers, RNG state, and the embedded `config`. For downstream feature extraction, use the `teacher` sub-dict's `backbone.*` weights (EMA teacher, matching the base backbone repo's recommendation). |
| `avg_fiber_embedding__864d_backbone.npz` | ~3.7 KB | Supplementary. Single array `avg_embedding`, shape `(864,)`, float32, **L2-normalized** (verified: norm = 1.0000). A reference/prototype "fiber" embedding in this backbone's feature space, used as a cosine-similarity lookup vector by the dynamic pseudo-labeling pipeline of [`fiber_dinoguided_2class_step010000`](https://huggingface.co/scrollprize/fiber_dinoguided_2class_step010000) (its `ref_embedding` config field points at a file matching this one). The exact averaging/extraction procedure that produced it is not independently confirmed by us. |

Unlike the base backbone repo, this repo does not include a separate slim
"backbone-only" file — only the full training-checkpoint payload above.

## Usage

This checkpoint has the same top-level shape as the base repo's full training
checkpoint (`student`, `teacher`, `optimizer`, ...), plus the additional
`contrastive_head`. Loading the backbone for feature extraction is expected to
work with the same loader used for the base checkpoint:

```python
import torch
from huggingface_hub import hf_hub_download
from dinovol_2.eval import embedding_utils as eu

path = hf_hub_download(
    "scrollprize/dinovol_v2_ps8_supcon3class_step362500",
    "frozen_dino_backbone__3class_supcon__step_362500.pt",
)

loaded = eu.load_backbone_from_checkpoint(path, device="cuda")
backbone = loaded.backbone.eval()

vol = torch.randn(1, 1, 128, 128, 128, device="cuda")
with torch.no_grad():
    out = backbone.forward_features(vol, masks=None, view_kind="global")
patch_tokens = out["x_norm_patchtokens"]   # (B, num_patches, 864)
```

We inspected this checkpoint's structure directly (`torch.load(..., weights_only=False)`)
to write the description above, but did not re-run this exact loading path
end-to-end ourselves before publishing. If `load_backbone_from_checkpoint`
rejects the extra `contrastive_head` keys, load `teacher["backbone"]` (or
`student["backbone"]`) directly into the base architecture instead.

## Related models

- **Base backbone:** [`scrollprize/dinovol_v2_ps8_with_paris4_352500`](https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500) (step 352500, pre-contrastive-fine-tune)
- **Direct downstream consumer:** [`scrollprize/fiber_dinoguided_2class_step010000`](https://huggingface.co/scrollprize/fiber_dinoguided_2class_step010000)
- **Frozen fiber teacher used alongside this backbone in that downstream run:** [`scrollprize/fiber_selftrain_teacher_epoch30`](https://huggingface.co/scrollprize/fiber_selftrain_teacher_epoch30)

## Links

- **Code:** <https://github.com/ScrollPrize/villa> (fiber pipeline: PRs [#825](https://github.com/ScrollPrize/villa/pull/825), [#985](https://github.com/ScrollPrize/villa/pull/985)) · [`dinovol`](https://github.com/ScrollPrize/dinovol) (base pretraining)
- **W&B run (this fine-tune):** <https://wandb.ai/vesuvius-challenge/dinov2_pretrain/runs/58oj0suf>
- **Data:** <https://scrollprize.org/data_browser>
- **Vesuvius Challenge:** <https://scrollprize.org>

## Caveats

- The training script for this specific supervised-contrastive fine-tuning
  stage was not located in the available repository snapshot; the description
  above is reconstructed from the checkpoint's own embedded config and its W&B
  run, not from reading the training source directly (contrast with the
  downstream 4-class fiber/ink pipeline, whose source we did verify directly).
- The exact semantics of the "3-class" scheme (we infer `{air, fiber, ignore}`
  from logged metric names) and the precise procedure behind
  `avg_fiber_embedding__864d_backbone.npz` are not confirmed with certainty.
- Trained on single-channel Herculaneum micro-CT; behavior on other modalities
  is untested. This is a pretraining/fine-tuning checkpoint only — no
  segmentation head is included.

## License

MIT.
