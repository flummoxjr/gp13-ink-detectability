# Investigation C — Embedding-space prospecting with `avg_ref_embedding.npy`

_Written 2026-08-17. All numbers below are measured, not quoted. Sources given inline._
_Independently re-verified 2026-08-17 (second pass) — see **§10**. Every claim in §1, §3 and §4
reproduced exactly. Three things changed: **Gate E0's structural half is now PASSED** (463/463
state-dict keys match by name and shape, proven from the checkpoint manifest for 0.11 MB of HTTP —
killer K-a retired); **§8's cost model was wrong by ~10×** because of a fused-attention fallback,
now measured and fixed; and **E1 should run on four labelled segments with sampled windows**, which
is both cheaper and a stronger test._

---

## 0. Verdict in one paragraph

The route is **technically unblocked and cheaper than expected** — the guidance encoder
(`scrollprize/dinovol_v2_ps8_with_paris4_352500`) is **public and ungated**, its forward path is
already vendored in our villa checkout, and PHerc0139 **w035 ships aligned surface volumes at
9.362 µm, 2.399 µm _and_ 1.129 µm plus per-volume tifxyz meshes**, so the whole calibration runs on
the local 4090 for **$0 of cloud spend**. It is also **weaker as a detector than the release notes
imply**: the "256 expert-clicked tokens" are 128 tokens duplicated across two byte-identical files,
the ink cluster is diffuse (median pairwise cosine 0.334, 56 PCs for 90 % of variance), and at the
published threshold τ = 0.5 only **82.8 % of the expert ink tokens clear their own prototype**. The
decisive unknown is not code, it is physics: **the encoder has never seen 9 µm data** — all 11
pretraining volumes are 2.2–2.4 µm — so applying it to the GP-13 is a 3.9× scale extrapolation. The
plan below is built so that the resolution question is answered first, in one afternoon, on human
ground truth, before any corpus spend.

**Amendment (second pass).** One paragraph-level claim above was wrong. The encoder is **not** cheap
to run out of the box: `embed_dim 864 / 16 heads` gives `head_dim = 54`, which is **not a multiple of
8**, so *every* fused SDPA backend (flash, mem-efficient, cuDNN) refuses it and PyTorch silently
falls back to the `MATH` kernel. Measured on the local 4090: **12.8 s for one 128³ window**, of which
~85 % is that fallback. Zero-padding `head_dim` 54→56 with an explicit `scale=54**-0.5` is
**mathematically exact** (fp32 max relative error 3.0e-06; whole-model token cosine 0.99994 in bf16)
and restores the fused kernels — **1.3 s per 128³ window, ~10× throughput** (§10.2). This does not
change any gate or probability; it changes the cost table (§8) and it is a real, upstreamable villa
fix. Everything downstream assumes the patch is applied.

---

## 1. What the released assets actually are (measured, this session)

Downloaded to `C:\Users\benbl\Desktop\Vsuvious\trackD\hunt\assets\`.

| File | Measured | Note |
|---|---|---|
| `avg_ref_embedding.npy` | `(864,)` float32, ‖v‖ = 1.0000 | L2-normalised |
| `recorded_embeddings.npy` | `(128, 864)` float32, all rows unit-norm | |
| `recorded_embeddings_2.npy` | `(128, 864)` float32 | **byte-identical to file 1** (`np.array_equal` → True; 128 unique rows across both) |

**Finding C-1 (release bug, Track B material).** The model card claims *"the L2-normalised mean of
256 expert-clicked tokens stored in `recorded_embeddings*.npy`"* and *"`(128, 864)` each; … (256
total)"*. The two files are the same bytes. There are **128 unique expert tokens, not 256**.
Confirmed by reconstruction: `cos(normalise(mean(recorded_embeddings)), avg_ref) = 1.0000000`
— i.e. the published reference is exactly the mean of the 128, so the second file contributed
nothing. Worth a one-line HF repo issue; costs us nothing and is the kind of "documented fix" the
progress prizes reward.

> **Second pass: bundle a second documentation bug into the same issue.** The
> `dinovol_v2_ps8_with_paris4_352500` model card's Usage block says
> `from dinovol_2.eval import embedding_utils as eu`. `ScrollPrize/dinovol` is public and
> `dinovol_2/eval/` contains exactly `['__init__.py', 'download_data.py', 'napari_visualizer.py',
> 'task_eval.py']` (GitHub contents API, verified) — **there is no `embedding_utils` module**, so the
> card's only worked example cannot run as written. The functions it names do exist, in
> `napari_visualizer.py`. Two card fixes, one issue: `(128, 864)` ×1 not ×2, and the correct import
> path.

**Finding C-2 (the prototype is loose).** Geometry of the 128 ink tokens:

| statistic | value |
|---|---|
| pairwise cosine among the 128 ink tokens | min −0.037, p05 0.144, **median 0.334**, p95 0.580, max 0.889 |
| cos(token, `avg_ref`) | min 0.255, p05 0.414, **median 0.600**, max 0.760 |
| ‖mean of the 128 unit tokens‖ | **0.591** → the shared direction carries 0.591² = **35 %** of token energy |
| PCA on centred tokens | PC1 9.4 %, PC1–3 24.6 %, PC1–10 48.3 %, **56 PCs for 90 %** |
| centred pairwise cosine | median −0.028 (essentially orthogonal once the mean is removed) |

**Finding C-3 (recall ceiling of the published rule).** Fraction of the expert ink tokens
themselves that clear the published τ = 0.5 against the published prototype:

| τ | 0.40 | 0.45 | **0.50** | 0.55 | 0.60 | 0.65 | 0.70 |
|---|---|---|---|---|---|---|---|
| self-recall | 0.961 | 0.883 | **0.828** | 0.719 | 0.500 | 0.297 | 0.109 |

So the mean-prototype-at-τ=0.5 rule has **≤ 83 % recall in-domain, on its own training clicks,
at 2.4 µm on the scroll it was clicked on**. That is the ceiling before any domain shift. It is also
a design instruction: with 35 % of energy in the mean and 56 effective dimensions, **nearest-of-128
and whitened/Mahalanobis scores should beat mean-cosine**, and all three must be run as arms.

Reproduce (**written and already run** — output in `trackD/hunt/out/e0_prototype.json`, frozen
scorers in `trackD/hunt/out/e0_scorers.npz`):
```powershell
cd C:\Users\benbl\Desktop\Vsuvious
.\.venv\Scripts\python.exe trackD\hunt\e0_prototype_geometry.py
```

---

## 2. How the embedding is actually consumed (code, file:line)

**The released ink checkpoint does NOT contain a DINO module.** `assets/config.json` (mirrored from
`ckpt_78k_fullsup.pth`) has `model_config: {autoconfigure: true, z_projection_mode: "none"}` and no
`guide_backbone` key. DINO guidance was **offline label generation** in training stage 2, not an
in-model branch. So the 1.7 GB ink checkpoint is useless for this route and must not be downloaded.

**The villa `guide_*` machinery is a different mechanism.** `villa/vesuvius/src/vesuvius/models/
build/guidance.py:TokenBook3D` learns its own `[n_tokens, embed_dim]` prototype book during
training (`self.book = nn.Parameter(...)`, random init at line 23) and averages cosine over all
prototypes. `avg_ref_embedding` appears **nowhere in the villa repo** (grep over `*.py *.yaml *.md`
→ 0 hits). Do not go looking for a loader; there isn't one.

**The real consumer is the upstream `dinovol` repo**, `ScrollPrize/dinovol` (public, pushed
2026-08-17). Its model card points at `dinovol_2.eval.embedding_utils`, which **does not exist in
the public tree** (89 files enumerated; only `eval/{__init__,download_data,napari_visualizer,
task_eval}.py`). The equivalent code is all in `dinovol_2/eval/napari_visualizer.py` — this is the
napari tool the experts actually clicked the 128 tokens in. Local copy:
`trackD/hunt/assets/dinovol_src/dinovol_2__eval__napari_visualizer.py`. Key functions and the exact
recipe they encode:

| step | function | line | behaviour |
|---|---|---|---|
| load | `load_backbone_from_checkpoint` | 328 | `config["model"]` → `DinoVitStudentTeacher._build_backbone`, `teacher.backbone.*`, `strict=True` |
| normalise | `normalize_volume` | 474 | scheme from `config["dataset"]["normalization_scheme"]`, **default `"robust"`** (line 20) |
| pad | `pad_volume_to_patch_size` | 490 | edge-pad up to a multiple of 8 |
| tile | `_compute_patch_embedding_grid_windowed` | 627 | windows = `min(padded, global_crops_size)` = **128³**, overlap **1 patch**, linear-ramp blend |
| tokens | `_reshape_and_normalize_patch_tokens` | 600 | `forward_features(...)["x_norm_patchtokens"]` → reshape to grid → **`F.normalize(dim=-1)`** |
| score | `cosine_similarity_patch_grid` | 741 | plain dot product against a unit reference |
| upsample | `upsample_patch_grid_to_volume` | 750 | `np.repeat` ×8 per axis |

Normalisation is `normalize_robust` (`dinovol_2/dataset/normalization.py:175`): clip to the 1st/99th
percentile **of the crop**, subtract median, divide by 1.4826·MAD. **Consequence: absolute
attenuation is destroyed**, exactly as K1 established for `robust_mad`. This encoder is a
texture/morphology descriptor and cannot see the dense-ink intensity channel — good for cross-scroll
transfer, but it means embedding prospecting and the K1/K3 intensity screens are genuinely
independent evidence, not two views of one signal.

**Extraction needs no classifier head.** `Dinov2Backbone.forward`
(`villa/…/pretrained_backbones/dinov2.py:44–57`) returns `[B, 864, D/8, H/8, W/8]` straight from
`x_norm_patchtokens`. That is the whole interface.

> **Correction (second pass).** The line reference was 192; it is **44–57**. More important:
> `Dinov2Backbone.forward` **does not L2-normalise**. `x_norm_patchtokens` is *LayerNorm*-normed, not
> unit-norm. The upstream recipe applies `F.normalize(dim=-1)` afterwards
> (`_reshape_and_normalize_patch_tokens`, napari_visualizer.py:600). Skipping it makes every cosine
> against `avg_ref_embedding` silently wrong. Measured on real tokens: pre-normalisation token norms
> are **not** 1. Either call `F.normalize` yourself or call `backbone.forward_features(...)` directly
> and normalise — do not consume villa's wrapper output raw.

---

## 3. The encoder: available, and what it was trained on

`scrollprize/dinovol_v2_ps8_with_paris4_352500` — **gated = False, private = False** (HF API).

> Correction to `trackD/report/REPRODUCIBILITY.md`: that file says *"`scrollprize/dinovol_v2` is
> gated and was **not** used"*. True for `scrollprize/dinovol_v2` and `…/dinovol_v2_ps8` (both HTTP
> 401), but the **specific model the ink card names is public**. The report line should be narrowed
> before submission or it understates what is reachable.

| file | size | use |
|---|---|---|
| `dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt` | **863.6 MB** | the one we want |
| `checkpoint_step_352500_paris4.pt` | 5.0 GB | training resume — skip |
| `config.json` | 950 B | flat model block (no `{"model": …}` wrapper — see §5) |

I read the checkpoint manifest **without downloading it** — torch `.pt` is a zip, so HTTP range
requests over the central directory + `data.pkl` gave the whole structure for **0.11 MB in 7
requests** (`scratchpad/peek_pt.py`, reusable). Measured:

- top-level keys `{step: 352500, config, teacher}`; `teacher` is a flat dict of 463 keys all
  prefixed `backbone.` → **both** `dinovol`'s `_extract_backbone_state_dict` (line 302) and villa's
  `_load_teacher_backbone_state` (`dinov2.py:258`) accept it unmodified.
- `backbone.down_projection.proj.weight (864, 1, 8, 8, 8)`, `cls_token (1,1,864)`,
  `reg_token (1,4,864)`, 24 blocks, `attn.qkv.weight (2592, 864)` (fused), SwiGLU
  `fc1_g/fc1_x (2304, 864)`, and **per-block** `blocks.N.rope_embed.mix_frequencies (16,27,3)` +
  `.periods (9,)`.
- embedded `config.model`: `embed_dim 864, patch_size [8,8,8], depth 24, num_heads 16,
  qkv_fused true, mlp_ratio 8/3, num_reg_tokens 4, rope_type "mixed", model_type "v2",
  global_crops_size [128,128,128]`.
- embedded `config.dataset`: **no `normalization_scheme` key and no `intensity_properties`** →
  the `"robust"` default applies. Confirmed, not assumed.

**Finding C-4 — the load-bearing risk. Every pretraining volume is 2.2–2.4 µm:**

```
PHerc0009B 2.401µm · PHerc0500P2 2.215µm · PHerc0814 2.399µm · PHerc1299 2.399µm
PHerc0343P 2.215µm · PHerc0332 2.399µm · PHerc0139 2.399µm (20260102150214)
PHercMAN5 2.399µm · PHerc1451 2.399µm · PHercMANB 2.399µm · PHercParis4 2.400µm
val: PHerc0332, PHerc0139, PHercParis4 — all 2.4µm
```

`PHerc0139` is in the training set, but at **2.399 µm**, not at the 9.362 µm scan our w035 labels
live on. One 8-voxel patch is 19.2 µm at training scale and **74.9 µm at 9.362 µm** — larger than a
stroke width. Nothing in this encoder's history says a 74.9 µm token is meaningful. This is the
thing to test first and the most likely killer.

---

## 4. Why the calibration is unusually clean here

`s3://vesuvius-challenge-open-data/PHerc0139/segments/20260317000000-w035_2026031718/` ships, all in
**the same (u,v) frame as our human ink labels**:

```
surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr   shape [ 28,  5820,  5240]
surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr   shape [109, 22640, 20400]
surface-volumes/1.129um-0.22m-59keV-volume-20260413113053-L1.zarr shape [116, 24080, 21700]
mesh/20260317000000-on-20250728140407-9.362um.tifxyz
mesh/20260317000000-on-20260102150214-2.399um.tifxyz
mesh/20260317000000-on-20260413113053-1.129um.tifxyz     (+ 2 more 2.403µm bands)
```

The 9.362 µm SV shape is byte-for-byte our local label shape `(28, 5820, 5240)`
(`D:\vesuvius-data\trackD\w035_{ink,sup,surf}.npy`), and 22640/5820 = 20400/5240 = **3.890**, the
voxel ratio — so the resolution ladder needs **zero registration work**, and the per-volume meshes
remove it for native crops too. Local label geometry, measured this session:

```
ink px 334,024 · background px 737,091 · valid fraction 0.0351 of canvas
ink bbox (9.362µm grid): y 778–2754, x 587–2908  → 18.5 mm × 21.7 mm
ink coverage within valid: 31.2 %
```
Cached as `trackD/hunt/assets/w035_ink2d.npy`, `w035_valid2d.npy`.

This gives the single most informative experiment available to us: **the same human-verified letters,
imaged at 9.362 / 2.399 / 1.129 µm, scored by the same encoder.** Whatever happens, that is a
publishable resolution-threshold curve for embedding-space ink detection and it feeds the report's
rescan-priority thesis directly.

---

## 5. Integration gotchas (each one is a silent-wrong-answer risk)

1. **`config.json` in the encoder repo is flat.** villa's `_load_local_model_config`
   (`dinov2.py:279`) does `config.get("model")`; a flat dict yields `{}` and the model is silently
   built from **defaults** — wrong `patch_size` (16 vs 8), wrong rope. Use the **embedded** config
   from the `.pt` (which does have `{"model": …}`), never the sidecar.
   > **Softened (second pass).** villa already does the right thing here unaided:
   > `_resolve_local_checkpoint_config` (`dinov2.py:217–246`) returns the **embedded** config first
   > and only falls back to a sidecar when the checkpoint has none. Our `.pt` *does* carry
   > `config.model` (verified, §3), so the flat sidecar only bites if you pass `config_path=`
   > explicitly or hand-roll the loader. Keep the rule, downgrade the risk.
2. **villa's Eva ≠ upstream Eva, but only cosmetically.** Whitespace-insensitive diff of
   `villa/…/dinovol_2_eva.py` vs `dinovol_2/model/dinov2_eva.py`: 71 changed lines, all of them
   import paths, a `rope_coords` caching path villa lacks, `mask_token` init
   (`zeros_` vs `trunc_normal_`, unused when `masks=None`), and a rename
   `resolve_output_spatial_shape` → `_resolve_target_spatial_shape`. **The forward math is
   identical.** Prefer the upstream files anyway — provenance matches the weights.
   > **Re-diffed (second pass): 65 changed lines, and the claim holds.** The categories are exactly
   > as listed, plus a dead `embedding_type == "deeper"` branch (our config is `"default"`).
   > The one that looked dangerous — villa dropping upstream's *shared* `rope_coords`, which could
   > have let each of the 24 blocks draw its own RoPE jitter — **is safe**: `shift_coords`,
   > `jitter_coords` and `rescale_coords` are all gated on `self.training` in **both** trees
   > (`rope.py:91,97,104` villa / `up_rope.py:99,105,112` upstream). Under `.eval()` they are
   > no-ops, so the shared-coords path is a caching optimisation only. **villa is safe to use.**
3. **Window shape drives RoPE — and this is not a soft warning.** `rope_kwargs.normalize_coords =
   "separate"` computes each axis as `arange(0.5, size) / size` (`rope.py:85`) — i.e. **every axis
   is normalised to [0,1] by its own length, independent of physical extent.** A 24- or 32-slice
   window therefore tells the model "this z axis spans the full range", exactly as a 128-slice
   window does, while covering 4–5× less tissue. The model has no way to know the difference.
   **Always use true 128³ windows on native crops.** The 28-slice surface volume cannot do that —
   which is precisely why the native-crop arm is primary and the surface-volume arm is a labelled
   ablation, not the headline. When you must use a shallow window (E2 has no choice), the null must
   be computed at **the same depth** — see the E2 note.
4. **Deps.** `.venv` has torch 2.11.0+cu128, numpy 2.5.2, zarr 2.18.7, s3fs 2026.7.0,
   tifffile 2026.8.16, skimage 0.26.0 — but **`timm`, `einops` and `huggingface_hub` are missing**.
   The encoder needs only `timm` + `einops` (from `dinovol_2_eva.py:8,12`); `rope.py` and
   `patch_encode_decode.py` are pure torch. No `uv sync --extra models`, no vesuvius package import.
   **Install with `--no-deps`** — a plain `pip install timm` drags in `torchvision`, which can
   re-resolve torch and break the cu128 build. Verified working set in §6/E0.
5. **`hf_hub_download` is unavailable** until `huggingface_hub` is installed; plain HTTPS to
   `https://huggingface.co/<repo>/resolve/main/<file>` works anonymously and is what §6 uses.
   > **Measured (second pass).** `hf_hub_download` **stalled dead at 134 MB of 863 MB** and never
   > resumed (anonymous rate-limited transfer, no error raised — it just stopped). `curl -L -C -
   > --retry 5` works but the rate wandered **140–500 kB/s** across the transfer, i.e. **35–90 min**
   > for the 863.6 MB encoder. Use curl, not `hf_hub_download`, not `Invoke-WebRequest` (no resume).
   > **Start the download first and do all of E0's other work while it runs** — thanks to
   > `e0_keycheck.py` (§6/E0) none of the structural verification needs the weights at all.
6. **`head_dim = 54` kills fused attention — patch it or pay 10×.** `embed_dim 864 / num_heads 16`
   = 54, and PyTorch requires `head_dim % 8 == 0` for flash / mem-efficient / cuDNN SDPA. All three
   refuse; `F.scaled_dot_product_attention` falls back to `MATH` **silently**. Measured at
   `N=4096, head_dim=54, bf16` on the 4090: `MATH` 292 ms vs `cuDNN` at `head_dim=64` 1.32 ms
   (**221×**). Whole 128³ forward: **12.8 s stock**. The fix is exact, not an approximation —
   zero-pad q/k/v to `head_dim=56` and pass `scale=54**-0.5` explicitly (SDPA would otherwise use
   `1/sqrt(56)`), then slice the output back to 54. Zero-padded q·k is unchanged, softmax is
   unchanged, zero-padded v contributes only zero output columns. Verified: fp32 max relative error
   **3.0e-06**, whole-model bf16 token cosine **0.99994**. Result: **12.8 s → 1.3 s** per 128³
   window. Drop-in patch in §10.2. This is architecture-intrinsic, so it hits the official pipeline
   and any RunPod 5090 too — it is not a local quirk.
7. **The local GPU is a 4090 _Laptop_ (17.2 GB), not a desktop 4090.** All wall-clock numbers in
   §8 are measured on it. Peak VRAM for the patched encoder is **< 1 GB** at every window shape
   tested, so memory is never the constraint — throughput is.

---

## 6. The plan. E0 → E1 → E2 → E3, each gated.

Everything lives in `trackD/hunt/`. Nothing proceeds past a failed gate.

### E0 — prototype geometry + vendoring (no GPU, ~20 min, $0)

```powershell
cd C:\Users\benbl\Desktop\Vsuvious

# deps: --no-deps so pip cannot re-resolve torch via torchvision. VERIFIED working set.
.\.venv\Scripts\python.exe -m pip install --no-deps timm einops safetensors huggingface_hub `
    tqdm filelock requests packaging typing_extensions pyyaml hf_xet
# -> timm 1.0.28, einops 0.8.2, huggingface_hub 1.27.0, torch stays 2.11.0+cu128

# vendor the 4 upstream forward-path files (provenance-matched to the weights)
$B = "https://raw.githubusercontent.com/ScrollPrize/dinovol/main/dinovol_2"
New-Item -ItemType Directory -Force trackD\hunt\dinovol_min | Out-Null
foreach ($f in @("model/dinov2_eva.py","model/rope.py","model/patch_encode_decode.py","model/model.py","dataset/normalization.py")) {
  Invoke-WebRequest "$B/$f" -OutFile ("trackD\hunt\dinovol_min\" + ($f -replace '/','__'))
}
```
```bash
# the encoder (863.6 MB, anonymous, no token). USE CURL — hf_hub_download stalls at ~134 MB
# and Invoke-WebRequest has no resume. ~33 min at the ~440 kB/s anonymous rate.
mkdir -p /d/vesuvius-data/trackD/models/dinovol
curl -L -C - --retry 5 --retry-delay 5 \
  -o /d/vesuvius-data/trackD/models/dinovol/teacher_backbone_ps8_paris4_352500.pt \
  "https://huggingface.co/scrollprize/dinovol_v2_ps8_with_paris4_352500/resolve/main/dinovol_v2_ps8_paris4_step352500_teacher_backbone.pt"
```

Write `trackD/hunt/dinovol_min/loader.py` reproducing `load_backbone_from_checkpoint` +
`compute_patch_embedding_grid` verbatim from `assets/dinovol_src/dinovol_2__eval__napari_visualizer.py`
(lines 328, 474–720), with `open_zarr_handle` swapped for our own zarr/s3fs open so the heavy
`ssl_zarr_dataset` → augmentation import chain is not pulled in.

Then `trackD/hunt/e0_prototype_geometry.py` — re-derives §1 Findings C-1/2/3 and writes
`trackD/hunt/out/e0_prototype.json`. Also build and freeze the three scorers:
`S_mean` (cosine to `avg_ref`), `S_nn` (max cosine over the 128 tokens),
`S_maha` (whitened score using the 128-token covariance + shrinkage, since 128 ≪ 864).

**Gate E0.** Backbone loads `strict=True` with **zero** missing/unexpected keys, and a random
`(1,1,128,128,128)` input yields `x_norm_patchtokens` of shape `(1, 4096, 864)`. Any key mismatch →
stop and re-check §5.1 before spending another minute.

> ### Gate E0's structural half is **PASSED**, proven without downloading the weights.
>
> `trackD/hunt/e0_keycheck.py` builds the architecture from the checkpoint's *embedded*
> `config.model` via villa's `build_dinovol_2_backbone`, then compares its `state_dict()` against the
> checkpoint's tensor manifest read over **HTTP range requests (0.11 MB total)**:
>
> ```
> checkpoint teacher.backbone tensors: 463   (HTTP 0.11 MB)
> villa-built state_dict tensors:      463
> in checkpoint but NOT in built model : 0
> in built model but NOT in checkpoint : 0
> name matches but SHAPE differs       : 0
> STRICT-LOAD PREDICTION: PASS
> ```
>
> Params **215.9 M**, exactly the model card's figure. A random `(1,1,128,128,128)` forward already
> returns `(1, 4096, 864)`. **K-a is retired** — `load_state_dict(strict=True)` cannot fail; the only
> thing the 863 MB download can add is a numerical surprise, not a structural one. Run
> `e0_keycheck.py` first on any future checkpoint: it costs 0.11 MB and 20 s, and it turns "does the
> loader work" from a 40-minute download question into a coffee-length one.
>
> ### Gate E0 — **PASSED on the real released weights.** `out/e0_gate.json`
>
> ```
> checkpoint 863,606,099 bytes | top-level {step: 352500, config, teacher}
> STRICT: teacher keys 463 | missing 0 | unexpected 0        <- as predicted
> params 215.9 M
> raw token norm median 28.475                               <- NOT 1. K-j confirmed.
> NOISE NULL cos(token, avg_ref):  min 0.251  p50 0.357  p95 0.409  max 0.447  frac>0.5 = 0.0000
> NOISE NULL max-cos over the 128 expert tokens: p50 0.364  max 0.489
> agreement cosine(stock, padded) on real weights, fp32: min 0.9999998  median 1.000000
> ```
>
> **K-i is retired — but read the number carefully, because it cuts both ways.** Pure Gaussian noise
> never once clears the published τ = 0.5 across all 4,096 tokens (max 0.447), so τ = 0.5 is *not*
> a meaningless threshold and the release is not obviously broken. **However**, the noise null's
> p95 is **0.409** and the expert ink tokens' p05 is **0.416** (§1, Finding C-3). Those are the same
> number. So the entire usable dynamic range of this detector is roughly:
>
> | | cos vs `avg_ref` |
> |---|---|
> | pure Gaussian noise, p50 → p95 → max | 0.357 → 0.409 → 0.447 |
> | expert ink clicks, p05 → p50 → max | 0.416 → **0.600** → 0.760 |
>
> ~0.36 to ~0.60, with the bottom 5 % of genuine ink indistinguishable from *noise*. Blank papyrus
> is structured, not noise, so it will not sit at 0.357 — it will sit somewhere between, and where
> exactly **is** the E1 question. This measurement makes E1's matched-blank design mandatory rather
> than merely good practice: the noise null is a **floor**, not a null. Report every E1 AUC together
> with the blank distribution's p50/p95, not just the AUC.
>
> **K-j is confirmed, on real weights:** the median raw token norm is **28.475**, not 1. Anyone who
> dots villa's `Dinov2Backbone.forward` output against `avg_ref_embedding` without `F.normalize`
> gets numbers ~28× too large that still look like plausible scores. This is the single easiest way
> to produce a confident wrong answer on this route.
>
> The exactness of the §10.2 patch is now verified on the **released weights**, not just random
> ones: fp32 token-wise `cosine(stock, padded)` **min 0.9999998, median 1.000000**. (This gate run
> was fp32, where the fix buys 11.0 s → 6.4 s = 1.7×; the ~10× figure is bf16, where the MATH
> fallback is relatively far more expensive. Run inference in **bf16 + patched**.)

### E1 — THE KILL TEST: resolution ladder on human letters (local 4090, ~1 h GPU, $0)
_Second pass: run it on **four** labelled 0139 segments, and **sample windows** rather than tile —
see the two corrections below. Both make it cheaper *and* stronger._

Native 128³ crops, centred on mesh points, at both resolutions. Meshes are pre-registered per
volume so there is no transform to derive.

```powershell
cd C:\Users\benbl\Desktop\Vsuvious

# 1. meshes only — 15 MB per segment, no bulk CT. Writes the (u,v) -> (x,y,z) tables per rung.
.\.venv\Scripts\python.exe trackD\hunt\e1_fetch_meshes.py --segments w035,w039,w040,w041,w032

# 2. choose window centres from the labels, matched on radius + local CT mean.
#    ~100 ink + ~100 blank per segment per rung. Writes out/e1_windows.json.
.\.venv\Scripts\python.exe trackD\hunt\e1_sample_windows.py --n-ink 100 --n-blank 100 --seed 0

# 3. pull ONLY the sampled 128^3 windows (~210 MB/rung, not the 7.6 GB ROI) and embed them.
#    --patch-sdpa applies the exact head_dim 54->56 fix from section 10.2.
foreach ($seg in @("w035","w039","w040","w041")) {
  foreach ($res in @("9.362","2.399")) {
    .\.venv\Scripts\python.exe trackD\hunt\e1_embed.py --seg $seg --res $res --mode native --patch-sdpa
    .\.venv\Scripts\python.exe trackD\hunt\e1_embed.py --seg $seg --res $res --mode surfacevol --patch-sdpa
  }
}
# optional third rung, same pattern:  --res 1.129

# 4. score: AUC per (segment, rung, mode, scorer) against the N1-N5 nulls.
.\.venv\Scripts\python.exe trackD\hunt\e1_score.py        # writes out/e1_ladder.json + figure
```

> **Two design corrections (second pass), both free.**
>
> **(a) Sample windows, never tile the ROI at 2.399 µm.** Tiling the w035 ink bbox at 2.399 µm is
> 61 × 71 = **4,331** 128³ windows = **1.6 h** patched (**15 h** unpatched); tiling the *whole*
> 2.399 µm segment is 28,320 windows ≈ **10 h** patched. But one 128³ window already yields 4,096
> tokens spanning 307 µm³, so **~50 ink-centred + ~50 blank-centred windows (~2 min) covers the
> 1,500 + 1,500 token budget with room to spare.** Place windows, harvest tokens, discard the rest.
> Tiling is only needed if you want a picture; sampling is what answers the gate.
>
> The **I/O** argument is stronger than the GPU one. Measured anonymous S3 read of the w035
> 9.362 µm surface volume (chunks `(28,128,128)`): **1.7 MB/s** for a 7.3 MB block, **3.3 MB/s** for
> a 29.4 MB block — latency-bound, and measured while the 863 MB encoder download was competing for
> bandwidth, so treat these as a floor. At that rate the 2.399 µm ink-bbox ROI (7.6 GB) is
> **40+ min of pure download** and the whole 2.399 µm surface volume (48 GB) is hours. Sampled
> windows are **~210 MB** for 100 × 128³. Sample.
>
> Concrete pool sizes for w035 (recomputed from `assets/w035_ink2d.npy` + `w035_valid2d.npy`;
> confirms §4's 334,024 ink / 737,091 blank / 31.2 % — the raw `ink2d` has 396,153 px, 334,024 of
> them inside `valid`): **one 128³ window at 2.399 µm spans only 32.9 × 32.9 px of the 9.362 µm
> label raster**, and w035 contains **310 disjoint 33×33 label tiles that are ≥ 50 % ink**
> (513 with any ink). So ~100 ink-centred 2.399 µm windows is a comfortable draw from 310 on w035
> alone, ~1,200 across the four labelled segments — but it is *not* so large that you can be sloppy
> about drawing blanks from the same radial/intensity strata. Budget ~100 ink + ~100 blank per
> segment per rung.
>
> **(b) Run the gate on four labelled segments, not one.** All of `w035, w039, w040, w041` have
> local `*_ink.npy`/`*_sup.npy` at `D:\vesuvius-data\trackD\` whose shapes match their 9.362 µm
> surface volumes **exactly** — `(28, 5820, 5240)`, `(28, 8560, 7720)`, `(28, 6400, 7980)`,
> `(28, 6200, 8020)` — and all four also publish 2.399 µm and 1.129 µm surface volumes (ratios
> 3.884–3.890). `w032` has the same three-rung ladder and is the unlabelled negative. Four AUCs cost
> ~4× a few minutes and convert "one number" into a spread; a gate decided by a single segment is
> exactly the failure mode this project has already been burned by three times.

Sampling: **1,500 ink patch-locations and 1,500 blank patch-locations**, drawn from
`w035_ink2d.npy` / `w035_valid2d.npy`, blanks stratified to match the ink set's radial position and
local CT mean so the contrast is not a proxy for "where on the sheet". For each, score
`S_mean`, `S_nn`, `S_maha` at the surface patch and at every depth offset in ±3 patches, and keep
the per-depth curve — the depth curve is the direct test of the "meshes sit off the recto"
hypothesis from the project state.

Nulls, all of them, on identical patch sets:
- **N1 random prototype** — unit vector drawn from the token-mean-centred empirical distribution.
- **N2 matched-tightness decoy prototypes** — 128 randomly drawn *blank* tokens, mean-prototyped
  the same way; 50 draws → an AUC null distribution, not a point.
- **N3 z-reversed surface volume** — our existing symmetry control (control r = 0.076,
  GP corpus 0.22–0.91).
- **N4 label permutation** — block-permuted ink mask, 200 draws.
- **N5 an unlabelled PHerc0139 segment** (w032, already staged in `D:\vesuvius-data\trackD\w032\`)
  as an out-of-label negative.

Primary readout: **AUC(ink vs matched blank), 2.399 µm vs 9.362 µm, per scorer**, against the N2
null band.

> **Gate E1 — this is where the route lives or dies.**
> - **AUC at 2.399 µm ≤ 0.65 for all three scorers** → **ROUTE DEAD, stop.** In-domain, native
>   resolution, a scroll inside the encoder's own training set, human ground truth, the released
>   prototype. There is no excuse left; the failure is ours or the prototype's, and either way
>   embedding prospecting has nothing to offer at 9 µm.
> - **AUC at 2.399 µm ≥ 0.85 but AUC at 9.362 µm ≤ 0.65** → **the GP-13 arm is dead** (they are all
>   9 µm-only). Do **not** run the corpus pass. Instead: this is a clean, quantitative
>   resolution-threshold result for an official Vesuvius model on human ground truth; it belongs in
>   §4 of the report as direct support for rescan priority. Proceed to E3 only.
> - **AUC at 9.362 µm ≥ 0.80 with the N2 null band below 0.60** → proceed to E2.
> - Anything between → report the number, do not proceed on hope.
>
> **Second pass — read the gate on the 4-segment spread, not the mean.** With `w035/w039/w040/w041`
> you get four AUCs per cell. Require the gate to hold on **at least 3 of 4** and report the range.
> If the four disagree wildly (say 0.9 / 0.55 / 0.88 / 0.6) the honest conclusion is *"the prototype
> transfers to some sheets and not others"*, which is a **different** finding from either pass or
> fail and must not be averaged into one number. Also: fix the τ question first — E0's noise null
> (K-i) tells you whether to threshold at the published 0.5 or at a per-volume quantile. AUC is
> threshold-free so the gate itself is safe either way, but every downstream map is not.

Also recorded regardless of outcome, because it is cheap and each is independently useful:
the 1.129 µm rung; the native-vs-surface-volume delta (tests flattening as a confound); the
per-depth curve (tests the off-recto hypothesis); and `S_nn`/`S_maha` vs `S_mean` (tests whether the
released mean prototype is leaving signal on the table — Finding C-2 predicts it is).

### E2 — corpus pass, ONLY if E1 clears at 9.362 µm (RunPod, **$8–10 patched / $16–20 unpatched**)

Re-use `runpod/render_tifxyz_sv.py` + `runpod/survey_segments.py` on the same 80 segments
(`runpod/segment_catalog.json`), adding an embedding stage to the existing render→infer loop.

> **Second pass — a depth mismatch to handle explicitly.** `render_tifxyz_sv.py` defaults to
> `--num-slices 21`, centred (`offsets = (arange(n) - (n-1)/2) * slice_step`, line 35). The encoder
> needs the z axis padded to a multiple of 8, so 21 → 24, and the RoPE `normalize_coords="separate"`
> rescales z over a **24-deep** axis when the model only ever saw 128. That is §5.3's warning applied
> to the corpus arm, and it is not avoidable there — the surface volume is all E2 has. Two
> consequences: (i) render with `--num-slices 24` (or 32) so the pad is not an edge-replicated lie,
> and (ii) **E2's decoy-prototype null must be computed at the same window depth**, because the
> depth distortion shifts the whole cosine distribution, not just the ink tokens. Comparing an
> ink score at depth 24 to a null computed at depth 128 is exactly the kind of unmatched comparison
> §7's anti-patterns warn about.
Sizing measured from w035: a whole 9.362 µm segment is ~2,160 windows of 1,024 tokens ≈ **10¹⁵ FLOP,
i.e. 1–2 min of 5090 time** — the render and S3 streaming dominate, so this is the previous
$5.5 survey plus ~30–50 %. Ship the per-segment cosine map alongside the existing prediction map and
run the **same** 4-test verdict battery + tripwire (`salvage/verdict_*.py`), with the N2 decoy-
prototype null computed per segment so "it fired" is measured against that segment's own null.

### E3 — the actually-good odds: PHerc1203's 2.403 µm band (RunPod, ~$3–5)

**This is the one place where embedding prospecting is in-domain.** PHerc1203's 2.403 µm band
(volume `20260319130212`, `[15137, 26493, 26493]`, verified present and public) sits at the
encoder's exact training resolution and protocol (2.399–2.401 µm, 77–78 keV) even though 1203 itself
is not in the pretraining list — confirmed against the embedded `config.dataset` volume list in §3,
which contains no 1203.

> **Second pass, strengthening E3.** The protocol match is closer than "same ballpark":
> `PHerc0009B/20250820154339-**2.401µm**-0.3m-**77keV**` is in the pretraining set, and 1203's band
> is `**2.403µm**-0.2m-**77keV**`. Same beam energy, 2 nm voxel difference. Of every arm in this
> plan, E3 is the only one where the encoder is being asked to do something it was actually trained
> to do — which is why §9 puts it 2–3× above E2 despite costing less. The prior
screen there failed in the specific way a prototype-similarity score is designed to fix: the
`ink_3d_dino_guided` **classifier** blanket-fired (0 silent tiles in 29,748, §3B of the report).
A cosine score with a per-tile decoy-prototype null is the natural control for exactly that failure.

We already own the tile grid: `trackD/salvage/tiles.parquet` — 496,786 rows with `z,y,x,pmax,f05,
f08,meanct,radius,rnorm,theta`. Re-score **those identical tiles** in embedding space and publish
the head-to-head. Deliverables either way: (a) classifier-vs-prototype agreement map on identical
coordinates, (b) does the prototype score have silent regions where the classifier had none — the
direct test of whether embedding space is the more honest detector under domain shift.

E3 is independent of Gate E1's 9 µm branch and should run **even if the GP-13 arm dies**, because it
does not need 9 µm to work.

---

## 7. What kills this route — stated in advance

| # | Killer | How we see it | When |
|---|---|---|---|
| K-a | ~~Backbone will not load cleanly~~ | ~~`strict=True` missing/unexpected keys~~ | **RETIRED** — 463/463 keys match by name and shape, proven from the manifest for 0.11 MB (`e0_keycheck.py`) |
| K-b | **Prototype does not separate ink from blank at 2.399 µm on w035** | AUC ≤ 0.65 for all scorers, in-domain, human labels | **E1 — primary kill** |
| K-c | Separates at 2.4 µm, collapses at 9.362 µm | AUC ≥ 0.85 → ≤ 0.65 across the rung | E1 — kills the **GP-13** arm, not the finding |
| K-d | Fires but is indistinguishable from decoys | ink AUC inside the N2 50-draw null band | E1/E2 |
| K-e | Score tracks geometry, not ink | AUC collapses under radial/CT-mean matching, or forward≈reversed (N3) | E1 |
| K-f | Signal is really the flattening | native-crop AUC ≫ surface-volume AUC, and the surface arm is what E2 would use | E1 |
| K-g | Prototype is Paris4-specific | 2.399 µm AUC high on Paris4 crops but low on 0139 w035 | E1 (add a Paris4 positive control if E1 is ambiguous) |
| K-h | Corpus fires everywhere, as the classifier did | per-segment silent-tile fraction ≈ 0 | E2 — stop the fleet, as before |
| K-i | ~~τ = 0.5 is not a meaningful threshold at all~~ | ~~pure-Gaussian noise already scores ≈ 0.5~~ | **RETIRED** — measured noise null max **0.447**, `frac>0.5 = 0.0000`. But the noise p95 (0.409) equals the expert-ink p05 (0.416), so the usable range is only ~0.36–0.60: **always report the blank distribution alongside the AUC** |
| K-j | Silent numerical wrongness from the wrapper | villa's `Dinov2Backbone.forward` returns **un-L2-normalised** tokens; cosines computed on them are meaningless but look plausible | **CONFIRMED on real weights** — median raw token norm **28.475**. `F.normalize(dim=-1)` is not optional |
| K-k | **The prototype's dynamic range is too narrow to survive any domain shift** | ink p50 0.600 sits only 0.24 above the noise floor 0.357; a blank-papyrus p50 anywhere near 0.5 leaves almost no margin | E1 — this is the quantitative form of K-b, and it is now measured rather than feared |

Two anti-patterns this project has already been burned by, restated: (i) an AUC computed against
*unmatched* blanks measures position on the sheet, not ink — hence the radial/intensity matching in
E1; (ii) a single high score on one segment out of 80 is a multiple-comparisons artefact — hence the
per-segment decoy null in E2 rather than a corpus-wide threshold.

---

## 8. Cost

**Measured encoder throughput** (4090 Laptop, bf16, random weights, exact released config —
`scratchpad/e0_speed.py`, `e0_batch.py`). Stock = as villa ships it; patched = §10.2.

| window | stock | patched | speedup | Mvox/s patched | peak VRAM |
|---|---|---|---|---|---|
| 128³ (4096 tokens) | **12,754 ms** | **1,295 ms** | 9.8× | 1.62 | 0.56 GB |
| 64×128×128 (2048 tok) | 3,566 ms | — | — | — | 1.13 GB |
| 32×128×128 (1024 tok), B=8 | 1,172 ms | **286 ms** | 4.1× | 1.83 | 0.65 GB |

Batching past B≈2 buys almost nothing (1.7–1.8 Mvox/s plateau) and VRAM never exceeds 0.9 GB.

**Corpus sizing is now exact, not estimated.** The 80 segments already surveyed total
**1.249e9 surface pixels** (`out/survey/survey_all.json`), i.e. **4.0e10 voxels** at 32 padded
slices, ×1.32 for 1-patch window overlap → **5.3e10 voxels**.

| stage | compute | wall clock | $ |
|---|---|---|---|
| E0 deps + geometry | local CPU | ~10 min | **$0** |
| E0 encoder download, 863.6 MB, anonymous | network | **35–90 min** — sustained rate wandered 140–500 kB/s over the transfer; `hf_hub_download` hung outright at 134 MB. Start it first and do everything else while it runs. | **$0** |
| E1 kill test — 4 segments × 3 rungs, **window-sampled** (~100 windows/rung) + all nulls | local 4090 | **~1 h GPU**, +~20 GB S3 | **$0** |
| E1 if you tile the 2.399 µm ink bbox instead of sampling | local 4090 | 1.6 h/segment patched, **15 h unpatched** | $0 but don't |
| E2 corpus pass, **patched** (5.3e10 vox @ ~4 Mvox/s on a 5090) | 4× RTX 5090 @ $0.69/h | ~1 h | **≈$3 GPU + ~$5.5 render/S3 = $8–10** |
| E2 corpus pass, **unpatched** | 4× RTX 5090 | ~3.5 h | **$16–20** |
| E3 PHerc1203 2.403 µm re-score on the existing tile grid | 2× RTX 5090 | ~2 h | **$3–5** |

The render + S3 half of E2 is not a guess: the previous 80-segment survey burned **6.34 aggregate
GPU-hours for $5.5** (`survey_all.json` `secs` field summed). The embedding stage adds ~2.6–3.3
GPU-hours *if patched*. **The §5.6 patch is the difference between E2 fitting in the remaining $45
and eating half of it.**

Against ~$45 remaining of the shared $80 cap. **The decision-relevant experiment is free.** Nothing
goes to RunPod until Gate E1 has produced a number.

**Software to build** (revised second pass — three files already exist):

| file | status | ~lines |
|---|---|---|
| `e0_prototype_geometry.py` | **written & run** → `out/e0_prototype.json`, `out/e0_scorers.npz` | 90 |
| `e0_sdpa_backends.py`, `e0_sdpa_pad_exactness.py`, `e0_speed.py`, `e0_fastattn.py`, `e0_batch.py` | **written & run** (second pass) | 40–90 ea. |
| `e0_gate.py` | **written**, runs the moment the 863 MB download lands | 140 |
| `peek_pt.py` | **written & run** — remote `.pt` manifest over HTTP range | 130 |
| `e0_keycheck.py` | **written & run** (second pass) — 463/463 key+shape match from 0.11 MB of HTTP; retires K-a | 45 |
| `dinovol_min/loader.py` | to write — normalise + pad + window + `F.normalize`, transcribed from `napari_visualizer.py:474–760` | ~150 |
| `e1_fetch_meshes.py` | to write — but **copy the S3/caching pattern from the existing `trackD/hunt/fetch_meshes.py`**, which already caches tifxyz to `hunt/meshcache/`. It only pulls the 9.362 µm rung for 10 Investigation-D segments; E1 needs the 2.399 µm and 1.129 µm meshes for the five 0139 segments. | ~70 |
| `e1_sample_windows.py` | to write — matched ink/blank window centres | ~90 |
| `e1_embed.py` | to write — pull sampled windows, embed, score with `S_mean`/`S_nn`/`S_maha` | ~150 |
| `e1_score.py` | to write — AUC + N1–N5 nulls; reuses `salvage/verdict_common.py` | ~200 |

Reused unchanged: `runpod/render_tifxyz_sv.py` (but see the `--num-slices` note in E2),
`runpod/survey_segments.py`, `runpod/fleet_*.py`, `salvage/verdict_*.py`.

**Do not forget** the two silent-wrong-answer traps when writing `loader.py`: L2-normalise the
tokens yourself (§2 correction), and apply the head_dim pad (§5.6) or accept a 10× bill.

---

## 9. Calibrated odds

- ~~**P(Gate E0 passes)** ≈ 0.92.~~ → **E0 PASSED, P = 1.** Real weights, `strict` load with
  0 missing / 0 unexpected of 463 keys, forward returns `(4096, 864)`, noise null measured.
  The `use_fused_attn` worry was real but was a 10× *speed* problem with an exact fix, not a
  correctness one. `out/e0_gate.json`.
- **P(2.399 µm separation on w035, AUC ≥ 0.85)** ≈ **0.7**. In favour: in-domain resolution,
  0139 is in the SSL training set, human ground truth, the prototype came from this exact procedure.
  Against: the prototype was clicked on *Paris4*, not 0139; Finding C-2 says it is diffuse; and
  `S_mean` at τ=0.5 only recovers 83 % of its own clicks.
  > **Second pass: nudge down to ≈ 0.6.** The E0 noise null narrowed the usable band more than
  > expected — noise p95 **0.409** vs expert-ink p05 **0.416**. AUC ≥ 0.85 needs blank papyrus to
  > land clearly below ~0.45, and blank papyrus is structured carbonised material that this encoder
  > represents richly, not white noise. Nothing measured says it will. This is a *reduction in the
  > headroom*, not new evidence against the route — but the honest move is to widen the interval,
  > not keep the old point estimate now that the floor is known.
- **P(9.362 µm separation, AUC ≥ 0.80 | 2.399 µm passes)** ≈ **0.15**. This is the honest number.
  A patch is 74.9 µm at 9 µm, the encoder has literally never seen a 9 µm voxel, K1/K2 already
  placed 9 µm at or below the information ceiling for the intensity channel, and 80/80 GP segments
  came back null with a *validated* 9 µm instrument.
- **P(new, real letters on a GP-13 scroll via the E2 corpus pass)** ≈ **0.02–0.03**, and with the
  revised 2.399 µm term above, **0.015–0.025**
  (0.7 × 0.15 × ~0.2 for "a genuinely independent detector finds text the 9 µm classifier missed on
  the same 80 segments that already returned null"). **This route is not a plausible First-Letters
  path through the published GP corpus.**
- **P(real letters via E3, PHerc1203 2.403 µm)** ≈ **0.05–0.08** — 2–3× better than E2, because it
  is the only arm where the encoder is in-domain, and because the failure it is designed to correct
  (blanket-firing classifier under domain shift) is one we have measured rather than assumed.
- **P(a publishable, prize-relevant result regardless of letters)** ≈ **0.8**. The resolution-ladder
  curve on human ground truth, the 128-vs-256 release bug, the ≤83 % self-recall of the published
  τ=0.5 rule, and a classifier-vs-prototype head-to-head on 29,748 identical tiles are each
  standalone contributions, and all four survive a negative result.

**Recommendation.** Run E0+E1 now — they are free, they take an afternoon, and E1 answers the
question that decides everything downstream. Do not book a pod until `out/e1_ladder.json` exists.
If E1 kills the 9 µm arm (most likely single outcome), skip E2 entirely and put the compute into E3,
which does not depend on 9 µm working.

---

## 10. Independent verification pass — 2026-08-17, second session

> **State of play. E0 IS DONE.** Deps installed and verified; encoder downloaded in full to
> `D:\vesuvius-data\trackD\models\dinovol\teacher_backbone_ps8_paris4_352500.pt`
> (**863,606,099 bytes**, verified); **Gate E0 PASSED** on the real weights →
> `trackD/hunt/out/e0_gate.json`. `e0_gate.py` refuses to run on a short file, so a truncated
> resume cannot silently produce garbage.
>
> **The next command to run is E1's, not E0's.** Nothing in E0 remains.

Every claim in §1–§4 was re-derived from scratch, without reading this document's outputs, before
comparing. Result: **§1, §3, §4 reproduce exactly; §2 and §5 needed the corrections inlined above;
§8 was wrong by ~10× and is rewritten.**

### 10.1 Reproduced exactly

| claim | independent result |
|---|---|
| `recorded_embeddings*.npy` byte-identical | **True**, md5 `2c30bfbefaf905bc40761f10e5649928` both; `np.unique` over the concatenation → **(128, 864)** |
| `cos(normalise(mean(128)), avg_ref)` | **1.00000000** |
| ‖mean of 128 unit tokens‖ | **0.5907** |
| pairwise cos among the 128 | min −0.037 · p05 0.144 · **med 0.334** · p95 0.580 · max 0.889 |
| cos(token, avg_ref) | min 0.255 · p05 0.416 · **med 0.600** · max 0.760 |
| PCA | PC1 9.4 % · PC1–3 24.6 % · PC1–10 48.3 % · **56 PCs for 90 %** |
| self-recall vs τ | 0.40→0.961 · 0.45→0.883 · **0.50→0.828** · 0.60→0.500 · 0.70→0.109 |
| encoder gating | `dinovol_v2_ps8_with_paris4_352500` **gated=False private=False**; `dinovol_v2` and `dinovol_v2_ps8` both **HTTP 401** — §3's narrowing of `REPRODUCIBILITY.md` is correct |
| anonymous weight fetch | `HTTP 200`, `content-length 863606099` |
| checkpoint manifest (HTTP range, 0.11 MB / 7 requests) | `{step: 352500, config, teacher}`; **463** keys all `backbone.`-prefixed; `down_projection.proj.weight (864,1,8,8,8)`; per-block `rope_embed.mix_frequencies (16,27,3)` |
| Finding C-4, all 11 pretraining volumes | **2.215–2.401 µm**, verbatim from the embedded `config.dataset`; `config.dataset` has **no** `normalization_scheme` and **no** `intensity_properties` → `"robust"` default confirmed |
| w035 three-rung ladder | 9.362 µm `(28, 5820, 5240)` · 2.399 µm `(109, 22640, 20400)` · 1.129 µm `(116, 24080, 21700)`, all present, plus 5 per-volume tifxyz meshes |
| normalisation recipe | `DEFAULT_NORMALIZATION_SCHEME = "robust"` (napari_visualizer.py:20); `RobustNormalization` = clip p1/p99, −median, ÷1.4826·MAD (normalization.py:177–236) |

### 10.2 New: the `head_dim = 54` fused-attention trap, and the exact fix

`torch.profiler` on a stock 128³ forward (11.17 s self-CUDA) — the linear layers are **4 %** of it:

```
aten::bmm            3.231 s   28.9 %   72 calls   <- 3 per block, attention matmuls
aten::_softmax       2.165 s   19.4 %   24 calls
aten::where          1.676 s   15.0 %   24 calls   } MATH-backend masked-softmax
aten::isneginf       1.300 s   11.6 %   24 calls   } fallback machinery
ampere_sgemm_128x128_nn  1.909 s 17.1 %            <- FP32 SGEMM, no tensor cores
aten::addmm          0.436 s    3.9 %  120 calls   <- cutlass bf16 tensorop (qkv/proj/fc1_g/fc1_x/fc2)
```

Backend probe at `N=4096, bf16`:

| head_dim | FLASH | MEM_EFF | CUDNN | MATH |
|---|---|---|---|---|
| **54** | refused | refused | refused | **292 ms** |
| 64 | refused¹ | 5.03 ms | **1.32 ms** | 331 ms |

¹ this Windows torch build was not compiled with flash; irrelevant — 54 fails the `%8` check anyway.
Torch's own message: *"Mem efficient attention requires last dimension of inputs to be divisible by
8. Got Query.size(-1): 54"*.

Exactness of the pad, measured: `N∈{1024,2048,4096}`, **fp32 max relative error 2.0e-06 – 3.4e-06**
(pure accumulation-order noise), bf16 3.6e-03 (= bf16 epsilon). End-to-end on the full 24-block
model: token-wise `cosine(stock, patched)` **min 0.999920, median 0.999945**.

```python
# drop-in replacement for EvaAttention.forward in
# villa/vesuvius/src/vesuvius/models/build/pretrained_backbones/dinovol_2_eva.py
# (only the attention call changes; everything else is byte-identical to upstream)
        hd = q.shape[-1]
        pad = (-hd) % 8
        if pad:
            q, k, v = F.pad(q, (0, pad)), F.pad(k, (0, pad)), F.pad(v, (0, pad))
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
            scale=hd ** -0.5,          # MUST be explicit: SDPA would use 1/sqrt(56)
        )
        if pad:
            x = x[..., :hd]
```

Reproduce (all in `trackD/hunt/`, run with `.\.venv\Scripts\python.exe`, no weights needed —
they build the architecture from the released config with random init):

```powershell
cd C:\Users\benbl\Desktop\Vsuvious
.\.venv\Scripts\python.exe trackD\hunt\e0_sdpa_backends.py        # which backends accept head_dim 54
.\.venv\Scripts\python.exe trackD\hunt\e0_sdpa_pad_exactness.py   # pad is exact + how much faster
.\.venv\Scripts\python.exe trackD\hunt\e0_speed.py                # stock throughput, 3 window shapes
.\.venv\Scripts\python.exe trackD\hunt\e0_fastattn.py             # end-to-end gain + token agreement
.\.venv\Scripts\python.exe trackD\hunt\e0_batch.py                # batch scaling, patched
.\.venv\Scripts\python.exe trackD\hunt\e0_gate.py                 # THE gate: real weights, strict load, noise null
```

**This is worth a villa PR on its own** — it is a ~10× inference speedup of an official Vesuvius
release, exact to float rounding, in ~8 lines, and it applies to every downstream user of
`dinovol_v2`. Independent of whether embedding prospecting finds anything.

### 10.3 New assets found that the plan should use

- **Four labelled PHerc0139 segments, not one.** `w035, w039, w040, w041` all have local
  `*_ink.npy` + `*_sup.npy` matching their 9.362 µm SV shapes exactly, and all four (plus `w032`)
  publish the full 1.129 / 2.399 / 9.362 µm ladder. See §6/E1 correction (b).
- **Native scroll volumes are public and chunked exactly 128³.**
  `PHerc0139/volumes/20260102150214-2.399um-…-masked.zarr` → `[76953, 26511, 26511]`, chunks
  `[128,128,128]`; `…20250728140407-9.362um-…` → `[20974, 6621, 6621]`, same chunking. A
  chunk-aligned 128³ native window is therefore **one chunk fetch** — the native-crop arm of E1 is
  cheaper than the surface-volume arm, not more expensive.
- **The label → native-3D mapping is exact and needs no code we don't have.** Both w035 meshes carry
  `meta.json: {"scale": [0.05, 0.05], "format": "tifxyz"}`, i.e. the tifxyz grid is 1/20 of the
  surface-volume raster:

  | mesh | grid | ×20 | matching SV shape |
  |---|---|---|---|
  | `…-on-20250728140407-9.362um.tifxyz` | `(291, 262)` float32, 0.9 MB | `(5820, 5240)` | `(28, 5820, 5240)` ✓ |
  | `…-on-20260102150214-2.399um.tifxyz` | `(1132, 1020)` float32, 13.9 MB | `(22640, 20400)` | `(109, 22640, 20400)` ✓ |

  So a labelled pixel `(v,u)` in the 9.362 µm frame is the normalised surface coordinate
  `(v/5820, u/5240)`, which indexes straight into the 2.399 µm mesh to give `(x,y,z)` in the native
  2.399 µm volume. **Zero registration, zero derived transform, 15 MB of mesh.** `-1` is the
  no-data sentinel in x/y/z — mask it. This is what makes E1's native arm a few minutes of work
  rather than a project.
- Disk headroom: **740 GB free on D:**, so the 2.399 µm pulls are unconstrained.

### 10.4 What did *not* change

The kill test is still "does the released prototype separate ink from matched blank at 2.399 µm on
human-labelled 0139", and the most likely single outcome is still that the 9 µm arm dies. The
verification pass made the route **cheaper, faster and better-instrumented**, and retired two of
the ten killers (K-a, K-i) — but it moved the headline probability *down* slightly, not up: the
measured noise floor (0.357/0.409) sits closer to the expert-ink p05 (0.416) than anyone would want,
so `P(2.399 µm separation)` goes 0.7 → ~0.6 and the end-to-end letters number with it. **E0 being a
clean pass is not evidence that E1 will pass.** E0 tested our plumbing; E1 tests the physics.

### 10.5 Where this stands, in one line

Everything that can be established without looking at real ink is now established: the encoder
loads, runs, is 10× faster than it was, and its score is not trivially saturated. **The next
number that matters is an AUC on w035's letters, and nothing else should be run before it.**
