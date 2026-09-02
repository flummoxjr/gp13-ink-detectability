# 1203 clean-ROI ink sweep — run ONLY when the GPU is free (Track A has priority).
# Runs ink_3d_dino_guided on the 6 surface-pred-verified clean-sheet ROIs,
# blends, sigmoids (finalize_outputs is broken upstream — see trackD/LOG.md),
# and renders overlays + on-sheet/in-void artifact metrics.
$env:TORCH_COMPILE_DISABLE = "1"
$py314 = "C:\Users\benbl\Desktop\Vsuvious\.venv314\Scripts"
$py = "C:\Users\benbl\Desktop\Vsuvious\.venv\Scripts\python.exe"
$model = "D:\vesuvius-data\trackD\models\ink3d\ckpt_78k_fullsup.pth"
$vol = "s3://vesuvius-challenge-open-data/PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr"
$outroot = "C:\Users\benbl\Desktop\Vsuvious\trackD\out\sweep_1203"

$bboxes = @(
  "10560:10944,9472:9856,16320:16704",
  "3200:3584,13120:13504,20032:20416",
  "8576:8960,8320:8704,11072:11456",
  "3008:3392,15040:15424,8128:8512",
  "13568:13952,13504:13888,19136:19520",
  "6208:6592,16704:17088,7808:8192"
)

$i = 0
foreach ($bbox in $bboxes) {
  $dir = "$outroot\roi$i"
  if (-not (Test-Path "$dir\merged_logits.zarr")) {
    & "$py314\vesuvius.predict.exe" --model_path $model --input_dir $vol `
      --output_dir $dir --bbox $bbox --device cuda --disable_tta --batch_size 1 `
      --patch_size 192,192,192 --input_anon
    & "$py314\vesuvius.blend_logits.exe" $dir "$dir\merged_logits.zarr"
  }
  $i++
}
& $py "C:\Users\benbl\Desktop\Vsuvious\trackD\sweep_1203_report.py"
