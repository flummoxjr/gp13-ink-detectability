# Step 0 identity rule for w042 (written 2026-08-25 BEFORE computing the correlations)

Question: is 20260206000000-w042 the same physical wrap as training segment
20260108000000-w041 (mesh NN median 9.2 vox @9.362um; true-duplicate calibration
w045/w046 = 4.4 vox, 46% <4 vox; adjacent-wrap calibration 17-20 vox, ~0.5% <4 vox;
w042 shows 7% <4 vox)?

Test: sample both segments' published ds8 ink maps (arm A = volume 20260102150214)
at 3D-corresponding mesh points (NN correspondence over the 9.362um tifxyz grids,
pairs restricted to 3D distance < 25 vox), Pearson r over pairs.

Controls, same machinery:
- same-wrap control r_ctrl = corr(w045, w046)  [expect high]
- adjacent-wrap nulls r_null = max over corr(w040, w041), corr(w043, w044)  [expect low]

Decision (fixed now):
- DROP w042 as duplicate-of-w041 if r(w042,w041) > (r_ctrl + r_null)/2.
- KEEP w042 if r(w042,w041) < (r_ctrl + r_null)/2, PROVIDED the controls separate:
  r_ctrl >= 0.50 and r_null <= 0.35. If controls fail to separate, the test is
  inconclusive and w042 is DROPPED anyway (premise not verified = not usable).
