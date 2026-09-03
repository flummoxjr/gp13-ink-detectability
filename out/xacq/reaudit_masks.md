# xacq null re-audit: mask convention (2026-09-03)

Frozen harness = joint-valid mask (value > 2/255 in both maps). Footprint = 11-px closing of each
map's valid region with enclosed holes < 64x64 px filled; the null rolls B's footprint WITH B's calls.

| stratum | n | conv | r real | r roll | r rot | L99 real | L99 roll | L99 rot | L95 | L90 | L99>=5 | survive roll | survive rot | zero-frac A/B in footprint |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 112 | jointvalid | 0.559 | 0.013 | -0.008 | 25.094 | 1.109 | 0.865 | 9.4 | 4.905 | 103 | 100 | 101 | 0.0/0.002 |
| all | 112 | footprint | 0.558 | 0.009 | -0.007 | 25.116 | 1.106 | 0.864 | 9.404 | 4.902 | 103 | 101 | 101 | 0.0/0.002 |
| main | 75 | jointvalid | 0.479 | 0.002 | -0.015 | 25.571 | 0.926 | 0.442 | 8.553 | 4.465 | 70 | 68 | 69 | 0.001/0.001 |
| main | 75 | footprint | 0.48 | 0.002 | -0.015 | 25.599 | 0.925 | 0.442 | 8.57 | 4.46 | 70 | 69 | 69 | 0.001/0.001 |
| Paris4 | 37 | jointvalid | 0.625 | 0.029 | -0.0 | 23.606 | 1.376 | 1.038 | 10.264 | 5.827 | 33 | 32 | 32 | 0.0/0.002 |
| Paris4 | 37 | footprint | 0.624 | 0.029 | -0.0 | 23.588 | 1.375 | 1.038 | 10.249 | 5.824 | 33 | 32 | 32 | 0.0/0.002 |
