# The detached-fragment surface-volume TIFFs do not share a byte layout — parse the IFD, never assume an offset

**Draft — not yet filed. All numbers below measured directly against dl.ash2txt.org on 2026-08-24
with 1 KB range reads and a hand-rolled IFD parser; nothing is inferred from documentation.**

## Summary

Anyone reading `surface_volume/NN.tif` slices by a fixed byte offset — which is the natural thing to
do for uncompressed single-strip TIFFs of this size, and which at least two independent analysis
sessions of ours did — silently corrupts their data, because the fragments ship **three different
TIFF layouts**, one of which changes **within a single fragment between layers 09 and 10**.

| fragment | byte order | IFD location | pixel data starts | dims | constant across layers? |
|---|---|---|---|---|---|
| Frag1 (`PHercParis2Fr47`, `54keV_exposed_surface`) | `MM` (big-endian) | front | **byte 260 for layers 00–09, byte 262 for layers 10–64** | 6330×8181 | **NO** |
| Frag3 (`PHercParis1Fr34`, `54keV_exposed_surface`) | `II` (little-endian) | **end of file** (offset 79,847,796) | byte 8 | 5249×7606 | yes |
| Frag6 (`PHerc51Cr4Fr8`, `…53keV_3.24um/surface_processing`) | `II` (little-endian) | **end of file** (offset 109,865,738) | byte 8 | 6205×8853 | yes |

All are uncompressed (`Compression=1`), single-strip (`RowsPerStrip` = image height), 16-bit.
`StripOffsets + StripByteCounts == Content-Length` holds for every file checked (Frag1) and
`width × height × 2 == StripByteCounts` holds everywhere.

## The Frag1 per-layer split, mechanism

Frag1's IFD sits at the front of the file and contains a variable-length ASCII tag carrying the
layer index. A one-digit index (`0`–`9`) makes the header 260 bytes; a two-digit index (`10`–`64`)
makes it 262. So:

- assuming **262** everywhere (from probing a late layer) shifts layers 00–09 by one pixel;
- assuming **260** everywhere (from probing layer 00) shifts layers 10–64 by one pixel.

We made **both** mistakes, in two independent sessions, before parsing the IFD per file. The
resulting error is a silent one-pixel lateral misregistration of a subset of layers against the
`inklabels.png` / `mask.png` ground truth — small enough to survive casual inspection, large enough
to matter for any per-pixel measurement.

Frag3 and Frag6 fail differently: their IFD is at the *end* of the file, so any code that parses
only the first kilobyte (or assumes the pixel data follows a front IFD) reads garbage — a loud
failure, at least, rather than a silent shift. But note their pixel data starts at byte **8**, so a
Frag1-derived constant applied to them is also wrong.

## Minimal repro (10 lines, no downloads beyond ~2 KB per file)

```python
import requests, struct
def strip_offset(url):
    head = requests.get(url, headers={'Range': 'bytes=0-15'}, timeout=60).content
    end = '<' if head[:2] == b'II' else '>'
    off = struct.unpack(end + 'I', head[4:8])[0]
    blob = requests.get(url, headers={'Range': f'bytes={off}-{off+2047}'}, timeout=60).content
    n = struct.unpack(end + 'H', blob[:2])[0]
    for i in range(n):
        tag, typ, cnt = struct.unpack(end + 'HHI', blob[2+i*12:10+i*12])
        if tag == 273:
            return struct.unpack(end + ('H' if typ == 3 else 'I'), blob[10+i*12:10+i*12+(2 if typ==3 else 4)])[0]

base = "https://dl.ash2txt.org/fragments/Frag1/PHercParis2Fr47.volpkg/working/54keV_exposed_surface/surface_volume"
print(strip_offset(f"{base}/05.tif"), strip_offset(f"{base}/32.tif"))   # -> 260 262
```

## Who this affects

Anyone byte-addressing the fragment surface volumes for streaming/range reads — which is the
efficient way to consume 5–7 GB stacks over HTTP. Standard TIFF readers that parse the IFD
(tifffile, PIL) are unaffected; the trap is exactly the optimization a careful engineer reaches for.

## The fix

One line of policy: **parse `StripOffsets` from each file's own IFD** (handling both byte orders
and end-of-file IFDs), never carry an offset from one file — or one layer — to another. The repro
above is a complete implementation.

## Verification trail

- Frag1: all 65 layers' IFDs parsed from the server (2026-08-21 session): layers 00–09 → 260
  (`Content-Length` 103,571,720), layers 10–64 → 262 (103,571,722), `StripByteCounts` 103,571,460
  in every case. Independently re-verified for layers 0/5/9/10/11/32/64 on 2026-08-22.
- Frag3: layers 00/09/10/32/64 parsed 2026-08-24 — `II`, IFD @ 79,847,796, data @ 8, all identical.
- Frag6: layers 00/09/10/32 parsed 2026-08-24 — `II`, IFD @ 109,865,738, data @ 8, all identical.
- A prior internal survey had recorded Frag3 as big-endian `MM`; that was wrong, and is corrected by
  the direct parse above — a live demonstration of why the assumption keeps failing.
