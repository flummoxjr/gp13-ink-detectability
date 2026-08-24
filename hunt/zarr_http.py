"""Minimal HTTP reader for the open-data OME-zarr volumes.

All PHerc1203 volume pyramids are zarr v2, uint8, 128^3 chunks, compressor=None,
dimension_separator='/'.  Uncompressed chunks mean we can Range-read sub-blocks.
"""
import concurrent.futures as cf
import json
import os
import time

import numpy as np
import requests

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
CACHE = r"C:\Users\benbl\AppData\Local\Temp\claude\C--Users-benbl-Desktop-Vsuvious\b3441997-0118-49b2-8364-cbdf28fc6397\scratchpad\zcache"
os.makedirs(CACHE, exist_ok=True)
_S = requests.Session()


class Zarr3D:
    def __init__(self, prefix, level):
        self.prefix = f"{prefix}/{level}"
        meta = json.loads(_S.get(f"{BUCKET}/{self.prefix}/.zarray", timeout=30).text)
        self.shape = tuple(meta["shape"])
        self.chunks = tuple(meta["chunks"])
        assert meta["compressor"] is None and meta["dtype"] == "|u1"
        self.cgrid = tuple(-(-s // c) for s, c in zip(self.shape, self.chunks))

    def _chunk(self, cz, cy, cx):
        key = f"{self.prefix}/{cz}/{cy}/{cx}"
        fn = os.path.join(CACHE, key.replace("/", "_"))
        if os.path.exists(fn):
            buf = open(fn, "rb").read()
        else:
            buf = b""
            for attempt in range(6):
                try:
                    r = requests.get(f"{BUCKET}/{key}", timeout=180)
                    buf = r.content if r.status_code == 200 else b""
                    break
                except Exception:
                    time.sleep(1.5 * (attempt + 1))
            open(fn, "wb").write(buf)
        n = int(np.prod(self.chunks))
        if len(buf) != n:
            return np.zeros(self.chunks, np.uint8)
        return np.frombuffer(buf, np.uint8).reshape(self.chunks)

    def read(self, z0, z1, y0, y1, x0, x1, workers=8):
        """Read [z0:z1, y0:y1, x0:x1] (clipped to shape)."""
        z1, y1, x1 = min(z1, self.shape[0]), min(y1, self.shape[1]), min(x1, self.shape[2])
        cz, cy, cx = self.chunks
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        jobs = [(i, j, k) for i in range(z0 // cz, (z1 - 1) // cz + 1)
                for j in range(y0 // cy, (y1 - 1) // cy + 1)
                for k in range(x0 // cx, (x1 - 1) // cx + 1)]
        with cf.ThreadPoolExecutor(workers) as ex:
            for (i, j, k), blk in zip(jobs, ex.map(lambda t: self._chunk(*t), jobs)):
                za, ya, xa = i * cz, j * cy, k * cx
                zs, ze = max(z0, za), min(z1, za + cz)
                ys, ye = max(y0, ya), min(y1, ya + cy)
                xs, xe = max(x0, xa), min(x1, xa + cx)
                if zs >= ze or ys >= ye or xs >= xe:
                    continue
                out[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = \
                    blk[zs - za:ze - za, ys - ya:ye - ya, xs - xa:xe - xa]
        return out
