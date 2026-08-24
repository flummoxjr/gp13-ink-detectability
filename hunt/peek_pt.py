"""Peek inside a remote torch .pt (zip) via HTTP range requests: read data.pkl only."""
import io
import json
import pickle
import sys
import urllib.request
import zipfile

URL = sys.argv[1]


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])
            self.final = r.url
        self.nbytes = 0
        self.nreq = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0:
            return b""
        end = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(
            self.final,
            headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "curl/8"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        self.nbytes += len(data)
        self.nreq += 1
        self.pos += len(data)
        return data


TENSORS = {}


class Stub:
    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return f"<{self.tag}>"


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, *a):
    return {"__tensor__": True, "dtype": storage.get("dtype"), "shape": tuple(size)}


def rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad, backward_hooks, dtype, *a):
    return {"__tensor__": True, "dtype": str(dtype), "shape": tuple(size)}


class Unp(pickle.Unpickler):
    def persistent_load(self, pid):
        # pid = ('storage', dtype_cls, key, location, numel)
        try:
            return {"dtype": str(pid[1]), "key": pid[2], "numel": pid[4]}
        except Exception:
            return {"dtype": None}

    def find_class(self, mod, name):
        if mod == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return rebuild_tensor_v2
            if name == "_rebuild_tensor_v3":
                return rebuild_tensor_v3
        if mod.startswith("torch") or mod.startswith("numpy") or mod.startswith("collections"):
            if name == "OrderedDict":
                return dict
            return lambda *a, **k: Stub(f"{mod}.{name}")
        return lambda *a, **k: Stub(f"{mod}.{name}")


f = HTTPRangeFile(URL)
print(f"remote size = {f.size/1e6:.1f} MB")
z = zipfile.ZipFile(f)
names = z.namelist()
pkls = [n for n in names if n.endswith("data.pkl")]
print("zip members:", len(names), "| data.pkl:", pkls)
raw = z.read(pkls[0])
print(f"data.pkl = {len(raw)/1e6:.2f} MB")
obj = Unp(io.BytesIO(raw)).load()
print(f"bytes fetched over HTTP: {f.nbytes/1e6:.2f} MB in {f.nreq} requests")
print("=" * 60)


def summarize(o, path="", depth=0):
    if depth > 2:
        return
    if isinstance(o, dict):
        if o.get("__tensor__"):
            print(f"{path}: TENSOR {o['dtype']} {o['shape']}")
            return
        print(f"{path or '<root>'}: dict, {len(o)} keys")
        for k in list(o)[:400]:
            summarize(o[k], f"{path}.{k}" if path else str(k), depth + 1)
    elif isinstance(o, (list, tuple)):
        print(f"{path}: {type(o).__name__} len={len(o)}")
    else:
        print(f"{path}: {o!r}"[:300])


summarize(obj)
if isinstance(obj, dict) and isinstance(obj.get("config"), dict):
    print("=" * 60)
    print("EMBEDDED CONFIG (json-safe subset):")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        return repr(o)

    print(json.dumps(clean(obj["config"]), indent=1)[:8000])
