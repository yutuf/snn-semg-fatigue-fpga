"""
Torch-free reader for a PyTorch .pth (zip) checkpoint.
Reconstructs each tensor as a numpy array by intercepting the pickle's
storage persistent-ids and rebuild calls. Works for plain state_dicts.
"""
import io
import pickle
import zipfile
import struct
import numpy as np

_DTYPE = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _Storage:
    def __init__(self, key, dtype):
        self.key = key
        self.dtype = dtype


def load_state_dict(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    root = names[0].split("/")[0]

    try:
        byteorder = zf.read(root + "/byteorder").decode().strip()
    except KeyError:
        byteorder = "little"

    raw_cache = {}

    def read_raw(key):
        if key not in raw_cache:
            raw_cache[key] = zf.read("%s/data/%s" % (root, key))
        return raw_cache[key]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name in (
                "_rebuild_tensor_v2", "_rebuild_tensor"):
                return self._rebuild_tensor
            if module == "torch._utils" and name == "_rebuild_parameter":
                return self._rebuild_parameter
            if name in _DTYPE:
                return name  # storage type as a string marker
            if module == "torch" and name in _DTYPE:
                return name
            try:
                return super().find_class(module, name)
            except Exception:
                return lambda *a, **k: None

        def persistent_load(self, pid):
            # pid = ('storage', <StorageTypeMarker>, key, location, numel)
            typ = pid[1]
            key = str(pid[2])
            dtype = _DTYPE.get(typ if isinstance(typ, str) else getattr(typ, "__name__", ""),
                               np.float32)
            return _Storage(key, dtype)

        @staticmethod
        def _rebuild_tensor(storage, storage_offset, size, stride, *rest):
            dtype = storage.dtype
            buf = read_raw(storage.key)
            flat = np.frombuffer(buf, dtype=dtype)
            size = tuple(int(s) for s in size)
            n = 1
            for s in size:
                n *= s
            off = int(storage_offset)
            # contiguous row-major assumption (standard for saved params)
            arr = flat[off:off + n].reshape(size) if n else flat[off:off + n]
            return np.array(arr)  # copy (frombuffer is read-only)

        @classmethod
        def _rebuild_parameter(cls, data, requires_grad, backward_hooks):
            return data

    data_pkl = zf.read(root + "/data.pkl")
    up = Unpickler(io.BytesIO(data_pkl))
    obj = up.load()
    return obj


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    p = r"C:\Users\Asuss\Downloads\oss-cad-suite\snn_fatigue_final.pth"
    obj = load_state_dict(p)

    if isinstance(obj, dict) and "state_dict" in obj:
        print("container keys:", list(obj.keys()))
        sd = obj["state_dict"]
    else:
        sd = obj

    print("\n=== STATE DICT (torch-free parse) ===")
    total = 0
    for k, v in sd.items():
        if isinstance(v, np.ndarray) and v.dtype != np.bool_ and v.size:
            n = v.size
            total += n
            a = v.astype(np.float64)
            print(f"{k:34s} {str(v.shape):20s} n={n:7d}  absmax={np.abs(a).max():.5f}  "
                  f"mean={a.mean():+.5f}  std={a.std():.5f}")
        else:
            try:
                print(f"{k:34s} scalar/other -> {np.array(v).ravel()[:6]}")
            except Exception:
                print(f"{k:34s} -> {type(v)}")
    print(f"\nTOTAL PARAMS: {total}")
