import h5py
import numpy as np
import sys

path = sys.argv[1]

print(f"\nInspecting: {path}\n")

with h5py.File(path, "r") as f:
    def walk(name, obj):
        if hasattr(obj, "shape"):
            arr = obj[()]
            print(f"{name}")
            print(f"  shape: {obj.shape}")
            print(f"  dtype: {obj.dtype}")
            try:
                print(f"  min/max: {np.nanmin(arr)} / {np.nanmax(arr)}")
            except Exception:
                pass
        else:
            print(f"{name}/")

    f.visititems(walk)