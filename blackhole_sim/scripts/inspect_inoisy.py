import h5py
from pathlib import Path

path = Path("../aart/inoisy.h5")

with h5py.File(path, "r") as f:
    print(f"Inspecting {path.resolve()}")
    print()

    def walk(name, obj):
        if hasattr(obj, "shape"):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{name}/")

    f.visititems(walk)