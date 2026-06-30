import numpy as np
from pathlib import Path

# Path to your .npz file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

filename = PROJECT_ROOT / "blackhole_sim_testing" / "observations_npz" / "frame_00.npz"


data = np.load(filename)

print(f"Contents of {filename}")
print("=" * 60)

for key in data.files:
    arr = data[key]

    print(f"\nArray: {key}")
    print(f"  Shape : {arr.shape}")
    print(f"  Dtype : {arr.dtype}")

    if arr.ndim == 0:
        print(f"  Value : {arr}")
    else:
        print("  First 10 entries:")
        print(arr[:10])

        # Optional statistics for numeric arrays
        if np.issubdtype(arr.dtype, np.number):
            if np.iscomplexobj(arr):
                print(f"  |x| min : {np.abs(arr).min():.3e}")
                print(f"  |x| max : {np.abs(arr).max():.3e}")
                print(f"  |x| mean: {np.abs(arr).mean():.3e}")
            else:
                print(f"  Min : {arr.min():.3e}")
                print(f"  Max : {arr.max():.3e}")
                print(f"  Mean: {arr.mean():.3e}")

print("\nDone.")
