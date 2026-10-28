"""Why did Kronos lose to the random walk? Cheap diagnostics on the saved Stage 3 paths (no model run)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = np.load(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs" / "stage3_paths.npz")  # optional: path to another dataset's paths
actual, k = d["actual"], d["kronos"]
print(f"windows: {len(actual)}, horizon: {actual.shape[1]}")

print("\n1) Size of moves (std across windows of the cumulative return)")
for name, p in [("actual", actual), ("kronos", k), ("drift", d["drift"]), ("momentum", d["momentum"])]:
    print(f"  {name:9s} step1 {p[:, 0].std():.5f}  end {p[:, -1].std():.5f}")

print("\n2) First forecast candle vs last observed close (should be tiny if the forecast starts where price is)")
print(f"  kronos step-1 mean {k[:, 0].mean():+.5f}, mean|.| {np.abs(k[:, 0]).mean():.5f}   actual step-1 mean|.| {np.abs(actual[:, 0]).mean():.5f}")

print("\n3) Bias (mean end return)")
print(f"  kronos {k[:, -1].mean():+.5f}   actual {actual[:, -1].mean():+.5f}")

print("\n4) Shrinking Kronos toward zero: path MAE of alpha * kronos (in-sample diagnostic only, not a tuned model)")
rw = np.abs(actual).mean()
for a in [0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    print(f"  alpha={a:4.2f}  path MAE {np.abs(a * k - actual).mean():.6f}  (random walk {rw:.6f})")

print("\n5) Step-level correlation between Kronos and actual cumulative returns (pooled, windows x steps)")
print(f"  pearson {np.corrcoef(k.ravel(), actual.ravel())[0, 1]:+.3f}")
