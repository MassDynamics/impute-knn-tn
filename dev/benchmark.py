"""Benchmark: R vs Python kNN-TN imputation."""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from impute_knn_tn import impute_knn

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "reference")


def load_and_run(name, distance, n_runs=3):
    """Load a reference dataset and time Python imputation."""
    config = f"{name}_{distance}"
    d = os.path.join(REF_DIR, config)
    inp = pd.read_csv(os.path.join(d, "input_matrix.csv"), index_col=0)
    log2_file = os.path.join(d, "log2.txt")
    use_log2 = os.path.exists(log2_file) and open(log2_file).read().strip() == "true"

    mat = inp.values.astype(np.float64)
    if use_log2:
        mat = np.log2(mat)

    # Warmup
    impute_knn(mat, k=5, distance=distance, perc=0.01)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        impute_knn(mat, k=5, distance=distance, perc=0.01)
        times.append(time.perf_counter() - t0)

    r_timing_file = os.path.join(d, "timing.txt")
    r_time = (
        float(open(r_timing_file).read().strip())
        if os.path.exists(r_timing_file)
        else None
    )

    return {
        "name": name,
        "distance": distance,
        "features": mat.shape[0],
        "samples": mat.shape[1],
        "nas": int(np.sum(np.isnan(mat))),
        "python_time": np.median(times),
        "r_time": r_time,
    }


def main():
    datasets = [
        ("targeted", "truncation"),
        ("targeted", "correlation"),
        ("untargeted", "truncation"),
        ("untargeted", "correlation"),
        ("real_data", "truncation"),
        ("sim", "truncation"),
        ("synthetic_1000", "truncation"),
        ("synthetic_1000", "correlation"),
        ("synthetic_5000", "truncation"),
        ("synthetic_5000", "correlation"),
    ]

    # Check which datasets exist
    available = []
    for name, dist in datasets:
        config = f"{name}_{dist}"
        if os.path.isdir(os.path.join(REF_DIR, config)):
            available.append((name, dist))

    print(
        f"{'Dataset':<20} {'Mode':<12} {'Features':>8} {'Samples':>8} {'NAs':>6} {'R (s)':>8} {'Py (s)':>8} {'Speedup':>8}"
    )
    print("-" * 90)

    for name, dist in available:
        result = load_and_run(name, dist)
        speedup = (
            result["r_time"] / result["python_time"]
            if result["r_time"]
            else float("nan")
        )
        print(
            f"{result['name']:<20} {result['distance']:<12} "
            f"{result['features']:>8} {result['samples']:>8} {result['nas']:>6} "
            f"{result['r_time'] or 0:>8.3f} {result['python_time']:>8.3f} "
            f"{speedup:>7.1f}x"
        )


if __name__ == "__main__":
    main()
