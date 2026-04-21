# Changelog

## 0.1.0 (2024-04-10)

Initial release.

- Python reimplementation of kNN-TN from the GSimp R package
- Truncated normal MLE via Newton-Raphson
- Correlation and truncation distance modes
- Cython-accelerated inner loop with pure-Python fallback
- Parity with R at `rtol=1e-10` across 14 reference datasets
- 1.3-17x faster than R depending on dataset size
