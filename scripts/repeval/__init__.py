"""Cross-condition activation analyses.

`CaptureView` loads activations + stats + labels from a capture folder.
`CONDITIONS` / `resolve` map friendly names to paths.

Each metric is its own script under `metrics/`. Outputs land at
`predictions/repeval/<metric>/<comparison>/`.
"""
from scripts.repeval.conditions import CONDITIONS, resolve
from scripts.repeval.data import CaptureView

__all__ = ["CaptureView", "CONDITIONS", "resolve"]
