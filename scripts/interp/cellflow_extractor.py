from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import torch

from scripts.interp.hooks import ActivationRecord


class CellflowActivationCapture:
    def __init__(
        self,
        vf_module: Any,
        params: dict,
        capture_names: list[str],
        sink: Any,
        capture_dtype: torch.dtype = torch.float32,
        timesteps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
        gate: Callable[[dict[str, str]], bool] | None = None,
    ) -> None:
        self.vf_module = vf_module
        self.params = params
        self.capture_names = list(capture_names)
        self.sink = sink
        self.capture_dtype = capture_dtype
        # Scalar `t` with rank-2 `x_t` would fail cellflow's concat at
        # _velocity_field.py:219 (rank-1 t_encoded vs rank-2 x_encoded).
        # We build a rank-2 t `(B, 1)` per run below, so cache nothing here.
        self.timesteps = tuple(float(t) for t in timesteps)
        self.gate = gate
        self.current_tags: dict[str, str] = {}
        # Cleared after every run(): cell ids differ per batch, silent reuse
        # of stale ids across batches would misalign h5 sidecars.
        self.current_per_cell: dict[str, torch.Tensor | np.ndarray] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.current_tags[key] = value

    def set_per_cell(
        self, per_cell: dict[str, torch.Tensor | np.ndarray]
    ) -> None:
        self.current_per_cell = dict(per_cell)

    def __enter__(self) -> CellflowActivationCapture:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        try:
            self.sink.close()
        except Exception:
            if exc_type is None:
                raise

    def run(
        self,
        x: Any,
        cond: dict[str, Any],
        encoder_noise: Any,
    ) -> None:
        per_cell = self.current_per_cell
        self.current_per_cell = {}

        B = x.shape[0]
        for t in self.timesteps:
            t_jnp = jnp.full((B, 1), t, dtype=jnp.float32)
            _, variables = self.vf_module.apply(
                {"params": self.params},
                t_jnp,
                x,
                cond,
                encoder_noise,
                train=False,
                mutable=["intermediates"],
            )
            # Single D2H sync for the whole pytree of taps; per-key
            # np.asarray would force N_taps separate accelerator→host
            # transfers.
            inters = jax.device_get(variables["intermediates"])
            tags = dict(self.current_tags)
            tags["t"] = f"{t:.2f}"
            if self.gate is not None:
                g = self.gate(tags)
                if not isinstance(g, bool):
                    raise TypeError(
                        f"gate returned {type(g).__name__}, expected bool"
                    )
                if not g:
                    continue
            for name in self.capture_names:
                # sow stores values in a 1-tuple per name — [0] unwraps it;
                # a future Flax change flipping this to a list is what
                # test_capture_faithful_fixed_t guards against.
                arr = inters[name][0]
                tensor = torch.from_numpy(np.asarray(arr)).to(self.capture_dtype)
                self.sink.write(
                    ActivationRecord(
                        name=name,
                        tensor=tensor,
                        metadata_tags=tags,
                        per_cell=per_cell,
                        layout="BD",
                    )
                )
