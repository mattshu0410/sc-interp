from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ActivationRecord:
    name: str
    tensor: torch.Tensor
    metadata_tags: dict[str, str]


class ActivationSink(Protocol):
    def write(self, record: ActivationRecord) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> ActivationSink: ...

    def __exit__(self, *exc: object) -> None: ...
