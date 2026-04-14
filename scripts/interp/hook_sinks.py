from __future__ import annotations

from scripts.interp.hooks import ActivationRecord


class MemoryActivationSink:
    def __init__(self) -> None:
        self.records: list[ActivationRecord] = []
        self._closed = False

    def write(self, record: ActivationRecord) -> None:
        if self._closed:
            raise RuntimeError("write() called on closed MemoryActivationSink")
        stored = ActivationRecord(
            name=record.name,
            tensor=record.tensor,
            metadata_tags=dict(record.metadata_tags),
        )
        self.records.append(stored)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> MemoryActivationSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
