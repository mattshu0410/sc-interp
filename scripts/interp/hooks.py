from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol

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


CaptureSpec = tuple[str, Callable[[Any], Any]]

_SENTINEL: object = object()


class HookManager:
    def __init__(
        self,
        nn_model: Any,
        capture: list[CaptureSpec],
        sink: ActivationSink,
        capture_dtype: torch.dtype = torch.float32,
        gate: Callable[[dict[str, str]], bool] | None = None,
        queue_depth: int = 32,
    ) -> None:
        self.nn_model = nn_model
        self.capture = capture
        self.sink = sink
        self.capture_dtype = capture_dtype
        self.gate = gate
        self.queue_depth = queue_depth
        self.current_tags: dict[str, str] = {}
        self._queue: queue.Queue[Any] | None = None
        self._writer: threading.Thread | None = None
        self._writer_exc: BaseException | None = None

    def set_tag(self, key: str, value: str) -> None:
        self.current_tags[key] = value

    def __enter__(self) -> HookManager:
        self._queue = queue.Queue(maxsize=self.queue_depth)
        self._writer_exc = None
        self._writer = threading.Thread(
            target=self._writer_loop, name="HookManagerWriter", daemon=True
        )
        self._writer.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        assert self._queue is not None and self._writer is not None
        try:
            self._queue.put(_SENTINEL)
            self._writer.join()
            if self._writer_exc is not None and exc_type is None:
                raise self._writer_exc
        finally:
            self.sink.close()

    def _writer_loop(self) -> None:
        assert self._queue is not None
        try:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    return
                self.sink.write(item)
        except BaseException as e:
            self._writer_exc = e
            while self._queue.get() is not _SENTINEL:
                pass

    @contextmanager
    def trace(self, *args: Any, **kwargs: Any) -> Iterator[None]:
        assert self._queue is not None, "HookManager.trace called outside with-block"
        saved: list[tuple[str, Any]] = []
        with self.nn_model.trace(*args, **kwargs):
            for name, accessor in self.capture:
                saved.append((name, accessor(self.nn_model).save()))
            yield
        tags = dict(self.current_tags)
        if self.gate is not None and not self.gate(tags):
            return
        for name, s in saved:
            value = getattr(s, "value", s)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Capture target {name!r} produced {type(value).__name__}, "
                    "expected torch.Tensor — accessor must resolve to a tensor "
                    "(e.g. `.output[0]` for tuple-returning modules)."
                )
            tensor = value.detach().to("cpu", dtype=self.capture_dtype)
            self._queue.put(
                ActivationRecord(name=name, tensor=tensor, metadata_tags=tags)
            )