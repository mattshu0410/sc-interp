from __future__ import annotations

from pathlib import Path

import pytest

from scripts.cli import common_parser


def _minimal(extra: list[str] | None = None) -> list[str]:
    return ["--dataset", "norman"] + (extra or [])


def test_defaults_leave_capture_inert() -> None:
    args = common_parser().parse_args(_minimal())
    assert args.capture_activations is False
    assert args.activation_out is None
    assert args.capture_dtype == "fp32"
    assert args.capture_format == "h5"


def test_capture_activations_flag_enables() -> None:
    args = common_parser().parse_args(_minimal(["--capture-activations"]))
    assert args.capture_activations is True


def test_activation_out_parses_as_path() -> None:
    args = common_parser().parse_args(
        _minimal(["--activation-out", "predictions/foo.h5"])
    )
    assert isinstance(args.activation_out, Path)
    assert args.activation_out == Path("predictions/foo.h5")


def test_capture_dtype_choices() -> None:
    args = common_parser().parse_args(_minimal(["--capture-dtype", "fp16"]))
    assert args.capture_dtype == "fp16"

    with pytest.raises(SystemExit):
        common_parser().parse_args(_minimal(["--capture-dtype", "fp64"]))


def test_capture_format_choices() -> None:
    args = common_parser().parse_args(_minimal(["--capture-format", "memory"]))
    assert args.capture_format == "memory"

    with pytest.raises(SystemExit):
        common_parser().parse_args(_minimal(["--capture-format", "parquet"]))


def test_limit_num_batches_default_none() -> None:
    args = common_parser().parse_args(_minimal())
    assert args.limit_num_batches is None


def test_limit_num_batches_parses_int() -> None:
    args = common_parser().parse_args(_minimal(["--limit-num-batches", "8"]))
    assert args.limit_num_batches == 8


def test_batches_per_shard_default_none() -> None:
    args = common_parser().parse_args(_minimal())
    assert args.batches_per_shard is None