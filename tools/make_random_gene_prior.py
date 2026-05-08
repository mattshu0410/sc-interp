"""Generate a random Gaussian gene-prior table matching the shape and per-row
statistics of an existing ESM safetensors. Used as a non-protein control.

Per-row matching: for gene i, sample from N(mean_i, std_i^2) where mean_i and
std_i are computed from row i of the reference table. Preserves per-gene
magnitude while destroying per-dim content.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True,
                   help="ESM safetensors to copy shape + dtype + per-row stats from")
    p.add_argument("--output", type=Path, required=True,
                   help="output safetensors path")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--key", default="embeddings")
    p.add_argument("--mode", choices=["per_row", "global"], default="per_row",
                   help="per_row: match each gene's row mean/std (preserves "
                        "magnitude variation). global: single Gaussian over all "
                        "elements (strictest non-protein null).")
    args = p.parse_args()

    ref = load_file(str(args.reference))
    if args.key not in ref:
        raise KeyError(f"{args.reference} has no key {args.key!r}; got {list(ref)}")
    e = ref[args.key]
    print(f"reference: {tuple(e.shape)} dtype={e.dtype}  "
          f"per-element mean={e.mean().item():.4f} std={e.std().item():.4f}  "
          f"per-row L2 norm mean={e.norm(dim=1).mean().item():.4f} "
          f"std={e.norm(dim=1).std().item():.4f}")

    g = torch.Generator().manual_seed(args.seed)
    if args.mode == "per_row":
        row_mean = e.mean(dim=1, keepdim=True)            # (n_rows, 1)
        row_std = e.std(dim=1, keepdim=True)              # (n_rows, 1)
        rand = torch.randn(e.shape, generator=g, dtype=e.dtype) * row_std + row_mean
    else:
        mean = e.mean().item()
        std = e.std().item()
        rand = torch.randn(e.shape, generator=g, dtype=e.dtype) * std + mean

    print(f"random:    {tuple(rand.shape)} dtype={rand.dtype}  mode={args.mode}  "
          f"per-element mean={rand.mean().item():.4f} std={rand.std().item():.4f}  "
          f"per-row L2 norm mean={rand.norm(dim=1).mean().item():.4f} "
          f"std={rand.norm(dim=1).std().item():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({args.key: rand}, str(args.output))
    print(f"==> wrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
