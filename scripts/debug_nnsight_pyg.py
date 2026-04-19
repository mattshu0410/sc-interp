import inspect
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

import nnsight
import torch_geometric
from nnsight import NNsight


REPO_ROOT = Path(__file__).resolve().parent.parent


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, data):
        keys_obj = getattr(data, "keys")
        keys = keys_obj() if callable(keys_obj) else keys_obj
        print(
            "  type:", type(data).__name__,
            "keys:", list(keys),
            "has pert_idx:", hasattr(data, "pert_idx"),
        )
        return self.lin(data.x)


def part1_synthetic_batch() -> None:
    print("=== PART 1: synthetic batch through trivial NNsight ===")
    d1 = Data(x=torch.tensor([[1.0, 2.0]]), pert_idx=[[5, 10]], pert="A+B")
    d2 = Data(x=torch.tensor([[3.0, 4.0]]), pert_idx=[[-1]], pert="ctrl")
    b = Batch.from_data_list([d1, d2])

    m = M()
    print("direct:")
    m(b)

    nm = NNsight(m)
    print("trace:")
    with nm.trace(b):
        nm.lin.output.save()
    print()


def part2_real_dataloader_batch() -> None:
    print("=== PART 2: real gears dataloader batch (pre-trace) ===")
    from gears import PertData

    pert_data = PertData(str(REPO_ROOT / "data"), default_pert_graph=False)
    pert_data.load(data_name="norman")
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=64, test_batch_size=64)

    loader = pert_data.dataloader["test_loader"]
    batch = next(iter(loader))

    keys_obj = getattr(batch, "keys")
    keys = keys_obj() if callable(keys_obj) else keys_obj
    print("  type:", type(batch).__name__)
    print("  keys:", list(keys))
    print("  num_graphs:", batch.num_graphs)
    print("  has pert_idx:", hasattr(batch, "pert_idx"))
    print("  has pert:", hasattr(batch, "pert"))
    print("  has de_idx:", hasattr(batch, "de_idx"))
    if hasattr(batch, "pert_idx"):
        pi = batch.pert_idx
        print("  pert_idx type:", type(pi).__name__,
              "len:", len(pi) if hasattr(pi, "__len__") else "N/A",
              "head:", pi[:3] if hasattr(pi, "__getitem__") else pi)

    print("\n--- now on CUDA (if available) ---")
    if torch.cuda.is_available():
        batch.to("cuda")
        keys_obj = getattr(batch, "keys")
        keys = keys_obj() if callable(keys_obj) else keys_obj
        print("  keys after .to('cuda'):", list(keys))
        print("  has pert_idx:", hasattr(batch, "pert_idx"))
    else:
        print("  (no CUDA; skipping)")


def main() -> None:
    print("PyG:", torch_geometric.__version__)
    print("nnsight:", nnsight.__version__)
    import torch_geometric.data.data as d_mod
    print("data.py Data.__getattr__ line:",
          inspect.getsourcelines(d_mod.Data.__getattr__)[1])
    print()

    part1_synthetic_batch()
    part2_real_dataloader_batch()


if __name__ == "__main__":
    main()
