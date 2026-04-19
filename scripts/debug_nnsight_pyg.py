import inspect

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

import nnsight
import torch_geometric
from nnsight import NNsight


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
            "pert_idx:", getattr(data, "pert_idx", "<missing>"),
        )
        return self.lin(data.x)


def main() -> None:
    print("PyG:", torch_geometric.__version__)
    print("nnsight:", nnsight.__version__)
    import torch_geometric.data.data as d_mod
    print("data.py:", d_mod.__file__)
    print("Data.__getattr__ line:", inspect.getsourcelines(d_mod.Data.__getattr__)[1])

    d1 = Data(x=torch.tensor([[1.0, 2.0]]), pert_idx=[[5, 10]], pert="A+B")
    d2 = Data(x=torch.tensor([[3.0, 4.0]]), pert_idx=[[-1]], pert="ctrl")
    b = Batch.from_data_list([d1, d2])

    m = M()
    print("=== direct ===")
    m(b)

    nm = NNsight(m)
    print("=== nnsight trace ===")
    with nm.trace(b):
        nm.lin.output.save()


if __name__ == "__main__":
    main()
