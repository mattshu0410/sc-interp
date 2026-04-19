import torch
from torch_geometric.data import Batch, Data

import torch_geometric


class GearsData(Data):
    # Tell PyG: don't try to concatenate pert_idx across cells. Keep it as
    # a per-cell list. Required because pert_idx is variable-length per cell
    # (combo=2, single=1, ctrl=1) and PyG's default collation drops it.
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "pert_idx":
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pert_idx":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def show(b, label):
    keys_obj = getattr(b, "keys")
    keys = keys_obj() if callable(keys_obj) else keys_obj
    print(f"  {label}: keys={list(keys)} pert_idx={getattr(b,'pert_idx','<missing>')}")


def main() -> None:
    print("PyG:", torch_geometric.__version__)

    # Real-shape pert_idx (flat list, variable length, like pertdata.py:530)
    print("\n=== plain Data (current GEARS) — flat variable-length pert_idx ===")
    d1 = Data(x=torch.tensor([[1.0, 2.0]]), pert_idx=[5, 10])
    d2 = Data(x=torch.tensor([[3.0, 4.0]]), pert_idx=[-1])
    show(Batch.from_data_list([d1, d2]), "after collate")

    print("\n=== GearsData subclass with __cat_dim__=None for pert_idx ===")
    g1 = GearsData(x=torch.tensor([[1.0, 2.0]]), pert_idx=[5, 10])
    g2 = GearsData(x=torch.tensor([[3.0, 4.0]]), pert_idx=[-1])
    show(Batch.from_data_list([g1, g2]), "after collate")


if __name__ == "__main__":
    main()
