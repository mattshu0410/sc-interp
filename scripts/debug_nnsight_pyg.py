import pickle
from pathlib import Path

import torch
import torch_geometric
from torch_geometric.data import Batch, Data


REPO_ROOT = Path(__file__).resolve().parent.parent


def show(b, label):
    keys_obj = getattr(b, "keys")
    keys = keys_obj() if callable(keys_obj) else keys_obj
    print(f"  {label}: keys={list(keys)} pert_idx={getattr(b,'pert_idx','<missing>')}")


def main() -> None:
    print("PyG:", torch_geometric.__version__)

    print("\n=== synthetic flat pert_idx through Batch.from_data_list ===")
    d1 = Data(x=torch.tensor([[1.0, 2.0]]), pert_idx=[5, 10])
    d2 = Data(x=torch.tensor([[3.0, 4.0]]), pert_idx=[-1])
    show(Batch.from_data_list([d1, d2]), "synthetic batch")

    print("\n=== inspect actual pkl on disk ===")
    pkl_path = REPO_ROOT / "data" / "norman" / "data_pyg" / "cell_graphs.pkl"
    print("pkl path:", pkl_path, "exists:", pkl_path.exists())
    if not pkl_path.exists():
        return
    ds = pickle.load(open(pkl_path, "rb"))
    print("pkl type:", type(ds).__name__,
          "len:", len(ds) if hasattr(ds, "__len__") else "?")
    if isinstance(ds, dict):
        keys_to_check = list(ds.keys())[:3]
        for k in keys_to_check:
            entries = ds[k]
            sample = entries[0] if isinstance(entries, list) and entries else entries
            keys_obj = getattr(sample, "keys")
            keys = keys_obj() if callable(keys_obj) else keys_obj
            print(f"  pert={k!r}: type={type(sample).__name__} keys={list(keys)}"
                  f" has_pert_idx={hasattr(sample, 'pert_idx')}"
                  f" pert_idx={getattr(sample, 'pert_idx', '<missing>')}")
    else:
        sample = ds[0] if hasattr(ds, "__getitem__") else next(iter(ds))
        keys_obj = getattr(sample, "keys")
        keys = keys_obj() if callable(keys_obj) else keys_obj
        print(f"  sample[0]: type={type(sample).__name__} keys={list(keys)}"
              f" has_pert_idx={hasattr(sample, 'pert_idx')}")


if __name__ == "__main__":
    main()
