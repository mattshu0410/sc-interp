# sc-interp

Benchmarking foundation models for single-cell perturbation prediction. Designed so that adding a new foundation model or a new dataset is a localised change rather than a cross-cutting refactor.

## Quickstart

```bash
git clone https://github.com/mattshu0410/sc-interp.git
cd sc-interp
./setup.sh
```

`setup.sh` builds the tools venv, the docs venv, and one venv per model under `models/<name>/.venv`. On a machine without an NVIDIA driver >= 580 it will install one and prompt you to reboot before re-running.

To run the Norman 2019 benchmark as a quick end-to-end sanity check, pulling our fine-tuned weights from HuggingFace instead of retraining:

```bash
./experiments/norman.sh
```

This downloads Norman, fetches scGPT and CellFlow weights from HuggingFace, runs inference, and scores both with cell-eval. Results land in `predictions/` and `eval_outputs/`. Takes about 25 minutes on a fresh VM.

## Motivation

Benchmarking single-cell foundation models runs into two kinds of friction.

**Model side:**

1. Different models expect different input shapes. scGPT takes GEARS's `PertData` wrapper with torch_geometric cell graphs. CellFlow takes an AnnData with gene_1/gene_2 columns, ESM2 embeddings, and a PCA projection. A third model will take something else.
2. Models need incompatible Python environments. scGPT pins torch 2.3 + torchtext + gears; CellFlow needs JAX + flax + optax. Each model lives in its own venv, and a shared framework has to be importable from all of them.
3. Repeated boilerplate across runners (CLI flags, cache-or-retrain logic, train-stats persistence, HuggingFace Hub upload) invites drift the moment a second runner exists.

**Data side:**

1. Runners can easily couple themselves to a specific data source's on-disk format. The moment a dataset arrives in a different shape (e.g. Tahoe 100M as a HuggingFace drug perturb-seq), any coupled runner needs a new read path.
2. Splits matter for test leakage. Single-cell perturbation benchmarks care deeply about holding out the right perturbations, cell lines, or gene combinations. The field has converged on specific split definitions (`simulation`, `combo_seen0`, `combo_seen1`), and the choice is not an afterthought.
3. Gene IDs vary: Ensembl (`ENSG00000157557`), gene symbols (`ETS2`), Entrez (`2114`). Column names vary across datasets. A silent translation error produces garbage ESM embeddings and wrong results, not a crash.

## Architecture

```
       python -m scripts.run <runner> --dataset <name>
                           │
                           ▼
                    scripts/run.py              dispatcher, lazy import
                           │
                           ▼
                  scripts/runner.py             run(spec, argv)
                           │                    calls 5 callables on spec
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
      scripts/run_scgpt.py       scripts/run_cellflow.py
         exports SPEC               exports SPEC
            │                             │
            └─────────────┬───────────────┘
                          ▼
            shared services, called from any runner:

            scripts/cli.py          common CLI flags
            scripts/manifest.py     Manifest (obs + var schemas)
            scripts/cache.py        cache_or_train
            scripts/data/splits.py  canonical split JSON read/write
            scripts/data/genes.py   symbol→id with validation


  dataset materialisation, runs once (tools venv):

  python -m scripts.data.gears --dataset norman
       │
       ▼
  data/norman/splits/simulation_42_0.75.json
```

`scripts/run.py` picks a runner by name from a `REGISTRY` dict and lazy-imports only that runner's module, so the dispatcher itself is safe to import from any venv. Each runner exports `SPEC = RunnerSpec(...)` with five callables (`add_args`, `load_inputs`, `train_or_load`, `predict`, `save_predictions`). The orchestrator in `scripts/runner.py` calls them in order and knows nothing about what's inside them. Shared services for CLI parsing, caching, manifest validation, split loading, and gene ID translation live once under `scripts/` and get called from every runner.

## Adding a new model

1. Create `models/setup_<name>.sh` that builds `models/<name>/.venv` with the upstream dep stack.
2. Write `scripts/run_<name>.py` with five functions (one per step of the runner flow) and a `SPEC = RunnerSpec(...)` at the bottom. The functions own everything model-specific: shaping the data into the model's expected input, training, inference, and writing the canonical prediction h5ad.
3. Add one line to `REGISTRY` in `scripts/run.py`:
   ```python
   REGISTRY = {
       "scgpt":    "scripts.run_scgpt",
       "cellflow": "scripts.run_cellflow",
       "<name>":   "scripts.run_<name>",
   }
   ```
4. Run it: `source models/<name>/.venv/bin/activate && python -m scripts.run <name> --dataset norman`.

Shared infrastructure (common flags, manifest schema, `cache_or_train`, wandb wrapper, prediction h5ad layout) comes from the framework for free.

## Adding a new dataset

1. Create `data/manifests/<name>.yaml` declaring `name`, `source` (the loader type, e.g. `gears` or `tahoe`), `obs` (perturbation and control columns), and optionally `var` (gene ID type and symbol column).
2. If the source is new, add `scripts/data/<source>.py` with a `materialise(manifest, ...)` function that downloads the raw data, computes the split, and calls `scripts.data.splits.write_split` to emit the canonical JSON.
3. Add a handler for the new source to each runner's `LOADERS` dict that needs to consume it.
4. Materialise once from the tools venv:
   ```bash
   source tools/.venv/bin/activate
   python -m scripts.data.<source> --dataset <name>
   ```

Runners then read the canonical split through `scripts.data.splits.load_split` and do not care which source produced it.
