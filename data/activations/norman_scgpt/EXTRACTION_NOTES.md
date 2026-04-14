# scGPT Activation Extraction

## What this does

Extracts per-gene-position hidden-state activations from every transformer
layer of the scGPT whole-human foundation model, for cells in a given
dataset split. Outputs are flat `.npy` memory-mapped arrays, optionally
uploaded to a HuggingFace dataset repo.

## Architecture

```
experiments/extract_scgpt.sh      # Entry point: downloads data, runs extraction
  -> data/download_norman.sh      # Materialises Norman 2019 Perturb-seq via GEARS
  -> scripts/run.py               # CLI dispatcher (registry-based)
     -> scripts/extractor.py      # Generic extract() orchestrator (ExtractSpec)
        -> scripts/extract_scgpt.py  # scGPT-specific: tokenise, load model, extract, upload
           -> scripts/run_scgpt.py   # Shared helpers: REPO_ROOT, DEFAULT_PRETRAINED, build_vocab
           -> scripts/hf.py          # HuggingFace upload/download helpers
```

### Data flow

1. **Tokenisation** -- each cell's expression vector is converted to scGPT
   token inputs: gene IDs sorted by expression, padded to `max_seq_len`.
2. **Extraction** -- nnsight wraps the `TransformerModel` and intercepts the
   output of every `TransformerEncoderLayer` during a forward pass. Only the
   non-padded gene positions are kept.
3. **Storage** -- per-layer numpy memmaps: `layer_XX_activations.npy`
   (positions x d_model), `layer_XX_gene_ids.npy`, `layer_XX_cell_ids.npy`.
4. **Upload** -- layers are pushed to HF incrementally; `upload_progress.csv`
   tracks what's done so a restart can skip already-uploaded layers.

### Output layout

```
activations/<dataset>_scgpt/
    layer_00_activations.npy   # (total_positions, 512)  float16
    layer_00_gene_ids.npy      # (total_positions,)      int32
    layer_00_cell_ids.npy      # (total_positions,)      int32
    ...
    layer_11_*
    extraction_metadata.json
    upload_progress.csv
```

## Bugs fixed during bring-up

### 1. `torch_geometric` import crash (regex overflow in `torch._dynamo`)

**Symptom:** `re.error: missing ), unterminated subpattern` when importing
`from gears import PertData` at module load time.

**Root cause:** `run_scgpt.py` imported `from gears import PertData` at the
top level. This pulled in `torch_geometric`, which in its current version
imports `torch._dynamo` at load time. In torch 2.3.0, `_dynamo.trace_rules`
builds a regex from all `SKIP_DIRS` that exceeds Python 3.11's regex engine
limits.

**Fix:** Made the `gears` import lazy in `run_scgpt.py`:
- Added `from __future__ import annotations` so type hints like
  `pert_data: PertData` aren't evaluated at class definition time.
- Guarded `from gears import PertData` under `TYPE_CHECKING` for static
  analysis; added a runtime import inside `_load_gears()` only.

### 2. nnsight `fn=` keyword conflict

**Symptom:** `TypeError: NNsight.interleave() got multiple values for argument 'fn'`

**Root cause:** The original code passed `fn=raw_model._encode` to
`nns.trace(...)`. In nnsight 0.3.3, `trace()` forwards all kwargs to
`interleave()`, which has its own `fn` positional parameter, causing the
collision.

**Fix:** Removed the `fn=` argument. Since the model was constructed with all
auxiliary flags disabled (`do_mvc=False`, `do_dab=False`, etc.), `forward()`
calls `_encode()` internally and the transformer encoder layer activations
are identical.

### 3. NestedTensor slicing error

**Symptom:** `NotImplementedError: Could not run 'aten::slice.Tensor' with
arguments from the 'NestedTensorCUDA' backend`

**Root cause:** PyTorch 2.3+ `nn.TransformerEncoder` converts inputs to
NestedTensors when `src_key_padding_mask` is provided (for memory
efficiency). The intercepted layer outputs were NestedTensors, which don't
support regular indexing.

**Fix:** Added a `hidden.is_nested` check and `.to_padded_tensor(0.0)`
conversion before slicing. Also set
`model.transformer_encoder.enable_nested_tensor = False` as a belt-and-suspenders measure.

## Usage

```bash
# Smoke test (50 cells)
bash experiments/extract_scgpt.sh

# Full test split (~8,000+ cells)
source models/scgpt/.venv/bin/activate
python -m scripts.run extract-scgpt \
    --dataset norman \
    --split test \
    --dtype float16 \
    --hf-repo kevinychou/scgpt-activations-norman
```
