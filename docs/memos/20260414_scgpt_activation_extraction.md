Memo: scGPT Activation Extraction — Status
                                                                                                               
What we built
                                                                                                                    
A pipeline that extracts per-gene-position hidden states from every transformer layer of scGPT and ships them to   
HuggingFace dataset repos.                                                                                         
                                                                                                                    
- Input (per cell): nonzero expressed genes, sorted by expression, padded to max_seq_len=1200.                     
- Output (per layer): layer_XX_activations.npy (total_positions, 512) + parallel gene_ids and cell_ids arrays.
Norman = 12 layers, 512-d.                                                                                         
- Code layout: ExtractSpec in scripts/extractor.py owns orchestration (memmaps, checkpoints, HF upload).
scripts/extract_scgpt.py owns scGPT-specific tokenisation, model load, and nnsight layer taps. Running and         
extracting are deliberately separate scripts (different data access, different compute, and run_scgpt is
fine-tuning only — we extract from the base model).                                                                
- Validation: scripts/tests/validate_scgpt_activations.py does structural, distinctness, and reconstruction checks.
                                                                                                                    
1000-cell Norman calibration run (today)                                                                           
                                                                                                                    
- Extraction: 220 s at ~4.5 cells/s → 9.3 GB local (12 layers @ float32).                                          
- Upload: all 12 layers pushed to kevinychou/scgpt-activations-norman-calib-1k-f32.
- Validation: structural + distinctness + reconstruction all pass.                                                 
- Storage estimate correction: the docstring assumed ~800 expressed genes/cell; real average is ~381. Full Norman ≈
~870 GB float32 / ~435 GB float16 — well within the 1.56 TB free on this machine, though still painful for local  
dev.                                                                                                               
                                                                                                                    
Projected full run                                        

- Test split (~10k cells): ~3 min extraction, 15–30 min upload.                                                    
- Full (~91k cells): ~30 min extraction, 2–5 hr upload for ~435 GB float16.
                                                                                                                    
7 commits on scgpt-activations                            
                                                                                                                    
1. .gitattributes — line-ending hygiene for Windows/WSL.                                                           
2. Bug fixes — torch 2.3 + nnsight 0.3.3 compat: lazy gears import (avoids torch._dynamo regex overflow), drop
conflicting fn= kwarg, handle NestedTensor layer outputs.                                                          
3. Move activations under data/activations/; .gitignore keeps the tree + notes + upload_progress.csv but excludes
arrays.                                                                                                            
4. --tag flag for side-by-side extractions (different dtypes/cell counts don't clobber each other).
5. Smoke-test experiment (experiments/20260413_extract_scgpt.sh) + EXTRACTION_NOTES.md.                            
6. Validator with 3-stage checks.                                                                                  
7. 1000-cell float32 calibration experiment.                                                                       
                                                                                                                    
Preprocessing fix: binning (today)

Discovered that the pretrained checkpoint (args.json) was trained with input_style=binned,
n_bins=51. Norman adata.X contains log1p-normalized floats (range ~0.68–5.4), not bin
integers — we were passing raw floats through ContinuousValueEncoder, which is
out-of-distribution relative to pretraining. Fixed by applying scgpt.preprocess.binning()
per cell before tokenisation: nonzero expressed genes → quantile bins 1..n_bins, zeros
stay 0. A --n-bins 51 CLI flag was added (defaults to 51, matching the checkpoint).

The 1000-cell calib activations extracted earlier today are invalid (wrong preprocessing)
and need to be re-extracted once preprocessing decisions are finalised.

Open questions to align on                                
                                                                                                                    
1. Zero-gene filtering. Currently only nonzero-expressed genes are tokenised (~381/cell).
Fine-tuning uses include_zero_gene=all (all 5045 Norman genes, randomly subsampled to
max_seq_len=1200). Including zeros gives the model full gene-context but means ~24% gene
coverage per cell at max_seq_len=1200. Decision pending before re-running extraction.
2. Layer selection. Most mech-interp is layer-by-layer — worth adding --layers 5,6,7 so local dev doesn't need to  
store all 12?
3. Pre-MLP activations. Currently extracting post-MLP layer outputs (layer.output). For
CLTs, pre-MLP activations (layer.norm1.output) are also needed. Can be captured in the
same nnsight trace at no extra inference cost — doubles storage. Worth adding alongside
the post-MLP arrays.
4. Generalisation to CellFlow / scLDM. These are diffusion-style models with timesteps, not discrete transformer   
layers, so the layer_XX naming and total_positions concept don't transfer. Proposal: parametrise extractor.py on   
n_components, d_model, total_positions generically; let each extract_<model>.py own tokenisation + component
naming. Shared gears/PertData loading can move into scripts/data/gears.py.                                         
5. Shared GEARS loading. run_scgpt._load_gears and extract_scgpt._load_inputs duplicate PertData.load + 
prepare_split. Consolidating is cheap and makes sense before the next model.  
6. Extracting fine-tuned models and SAE training. This has proven the pipeline works, next steps are to consider extracting from fine-tuned and training SAEs.