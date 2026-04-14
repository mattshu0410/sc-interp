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
                                                                                                                    
Open questions to align on                                
                                                                                                                    
1. Layer selection. Most mech-interp is layer-by-layer — worth adding --layers 5,6,7 so local dev doesn't need to  
store all 12?
2. Generalisation to CellFlow / scLDM. These are diffusion-style models with timesteps, not discrete transformer   
layers, so the layer_XX naming and total_positions concept don't transfer. Proposal: parametrise extractor.py on   
n_components, d_model, total_positions generically; let each extract_<model>.py own tokenisation + component
naming. Shared gears/PertData loading can move into scripts/data/gears.py.                                         
3. Shared GEARS loading. run_scgpt._load_gears and extract_scgpt._load_inputs duplicate PertData.load + 
prepare_split. Consolidating is cheap and makes sense before the next model.  
4. Extracting fine-tuned models and SAE training. This has proven the pipeline works, next steps are to consider extracting from fine-tuned and training SAEs.