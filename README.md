# Text to Image Generation

This project integrates a Transformer text encoder with Stable Diffusion 1.5 using a curated COCO image-caption subset. M1 establishes the data and generation baseline. M2 makes the conditioning path explicit, reweights selected token vectors, and compares guidance, sampling steps, samplers and beta schedules. No model parameters are trained.

## Data and model

The existing dataset contains 1,200 unique COCO 2017 validation images, with one cleaned caption per image and an internal 1,000/100/100 train/validation/test split. Images use RGB conversion, aspect-preserving resize and padding to 512 x 512. M2 reuses 12 fixed validation captions; the test split remains unused. Source and license records are retained.

SD uses its own tokenizer and token-level Transformer states of shape `[1, 77, 768]`. The separate pooled CLIP vectors from M1 are used for embedding checks. The SD revision is pinned in `configs/m2.json`.

## Notebooks

The notebooks contain saved results. Read 04 for conditioning and 05 for the experiments.

| Notebook | Contents |
| --- | --- |
| [01](01_prepare_dataset.ipynb) | Data preparation, deduplication and split checks |
| [02](02_clip_embeddings.ipynb) | CLIP image/text embedding checks on 32 validation pairs |
| [03](03_baseline_generation.ipynb) | M1 fixed-prompt baseline and reference comparisons |
| [04](04_conditioning_integration.ipynb) | Explicit conditions, token weighting and equivalence checks |
| [05](05_conditional_experiments.ipynb) | Study A analysis, Study B experiments and paired comparisons |

## M2 experiments

Study A compares eight predefined configurations, not a full factorial grid, varying guidance, steps, sampler, DDIM timestep spacing and selected-token weighting (12 captions × three seeds × eight configurations = 288 outputs). Token weighting scales contextual vectors for the spans declared in `configs/m2.json` by 1 or 1.2; other vectors remain unchanged.

Study B adds 108 outputs comparing scaled-linear, linear and cosine beta schedules with DDIM and other sampling settings fixed. The checkpoint remains frozen. Exact configurations and conditioning details are in the M2 report (pp. 2–3) and [Notebook 05](05_conditional_experiments.ipynb).

Token weighting helps selected omissions but can damage structure or relations. More guidance or steps do not consistently improve alignment. Linear often increases contrast, while cosine commonly softens detail; these beta results include the effect of a mismatch with the checkpoint's training schedule.

## Code and records

| File | Purpose |
| --- | --- |
| `p2_pipeline.py` | Data processing, embedding checks and M1 generation |
| `p2_experiments.py` | Explicit conditioning, smoke validation and Study A generation |
| `p2_noise_schedule.py` | Study B beta-schedule experiments |
| `config.json` | Original data and baseline settings |
| `configs/m2.json`, `configs/m2_prompts.json` | M2 settings, target spans and fixed inputs |
| `requirements.txt` | Recorded dependency versions |
| `artifacts/m2/` | Study A summary records and the `noise_schedule/` Study B records |

## Reproduction

Use Python 3.10, CUDA and the versions in `requirements.txt`. Run commands from the project directory. For initial data preparation and the M1 baseline, execute notebooks 01–03 in order.

M2 requires the curated dataset and the cached SD checkpoint specified in `configs/m2.json`:

1. Execute 04 to validate the raw prompt and the explicit conditions at weights 1 and 1.2.
2. Generate Study A: `python p2_experiments.py sweep`.
3. Execute 05 to analyze Study A and run the Study B comparisons.

Completed runs can be verified and reused. The saved 04 displays six records from the original smoke run. 
