# Text-to-Image Generation — Milestone 1

This project establishes a text-to-image baseline using a COCO image-caption subset, pretrained CLIP embeddings and Stable Diffusion 1.5. Milestone 1 covers data preparation, image/text embedding checks and generation from fixed captions. No model parameters are trained in this milestone.

## Dataset

The dataset contains 1,200 unique COCO 2017 validation images, with one cleaned caption per image. The internal split is 1,000 training, 100 validation and 100 test pairs. Embedding checks and generation prompts use validation pairs; the test split is reserved for later work.

Captions are cleaned and filtered, and exact image and caption duplicates are removed. Images are converted to RGB, resized with preserved aspect ratio and padded to 512 x 512 pixels. Source and license records are retained with the dataset.

## Notebooks

The executed notebooks include saved outputs for review. Start with 03 to view generated examples and the qualitative analysis.

| Notebook | Contents |
| --- | --- |
| [01 — Data preparation](01_prepare_dataset.ipynb) | Caption cleaning, image preprocessing, deduplication and split checks |
| [02 — CLIP embeddings](02_clip_embeddings.ipynb) | Paired text/image embeddings and similarity checks on 32 validation pairs |
| [03 — Baseline generation](03_baseline_generation.ipynb) | Fixed prompts, generated/reference comparisons and visual analysis |

The baseline generates 12 images using PNDM sampling, 30 steps and guidance scale 7.5. All samples are retained for comparison, including counting errors and omitted objects.

## Repository Structure

```text
*.ipynb             Data preparation, embeddings and baseline generation
p2_pipeline.py      Shared processing, model inference and image comparison utilities
config.json         Dataset, seed and generation settings
requirements.txt    Recorded package versions
environment.json    Recorded server environment
evidence/           Saved experiment data and comparison images
```

Execution writes processed data to `data/curated_v1/` and results to `results/m1_baseline_v1/`. The saved evidence contains the full dataset manifest and 12 reference images; the complete image dataset and model weights are stored separately from Git.

## Reproduce the Baseline

Use Python 3.10 with the server's existing GPU environment. `requirements.txt` records the installed versions used for the experiment.

From the repository root, check the environment:

```bash
python p2_pipeline.py environment
```

For a fresh run, execute notebooks 01–03 in order using the matching GPU kernel, and save their outputs. Notebook 01 downloads the required COCO annotations and selected images; later notebooks load the pretrained models. Settings are in [config.json](config.json).
