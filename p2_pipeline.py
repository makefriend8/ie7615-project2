"""COCO preparation, CLIP embeddings and a Stable Diffusion baseline for M1."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import re
import shlex
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile

try:
    from PIL import Image, ImageOps
except ImportError:
    # Allow environment diagnostics when Pillow is unavailable.
    Image = ImageOps = None

ROOT = Path(__file__).resolve().parent
# COCO annotation and image downloads.
BUCKET = "https://s3.amazonaws.com/images.cocodataset.org"
ENVIRONMENT_PACKAGES = {
    "torch": "torch", "diffusers": "diffusers", "transformers": "transformers",
    "accelerate": "accelerate", "huggingface-hub": "huggingface_hub",
    "safetensors": "safetensors", "numpy": "numpy", "Pillow": "PIL",
    "ipykernel": "ipykernel", "ipython": "IPython",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_config(path="config.json"):
    c = read_json(path)
    if not (1000 <= c["dataset_size"] <= 5000):
        raise ValueError("The graded dataset must contain 1,000–5,000 pairs.")
    if c["train_size"] < 1 or c["validation_size"] < c["prompt_count"]:
        raise ValueError("Invalid train/validation sizes.")
    if c["train_size"] + c["validation_size"] >= c["dataset_size"]:
        raise ValueError("Reserve at least one test image.")
    if not 10 <= c["prompt_count"] <= 20:
        raise ValueError("M1 requires 10–20 prompts.")
    if c["image_size"] != 512:
        raise ValueError("This SD 1.5 baseline uses 512 × 512 images.")
    return c


def local_path(c, key):
    p = Path(c[key]).expanduser()
    return p if p.is_absolute() else ROOT / p


def clean_caption(text, min_words=5, max_words=50):
    """Return normalized caption text, or None when the quality filters reject it."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    if not min_words <= len(text.split()) <= max_words:
        return None
    if re.search(r"https?://|www\.|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text, re.I):
        return None
    if any(unicodedata.category(x).startswith("C") for x in text):
        return None
    if sum(x.isalpha() for x in text) / max(len(text), 1) < 0.5:
        return None
    return text


def fetch(url, destination):
    """Reuse complete downloads; atomic writes prevent partial files looking valid."""
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "P2-course-M1/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            partial.replace(destination)
            return
        except (OSError, TimeoutError) as exc:
            last_error = exc
            # Missing/forbidden resources are permanent errors; retry only transient failures.
            if isinstance(exc, urllib.error.HTTPError) and 400 <= exc.code < 500 and exc.code not in (408, 429):
                break
            time.sleep(attempt + 1)
    raise RuntimeError(f"Download failed: {url}. {last_error}") from last_error


def annotation_files(c):
    raw = local_path(c, "raw_root")
    names = ["captions_val2017.json", "instances_val2017.json"]
    files = [raw / "annotations" / name for name in names]
    if not all(p.exists() for p in files):
        archive = raw / "annotations_trainval2017.zip"
        fetch(BUCKET + "/annotations/annotations_trainval2017.zip", archive)
        with zipfile.ZipFile(archive) as z:
            for name, destination in zip(names, files):
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Extract only named JSON members; never extract arbitrary archive paths.
                    destination.write_bytes(z.read("annotations/" + name))
    return files


def normalized_pixels(image):
    """Convert RGB pixels to a float32 CHW array in [-1, 1]."""
    import numpy as np
    return np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0


def manifest(c):
    return [json.loads(x) for x in (local_path(c, "dataset_dir") / "manifest.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()]


def verify_dataset(c):
    """Validate pair counts, split disjointness and hashes against the saved manifest."""
    base = local_path(c, "dataset_dir")
    summary = read_json(base / "dataset_summary.json")
    if summary["config_hash"] != digest(c):
        raise RuntimeError("Config changed. Use a new dataset_dir and run_dir for a new experiment.")
    if sha256(base / "manifest.jsonl") != summary["manifest_sha256"]:
        raise RuntimeError("Manifest changed after dataset preparation.")
    rows = manifest(c)
    if len(rows) != c["dataset_size"]:
        raise RuntimeError("Wrong pair count.")
    for key in ("image_id", "caption_key", "pixel_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            raise RuntimeError(f"Duplicate {key} across splits.")
    expected = {"train": c["train_size"], "validation": c["validation_size"],
                "test": c["dataset_size"] - c["train_size"] - c["validation_size"]}
    if dict(Counter(r["split"] for r in rows)) != expected:
        raise RuntimeError("Wrong split sizes.")
    for r in rows:
        image_path = base / r["image_path"]
        if sha256(image_path) != r["processed_sha256"]:
            raise RuntimeError(f"Image changed: {image_path}")
        with Image.open(image_path) as im:
            if im.mode != "RGB" or im.size != (c["image_size"],) * 2:
                raise RuntimeError(f"Invalid processed image: {image_path}")
    return summary


def prepare(c):
    """Curate COCO pairs, deduplicate before splitting and save the dataset summary."""
    base = local_path(c, "dataset_dir")
    if (base / "dataset_summary.json").exists():
        return verify_dataset(c)
    if base.exists() and any(base.iterdir()):
        raise RuntimeError("Incomplete dataset directory retained. Inspect it and choose a new dataset_dir.")
    captions_path, instances_path = annotation_files(c)
    captions, instances = read_json(captions_path), read_json(instances_path)
    # Reject truncation before selecting examples. Resolve the exact Hub snapshot once.
    from huggingface_hub import model_info
    from transformers import CLIPTokenizer
    revision = model_info(c["clip_model"], revision=c["clip_revision"]).sha
    tokenizer = CLIPTokenizer.from_pretrained(c["clip_model"], revision=revision)
    by_image = defaultdict(list)
    for ann in captions["annotations"]:
        by_image[ann["image_id"]].append(ann)
    categories = {r["id"]: r["name"] for r in instances["categories"]}
    image_categories = defaultdict(set)
    for ann in instances["annotations"]:
        image_categories[ann["image_id"]].add(categories[ann["category_id"]])
    licenses = {r["id"]: r for r in captions.get("licenses", [])}
    candidates = sorted(captions["images"], key=lambda r: r["id"])
    random.Random(c["seed"]).shuffle(candidates)
    rows, rejected = [], []
    seen_text, seen_pixels = set(), set()
    base.mkdir(parents=True, exist_ok=True)
    for info in candidates:
        if len(rows) == c["dataset_size"]:
            break
        chosen = None
        for ann in sorted(by_image[info["id"]], key=lambda r: r["id"]):
            text = clean_caption(ann["caption"], c["min_words"], c["max_words"])
            if text is None or text.casefold() in seen_text:
                continue
            if len(tokenizer(text, truncation=False)["input_ids"]) > tokenizer.model_max_length:
                continue
            chosen = ann, text
            break
        if chosen is None:
            rejected.append({"image_id": info["id"], "reason": "no_unique_quality_caption_within_token_limit"})
            continue
        filename = Path(info["file_name"]).name
        raw_image = local_path(c, "raw_root") / "val2017" / filename
        # Network failures stop the run instead of silently biasing the subset.
        fetch(BUCKET + "/val2017/" + filename, raw_image)
        try:
            with Image.open(raw_image) as im:
                original = ImageOps.exif_transpose(im).convert("RGB")
                original.load()
            if min(original.size) < 64:
                raise ValueError("Image shorter side <64 pixels")
        except (OSError, ValueError) as exc:
            rejected.append({"image_id": info["id"], "reason": "invalid_image", "detail": str(exc)})
            continue
        pixel_hash = hashlib.sha256(str(original.size).encode() + original.tobytes()).hexdigest()
        if pixel_hash in seen_pixels:
            rejected.append({"image_id": info["id"], "reason": "exact_pixel_duplicate"})
            continue
        # Preserve aspect ratio and pad with gray pixels.
        processed = ImageOps.pad(original, (c["image_size"],) * 2,
                                 method=Image.Resampling.LANCZOS, color=(127, 127, 127))
        relative = "images/" + Path(filename).stem + ".png"
        target = base / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        processed.save(target)
        ann, text = chosen
        rows.append({"image_id": info["id"], "caption_id": ann["id"], "caption": text,
                     "original_caption": ann["caption"], "caption_key": text.casefold(),
                     "image_path": relative, "original_size": list(original.size),
                     "source_url": BUCKET + "/val2017/" + filename,
                     "flickr_url": info.get("flickr_url"), "license": licenses.get(info.get("license")),
                     "categories": sorted(image_categories[info["id"]]),
                     "raw_sha256": sha256(raw_image), "pixel_sha256": pixel_hash,
                     "processed_sha256": sha256(target)})
        seen_text.add(text.casefold())
        seen_pixels.add(pixel_hash)
        if len(rows) % 100 == 0:
            print(f"Accepted {len(rows)}/{c['dataset_size']} pairs", flush=True)
    if len(rows) != c["dataset_size"]:
        raise RuntimeError(f"Only {len(rows)} valid pairs; no completion marker was written.")
    # Split by unique image after filtering and exact deduplication, with a separate RNG.
    random.Random(c["seed"] + 1).shuffle(rows)
    for i, row in enumerate(rows):
        row["split"] = ("train" if i < c["train_size"] else "validation"
                        if i < c["train_size"] + c["validation_size"] else "test")
    (base / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                        encoding="utf-8")
    write_json(base / "rejected.json", rejected)
    summary = {"status": "complete", "config_hash": digest(c), "pairs": len(rows),
               "splits": dict(Counter(r["split"] for r in rows)), "rejected_images": len(rejected),
               "category_image_counts": dict(Counter(cat for r in rows for cat in r["categories"])),
               "license_image_counts": dict(Counter(str((r["license"] or {}).get("id")) for r in rows)),
               "word_counts": dict(Counter(len(r["caption"].split()) for r in rows)),
               "clip_tokenizer_revision": revision,
               "source_hashes": {p.name: sha256(p) for p in [captions_path, instances_path]},
               "manifest_sha256": sha256(base / "manifest.jsonl"),
               "preprocessing": "EXIF transpose; RGB; Lanczos resize and gray letterbox to 512 square; PNG",
               "normalization": "CLIPProcessor model-specific normalization; diffusion CHW float32 x/127.5-1",
               "deduplication": "casefolded cleaned caption and exact decoded RGB pixels; no near-duplicate audit",
               "config": c}
    write_json(base / "dataset_summary.json", summary)
    return verify_dataset(c)


def runtime_info():
    """Collect interpreter, package, source and available CUDA provenance."""
    packages = {}
    for name in [*ENVIRONMENT_PACKAGES, "nbconvert"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    result = {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(), "packages": packages, "git_commit": commit,
            "source_sha256": sha256(__file__)}
    if packages["torch"]:
        try:
            import torch
            result.update({"torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                           "cuda_available": torch.cuda.is_available()})
            if torch.cuda.is_available():
                result["gpu"] = torch.cuda.get_device_name(0)
                result["gpu_total_gib"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception as exc:
            result["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def check_environment(save_requirements=False):
    """Check imports and CUDA; optionally save package versions and environment details."""
    result = runtime_info()
    missing, import_errors = [], {}
    for distribution, module in ENVIRONMENT_PACKAGES.items():
        if result["packages"].get(distribution) is None:
            missing.append(distribution)
            continue
        try:
            importlib.import_module(module)
        except Exception as exc:
            import_errors[distribution] = f"{type(exc).__name__}: {exc}"
    result.update({"missing_packages": missing, "import_errors": import_errors,
                   "ready": not missing and not import_errors and bool(result.get("cuda_available")),
                   "requirements_saved": False})
    installable = [name for name in missing if name != "torch"]
    if installable:
        result["missing_package_install_command"] = (
            shlex.quote(sys.executable) + " -m pip install --user "
            + " ".join(shlex.quote(name) for name in installable))
    if "torch" in missing or not result.get("cuda_available"):
        result["gpu_note"] = "A CUDA-enabled Python/kernel and an allocated GPU session are required."
    if save_requirements and result["ready"]:
        lines = [f"# Python {result['python']}"]
        lines += [f"{name}=={result['packages'][name]}" for name in ENVIRONMENT_PACKAGES]
        if result["packages"].get("nbconvert"):
            lines.append(f"nbconvert=={result['packages']['nbconvert']}")
        (ROOT / "requirements.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["requirements_saved"] = True
        write_json(ROOT / "environment.json", result)
    return result


def device_for(c):
    import torch
    if c["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA required. Check nvidia-smi and the installed PyTorch CUDA wheel.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_folder(c):
    """Bind result reuse to the configuration, input manifest and pipeline source."""
    dataset_summary = verify_dataset(c)
    run = local_path(c, "run_dir")
    run.mkdir(parents=True, exist_ok=True)
    signature = {"config_hash": digest(c), "manifest_sha256": dataset_summary["manifest_sha256"],
                 "source_sha256": sha256(__file__)}
    record = run / "run_signature.json"
    if record.exists():
        if read_json(record) != signature:
            raise RuntimeError("Run provenance differs. Choose a new run_dir; retain earlier evidence.")
    else:
        if any(run.iterdir()):
            raise RuntimeError("Unrecognized nonempty run directory.")
        write_json(record, signature)
        write_json(run / "config.json", c)
    return run


def snapshot_revision(run, key, model, requested):
    from huggingface_hub import model_info
    record = run / f"{key}_snapshot.json"
    if record.exists():
        return read_json(record)["resolved_revision"]
    revision = model_info(model, revision=requested).sha
    write_json(record, {"model_id": model, "requested_revision": requested, "resolved_revision": revision})
    return revision


def embed(c):
    """Encode validation pairs with frozen CLIP and save embeddings and cosine diagnostics."""
    import numpy as np
    import torch
    from transformers import CLIPModel, CLIPProcessor
    run = run_folder(c)
    if (run / "embedding_summary.json").exists():
        summary = read_json(run / "embedding_summary.json")
        for name, expected in summary["artifact_hashes"].items():
            if sha256(run / name) != expected:
                raise RuntimeError(f"Embedding artifact changed: {name}")
        return summary
    device = device_for(c)
    # Use exactly the tokenizer snapshot that accepted captions during curation.
    tokenizer_revision = read_json(local_path(c, "dataset_dir") / "dataset_summary.json")["clip_tokenizer_revision"]
    revision = snapshot_revision(run, "clip", c["clip_model"], tokenizer_revision)
    model = CLIPModel.from_pretrained(c["clip_model"], revision=revision).to(device).eval()
    processor = CLIPProcessor.from_pretrained(c["clip_model"], revision=revision, use_fast=False)
    rows = [r for r in manifest(c) if r["split"] == "validation"][:c["embedding_samples"]]
    if len(rows) < 2:
        raise RuntimeError("At least two embedding examples required.")
    image_vectors, text_vectors, tokens = [], [], []
    started = time.perf_counter()
    for start in range(0, len(rows), c["embedding_batch_size"]):
        batch = rows[start:start + c["embedding_batch_size"]]
        images = []
        for r in batch:
            with Image.open(local_path(c, "dataset_dir") / r["image_path"]) as im:
                images.append(im.convert("RGB"))
        captions = [r["caption"] for r in batch]
        lengths = [len(processor.tokenizer(t, truncation=False)["input_ids"]) for t in captions]
        if max(lengths) > processor.tokenizer.model_max_length:
            raise RuntimeError("Caption exceeds tokenizer limit; refusing silent truncation.")
        inputs = processor(text=captions, images=images, return_tensors="pt", padding=True,
                           truncation=False).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        image_vectors.append(outputs.image_embeds.float().cpu().numpy())
        text_vectors.append(outputs.text_embeds.float().cpu().numpy())
        tokens.extend({"image_id": r["image_id"], "caption": r["caption"], "input_ids": ids,
                       "token_count": length} for r, ids, length in zip(
                           batch, inputs["input_ids"].cpu().tolist(), lengths))
    image_vectors, text_vectors = np.concatenate(image_vectors), np.concatenate(text_vectors)
    for vectors in (image_vectors, text_vectors):
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise RuntimeError("CLIP embeddings are not finite unit vectors.")
    similarities = text_vectors @ image_vectors.T
    np.savez_compressed(run / "embeddings.npz", text=text_vectors, image=image_vectors,
                        cosine_similarity=similarities, image_ids=[r["image_id"] for r in rows])
    write_json(run / "embedding_tokens.json", tokens)
    summary = {"status": "complete", "sample_count": len(rows), "text_shape": list(text_vectors.shape),
               "image_shape": list(image_vectors.shape), "paired_cosine_mean": float(similarities.diagonal().mean()),
               "off_diagonal_cosine_mean": float(similarities[~np.eye(len(rows), dtype=bool)].mean()),
               "sample_text_to_image_top1": float((similarities.argmax(1) == np.arange(len(rows))).mean()),
               "seconds": time.perf_counter() - started, "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "scope": "sanity check on selected validation pairs; not held-out generation evaluation",
               "artifact_hashes": {n: sha256(run / n) for n in ["embeddings.npz", "embedding_tokens.json"]},
               "runtime": runtime_info()}
    write_json(run / "embedding_summary.json", summary)
    return summary


def select_prompts(c):
    """Select fixed validation captions and seeds using category coverage."""
    rows = [r for r in manifest(c) if r["split"] == "validation"]
    random.Random(c["seed"] + 2).shuffle(rows)
    chosen, covered = [], set()
    for row in rows:
        if set(row["categories"]) - covered:
            chosen.append(row)
            covered.update(row["categories"])
            if len(chosen) == c["prompt_count"]:
                break
    for row in rows:
        if len(chosen) == c["prompt_count"]:
            break
        if row not in chosen:
            chosen.append(row)
    return [{"index": i + 1, "image_id": r["image_id"], "caption_id": r["caption_id"],
             "prompt": r["caption"], "ground_truth": r["image_path"], "categories": r["categories"],
             "seed": c["seed"] + i, "license": r["license"], "flickr_url": r["flickr_url"]}
            for i, r in enumerate(chosen)]


def generate(c):
    """Generate images with pretrained SD 1.5 and save per-sample metadata."""
    import torch
    from diffusers import StableDiffusionPipeline
    run = run_folder(c)
    device = device_for(c)
    revision = snapshot_revision(run, "sd", c["sd_model"], c["sd_revision"])
    prompts = select_prompts(c)
    prompts_file = run / "prompts.json"
    if prompts_file.exists() and read_json(prompts_file) != prompts:
        raise RuntimeError("Fixed prompt list changed.")
    write_json(prompts_file, prompts)
    if (run / "generation_summary.json").exists():
        return verify_generation(c)
    pipe = StableDiffusionPipeline.from_pretrained(c["sd_model"], revision=revision,
           torch_dtype=torch.float16 if device == "cuda" else torch.float32, use_safetensors=True)
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    # Use the pretrained safety checker.
    if pipe.safety_checker is None:
        raise RuntimeError("Expected pretrained safety checker is missing.")
    for p in prompts:
        if len(pipe.tokenizer(p["prompt"], truncation=False)["input_ids"]) > pipe.tokenizer.model_max_length:
            raise RuntimeError("Prompt exceeds the generator tokenizer limit.")
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    write_json(run / "scheduler.json", dict(pipe.scheduler.config))
    (run / "generated").mkdir(exist_ok=True)
    records = []
    for p in prompts:
        stem = f"{p['index']:02d}_{p['image_id']}"
        target, meta = run / "generated" / f"{stem}.png", run / "generated" / f"{stem}.json"
        if meta.exists():
            record = read_json(meta)
            if record["prompt_record"] != p or sha256(target) != record["image_sha256"]:
                raise RuntimeError("Saved sample differs from its prompt or recorded hash.")
            records.append(record)
            continue
        if target.exists():
            raise RuntimeError(f"Image has no completion metadata: {target}. Retain it and use a new run_dir.")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = pipe(prompt=p["prompt"], height=c["image_size"], width=c["image_size"],
                          num_inference_steps=c["steps"], guidance_scale=c["guidance_scale"],
                          generator=torch.Generator(device="cpu").manual_seed(p["seed"]))
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        result.images[0].save(target)
        record = {"prompt_record": p, "generated_image": target.relative_to(run).as_posix(),
                  "image_sha256": sha256(target), "seconds": seconds,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
                  "nsfw_content_detected": bool(result.nsfw_content_detected[0])
                  if result.nsfw_content_detected is not None else None}
        write_json(meta, record)
        records.append(record)
        print(f"Generated {p['index']}/{len(prompts)} in {seconds:.1f}s", flush=True)
    summary = {"status": "complete", "sample_count": len(records), "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "total_inference_seconds": sum(r["seconds"] for r in records),
               "flagged_count": sum(r["nsfw_content_detected"] is True for r in records),
               "scheduler_class": type(pipe.scheduler).__name__, "model_revision": revision,
               "runtime": runtime_info()}
    write_json(run / "generation_summary.json", summary)
    verify_generation(c)
    return summary


def verify_generation(c):
    """Check completion, prompt identity, image hashes and generated dimensions."""
    run = run_folder(c)
    prompts = select_prompts(c)
    if read_json(run / "prompts.json") != prompts:
        raise RuntimeError("Prompt file does not match dataset/config.")
    summary = read_json(run / "generation_summary.json")
    if summary["sample_count"] != len(prompts):
        raise RuntimeError("Incomplete generation count.")
    for p in prompts:
        meta = run / "generated" / f"{p['index']:02d}_{p['image_id']}.json"
        r = read_json(meta)
        if r["prompt_record"] != p or sha256(run / r["generated_image"]) != r["image_sha256"]:
            raise RuntimeError(f"Invalid sample: {meta}")
        with Image.open(run / r["generated_image"]) as im:
            if im.size != (c["image_size"],) * 2:
                raise RuntimeError("Unexpected generated image dimensions.")
    return summary


def gallery(c):
    """Render reference/generated comparisons and contact sheets with a JSON index."""
    from PIL import ImageDraw, ImageFont
    import textwrap
    run = run_folder(c)
    verify_generation(c)
    comparison_dir = run / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    prompts = select_prompts(c)
    pages, items = [], []
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except OSError:
        try:
            font = ImageFont.load_default(size=16)
        except TypeError:
            font = ImageFont.load_default()
    for p in prompts:
        stem = f"{p['index']:02d}_{p['image_id']}"
        record = read_json(run / "generated" / f"{stem}.json")
        pair = Image.new("RGB", (1024, 548), "white")
        for x, path in [(0, local_path(c, "dataset_dir") / p["ground_truth"]),
                        (512, run / record["generated_image"])]:
            with Image.open(path) as image:
                pair.paste(image.convert("RGB"), (x, 36))
        draw = ImageDraw.Draw(pair)
        draw.text((12, 8), "Ground truth", fill="black", font=font)
        draw.text((524, 8), "Generated", fill="black", font=font)
        pair.save(comparison_dir / f"{stem}.png")
        items.append({"prompt_record": p, "generated_image": record["generated_image"],
                      "generated_metadata": f"generated/{stem}.json",
                      "comparison_image": f"comparisons/{stem}.png",
                      "comparison_sha256": sha256(comparison_dir / f"{stem}.png")})
        # Three landscape sheets, four pairs per sheet at the default 12 prompts.
        if (p["index"] - 1) % 4 == 0:
            pages.append(Image.new("RGB", (2048, 1280), "white"))
        page = pages[-1]
        i = (p["index"] - 1) % 4
        x, y = (i % 2) * 1024, (i // 2) * 640
        page.paste(pair, (x, y))
        label = f"{p['index']:02d} | seed {p['seed']} | {p['prompt']}"
        ImageDraw.Draw(page).multiline_text((x + 12, y + 554), "\n".join(textwrap.wrap(label, 105)),
                                           fill="black", font=font, spacing=3)
    contacts = []
    for number, page in enumerate(pages, start=1):
        name = f"contact_sheet_{number:02d}.png"
        page.save(run / name)
        contacts.append({"image": name, "sha256": sha256(run / name)})
    write_json(run / "gallery_index.json", {
        "sample_count": len(items), "reference_used_as_generation_input": False,
        "model": c["sd_model"], "steps": c["steps"], "guidance_scale": c["guidance_scale"],
        "items": items, "contact_sheets": contacts})
    return {"index": str(run / "gallery_index.json"), "comparisons": str(comparison_dir),
            "contact_sheets": [str(run / item["image"]) for item in contacts]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "embed", "generate", "gallery", "environment"])
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--save-requirements", action="store_true",
                        help="Save installed package versions and environment details.")
    args = parser.parse_args()
    if args.save_requirements and args.command != "environment":
        parser.error("--save-requirements is only used with environment")
    if args.command == "environment":
        result = check_environment(save_requirements=args.save_requirements)
    else:
        c = load_config(args.config)
        result = {"prepare": prepare, "verify": verify_dataset, "embed": embed, "generate": generate,
                  "gallery": gallery}[args.command](c)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "environment" and not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
