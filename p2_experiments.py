"""Explicit SD conditioning and controlled M2 inference experiments."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import p2_pipeline as p2

ROOT = Path(__file__).resolve().parent
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
BASE = dict(guidance=7.5, steps=30, sampler="PNDM", spacing="leading", weight=1.0)
SMOKE_MODES = ("raw", "explicit", "modified")
SIGNED_SOURCES = ("p2_experiments.py", "p2_pipeline.py")
LEGACY_SOURCE_DIR = "artifacts/m2/provenance/v1_sources"


def verify_sources(saved, names):
    """Check an entire source set against the active code or the archived run code."""
    if saved == {name: p2.sha256(ROOT / name) for name in names}:
        return "current"
    archive = ROOT / LEGACY_SOURCE_DIR
    if all((archive / name).is_file() for name in names) and saved == {
            name: p2.sha256(archive / name) for name in names}:
        return "v1"
    raise RuntimeError("Run sources do not match the active or archived source set.")


def path_from_root(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def environment():
    """Collect Python, package and CUDA versions for reproducibility."""
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "packages": {}}
    for name in ("torch", "diffusers", "transformers", "accelerate", "numpy",
                 "Pillow", "huggingface-hub", "safetensors", "ipykernel"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if result["packages"]["torch"]:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda)
        if torch.cuda.is_available():
            result.update(gpu=torch.cuda.get_device_name(0),
                          gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 1024**3)
    return result


def settings(config_path="configs/m2.json"):
    cfg = p2.read_json(path_from_root(config_path))
    if cfg["model_revision"] != REVISION:
        raise ValueError("M2 must retain the resolved M1 SD revision.")
    if cfg["seed_offsets"] != [0, 1000, 2000] or cfg["image_size"] != 512:
        raise ValueError("Unexpected paired-seed or image-size settings.")
    if not math.isfinite(cfg["condition_weight"]) or not 1 < cfg["condition_weight"] <= 2:
        raise ValueError("Use a finite condition weight in (1, 2].")
    return cfg


def configurations(cfg):
    return {"B": dict(BASE), "G-low": dict(BASE, guidance=3.0),
            "G-high": dict(BASE, guidance=12.0), "T-short": dict(BASE, steps=20),
            "T-long": dict(BASE, steps=50), "S-DDIM": dict(BASE, sampler="DDIM"),
            "N-trailing": dict(BASE, sampler="DDIM", spacing="trailing"),
            "C": dict(BASE, weight=cfg["condition_weight"])}


def fixed_inputs(cfg, data_config):
    prompts = p2.read_json(path_from_root(cfg["prompts_file"]))
    if prompts != p2.select_prompts(data_config):
        raise RuntimeError("Fixed prompts differ from the original dataset selection.")
    spans = cfg["spans"]
    if set(spans) != {str(p["index"]) for p in prompts}:
        raise ValueError("Each fixed prompt needs one predeclared span.")
    return prompts


def span_indices(tokenizer, text, span, occurrence=0):
    """Map a complete word/phrase to CLIP BPE positions, rejecting ambiguity."""
    if not span or not isinstance(occurrence, int) or occurrence < 0:
        raise ValueError("A nonempty span and nonnegative occurrence are required.")
    matches = list(re.finditer(r"(?<!\w)" + re.escape(span) + r"(?!\w)", text, re.I))
    if occurrence >= len(matches):
        raise ValueError(f"Span not found: {span!r}, occurrence {occurrence}.")
    match = matches[occurrence]
    full = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(full) > tokenizer.model_max_length:
        raise ValueError("Prompt exceeds the SD tokenizer limit.")
    prefix = tokenizer(text[:match.start()], add_special_tokens=False)["input_ids"]
    part = tokenizer(text[match.start():match.end()], add_special_tokens=False)["input_ids"]
    start = len(prefix) + 1  # CLIP BOS occupies position zero.
    indices = list(range(start, start + len(part)))
    if not part or full[1:start] != prefix or full[start:start + len(part)] != part:
        raise ValueError("Span does not align with the caption's BPE boundaries.")
    if any(full[i] in tokenizer.all_special_ids for i in indices):
        raise ValueError("A conditioning span cannot include special or padding tokens.")
    return indices


def reweight(hidden, indices, weight):
    """Scale selected contextual token vectors; leave all other positions intact."""
    if len(hidden.shape) != 3 or hidden.shape[0] != 1:
        raise ValueError("Expected a single [1, sequence, hidden] tensor.")
    if not math.isfinite(weight) or not 0 < weight <= 2:
        raise ValueError("Weight must be finite and in (0, 2].")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Token positions must be nonempty and unique.")
    if any(i <= 0 or i >= hidden.shape[1] for i in indices):
        raise ValueError("Token position out of range or BOS selected.")
    result = hidden.clone() if hasattr(hidden, "clone") else hidden.copy()
    result[:, indices, :] *= weight
    return result


def encode(pipe, text):
    """Use SD's own tokenizer and final text-encoder hidden states."""
    import torch
    if len(pipe.tokenizer(text, truncation=False)["input_ids"]) > pipe.tokenizer.model_max_length:
        raise ValueError("Prompt truncation is not allowed.")
    inputs = pipe.tokenizer(text, padding="max_length", truncation=False,
                            max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    device = pipe._execution_device
    mask = inputs.attention_mask.to(device) if getattr(
        pipe.text_encoder.config, "use_attention_mask", False) else None
    with torch.inference_mode():
        hidden = pipe.text_encoder(inputs.input_ids.to(device), attention_mask=mask)[0]
    hidden = hidden.to(device=device, dtype=pipe.text_encoder.dtype)
    if hidden.shape[-1] != pipe.unet.config.cross_attention_dim or not torch.isfinite(hidden).all():
        raise RuntimeError("Invalid text-encoder output for U-Net cross-attention.")
    return hidden, inputs.input_ids[0].tolist()


def load_pipeline(cfg):
    import torch
    from diffusers import StableDiffusionPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference.")
    if importlib.metadata.version("diffusers") != "0.32.2":
        raise RuntimeError("This implementation requires Diffusers 0.32.2.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model_id"], revision=cfg["model_revision"], torch_dtype=torch.float16,
        use_safetensors=True, local_files_only=cfg["local_files_only"]).to("cuda")
    pipe.enable_attention_slicing()
    if pipe.safety_checker is None:
        raise RuntimeError("The pretrained safety checker must be present.")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    return pipe, dict(pipe.scheduler.config), time.perf_counter() - started


def scheduler(pipe, original, spec):
    from diffusers import DDIMScheduler, PNDMScheduler
    cls = PNDMScheduler if spec["sampler"] == "PNDM" else DDIMScheduler
    options = {"timestep_spacing": spec["spacing"]}
    if cls is DDIMScheduler:
        options.update(clip_sample=False, set_alpha_to_one=False)
    pipe.scheduler = cls.from_config(original, **options)
    for key in ("num_train_timesteps", "beta_start", "beta_end", "beta_schedule",
                "prediction_type", "steps_offset", "trained_betas"):
        if pipe.scheduler.config[key] != original[key]:
            raise RuntimeError(f"Pretrained scheduler setting changed: {key}")


def safe_run_dir(value, data_config):
    run = path_from_root(value).resolve()
    legacy = p2.local_path(data_config, "run_dir").resolve()
    if run == legacy or legacy in run.parents or run in legacy.parents:
        raise ValueError("M2 output must be separate from the M1 run directory.")
    if not run.name.startswith("m2_"):
        raise ValueError("Use a distinct m2_ output directory.")
    return run


def bind_run(cfg, data_config, prompts, phase, env):
    summary = p2.verify_dataset(data_config)
    run = safe_run_dir(cfg[phase + "_dir"], data_config)
    signature = {"phase": phase, "config": cfg, "data_config": data_config,
                 "manifest_sha256": summary["manifest_sha256"],
                 "prompts_sha256": p2.digest(prompts), "environment": env,
                 "sources": {name: p2.sha256(ROOT / name) for name in
                             ("p2_experiments.py", "p2_pipeline.py")}}
    run.mkdir(parents=True, exist_ok=True)
    sig = run / "run_signature.json"
    if sig.exists():
        if p2.read_json(sig) != signature:
            raise RuntimeError("Run provenance differs from the saved signature.")
    elif any(run.iterdir()):
        raise RuntimeError("Unrecognized nonempty M2 directory.")
    else:
        p2.write_json(sig, signature)
        p2.write_json(run / "prompts.json", prompts)
        p2.write_json(run / "environment.json", env)
        p2.write_json(run / "dataset_summary.json", summary)
    return run


def condition(pipe, prompt, span, weight):
    import torch
    positive, ids = encode(pipe, prompt)
    negative, _ = encode(pipe, "")
    indices = span_indices(pipe.tokenizer, prompt, span)
    modified = reweight(positive, indices, weight)
    mask = torch.ones(positive.shape[1], device=positive.device, dtype=torch.bool)
    mask[indices] = False
    if not torch.equal(positive[:, mask], modified[:, mask]):
        raise RuntimeError("Unselected tokens were changed.")
    if weight != 1 and torch.equal(positive, modified):
        raise RuntimeError("Condition modification did not change the tensor.")
    details = {"span": span, "token_indices": indices, "weight": weight,
               "tokens": pipe.tokenizer.convert_ids_to_tokens(ids), "token_ids": ids,
               "shape": list(positive.shape), "negative_shape": list(negative.shape),
               "dtype": str(positive.dtype),
               "unet_cross_attention_dim": pipe.unet.config.cross_attention_dim,
               "max_abs_change": (modified - positive).abs().max().item(),
               "unselected_tokens_equal": True}
    return modified, negative, details


def sample(pipe, original, cfg, spec, prompt, seed, run, key, raw=False):
    """Generate one paired sample and retain diagnostics and completion metadata."""
    import numpy as np
    import torch
    target = run / "generated" / (key + ".png")
    meta = target.with_suffix(".json")
    inputs = {"prompt_record": prompt, "seed": seed, "configuration": spec, "raw_prompt": raw}
    if meta.exists():
        record = p2.read_json(meta)
        if record["inputs"] != inputs:
            raise RuntimeError("Saved sample inputs changed.")
        verify_record(run, record)
        return record
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(".npz").exists() or target.with_suffix(".failure.json").exists():
        raise RuntimeError("Incomplete sample artifacts exist without completion metadata.")
    scheduler(pipe, original, spec)
    pos, neg, details = condition(pipe, prompt["prompt"], cfg["spans"][str(prompt["index"])], spec["weight"])
    expected = torch.cat([neg, pos])
    trace, stats, last, calls, received = [], [], [None], [0], {}

    def check_condition(module, args, kwargs):
        calls[0] += 1
        if calls[0] == 1:
            actual = kwargs["encoder_hidden_states"]
            if not torch.equal(expected, actual):
                raise RuntimeError("U-Net did not receive the expected CFG condition tensor.")
            received.update(shape=list(actual.shape), matches_expected=True)

    def callback(pipeline, step, timestep, values):
        z = values["latents"].detach()
        f = z.float()
        delta = (f - last[0].float()).square().mean().sqrt() if last[0] is not None else f.new_zeros(())
        stats.append(torch.stack([f.square().mean().sqrt(), f.std(), delta,
                                  torch.isfinite(f).all().to(f.dtype)]))
        trace.append(step)
        last[0] = z.clone()
        return values

    handle = pipe.unet.register_forward_pre_hook(check_condition, with_kwargs=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        args = {"prompt": prompt["prompt"]} if raw else {
            "prompt_embeds": pos, "negative_prompt_embeds": neg}
        with torch.inference_mode():
            output = pipe(**args, height=cfg["image_size"], width=cfg["image_size"],
                          num_inference_steps=spec["steps"], guidance_scale=spec["guidance"],
                          eta=0.0, guidance_rescale=0.0,
                          generator=torch.Generator(device="cpu").manual_seed(seed),
                          callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"])
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 1024**3
        diagnostics = torch.stack(stats).cpu().numpy()
        if not np.isfinite(diagnostics).all() or not diagnostics[:, 3].all():
            raise RuntimeError("Nonfinite denoising trajectory.")
        timesteps = pipe.scheduler.timesteps.cpu().tolist()
        if calls[0] != len(timesteps) or len(trace) != len(timesteps):
            raise RuntimeError("Incomplete sampler diagnostics.")
        if output.nsfw_content_detected is None:
            raise RuntimeError("Safety result is missing.")
        np.savez_compressed(target.with_suffix(".npz"), stats=diagnostics,
                            final_latents=last[0].float().cpu().numpy())
        output.images[0].save(target)
        record = {"status": "complete", "inputs": inputs, "condition": details,
                  "unet_condition": received, "seconds": seconds, "peak_allocated_gib": peak,
                  "nsfw_content_detected": bool(output.nsfw_content_detected[0]),
                  "scheduler_class": type(pipe.scheduler).__name__,
                  "scheduler_config": dict(pipe.scheduler.config), "timesteps": timesteps,
                  "unet_calls": calls[0], "diagnostic_columns": ["latent_rms", "latent_std",
                  "update_rms", "finite"], "first_update_rms": "zero; no previous callback latent",
                  "eta": 0.0, "guidance_rescale": 0.0, "generator_device": "cpu",
                  "image": target.relative_to(run).as_posix(), "image_sha256": p2.sha256(target),
                  "diagnostics": target.with_suffix(".npz").relative_to(run).as_posix(),
                  "diagnostics_sha256": p2.sha256(target.with_suffix(".npz")),
                  "utc": datetime.now(timezone.utc).isoformat()}
        p2.write_json(meta, record)
        return record
    except Exception as exc:
        p2.write_json(target.with_suffix(".failure.json"), dict(inputs=inputs,
                      error=f"{type(exc).__name__}: {exc}", unet_calls=calls[0]))
        raise
    finally:
        handle.remove()


def verify_record(run, record):
    import numpy as np
    from PIL import Image
    if record["status"] != "complete" or not record["unet_condition"]["matches_expected"]:
        raise RuntimeError("Incomplete sample or condition check.")
    for key in ("image", "diagnostics"):
        file = (run / record[key]).resolve()
        if run.resolve() not in file.parents or p2.sha256(file) != record[key + "_sha256"]:
            raise RuntimeError("Saved artifact path or hash differs.")
    with Image.open(run / record["image"]) as image:
        if image.size != (512, 512) or image.mode != "RGB":
            raise RuntimeError("Invalid generated image.")
    with np.load(run / record["diagnostics"], allow_pickle=False) as data:
        if data["stats"].shape != (len(record["timesteps"]), 4):
            raise RuntimeError("Diagnostic length differs from timesteps.")
        if not np.isfinite(data["stats"]).all() or not data["stats"][:, 3].all():
            raise RuntimeError("Invalid saved diagnostics.")
        if not np.isfinite(data["final_latents"]).all():
            raise RuntimeError("Invalid final latents.")


def smoke(cfg, data_config):
    import numpy as np
    run = safe_run_dir(cfg["smoke_dir"], data_config)
    if (run / "summary.json").is_file():
        records = verify_phase(cfg, "smoke")
        saved = p2.read_json(run / "summary.json")
        checks = [dict(check, image_hashes={mode: check["image_hashes"][mode]
                                          for mode in SMOKE_MODES}) for check in saved["checks"]]
        return dict(saved, checks=checks, sample_count=len(SMOKE_MODES) * len(checks),
                    source_sample_count=len(records), reused=True)

    import torch
    prompts = fixed_inputs(cfg, data_config)
    run = bind_run(cfg, data_config, prompts, "smoke", environment())
    pipe, original, load_seconds = load_pipeline(cfg)
    p2.write_json(run / "sessions" / f"load_{time.time_ns()}.json", {"seconds": load_seconds, "revision": cfg["model_revision"],
                                            "scheduler": original})
    spans = []
    for prompt in prompts:
        _, _, details = condition(pipe, prompt["prompt"], cfg["spans"][str(prompt["index"])],
                                  cfg["condition_weight"])
        spans.append(dict(prompt_id=prompt["index"], **details))
    p2.write_json(run / "span_validation.json", spans)
    checks = []
    for prompt_id in cfg["smoke_prompt_ids"]:
        prompt = next(p for p in prompts if p["index"] == prompt_id)
        manual, _ = encode(pipe, prompt["prompt"])
        with torch.inference_mode():
            native, native_neg = pipe.encode_prompt(prompt["prompt"], pipe._execution_device, 1, True)
        negative, _ = encode(pipe, "")
        if not torch.equal(manual, native) or not torch.equal(negative, native_neg):
            raise RuntimeError("Manual encoding differs from pipeline.encode_prompt.")
        outputs = {}
        for mode in SMOKE_MODES:
            spec = dict(BASE, weight=cfg["condition_weight"] if mode == "modified" else 1.0)
            outputs[mode] = sample(pipe, original, cfg, spec, prompt, prompt["seed"], run,
                                   f"{prompt_id:02d}_{mode}", raw=mode == "raw")
        arrays = {}
        for mode, record in outputs.items():
            with np.load(run / record["diagnostics"], allow_pickle=False) as data:
                arrays[mode] = data["final_latents"].copy()
        equal = np.array_equal(arrays["raw"], arrays["explicit"])
        changed = not np.array_equal(arrays["explicit"], arrays["modified"])
        check = {"prompt_id": prompt_id, "manual_encoding_exact": True,
                 "identity_latents_exact": equal, "modified_latents_differ": changed,
                 "image_hashes": {m: r["image_sha256"] for m, r in outputs.items()},
                 "max_abs_modified_latent_change": float(np.max(np.abs(
                     arrays["explicit"] - arrays["modified"]))) }
        checks.append(check)
        p2.write_json(run / "checks.json", checks)
        if not equal or not changed:
            raise RuntimeError("Conditioning equivalence/modification gate failed. Inspect checks.json.")
    result = {"status": "passed", "phase": "smoke", "checks": checks,
              "sample_count": len(SMOKE_MODES) * len(checks), "training_performed": False}
    p2.write_json(run / "summary.json", result)
    return dict(result, source_sample_count=result["sample_count"], reused=False)


def smoke_gate(cfg, data_config, prompts):
    run = safe_run_dir(cfg["smoke_dir"], data_config)
    signature = p2.read_json(run / "run_signature.json")
    if signature["config"] != cfg or signature["data_config"] != data_config:
        raise RuntimeError("Smoke configuration changed.")
    if signature["environment"] != environment() or signature["prompts_sha256"] != p2.digest(prompts):
        raise RuntimeError("Smoke environment or prompts changed.")
    if signature["manifest_sha256"] != p2.sha256(p2.local_path(data_config, "dataset_dir") / "manifest.jsonl"):
        raise RuntimeError("Smoke manifest changed.")
    verify_sources(signature["sources"], SIGNED_SOURCES)
    result = p2.read_json(run / "summary.json")
    if result["status"] != "passed" or len(result["checks"]) != len(cfg["smoke_prompt_ids"]):
        raise RuntimeError("Complete smoke validation is required before the sweep.")
    verify_phase(cfg, "smoke")
    return {"signature_sha256": p2.sha256(run / "run_signature.json"),
            "summary_sha256": p2.sha256(run / "summary.json")}


def sweep(cfg, data_config):
    run = safe_run_dir(cfg["sweep_dir"], data_config)
    if (run / "summary.json").is_file():
        verify_phase(cfg, "sweep")
        return p2.read_json(run / "summary.json")

    prompts = fixed_inputs(cfg, data_config)
    gate = smoke_gate(cfg, data_config, prompts)
    run = bind_run(cfg, data_config, prompts, "sweep", environment())
    p2.write_json(run / "smoke_gate.json", gate)
    pipe, original, load_seconds = load_pipeline(cfg)
    p2.write_json(run / "sessions" / f"load_{time.time_ns()}.json", {"seconds": load_seconds, "revision": cfg["model_revision"],
                                            "scheduler": original})
    # Keep a short warm-up separate from the 288 paired experimental outputs.
    sample(pipe, original, cfg, dict(BASE, steps=5), prompts[0], prompts[0]["seed"], run,
           f"warmup_{time.time_ns()}")
    records = []
    for name, spec in configurations(cfg).items():
        for prompt in prompts:
            for offset in cfg["seed_offsets"]:
                seed = prompt["seed"] + offset
                record = sample(pipe, original, cfg, spec, prompt, seed, run,
                                f"{name}_{prompt['index']:02d}_{seed}")
                records.append(record)
                print(f"{len(records)}/288 {name} prompt={prompt['index']} seed={seed}", flush=True)
    fields = ["configuration", "prompt_id", "image_id", "seed", "guidance", "steps", "sampler",
              "spacing", "weight", "seconds", "peak_allocated_gib", "unet_calls", "flagged", "image"]
    with (run / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            inputs = record["inputs"]
            spec = inputs["configuration"]
            name = next(n for n, s in configurations(cfg).items() if s == spec)
            writer.writerow(dict(configuration=name, prompt_id=inputs["prompt_record"]["index"],
                image_id=inputs["prompt_record"]["image_id"], seed=inputs["seed"], **spec,
                seconds=record["seconds"], peak_allocated_gib=record["peak_allocated_gib"],
                unet_calls=record["unet_calls"], flagged=record["nsfw_content_detected"], image=record["image"]))
    result = {"status": "complete", "phase": "sweep", "sample_count": len(records),
              "training_performed": False, "configurations": configurations(cfg),
              "total_inference_seconds": sum(r["seconds"] for r in records),
              "flagged_count": sum(r["nsfw_content_detected"] for r in records),
              "timing_scope": "pipeline call including per-step diagnostics, VAE and safety; excludes encoding/save/load"}
    p2.write_json(run / "summary.json", result)
    return result


def verify_phase(cfg, phase):
    """Verify a returned run without requiring model weights or full COCO data."""
    import numpy as np
    data_config = p2.load_config(ROOT / "config.json")
    data_config["dataset_dir"] = cfg["dataset_dir"]
    run = safe_run_dir(cfg[phase + "_dir"], data_config)
    signature = p2.read_json(run / "run_signature.json")
    if signature["phase"] != phase or signature["config"] != cfg or signature["data_config"] != data_config:
        raise RuntimeError("Returned configuration differs from the signed run.")
    source_version = verify_sources(signature["sources"], SIGNED_SOURCES)
    prompts = p2.read_json(path_from_root(cfg["prompts_file"]))
    if p2.read_json(run / "prompts.json") != prompts or signature["prompts_sha256"] != p2.digest(prompts):
        raise RuntimeError("Returned prompts changed.")
    summary = p2.read_json(run / "summary.json")
    records = []

    def check(key, prompt, seed, spec, raw=False):
        record = p2.read_json(run / "generated" / (key + ".json"))
        if record["inputs"] != dict(prompt_record=prompt, seed=seed, configuration=spec, raw_prompt=raw):
            raise RuntimeError("Returned sample inputs differ from the design.")
        verify_record(run, record)
        if record["unet_calls"] != len(record["timesteps"]):
            raise RuntimeError("Returned U-Net call count differs.")
        records.append(record)
        return record

    if phase == "smoke":
        if summary["status"] != "passed":
            raise RuntimeError("Smoke did not pass.")
        modes = ("raw", "explicit", "identity", "modified") if source_version == "v1" else SMOKE_MODES
        if [item["prompt_id"] for item in summary["checks"]] != cfg["smoke_prompt_ids"]:
            raise RuntimeError("Smoke summary does not cover the expected prompts.")
        for index in cfg["smoke_prompt_ids"]:
            prompt = next(p for p in prompts if p["index"] == index)
            arrays = {}
            for mode in modes:
                spec = dict(BASE, weight=cfg["condition_weight"] if mode == "modified" else 1.0)
                record = check(f"{index:02d}_{mode}", prompt, prompt["seed"], spec, mode == "raw")
                with np.load(run / record["diagnostics"], allow_pickle=False) as data:
                    arrays[mode] = data["final_latents"].copy()
            if not all(np.array_equal(arrays["raw"], arrays[m]) for m in modes if m in ("explicit", "identity")):
                raise RuntimeError("Returned identity latents differ.")
            if np.array_equal(arrays["explicit"], arrays["modified"]):
                raise RuntimeError("Returned modified latents did not change.")
    else:
        if summary["status"] != "complete":
            raise RuntimeError("Sweep is incomplete.")
        smoke_run = safe_run_dir(cfg["smoke_dir"], data_config)
        smoke_signature = p2.read_json(smoke_run / "run_signature.json")
        verify_sources(smoke_signature["sources"], SIGNED_SOURCES)
        if p2.read_json(run / "smoke_gate.json") != {
            "signature_sha256": p2.sha256(smoke_run / "run_signature.json"),
            "summary_sha256": p2.sha256(smoke_run / "summary.json") }:
            raise RuntimeError("Sweep's conditioning gate evidence differs.")
        for name, spec in configurations(cfg).items():
            for prompt in prompts:
                for offset in cfg["seed_offsets"]:
                    seed = prompt["seed"] + offset
                    check(f"{name}_{prompt['index']:02d}_{seed}", prompt, seed, spec)
    if summary["sample_count"] != len(records):
        raise RuntimeError("Returned sample count differs.")
    return records


def show_comparison(cfg, prompt_id=1, seed_offset=0, names=None):
    """Display paired outputs without hiding failed or safety-flagged examples."""
    from IPython.display import display
    from PIL import Image, ImageDraw
    names = names or list(configurations(cfg))
    prompts = p2.read_json(path_from_root(cfg["prompts_file"]))
    prompt = next(p for p in prompts if p["index"] == prompt_id)
    run = path_from_root(cfg["sweep_dir"])
    columns = min(4, len(names))
    sheet = Image.new("RGB", (256 * columns, 300 * math.ceil(len(names) / columns)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, name in enumerate(names):
        x, y = 256 * (i % columns), 300 * (i // columns)
        seed = prompt["seed"] + seed_offset
        record = p2.read_json(run / "generated" / f"{name}_{prompt_id:02d}_{seed}.json")
        verify_record(run, record)
        with Image.open(run / record["image"]) as image:
            sheet.paste(image.resize((256, 256)), (x, y + 44))
        s = record["inputs"]["configuration"]
        label = f"{name} g={s['guidance']} n={s['steps']} w={s['weight']}\n{s['sampler']} {s['spacing']}"
        if record["nsfw_content_detected"]:
            label += " FLAGGED"
        draw.text((x + 4, y + 2), label, fill="black")
    print(f"{prompt_id:02d} | seed {prompt['seed'] + seed_offset} | {prompt['prompt']}")
    display(sheet)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["environment", "smoke", "sweep"])
    parser.add_argument("--config", default="configs/m2.json")
    args = parser.parse_args()
    if args.phase == "environment":
        result = environment()
    else:
        cfg = settings(args.config)
        data_config = p2.load_config(ROOT / "config.json")
        data_config["dataset_dir"] = cfg["dataset_dir"]
        result = (smoke if args.phase == "smoke" else sweep)(cfg, data_config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
