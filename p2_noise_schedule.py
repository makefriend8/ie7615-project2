"""Frozen-checkpoint beta-schedule comparisons with paired DDIM sampling."""
import base64
import csv
import html
import io
import time

import p2_pipeline as p2
import p2_experiments as m2

RUN_DIR = "results/m2_noise_schedule_v1"
SCHEDULES = {"N-scaled": "scaled_linear", "N-linear": "linear",
             "N-cosine": "squaredcos_cap_v2"}
SOURCES = ("p2_pipeline.py", "p2_experiments.py", "p2_noise_schedule.py")
RUNTIME_PACKAGES = ("torch", "diffusers", "transformers", "accelerate", "numpy",
                    "Pillow", "huggingface-hub", "safetensors")


def check_runtime(recorded, current):
    """Check inference dependencies; report host and notebook-runtime differences."""
    differences, incompatible = {}, []
    for key in sorted(set(recorded) | set(current)):
        if key == "packages":
            old, new = recorded.get(key, {}), current.get(key, {})
            fields = [(f"packages.{k}", old.get(k), new.get(k)) for k in sorted(set(old) | set(new))]
        else:
            fields = [(key, recorded.get(key), current.get(key))]
        for name, old, new in fields:
            if old != new:
                differences[name] = {"recorded": old, "current": new}
    for key in RUNTIME_PACKAGES:
        name = f"packages.{key}"
        if name in differences:
            incompatible.append(name)
    if "torch_cuda" in differences:
        incompatible.append("torch_cuda")
    if str(recorded.get("python", "")).split(".")[:2] != str(current.get("python", "")).split(".")[:2]:
        incompatible.append("python")
    if current.get("cuda_available") is False:
        incompatible.append("cuda_available")
        differences.setdefault("cuda_available", {"recorded": recorded.get("cuda_available"), "current": False})
    if incompatible:
        details = "; ".join(f"{k}: {differences[k]['recorded']!r} -> {differences[k]['current']!r}"
                            for k in incompatible)
        raise RuntimeError("Incompatible inference runtime: " + details)
    return {"compatible": True, "differences": differences}


def baseline_gate(cfg, data_config, prompts, env=None):
    """Read-only verification of the completed smoke/sweep and current data."""
    current = m2.environment() if env is None else env
    dataset = p2.verify_dataset(data_config)
    report = {}
    for phase in ("smoke", "sweep"):
        folder = m2.safe_run_dir(cfg[phase + "_dir"], data_config)
        saved = p2.read_json(folder / "run_signature.json")
        if saved["config"] != cfg or saved["data_config"] != data_config:
            raise RuntimeError(f"{phase} configuration changed.")
        if saved["prompts_sha256"] != p2.digest(prompts):
            raise RuntimeError(f"{phase} prompts changed.")
        if saved["manifest_sha256"] != dataset["manifest_sha256"]:
            raise RuntimeError(f"{phase} manifest changed.")
        runtime = check_runtime(saved["environment"], current)
        records = m2.verify_phase(cfg, phase)
        version = m2.verify_sources(saved["sources"], m2.SIGNED_SOURCES)
        expected_count = 288 if phase == "sweep" else (
            4 if version == "v1" else len(m2.SMOKE_MODES)) * len(cfg["smoke_prompt_ids"])
        if len(records) != expected_count:
            raise RuntimeError(f"{phase} output coverage changed.")
        if phase == "smoke":
            summary = p2.read_json(folder / "summary.json")
            if len(summary["checks"]) != len(cfg["smoke_prompt_ids"]):
                raise RuntimeError("Smoke checks are incomplete.")
        report[phase] = {"samples_verified": len(records), "runtime": runtime,
                         "signature_sha256": p2.sha256(folder / "run_signature.json")}
    return report


def configurations():
    return {name: dict(m2.BASE, sampler="DDIM", beta_schedule=beta)
            for name, beta in SCHEDULES.items()}


def run_path(data_config):
    return m2.safe_run_dir(RUN_DIR, data_config)


def signature(cfg, data_config, prompts, env, source_hashes=None):
    baseline = m2.path_from_root(cfg["sweep_dir"])
    old = p2.read_json(baseline / "run_signature.json")
    return {"phase": "noise_schedule", "config": cfg, "data_config": data_config,
            "configurations": configurations(), "environment": env,
            "manifest_sha256": old["manifest_sha256"],
            "prompts_sha256": p2.digest(prompts),
            "baseline_signature_sha256": p2.sha256(baseline / "run_signature.json"),
            "baseline_gate_sha256": p2.sha256(baseline / "smoke_gate.json"),
            "sources": (source_hashes if source_hashes is not None else
                        {name: p2.sha256(m2.ROOT / name) for name in SOURCES})}


def bind_run(cfg, data_config, prompts, env):
    run = run_path(data_config)
    expected = signature(cfg, data_config, prompts, env)
    run.mkdir(parents=True, exist_ok=True)
    saved = run / "run_signature.json"
    if saved.exists():
        recorded = p2.read_json(saved)
        expected["environment"] = recorded["environment"]
        if recorded != expected:
            raise RuntimeError("Noise-schedule run provenance changed.")
        check_runtime(recorded["environment"], env)
        if (run / "generated").exists() and not (run / "summary.json").exists():
            if any(recorded["environment"].get(k) != env.get(k) for k in ("gpu", "gpu_total_gib")):
                raise RuntimeError("Partial noise-schedule run GPU changed; retain its outputs and use a separate run.")
    elif any(run.iterdir()):
        raise RuntimeError("Unrecognized nonempty noise-schedule directory.")
    else:
        p2.write_json(saved, expected)
        p2.write_json(run / "prompts.json", prompts)
        p2.write_json(run / "environment.json", env)
    return run


def write_curves(pipe, original, run):
    """Retain the actual beta, cumulative-alpha and SNR arrays used in inference."""
    import numpy as np
    entries = []
    for name, spec in configurations().items():
        variant = dict(original, beta_schedule=spec["beta_schedule"])
        m2.scheduler(pipe, variant, spec)
        betas = pipe.scheduler.betas.detach().cpu().numpy()
        alpha = pipe.scheduler.alphas_cumprod.detach().cpu().numpy()
        snr = alpha.astype(np.float64) / (1 - alpha.astype(np.float64))
        if not np.isfinite(snr).all() or not ((betas > 0) & (betas < 1)).all():
            raise RuntimeError("Invalid beta schedule.")
        file = run / f"{name}_curves.npz"
        arrays = dict(betas=betas, alphas_cumprod=alpha, snr=snr)
        if file.exists():
            with np.load(file, allow_pickle=False) as saved:
                if any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                    raise RuntimeError("Saved noise curve differs from the scheduler.")
        else:
            np.savez_compressed(file, **arrays)
        entries.append({"configuration": name, "beta_schedule": spec["beta_schedule"],
                        "scheduler_config": {k: v for k, v in pipe.scheduler.config.items()
                                             if not k.startswith("_")},
                        "file": file.name, "sha256": p2.sha256(file),
                        "beta_first": float(betas[0]), "beta_last": float(betas[-1]),
                        "terminal_snr": float(snr[-1])})
    path = run / "curves.json"
    if path.exists() and p2.read_json(path) != entries:
        raise RuntimeError("Saved curve metadata changed.")
    if not path.exists():
        p2.write_json(path, entries)
    return entries


def check_baseline(cfg, run, record):
    """Compare the repeated baseline; cross-runtime equality is diagnostic."""
    import numpy as np
    old_run = m2.path_from_root(cfg["sweep_dir"])
    inputs = record["inputs"]
    if inputs["configuration"]["beta_schedule"] != "scaled_linear":
        raise RuntimeError("Repeated DDIM baseline requires scaled_linear.")
    prompt_id, seed = inputs["prompt_record"]["index"], inputs["seed"]
    old = p2.read_json(old_run / "generated" / f"S-DDIM_{prompt_id:02d}_{seed}.json")
    if (old["inputs"]["prompt_record"] != inputs["prompt_record"] or
            old["inputs"]["seed"] != seed or old["inputs"]["raw_prompt"]):
        raise RuntimeError("Earlier DDIM baseline inputs differ.")
    expected = dict(inputs["configuration"])
    expected.pop("beta_schedule")
    if old["inputs"]["configuration"] != expected:
        raise RuntimeError("Earlier DDIM baseline settings differ.")
    m2.verify_record(old_run, old)
    with np.load(old_run / old["diagnostics"], allow_pickle=False) as a:
        with np.load(run / record["diagnostics"], allow_pickle=False) as b:
            latent_equal = np.array_equal(a["final_latents"], b["final_latents"])
            max_change = float(np.max(np.abs(a["final_latents"] - b["final_latents"])))
    result = {"prompt_id": prompt_id, "seed": seed,
              "image_exact": record["image_sha256"] == old["image_sha256"],
              "latents_exact": bool(latent_equal), "latent_max_abs_diff": max_change}
    return result


def verify(cfg, data_config):
    """Verify all 108 outputs and their connection to the original experiment."""
    import numpy as np
    run = run_path(data_config)
    saved = p2.read_json(run / "run_signature.json")
    prompts = p2.read_json(m2.path_from_root(cfg["prompts_file"]))
    m2.verify_sources(saved["sources"], SOURCES)
    if saved != signature(cfg, data_config, prompts, saved["environment"], saved["sources"]):
        raise RuntimeError("Noise-schedule signature differs from current sources or inputs.")
    if p2.read_json(run / "prompts.json") != prompts:
        raise RuntimeError("Noise-schedule prompts changed.")
    summary = p2.read_json(run / "summary.json")
    if summary["status"] != "complete" or summary["sample_count"] != 108:
        raise RuntimeError("Noise-schedule experiment is incomplete.")
    curves = p2.read_json(run / "curves.json")
    if len(curves) != 3 or summary["curves_sha256"] != p2.sha256(run / "curves.json"):
        raise RuntimeError("Noise-curve metadata differs.")
    for entry, (name, spec) in zip(curves, configurations().items()):
        file = run / f"{name}_curves.npz"
        if (entry["configuration"] != name or entry["file"] != file.name or
                entry["beta_schedule"] != spec["beta_schedule"] or
                entry["sha256"] != p2.sha256(file)):
            raise RuntimeError("Noise-curve identity or hash differs.")
        with np.load(file, allow_pickle=False) as arrays:
            for key in ("betas", "alphas_cumprod", "snr"):
                if arrays[key].shape != (1000,) or not np.isfinite(arrays[key]).all():
                    raise RuntimeError("Invalid saved noise curve.")
    records, checks = [], []
    for name, spec in configurations().items():
        for prompt in prompts:
            for offset in cfg["seed_offsets"]:
                seed = prompt["seed"] + offset
                record = p2.read_json(run / "generated" / f"{name}_{prompt['index']:02d}_{seed}.json")
                if record["inputs"] != dict(prompt_record=prompt, seed=seed,
                                             configuration=spec, raw_prompt=False):
                    raise RuntimeError("Noise-schedule sample inputs differ.")
                m2.verify_record(run, record)
                if (record["scheduler_class"] != "DDIMScheduler" or record["unet_calls"] != 30 or
                        record["scheduler_config"]["beta_schedule"] != spec["beta_schedule"] or
                        record["scheduler_config"]["prediction_type"] != "epsilon"):
                    raise RuntimeError("Noise-schedule sampling settings differ.")
                records.append(record)
                if name == "N-scaled":
                    checks.append(check_baseline(cfg, run, record))
    if len(records) != 108 or p2.read_json(run / "baseline_reproduction.json") != checks:
        raise RuntimeError("Noise-schedule coverage or baseline checks differ.")
    exact = sum(c["image_exact"] and c["latents_exact"] for c in checks)
    if summary["baseline_checked_count"] != len(checks) or summary["baseline_exact_count"] != exact:
        raise RuntimeError("Baseline reproduction summary differs.")
    return records


def run(cfg, data_config):
    folder = run_path(data_config)
    if (folder / "summary.json").is_file():
        verify(cfg, data_config)
        return p2.read_json(folder / "summary.json")

    prompts = m2.fixed_inputs(cfg, data_config)
    if len(prompts) != 12:
        raise ValueError("Expected the original 12 validation prompts.")
    env = m2.environment()
    validation = baseline_gate(cfg, data_config, prompts, env)
    folder = bind_run(cfg, data_config, prompts, env)
    pipe, original, load_seconds = m2.load_pipeline(cfg)
    if (original["beta_schedule"] != "scaled_linear" or original["trained_betas"] is not None
            or original["prediction_type"] != "epsilon" or original["num_train_timesteps"] != 1000):
        raise RuntimeError("Unexpected pretrained noise configuration.")
    p2.write_json(folder / "sessions" / f"load_{time.time_ns()}.json",
                  {"seconds": load_seconds, "scheduler": original,
                   "environment": env, "baseline_validation": validation})
    write_curves(pipe, original, folder)
    m2.sample(pipe, original, cfg, dict(m2.BASE, sampler="DDIM", steps=5),
              prompts[0], prompts[0]["seed"], folder, f"warmup_{time.time_ns()}")

    def generate(name, spec, prompt, seed):
        variant = dict(original, beta_schedule=spec["beta_schedule"])
        record = m2.sample(pipe, variant, cfg, spec, prompt, seed, folder,
                           f"{name}_{prompt['index']:02d}_{seed}")
        if name == "N-scaled":
            check_baseline(cfg, folder, record)
        return record

    # These six finite-trajectory checks are part of the 108 formal outputs.
    for name, spec in configurations().items():
        for prompt in (prompts[0], prompts[-1]):
            generate(name, spec, prompt, prompt["seed"])
    print("Six pilot outputs verified; continuing paired comparisons.", flush=True)
    records, rows, checks = [], [], []
    for name, spec in configurations().items():
        for prompt in prompts:
            for offset in cfg["seed_offsets"]:
                seed = prompt["seed"] + offset
                record = generate(name, spec, prompt, seed)
                records.append(record)
                rows.append(dict(configuration=name, prompt_id=prompt["index"], seed=seed,
                                 **spec, seconds=record["seconds"],
                                 peak_allocated_gib=record["peak_allocated_gib"],
                                 unet_calls=record["unet_calls"],
                                 flagged=record["nsfw_content_detected"], image=record["image"]))
                if name == "N-scaled":
                    checks.append(check_baseline(cfg, folder, record))
                print(f"{len(records)}/108 {name} prompt={prompt['index']} seed={seed}", flush=True)
    with (folder / "samples.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p2.write_json(folder / "baseline_reproduction.json", checks)
    summary = {"status": "complete", "phase": "noise_schedule", "sample_count": len(records),
               "training_performed": False, "checkpoint_frozen": True,
               "configurations": configurations(), "baseline_checked_count": len(checks),
               "baseline_exact_count": sum(c["image_exact"] and c["latents_exact"] for c in checks),
               "total_inference_seconds": sum(r["seconds"] for r in records),
               "flagged_count": sum(r["nsfw_content_detected"] for r in records),
               "curves_sha256": p2.sha256(folder / "curves.json"),
               "timing_scope": "pipeline call including diagnostics, VAE and safety; excludes encoding/save/load"}
    p2.write_json(folder / "summary.json", summary)
    verify(cfg, data_config)
    return summary


def gallery(cfg, data_config, records):
    """Export all comparisons; return only three small notebook previews."""
    from PIL import Image, ImageDraw
    run = run_path(data_config)
    prompts = p2.read_json(m2.path_from_root(cfg["prompts_file"]))
    lookup = {(r["inputs"]["configuration"]["beta_schedule"],
               r["inputs"]["prompt_record"]["index"], r["inputs"]["seed"]): r for r in records}
    entries, previews = [], {}
    target = run / "gallery.html"
    partial = target.with_suffix(".html.partial")
    with partial.open("w", encoding="utf-8") as output:
        output.write('<!doctype html><html lang="en"><meta charset="utf-8">'
                     '<title>Beta-schedule comparisons</title><style>'
                     'body{font:16px sans-serif;max-width:800px;margin:24px auto}'
                     'figure{margin:24px 0;break-inside:avoid}img{width:100%}</style>'
                     '<h1>Frozen-checkpoint beta-schedule comparisons</h1>'
                     '<p>DDIM, 30 steps, guidance 7.5, token weight 1, leading spacing, eta 0. '
                     '108 outputs: 12 prompts, 3 seeds, 3 beta schedules. No retraining.</p>')
        for offset in cfg["seed_offsets"]:
            for prompt in prompts:
                seed = prompt["seed"] + offset
                sheet = Image.new("RGB", (768, 300), "white")
                draw, sources = ImageDraw.Draw(sheet), []
                for i, (name, beta) in enumerate(SCHEDULES.items()):
                    record = lookup[beta, prompt["index"], seed]
                    with Image.open(run / record["image"]) as image:
                        sheet.paste(image.resize((256, 256)), (i * 256, 44))
                    label = f"{name}\n{beta}" + (" FLAGGED" if record["nsfw_content_detected"] else "")
                    draw.text((i * 256 + 4, 4), label, fill="black")
                    sources.append({"configuration": name, "image": record["image"],
                                    "image_sha256": record["image_sha256"]})
                buffer = io.BytesIO()
                sheet.save(buffer, format="JPEG", quality=85)
                jpeg = buffer.getvalue()
                caption = html.escape(f"{prompt['index']:02d} | seed {seed} | {prompt['prompt']}", quote=True)
                output.write(f'<figure><figcaption>{caption}</figcaption><img alt="{caption}" '
                             f'src="data:image/jpeg;base64,{base64.b64encode(jpeg).decode("ascii")}"></figure>')
                entries.append({"prompt_id": prompt["index"], "seed": seed, "sources": sources})
                if offset == 0 and prompt["index"] in (2, 3, 12):
                    previews[prompt["index"]] = jpeg
        output.write('</html>')
    if len(entries) != 36 or len(lookup) != 108:
        raise RuntimeError("Incomplete noise-schedule gallery.")
    partial.replace(target)
    p2.write_json(run / "gallery_index.json", {"sample_count": 108, "gallery": target.name,
                  "gallery_sha256": p2.sha256(target), "groups": entries})
    return previews
