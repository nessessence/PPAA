#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


PROMPTS: tuple[str, ...] = (
    "a photo of Vegeta",
    "a photo of Sasuke",
    "a photo of Luffy",
    "a photo of Naruto",
)


@dataclass(frozen=True)
class Experiment:
    index: int
    exp_name: str
    checkpoint_path: Path
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate extra prompts for completed Goku PPAA experiments."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--n-generation", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=7.50)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--gen-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gen-batch", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)

    parser.add_argument("--model-id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--save-dir", type=Path, default=Path("data_root/logs/ppaa"))
    parser.add_argument("--output-base", type=Path, default=Path("data_root/generated"))
    parser.add_argument("--force-generate", action="store_true")

    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    if args.n_generation < 1:
        raise ValueError("--n-generation must be at least 1.")
    if args.gen_batch < 1:
        raise ValueError("--gen-batch must be at least 1.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100].")


def prompt_path_name(prompt: str) -> str:
    clean = re.sub(r"[\\/]+", "_", prompt.strip())
    return clean or "prompt"


def cfg_dir_name(args: argparse.Namespace) -> str:
    return f"{args.cfg_scale:.2f}"


def image_path(exp: Experiment, prompt: str, image_seed: int, args: argparse.Namespace) -> Path:
    return (
        exp.output_root
        / "step1000"
        / prompt_path_name(prompt)
        / cfg_dir_name(args)
        / f"{image_seed:04d}.jpg"
    )


def discover_experiments(args: argparse.Namespace) -> list[Experiment]:
    checkpoint_paths = sorted(
        args.save_dir.glob("goku_to_anime-character*/step1000.safetensors")
    )
    experiments: list[Experiment] = []
    for checkpoint_path in checkpoint_paths:
        exp_name = checkpoint_path.parent.name
        output_root = args.output_base / "ppaa" / exp_name
        if not output_root.is_dir():
            continue
        experiments.append(
            Experiment(
                index=len(experiments),
                exp_name=exp_name,
                checkpoint_path=checkpoint_path,
                output_root=output_root,
            )
        )
    return experiments


def shard_experiments(experiments: list[Experiment], args: argparse.Namespace) -> list[Experiment]:
    return [
        exp
        for index, exp in enumerate(experiments)
        if index % args.num_shards == args.shard_index
    ]


def generation_seeds(args: argparse.Namespace) -> list[int]:
    return list(range(args.n_generation))


def prompt_complete(exp: Experiment, prompt: str, args: argparse.Namespace) -> bool:
    if args.force_generate:
        return False
    return all(image_path(exp, prompt, seed, args).exists() for seed in generation_seeds(args))


def pending_prompts(exp: Experiment, args: argparse.Namespace) -> list[str]:
    return [prompt for prompt in PROMPTS if not prompt_complete(exp, prompt, args)]


def inference_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_name_or_path=args.model_id,
        gen_dtype=args.gen_dtype,
        device=args.device,
    )


def save_jpeg(image, path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, format="JPEG", quality=args.jpeg_quality)


def generate_prompt_images(pipeline, exp: Experiment, prompt: str, args: argparse.Namespace) -> None:
    import torch

    seeds = generation_seeds(args)
    for start in range(0, len(seeds), args.gen_batch):
        batch_seeds = seeds[start : start + args.gen_batch]
        batch_paths = [image_path(exp, prompt, seed, args) for seed in batch_seeds]
        pending = [
            (seed, path)
            for seed, path in zip(batch_seeds, batch_paths)
            if args.force_generate or not path.exists()
        ]
        if not pending:
            for seed in batch_seeds:
                print(f"  [skip] {prompt!r} seed{seed}")
            continue

        batch_prompts = [prompt] * len(pending)
        generators = [
            torch.Generator(device=args.device).manual_seed(seed) for seed, _ in pending
        ]
        result = pipeline(
            batch_prompts,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.cfg_scale,
            generator=generators,
        )
        for (seed, path), image in zip(pending, result.images):
            save_jpeg(image, path, args)
            print(f"  [done] {prompt!r} seed{seed}")


def cleanup_pipeline(pipeline) -> None:
    del pipeline
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_dry_run(
    all_experiments: list[Experiment],
    selected_experiments: list[Experiment],
    args: argparse.Namespace,
) -> None:
    total_images = len(all_experiments) * len(PROMPTS) * args.n_generation
    selected_images = len(selected_experiments) * len(PROMPTS) * args.n_generation
    print(
        f"total experiments: {len(all_experiments)} | "
        f"selected for shard {args.shard_index}/{args.num_shards}: "
        f"{len(selected_experiments)}"
    )
    print(f"prompts per experiment: {len(PROMPTS)}")
    print(f"generation seeds: 0..{args.n_generation - 1}")
    print(f"total images: {total_images} | selected images: {selected_images}")
    print(
        "image layout: "
        f"{args.output_base}/ppaa/{{exp_name}}/step1000/{{prompt}}/"
        f"{args.cfg_scale:.2f}/0000.jpg"
    )
    for prompt in PROMPTS:
        print(f"prompt: {prompt!r}")
    for exp in selected_experiments[:20]:
        pending = pending_prompts(exp, args)
        print(
            f"{exp.index:03d} | {exp.exp_name} | "
            f"pending_prompts={len(pending)}/{len(PROMPTS)}"
        )
    if len(selected_experiments) > 20:
        print(f"... {len(selected_experiments) - 20} more experiments")


def main() -> None:
    from inference_ppaa import build_pipeline, load_unet_weights

    args = parse_args()
    all_experiments = discover_experiments(args)
    selected_experiments = shard_experiments(all_experiments, args)

    if args.dry_run:
        print_dry_run(all_experiments, selected_experiments, args)
        return

    print(
        f"Running {len(selected_experiments)} Goku experiments on {args.device} "
        f"(shard {args.shard_index}/{args.num_shards})."
    )
    print(
        "image layout: "
        f"{args.output_base}/ppaa/{{exp_name}}/step1000/{{prompt}}/"
        f"{args.cfg_scale:.2f}/0000.jpg"
    )

    runnable = [(exp, pending_prompts(exp, args)) for exp in selected_experiments]
    runnable = [(exp, prompts) for exp, prompts in runnable if prompts]
    if not runnable:
        print("[skip] all selected Goku prompt images already exist")
        return

    pipeline = build_pipeline(inference_args(args))
    try:
        for exp, prompts in runnable:
            print("=" * 88)
            print(f"[load] exp_name={exp.exp_name}")
            print(f"checkpoint: {exp.checkpoint_path}")
            load_unet_weights(pipeline, str(exp.checkpoint_path))
            for prompt in prompts:
                print(f"[generate] {prompt!r}")
                generate_prompt_images(pipeline, exp, prompt, args)
    finally:
        cleanup_pipeline(pipeline)


if __name__ == "__main__":
    main()
