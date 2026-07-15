#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path
from types import SimpleNamespace


PROMPTS: tuple[str, ...] = (
    "a photo of Naruto",
    "a photo of Sasuke Uchiha",
    "a photo of Sasuke",
    "a photo of Killua Zoldyck",
    "a photo of Kurapika",
    "a photo of Griffith",
    "a photo of Spike Spiegel",
    "a photo of Inuyasha",
    "a photo of Dio Brando",
    "a photo of Itachi Uchiha",
    "a photo of Ichigo Kurosaki",
    "a photo of Vegeta",
    "a photo of Sanji",
    "a photo of Luffy",
    "a photo of Edward Elric",
    "a photo of All Might",
    "a photo of Deidara",
    "a photo of Minato Namikaze",
    "a photo of Giorno Giovanna",
    "a photo of Great Teacher Onizuka",
    "a photo of Yagami Light",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate original SD1.4 images for a fixed prompt list."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--n-generation", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=7.50)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--gen-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--model-id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--output-base", type=Path, default=Path("data_root/generated"))
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Only consider prompts from this PROMPTS index onward before sharding.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards.")
    if args.n_generation < 1:
        raise ValueError("--n-generation must be at least 1.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100].")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    return args


def selected_prompts(args: argparse.Namespace) -> list[str]:
    return [
        prompt
        for index, prompt in enumerate(PROMPTS[args.start_index :])
        if index % args.num_shards == args.shard_index
    ]


def prompt_path_name(prompt: str) -> str:
    clean = re.sub(r"[\\/]+", "_", prompt.strip())
    return clean or "prompt"


def cfg_dir_name(args: argparse.Namespace) -> str:
    return f"{args.cfg_scale:.2f}"


def image_path(prompt: str, image_seed: int, args: argparse.Namespace) -> Path:
    return (
        args.output_base
        / "ppaa"
        / "original_pretrained_sd1.4"
        / prompt_path_name(prompt)
        / cfg_dir_name(args)
        / f"{image_seed:04d}.jpg"
    )


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


def cleanup_pipeline(pipeline) -> None:
    del pipeline
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    import torch
    from inference_ppaa import build_pipeline

    args = parse_args()
    prompts = selected_prompts(args)
    seeds = list(range(args.n_generation))
    print(
        f"Running {len(prompts)} original prompts on {args.device} "
        f"(shard {args.shard_index}/{args.num_shards})."
    )
    print(f"generation seeds: 0..{args.n_generation - 1}")
    print(
        "original image layout: "
        f"{args.output_base}/ppaa/original_pretrained_sd1.4/{{prompt}}/"
        f"{args.cfg_scale:.2f}/0000.jpg"
    )
    for prompt in prompts:
        print(f"[prompt] {prompt!r}")

    pending = []
    for prompt in prompts:
        paths = [image_path(prompt, seed, args) for seed in seeds]
        if args.force_generate or not all(path.exists() for path in paths):
            pending.append((prompt, paths))
    if not pending:
        print("[skip] all original images already exist")
        return

    pipeline = build_pipeline(inference_args(args))
    try:
        for prompt, paths in pending:
            print(f"[generate] {prompt!r}")
            for image_seed, path in zip(seeds, paths):
                if path.exists() and not args.force_generate:
                    print(f"  [skip] {prompt!r} seed{image_seed}")
                    continue
                generator = torch.Generator(device=args.device).manual_seed(image_seed)
                result = pipeline(
                    [prompt],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.cfg_scale,
                    generator=generator,
                )
                save_jpeg(result.images[0], path, args)
                print(f"  [done] {prompt!r} seed{image_seed}")
    finally:
        cleanup_pipeline(pipeline)


if __name__ == "__main__":
    main()
