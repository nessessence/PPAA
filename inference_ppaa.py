#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stable Diffusion images with an optional PPAA/ESD UNet checkpoint."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        required=True,
        help='Semicolon-separated prompts, e.g. "a photo of Barack Obama;a photo of Will Smith".',
    )
    parser.add_argument("--gen_image_path", type=str, required=True)
    parser.add_argument("--load_unet_weight_path", type=str, default=None)
    parser.add_argument("--wait_weight", action="store_true", default=False)

    parser.add_argument("--num_validation_images", type=int, default=100)
    parser.add_argument("--cfg_scale", type=str, default="7.50")
    parser.add_argument("--gen_batch", type=int, default=10)
    parser.add_argument("--sampler", type=str, default="DDIM", choices=["DDIM"])

    parser.add_argument("--gen_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_validation_images < 1:
        raise ValueError("--num_validation_images must be at least 1.")
    if args.gen_batch < 1:
        raise ValueError("--gen_batch must be at least 1.")
    if args.num_inference_steps < 1:
        raise ValueError("--num_inference_steps must be at least 1.")
    if not split_prompts(args.instance_prompt):
        raise ValueError("--instance_prompt must contain at least one non-empty prompt.")


def split_prompts(raw_prompt: str) -> list[str]:
    return [prompt.strip() for prompt in raw_prompt.split(";") if prompt.strip()]


def dtype_from_name(dtype_name: str) -> Any:
    import torch

    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def wait_for_weight(weight_path: str) -> None:
    while not os.path.exists(weight_path):
        print(f"waiting for unet weight: {weight_path}")
        time.sleep(30)


def load_unet_weights(pipeline: Any, weight_path: str) -> None:
    import torch
    from safetensors.torch import load_file

    print(f"loading UNet weight from: {weight_path}")
    if weight_path.endswith(".safetensors") or weight_path.endswith(".safetensor"):
        state_dict = load_file(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    missing_keys, unexpected_keys = pipeline.unet.load_state_dict(state_dict, strict=False)
    print(
        "UNet weight loaded "
        f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
    )
    pipeline.unet.eval()


def build_pipeline(args: argparse.Namespace) -> Any:
    from diffusers import DDIMScheduler, DiffusionPipeline

    dtype = dtype_from_name(args.gen_dtype)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline = pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def save_prompt_images(
    pipeline: Any,
    prompt: str,
    output_root: Path,
    cfg_scale: float,
    num_images: int,
    batch_size: int,
    num_inference_steps: int,
    generator: Any,
) -> None:
    from tqdm.auto import tqdm

    save_dir = output_root / prompt / f"{cfg_scale:.2f}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save_image_path_dir: {save_dir}")

    for start in tqdm(range(0, num_images, batch_size), desc=prompt):
        stop = min(start + batch_size, num_images)
        prompts = [prompt] * (stop - start)
        images = pipeline(
            prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
        ).images

        for offset, image in enumerate(images):
            image.save(save_dir / f"{start + offset:04}.png")


def generate(args: argparse.Namespace) -> None:
    import torch

    validate_args(args)
    prompts = split_prompts(args.instance_prompt)
    cfg_scales = [float(cfg.strip()) for cfg in args.cfg_scale.split(",") if cfg.strip()]
    output_root = Path(args.gen_image_path)
    output_root.mkdir(parents=True, exist_ok=True)

    weight_path = args.load_unet_weight_path
    if weight_path == "":
        weight_path = None
    if weight_path is not None and args.wait_weight:
        wait_for_weight(weight_path)

    print(f"prompts: {prompts}")
    print(f"cfg scales: {[f'{cfg:.2f}' for cfg in cfg_scales]}")
    print(f"output root: {output_root}")
    print(f"gen dtype: {args.gen_dtype}")
    print(f"device: {args.device}")

    pipeline = build_pipeline(args)
    if weight_path is not None:
        load_unet_weights(pipeline, weight_path)
    else:
        print("not loading UNet weight")

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    for prompt in prompts:
        for cfg_scale in cfg_scales:
            print(f"prompt: {prompt} cfg: {cfg_scale:.2f}")
            save_prompt_images(
                pipeline=pipeline,
                prompt=prompt,
                output_root=output_root,
                cfg_scale=cfg_scale,
                num_images=args.num_validation_images,
                batch_size=args.gen_batch,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )


def main() -> None:
    generate(parse_args())


if __name__ == "__main__":
    main()
