from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from safetensors.torch import save_file
from tqdm.auto import tqdm


STYLE_PREFIX = "a painting in the style of "
TERMINAL_TIMESTEP = 999
IMAGE_SIZE = 512
LATENT_SIZE = 64
WANDB_PROJECT = "ppaa"
TOKEN_SELECTION_CHOICES = ("C", "Cs", "E", "CE", "CsE")


@dataclass(frozen=True)
class TrainConfig:
    erase_concept: str
    generic_concept: str
    token_selection: str = "CE"
    pull_token_selection: str = ""
    semantic_preserve_token: str = "all"
    m_push: float = 0.40
    m_pull: float = 0.40
    pull_option: str = "relative"
    push_weight: float = 1.0
    pull_weight: float = 1.0
    semantic_preserve_weight: float = 32.0
    denoise_preserve_weight: float = 1.0
    lr: float = 5e-5
    steps: int = 1000
    batch_size: int = 4
    seed: int = 0
    device: str = "cuda:0"
    precision: str = "bf16"
    save_dir: str = "data_root/logs/ppaa"
    save_every: int = 1000
    model_id: str = "CompVis/stable-diffusion-v1-4"
    report_to: str = "wandb"


@dataclass(frozen=True)
class PromptBundle:
    erase_prompt: str
    generic_prompt: str
    erase_concept_text: str
    generic_concept_text: str


@dataclass(frozen=True)
class TokenPlan:
    erase_concept_indices: tuple[int, ...]
    generic_concept_indices: tuple[int, ...]
    erase_eos_index: int
    generic_eos_index: int
    push_indices: tuple[int, ...]
    pull_pairs: tuple[tuple[int, int], ...]
    semantic_preserve_indices: tuple[int, ...]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a clean PPAA concept eraser for a single concept."
    )
    parser.add_argument("--erase_concept", type=str, required=True)
    parser.add_argument("--generic_concept", type=str, required=True)

    parser.add_argument(
        "--token_selection",
        type=str,
        default="CE",
        choices=TOKEN_SELECTION_CHOICES,
    )
    parser.add_argument(
        "--pull_token_selection",
        type=str,
        default="",
        choices=("", *TOKEN_SELECTION_CHOICES),
        help=(
            "Token positions used for pull pairs. Empty means reuse "
            "--token_selection."
        ),
    )
    parser.add_argument(
        "--semantic_preserve_token",
        type=str,
        default="all",
        choices=["all", "till_eos"],
        help="Token positions used for semantic preservation.",
    )
    parser.add_argument("--m_push", type=float, default=0.40)
    parser.add_argument("--m_pull", type=float, default=0.40)
    parser.add_argument(
        "--pull_option", type=str, default="relative", choices=["relative", "absolute"]
    )
    parser.add_argument("--push_weight", type=float, default=1.0)
    parser.add_argument("--pull_weight", type=float, default=1.0)
    parser.add_argument("--semantic_preserve_weight", type=float, default=32.0)
    parser.add_argument("--denoise_preserve_weight", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--save_dir", type=str, default="data_root/logs/ppaa")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--report_to", type=str, default="wandb", choices=["none", "wandb"])

    cfg = TrainConfig(**vars(parser.parse_args()))
    validate_config(cfg)
    return cfg


def validate_config(cfg: TrainConfig) -> None:
    if cfg.token_selection not in TOKEN_SELECTION_CHOICES:
        raise ValueError(
            f"--token_selection must be one of: {', '.join(TOKEN_SELECTION_CHOICES)}."
        )
    if cfg.pull_token_selection not in ("", *TOKEN_SELECTION_CHOICES):
        raise ValueError(
            "--pull_token_selection must be empty or one of: "
            f"{', '.join(TOKEN_SELECTION_CHOICES)}."
        )
    if cfg.semantic_preserve_token not in {"all", "till_eos"}:
        raise ValueError("--semantic_preserve_token must be one of: all, till_eos.")
    if cfg.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if cfg.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")
    if cfg.save_every < 1:
        raise ValueError("--save_every must be at least 1.")
    if cfg.lr <= 0:
        raise ValueError("--lr must be positive.")
    for name in ("m_push", "m_pull"):
        value = getattr(cfg, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1].")
    for name in (
        "push_weight",
        "pull_weight",
        "semantic_preserve_weight",
        "denoise_preserve_weight",
    ):
        value = getattr(cfg, name)
        if value < 0.0:
            raise ValueError(f"--{name} must be non-negative.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def strip_style_prefix(text: str) -> str:
    if text.startswith(STYLE_PREFIX):
        return text[len(STYLE_PREFIX) :].strip()
    return text.strip()


def build_prompts(erase_concept: str, generic_concept: str) -> PromptBundle:
    is_style = erase_concept.startswith(STYLE_PREFIX) or generic_concept.startswith(
        STYLE_PREFIX
    )
    if is_style:
        erase_text = strip_style_prefix(erase_concept)
        generic_text = strip_style_prefix(generic_concept)
        return PromptBundle(
            erase_prompt=f"{STYLE_PREFIX}{erase_text}",
            generic_prompt=f"{STYLE_PREFIX}{generic_text}",
            erase_concept_text=erase_text,
            generic_concept_text=generic_text,
        )
    return PromptBundle(
        erase_prompt=erase_concept,
        generic_prompt=generic_concept,
        erase_concept_text=erase_concept,
        generic_concept_text=generic_concept,
    )


def first_eos_index(tokenizer, token_ids: list[int]) -> int:
    try:
        return token_ids.index(tokenizer.eos_token_id)
    except ValueError as exc:
        raise ValueError("Prompt tokenization did not contain an EOS token.") from exc


def find_token_span(tokenizer, prompt: str, concept_text: str) -> tuple[int, ...]:
    full_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    concept_ids = tokenizer(concept_text, add_special_tokens=False)["input_ids"]
    if not concept_ids:
        raise ValueError(f"Concept text tokenized to an empty span: {concept_text!r}")

    for start in range(len(full_ids) - len(concept_ids) + 1):
        end = start + len(concept_ids)
        if full_ids[start:end] == concept_ids:
            return tuple(range(start, end))

    full_tokens = tokenizer.convert_ids_to_tokens(full_ids)
    concept_tokens = tokenizer.convert_ids_to_tokens(concept_ids)
    raise ValueError(
        "Could not find concept token span.\n"
        f"Prompt: {prompt!r}\n"
        f"Concept: {concept_text!r}\n"
        f"Prompt tokens: {full_tokens}\n"
        f"Concept tokens: {concept_tokens}"
    )


def dedupe_ordered(values: Iterable[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def semantic_preserve_indices(
    tokenizer, generic_eos: int, semantic_preserve_token: str
) -> tuple[int, ...]:
    if semantic_preserve_token == "all":
        max_length = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError(
                "Tokenizer must expose a positive integer model_max_length "
                "when --semantic_preserve_token=all."
            )
        return tuple(range(max_length))
    if semantic_preserve_token == "till_eos":
        return tuple(range(generic_eos + 1))
    raise ValueError(f"Unsupported semantic preserve token mode: {semantic_preserve_token}")


def build_token_plan(
    tokenizer,
    prompts: PromptBundle,
    token_selection: str,
    pull_token_selection: str = "",
    semantic_preserve_token: str = "all",
) -> TokenPlan:
    erase_ids = tokenizer(prompts.erase_prompt, add_special_tokens=True)["input_ids"]
    generic_ids = tokenizer(prompts.generic_prompt, add_special_tokens=True)["input_ids"]
    erase_eos = first_eos_index(tokenizer, erase_ids)
    generic_eos = first_eos_index(tokenizer, generic_ids)

    erase_span = find_token_span(
        tokenizer, prompts.erase_prompt, prompts.erase_concept_text
    )
    generic_span = find_token_span(
        tokenizer, prompts.generic_prompt, prompts.generic_concept_text
    )
    erase_last = erase_span[-1]
    generic_last = generic_span[-1]

    effective_pull_token_selection = pull_token_selection or token_selection
    push_indices = push_indices_for_selection(
        token_selection, erase_span, erase_eos
    )
    pull_pairs = pull_pairs_for_selection(
        effective_pull_token_selection,
        erase_span,
        generic_span,
        erase_eos,
        generic_eos,
    )

    preserve_indices = semantic_preserve_indices(
        tokenizer, generic_eos, semantic_preserve_token
    )
    return TokenPlan(
        erase_concept_indices=erase_span,
        generic_concept_indices=generic_span,
        erase_eos_index=erase_eos,
        generic_eos_index=generic_eos,
        push_indices=push_indices,
        pull_pairs=pull_pairs,
        semantic_preserve_indices=preserve_indices,
    )


def push_indices_for_selection(
    token_selection: str,
    erase_span: tuple[int, ...],
    erase_eos: int,
) -> tuple[int, ...]:
    erase_last = erase_span[-1]
    if token_selection == "C":
        return (erase_last,)
    if token_selection == "Cs":
        return erase_span
    if token_selection == "E":
        return (erase_eos,)
    if token_selection == "CE":
        return dedupe_ordered((erase_last, erase_eos))
    if token_selection == "CsE":
        return dedupe_ordered((*erase_span, erase_eos))
    raise ValueError(f"Unsupported token selection: {token_selection}")


def pull_pairs_for_selection(
    token_selection: str,
    erase_span: tuple[int, ...],
    generic_span: tuple[int, ...],
    erase_eos: int,
    generic_eos: int,
) -> tuple[tuple[int, int], ...]:
    erase_last = erase_span[-1]
    generic_last = generic_span[-1]
    if token_selection == "C":
        return ((erase_last, generic_last),)
    if token_selection == "Cs":
        return tuple((e_idx, g_idx) for e_idx in erase_span for g_idx in generic_span)
    if token_selection == "E":
        return ((erase_eos, generic_eos),)
    if token_selection == "CE":
        return ((erase_last, generic_last), (erase_eos, generic_eos))
    if token_selection == "CsE":
        return tuple(
            (e_idx, g_idx) for e_idx in erase_span for g_idx in generic_span
        ) + ((erase_eos, generic_eos),)
    raise ValueError(f"Unsupported token selection: {token_selection}")


def encode_prompts(
    pipe: StableDiffusionPipeline,
    prompts: list[str],
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    return prompt_embeds.detach().to(device=device, dtype=dtype)


def select_trainable_kv_modules(unet: UNet2DConditionModel) -> list[str]:
    module_names: list[str] = []
    for name, module in unet.named_modules():
        if "attn2.to_k" not in name and "attn2.to_v" not in name:
            continue
        if module.__class__.__name__ not in {
            "Linear",
            "LoRACompatibleLinear",
            "PatchedLoraProjection",
        }:
            continue
        module_names.append(name)
    if not module_names:
        raise RuntimeError("No cross-attention K/V modules were found.")
    return module_names


def trainable_parameters(
    unet: UNet2DConditionModel, module_names: list[str]
) -> tuple[list[str], list[torch.nn.Parameter]]:
    selected_prefixes = tuple(f"{name}." for name in module_names)
    param_names: list[str] = []
    params: list[torch.nn.Parameter] = []
    for name, param in unet.named_parameters():
        if name.startswith(selected_prefixes):
            param.requires_grad_(True)
            param_names.append(name)
            params.append(param)
        else:
            param.requires_grad_(False)
    if not params:
        raise RuntimeError("No trainable K/V parameters were selected.")
    return param_names, params


def cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x.float(), y.float(), dim=-1)


def compute_feature_losses(
    train_unet: UNet2DConditionModel,
    base_unet: UNet2DConditionModel,
    module_names: list[str],
    p_e: torch.Tensor,
    p_g: torch.Tensor,
    token_plan: TokenPlan,
    m_push: float,
    m_pull: float,
    pull_option: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_modules = dict(train_unet.named_modules())
    base_modules = dict(base_unet.named_modules())
    device = p_e.device

    push_terms: list[torch.Tensor] = []
    pull_terms: list[torch.Tensor] = []
    preserve_terms: list[torch.Tensor] = []

    push_idx = torch.tensor(token_plan.push_indices, device=device, dtype=torch.long)
    pull_e_idx = torch.tensor(
        [pair[0] for pair in token_plan.pull_pairs], device=device, dtype=torch.long
    )
    pull_g_idx = torch.tensor(
        [pair[1] for pair in token_plan.pull_pairs], device=device, dtype=torch.long
    )
    preserve_idx = torch.tensor(
        token_plan.semantic_preserve_indices, device=device, dtype=torch.long
    )

    for name in module_names:
        train_module = train_modules[name]
        base_module = base_modules[name]

        updated_e = train_module(p_e)
        updated_g = train_module(p_g)

        with torch.no_grad():
            base_e = base_module(p_e)
            base_g = base_module(p_g)

        push_cos = cosine(updated_e[0, push_idx], base_e[0, push_idx])
        push_terms.append(torch.clamp(push_cos - m_push, min=0.0).mean())

        pull_cos = cosine(updated_e[0, pull_e_idx], base_g[0, pull_g_idx])
        if pull_option == "relative":
            with torch.no_grad():
                base_pull_cos = cosine(base_e[0, pull_e_idx], base_g[0, pull_g_idx])
                safe_denom = torch.where(
                    base_pull_cos.abs() < 1e-6,
                    torch.full_like(base_pull_cos, 1e-6),
                    base_pull_cos,
                )
            pull_ratio = pull_cos / safe_denom
            negative_denom = base_pull_cos < 0.0
            shifted_pull_cos = (1.0 + pull_cos) / 2.0 + 1e-6
            shifted_base_pull_cos = (1.0 + base_pull_cos) / 2.0 + 1e-6
            pull_ratio = torch.where(
                negative_denom,
                shifted_pull_cos / shifted_base_pull_cos,
                pull_ratio,
            )
            pull_terms.append(torch.clamp(m_pull - pull_ratio, min=0.0).mean())
        elif pull_option == "absolute":
            pull_terms.append(torch.clamp(m_pull - pull_cos, min=0.0).mean())
        else:
            raise ValueError(f"Unsupported pull option: {pull_option}")

        preserve_cos = cosine(updated_g[0, preserve_idx], base_g[0, preserve_idx])
        preserve_terms.append((1.0 - preserve_cos).mean())

    return (
        torch.stack(push_terms).mean(),
        torch.stack(pull_terms).mean(),
        torch.stack(preserve_terms).mean(),
    )


def compute_denoise_preservation_loss(
    train_unet: UNet2DConditionModel,
    base_unet: UNet2DConditionModel,
    prompt_embeds: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    batch_size = prompt_embeds.shape[0]
    latents = torch.randn(
        (batch_size, train_unet.config.in_channels, LATENT_SIZE, LATENT_SIZE),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    timestep = torch.full(
        (batch_size,), TERMINAL_TIMESTEP, device=device, dtype=torch.long
    )

    with torch.no_grad():
        base_pred = base_unet(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]

    train_pred = train_unet(
        latents,
        timestep,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=None,
        cross_attention_kwargs=None,
        added_cond_kwargs=None,
        return_dict=False,
    )[0]
    return F.mse_loss(train_pred.float(), base_pred.float())


def sanitize_for_path(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "concept"


def format_loss_weight(value: float) -> str:
    return f"{value:g}"


def weighted_margin_name(
    name: str,
    margin: float,
    weight: float,
    default_weight: float = 1.0,
) -> str:
    if weight == default_weight:
        return f"{name}{margin:.2f}"
    return f"{format_loss_weight(weight)}{name}{margin:.2f}"


def effective_pull_token_selection(
    token_selection: str, pull_token_selection: str
) -> str:
    return pull_token_selection or token_selection


def token_selection_run_name(token_selection: str, pull_token_selection: str) -> str:
    effective_pull = effective_pull_token_selection(
        token_selection, pull_token_selection
    )
    if effective_pull == token_selection:
        return token_selection
    return f"{token_selection}.{effective_pull}"


def build_run_dir(cfg: TrainConfig) -> Path:
    push_name = weighted_margin_name("push", cfg.m_push, cfg.push_weight)
    if cfg.pull_weight == 0:
        pull_name = ""
    else:
        pull_name = "_" + weighted_margin_name("pull", cfg.m_pull, cfg.pull_weight)
    semantic_token_suffix = (
        "_seos" if cfg.semantic_preserve_token == "till_eos" else ""
    )

    run_name = (
        f"{sanitize_for_path(cfg.erase_concept)}_to_{sanitize_for_path(cfg.generic_concept)}"
        f"_{push_name}{pull_name}"
        f"_{cfg.pull_option}_"
        f"{token_selection_run_name(cfg.token_selection, cfg.pull_token_selection)}"
        f"{semantic_token_suffix}_seed{cfg.seed}"
    )
    return Path(cfg.save_dir) / run_name


def save_checkpoint(
    run_dir: Path,
    train_unet: UNet2DConditionModel,
    trainable_param_names: list[str],
    step: int,
) -> Path:
    params = dict(train_unet.named_parameters())
    state = {
        name: params[name].detach().cpu()
        for name in trainable_param_names
        if name in params
    }
    checkpoint_path = run_dir / f"step{step}.safetensors"
    save_file(state, str(checkpoint_path))
    return checkpoint_path


def write_config(
    run_dir: Path,
    cfg: TrainConfig,
    prompts: PromptBundle,
    token_plan: TokenPlan,
    module_names: list[str],
) -> None:
    payload = {
        "config": asdict(cfg),
        "effective_defaults": {
            "train_method": "cross_attention_kv",
            "terminal_timestep": TERMINAL_TIMESTEP,
            "image_size": IMAGE_SIZE,
            "latent_size": LATENT_SIZE,
            "preservation_prompts": ["", cfg.generic_concept],
            "template_behavior": "basic",
        },
        "prompts": asdict(prompts),
        "token_plan": {
            "push_token_selection": cfg.token_selection,
            "pull_token_selection": cfg.pull_token_selection,
            "effective_pull_token_selection": effective_pull_token_selection(
                cfg.token_selection, cfg.pull_token_selection
            ),
            "erase_concept_indices": list(token_plan.erase_concept_indices),
            "generic_concept_indices": list(token_plan.generic_concept_indices),
            "erase_eos_index": token_plan.erase_eos_index,
            "generic_eos_index": token_plan.generic_eos_index,
            "push_indices": list(token_plan.push_indices),
            "pull_pairs": [list(pair) for pair in token_plan.pull_pairs],
            "semantic_preserve_indices": list(token_plan.semantic_preserve_indices),
        },
        "num_trainable_modules": len(module_names),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def maybe_init_wandb(cfg: TrainConfig, run_dir: Path):
    if cfg.report_to != "wandb":
        return None
    try:
        import wandb

        return wandb.init(
            project=WANDB_PROJECT,
            name=run_dir.name,
            config={
                **asdict(cfg),
                "terminal_timestep": TERMINAL_TIMESTEP,
                "train_method": "cross_attention_kv",
            },
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        print(f"[wandb] init failed: {exc}")
        return None


def sample_preservation_embeds(
    library_embeds: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    indices = torch.randint(
        low=0,
        high=library_embeds.shape[0],
        size=(batch_size,),
        generator=generator,
        device=library_embeds.device,
    )
    return library_embeds[indices]


def train(cfg: TrainConfig) -> None:
    validate_config(cfg)
    set_seed(cfg.seed)
    dtype = dtype_from_precision(cfg.precision)
    device = torch.device(cfg.device)
    run_dir = build_run_dir(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    base_unet = UNet2DConditionModel.from_pretrained(
        cfg.model_id, subfolder="unet", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    base_unet.requires_grad_(False)
    base_unet.eval()

    train_unet = UNet2DConditionModel.from_pretrained(
        cfg.model_id, subfolder="unet", torch_dtype=dtype, use_safetensors=True
    ).to(device)
    train_unet.train()

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        unet=base_unet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    module_names = select_trainable_kv_modules(train_unet)
    trainable_param_names, params = trainable_parameters(train_unet, module_names)
    optimizer = torch.optim.Adam(params, lr=cfg.lr)

    prompts = build_prompts(cfg.erase_concept, cfg.generic_concept)
    token_plan = build_token_plan(
        pipe.tokenizer,
        prompts,
        cfg.token_selection,
        cfg.pull_token_selection,
        cfg.semantic_preserve_token,
    )
    write_config(run_dir, cfg, prompts, token_plan, module_names)

    p_e = encode_prompts(pipe, [prompts.erase_prompt], device, dtype)
    p_g = encode_prompts(pipe, [prompts.generic_prompt], device, dtype)
    preservation_library = encode_prompts(
        pipe, ["", cfg.generic_concept], device, dtype
    )

    wandb_run = maybe_init_wandb(cfg, run_dir)
    torch_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    preserve_generator = torch.Generator(device=device).manual_seed(cfg.seed + 2)

    print(f"Run directory: {run_dir}")
    print(f"Erase prompt: {prompts.erase_prompt}")
    print(f"Generic prompt: {prompts.generic_prompt}")
    print(f"Token selection: {cfg.token_selection}")
    print(f"Pull token selection: {cfg.pull_token_selection}")
    print(
        "Effective pull token selection: "
        f"{effective_pull_token_selection(cfg.token_selection, cfg.pull_token_selection)}"
    )
    print(f"Semantic preserve token mode: {cfg.semantic_preserve_token}")
    print(f"Semantic preserve indices: {token_plan.semantic_preserve_indices}")
    print(f"Push indices: {token_plan.push_indices}")
    print(f"Pull pairs: {token_plan.pull_pairs}")
    print(f"Trainable K/V modules: {len(module_names)}")
    print(f"Trainable K/V tensors: {len(trainable_param_names)}")

    last_saved_step = 0
    progress = tqdm(range(1, cfg.steps + 1), desc="Training PPAA")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)

        push_loss, pull_loss, semantic_preserve_loss = compute_feature_losses(
            train_unet=train_unet,
            base_unet=base_unet,
            module_names=module_names,
            p_e=p_e,
            p_g=p_g,
            token_plan=token_plan,
            m_push=cfg.m_push,
            m_pull=cfg.m_pull,
            pull_option=cfg.pull_option,
        )

        denoise_embeds = sample_preservation_embeds(
            preservation_library, cfg.batch_size, preserve_generator
        )
        denoise_preserve_loss = compute_denoise_preservation_loss(
            train_unet=train_unet,
            base_unet=base_unet,
            prompt_embeds=denoise_embeds,
            device=device,
            dtype=dtype,
            generator=torch_generator,
        )

        weighted_push_loss = cfg.push_weight * push_loss
        weighted_pull_loss = cfg.pull_weight * pull_loss
        weighted_semantic_preserve_loss = (
            cfg.semantic_preserve_weight * semantic_preserve_loss
        )
        weighted_denoise_preserve_loss = (
            cfg.denoise_preserve_weight * denoise_preserve_loss
        )
        total_loss = (
            weighted_push_loss
            + weighted_pull_loss
            + weighted_semantic_preserve_loss
            + weighted_denoise_preserve_loss
        )
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total loss at step {step}: {total_loss}")

        total_loss.backward()
        optimizer.step()

        progress_logs = {
            "push": float(push_loss.detach().item()),
            "pull": float(pull_loss.detach().item()),
            "semantic_preserve": float(semantic_preserve_loss.detach().item()),
            "denoise_preserve": float(denoise_preserve_loss.detach().item()),
            "total": float(total_loss.detach().item()),
        }
        wandb_logs = {
            "push_loss": float(push_loss.detach().item()),
            "pull_loss": float(pull_loss.detach().item()),
            "semantic_preserve_loss": float(semantic_preserve_loss.detach().item()),
            "denoise_preserve_loss": float(denoise_preserve_loss.detach().item()),
            "weighted_push_loss": float(weighted_push_loss.detach().item()),
            "weighted_pull_loss": float(weighted_pull_loss.detach().item()),
            "weighted_semantic_preserve_loss": float(
                weighted_semantic_preserve_loss.detach().item()
            ),
            "weighted_denoise_preserve_loss": float(
                weighted_denoise_preserve_loss.detach().item()
            ),
            "total_loss": float(total_loss.detach().item()),
            "total_weighted_loss": float(total_loss.detach().item()),
            "timestep": TERMINAL_TIMESTEP,
            "training_step": step,
            "lr": cfg.lr,
            "m_push": cfg.m_push,
            "m_pull": cfg.m_pull,
        }
        progress.set_postfix(
            {key: f"{value:.4f}" for key, value in progress_logs.items()}
        )

        if wandb_run is not None:
            try:
                wandb_run.log(wandb_logs, step=step)
            except Exception as exc:  # pragma: no cover - best effort logging
                print(f"[wandb] log failed: {exc}")

        if step % cfg.save_every == 0:
            checkpoint_path = save_checkpoint(
                run_dir, train_unet, trainable_param_names, step
            )
            last_saved_step = step
            print(f"Saved checkpoint: {checkpoint_path}")

    if last_saved_step != cfg.steps:
        checkpoint_path = save_checkpoint(
            run_dir, train_unet, trainable_param_names, cfg.steps
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train(parse_args())
