#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from .configuration_cosmos3 import Cosmos3Config
from .processor_cosmos3 import (
    COSMOS3_ACTION_CONDITION,
    COSMOS3_ACTION_CONDITION_MASK,
    COSMOS3_ACTION_DOMAIN_ID,
    COSMOS3_CLEAN_ACTION,
    COSMOS3_COND_INPUT_IDS,
    COSMOS3_CONDITIONING_FPS,
    COSMOS3_RAW_ACTION_DIM,
    COSMOS3_TRAINING_SIGMA,
    COSMOS3_UNCOND_INPUT_IDS,
    COSMOS3_VIDEO,
    classify_cosmos3_action_size,
    format_cosmos3_action_prompt,
)


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported Cosmos3 dtype={dtype_name!r}")


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _retrieve_latents(encoder_output: Any, *, sample_mode: str = "argmax") -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        latent_dist = encoder_output.latent_dist
    elif isinstance(encoder_output, tuple):
        latent_dist = encoder_output[0]
    else:
        raise TypeError(f"Unexpected VAE encoder output type: {type(encoder_output)!r}")

    if sample_mode == "argmax":
        return latent_dist.mode()
    if sample_mode == "sample":
        return latent_dist.sample()
    raise ValueError(f"Unsupported VAE latent sample_mode={sample_mode!r}.")


def _read_safetensors_index(index_path: Path) -> list[Path]:
    with index_path.open() as f:
        index = json.load(f)
    shards = sorted(set(index["weight_map"].values()))
    return [index_path.parent / shard for shard in shards]


def _load_safetensors_state_dict(path: Path) -> dict[str, Tensor]:
    if path.is_file():
        shard_paths = [path]
    elif (path / "diffusion_pytorch_model.safetensors.index.json").is_file():
        shard_paths = _read_safetensors_index(path / "diffusion_pytorch_model.safetensors.index.json")
    elif (path / "model.safetensors.index.json").is_file():
        shard_paths = _read_safetensors_index(path / "model.safetensors.index.json")
    elif (path / "diffusion_pytorch_model.safetensors").is_file():
        shard_paths = [path / "diffusion_pytorch_model.safetensors"]
    elif (path / "model.safetensors").is_file():
        shard_paths = [path / "model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors checkpoint found under {path}")

    state_dict: dict[str, Tensor] = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                state_dict[key] = f.get_tensor(key)
    return state_dict


def get_3d_mrope_ids_text_tokens(
    num_tokens: int,
    temporal_offset: int | float,
    use_float_positions: bool = False,
) -> tuple[torch.Tensor, int | float]:
    if use_float_positions:
        ids = torch.arange(num_tokens, dtype=torch.float32) + temporal_offset
    else:
        ids = torch.arange(num_tokens, dtype=torch.long) + int(temporal_offset)

    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    next_temporal_offset = temporal_offset + num_tokens
    return mrope_ids, next_temporal_offset


def get_3d_mrope_ids_vae_tokens(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    reset_spatial_indices: bool = True,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    fps_modulation_enabled = fps is not None and grid_t > 1
    effective_base_tcf = (
        base_temporal_compression_factor
        if base_temporal_compression_factor is not None
        else temporal_compression_factor
    )

    if fps_modulation_enabled:
        tps = fps / temporal_compression_factor
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        scaled_t = (frame_indices + start_frame_offset) / tps * base_tps + temporal_offset
        t_index = scaled_t.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if not reset_spatial_indices:
        spatial_offset = int(temporal_offset)
        h_index = h_index + spatial_offset
        w_index = w_index + spatial_offset

    if fps_modulation_enabled:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_temporal_offset = math.ceil(mrope_ids.max().item()) + 1
    return mrope_ids, next_temporal_offset


def _arch_invariant_rand(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    seed: int,
) -> torch.Tensor:
    random_array = np.random.RandomState(seed).standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


def _prepare_native_action_video_conditioning(
    video: Tensor,
    *,
    resolution_tier: int,
    num_frames: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, int, int]:
    if video.dtype != torch.uint8:
        raise ValueError(f"Cosmos3 action video input must be uint8, got dtype={video.dtype}.")
    if video.ndim != 4:
        raise ValueError(f"Expected Cosmos3 action video shape [C,T,H,W], got shape={tuple(video.shape)}.")

    frames = video.detach().to(device=device)
    source_h, source_w = frames.shape[-2:]
    target_h, target_w, content_h, content_w = classify_cosmos3_action_size(
        source_h,
        source_w,
        resolution_tier=resolution_tier,
    )

    if frames.shape[1] < num_frames:
        frames = torch.cat([frames, frames[:, -1:].expand(-1, num_frames - frames.shape[1], -1, -1)], dim=1)
    else:
        frames = frames[:, :num_frames]

    frames_t = frames.permute(1, 0, 2, 3).to(dtype=torch.float32)
    if content_h != source_h or content_w != source_w:
        frames_t = F.interpolate(
            frames_t,
            size=(content_h, content_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    pad_right = target_w - content_w
    pad_bottom = target_h - content_h
    if pad_right or pad_bottom:
        pad_mode = "replicate" if pad_right >= content_w or pad_bottom >= content_h else "reflect"
        frames_t = F.pad(frames_t, (0, pad_right, 0, pad_bottom), mode=pad_mode)
    frames = frames_t.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=dtype) / 127.5 - 1.0
    image_size = torch.tensor([target_h, target_w, content_h, content_w], device=device, dtype=torch.float32)
    return frames, image_size, target_h, target_w


class Cosmos3Policy(PreTrainedPolicy):
    """LeRobot policy wrapper for Cosmos3 DROID action generation."""

    config_class = Cosmos3Config
    name = "cosmos3"

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path: str | Path, *args, config=None, **kwargs):
        if config is None:
            config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
        if not isinstance(config, Cosmos3Config):
            raise TypeError(f"Expected Cosmos3Config, got {type(config)!r}.")
        config.pretrained_path = Path(pretrained_name_or_path)
        return super().from_pretrained(pretrained_name_or_path, *args, config=config, **kwargs)

    def __init__(self, config: Cosmos3Config, **kwargs):
        require_package("diffusers", extra="cosmos3")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = Cosmos3ActionModel(
            config,
            transformer=kwargs.pop("transformer", None),
            vae=kwargs.pop("vae", None),
            scheduler=kwargs.pop("scheduler", None),
        )
        self.to(config.device)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self.model.reset_generation()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.model(batch)

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        return self.model.sample_actions(batch, **kwargs)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        actions = self.sample_actions(batch, **kwargs).to(torch.float32)
        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_future_video(self, batch: dict[str, Tensor], **kwargs) -> Tensor | None:
        return self.model.predict_future_video(batch, **kwargs)

    def get_optim_params(self) -> dict:
        return self.parameters()


class Cosmos3ActionModel(nn.Module):
    """Cosmos3 action model built from public Diffusers model-level components."""

    def __init__(
        self,
        config: Cosmos3Config,
        *,
        transformer: nn.Module | None = None,
        vae: nn.Module | None = None,
        scheduler: Any | None = None,
    ):
        super().__init__()
        self.config = config
        self.transformer = transformer if transformer is not None else self._build_transformer()
        self.vae = vae if vae is not None else self._build_vae()
        self.scheduler = scheduler if scheduler is not None else self._build_scheduler()
        if self.config.freeze_vae:
            self.vae.eval().requires_grad_(False)
        self.reset_generation()

    def reset_generation(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)

    def _next_seed(self) -> int:
        if self.config.deterministic_seed:
            return int(self.config.seed)
        return int(self._rng.integers(0, 2**31))

    def _pretrained_source_path(self) -> Path | None:
        source = self.config.diffusers_model_name_or_path or self.config.base_model_name_or_path
        if source is None:
            return None
        source_path = Path(source)
        return source_path if source_path.is_dir() else None

    def _build_transformer(self) -> nn.Module:
        from diffusers import Cosmos3OmniTransformer

        source_path = self._pretrained_source_path()
        torch_dtype = _torch_dtype(self.config.dtype)
        if (
            source_path is not None
            and self.config.load_pretrained_weights
            and self.config.pretrained_path is None
            and not self._must_filter_source_sound_modules(source_path)
        ):
            return Cosmos3OmniTransformer.from_pretrained(
                source_path,
                subfolder="transformer",
                torch_dtype=torch_dtype,
                local_files_only=self.config.local_files_only,
            )

        transformer = Cosmos3OmniTransformer(**self.config.transformer_backbone_config)
        if (
            source_path is not None
            and self.config.load_pretrained_weights
            and self.config.pretrained_path is None
        ):
            self._load_diffusers_transformer_weights(transformer, source_path / "transformer")
        elif self.config.qwen3_vl_name_or_path is not None and self.config.pretrained_path is None:
            self._load_qwen3vl_understanding_weights(transformer, Path(self.config.qwen3_vl_name_or_path))

        if self.config.copy_understanding_to_generation_expert:
            self._copy_understanding_to_generation_expert(transformer)
        return transformer.to(dtype=torch_dtype)

    def _must_filter_source_sound_modules(self, source_path: Path) -> bool:
        config_path = source_path / "transformer" / "config.json"
        if not self.config.drop_sound_modules or not config_path.is_file():
            return False
        with config_path.open() as f:
            source_config = json.load(f)
        return bool(source_config.get("sound_gen"))

    def _load_diffusers_transformer_weights(self, transformer: nn.Module, transformer_path: Path) -> None:
        state_dict = _load_safetensors_state_dict(transformer_path)
        if self.config.drop_sound_modules:
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith(("audio_proj_in.", "audio_proj_out.", "audio_modality_embed"))
            }
        incompatible = transformer.load_state_dict(state_dict, strict=False)
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith(("audio_proj_in.", "audio_proj_out.", "audio_modality_embed"))
        ]
        if unexpected:
            raise RuntimeError(
                f"Unexpected Cosmos3 transformer keys after sound filtering: {unexpected[:16]}"
            )

    def _load_qwen3vl_understanding_weights(self, transformer: nn.Module, qwen_path: Path) -> None:
        if not qwen_path.exists():
            return
        source_state = _load_safetensors_state_dict(qwen_path)
        remapped: dict[str, Tensor] = {}
        target_shapes = {key: value.shape for key, value in transformer.state_dict().items()}
        for key, value in source_state.items():
            new_key = self._remap_qwen3vl_key(key)
            if new_key is not None and new_key in target_shapes and target_shapes[new_key] == value.shape:
                remapped[new_key] = value
        transformer.load_state_dict(remapped, strict=False)

    def _remap_qwen3vl_key(self, key: str) -> str | None:
        prefixes = ("model.language_model.", "language_model.", "model.")
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.startswith("visual.") or key.startswith("vision_model."):
            return None
        replacements = [
            (".self_attn.q_proj.", ".self_attn.to_q."),
            (".self_attn.k_proj.", ".self_attn.to_k."),
            (".self_attn.v_proj.", ".self_attn.to_v."),
            (".self_attn.o_proj.", ".self_attn.to_out."),
            (".self_attn.q_norm.", ".self_attn.norm_q."),
            (".self_attn.k_norm.", ".self_attn.norm_k."),
        ]
        for old, new in replacements:
            if old in key:
                return key.replace(old, new)
        return key

    def _copy_understanding_to_generation_expert(self, transformer: nn.Module) -> None:
        state = transformer.state_dict()
        replacements = [
            (".self_attn.add_q_proj.", ".self_attn.to_q."),
            (".self_attn.add_k_proj.", ".self_attn.to_k."),
            (".self_attn.add_v_proj.", ".self_attn.to_v."),
            (".self_attn.to_add_out.", ".self_attn.to_out."),
            (".self_attn.norm_added_q.", ".self_attn.norm_q."),
            (".self_attn.norm_added_k.", ".self_attn.norm_k."),
            (".mlp_moe_gen.", ".mlp."),
            (".input_layernorm_moe_gen.", ".input_layernorm."),
            (".post_attention_layernorm_moe_gen.", ".post_attention_layernorm."),
        ]
        with torch.no_grad():
            for target_key, target_tensor in state.items():
                source_key = None
                for old, new in replacements:
                    if old in target_key:
                        source_key = target_key.replace(old, new)
                        break
                if target_key == "norm_moe_gen.weight":
                    source_key = "norm.weight"
                if (
                    source_key is not None
                    and source_key in state
                    and state[source_key].shape == target_tensor.shape
                ):
                    target_tensor.copy_(state[source_key])

    def _build_vae(self) -> nn.Module:
        from diffusers import AutoencoderKLWan

        source_path = self._pretrained_source_path()
        torch_dtype = _torch_dtype(self.config.dtype)
        if (
            source_path is not None
            and self.config.load_pretrained_weights
            and self.config.pretrained_path is None
        ):
            return AutoencoderKLWan.from_pretrained(
                source_path,
                subfolder="vae",
                torch_dtype=torch_dtype,
                local_files_only=self.config.local_files_only,
            )
        if self.config.vae_config is not None:
            return AutoencoderKLWan(**self.config.vae_config).to(dtype=torch_dtype)
        return AutoencoderKLWan().to(dtype=torch_dtype)

    def _build_scheduler(self) -> Any:
        from diffusers import UniPCMultistepScheduler

        source_path = self._pretrained_source_path()
        if (
            source_path is not None
            and self.config.load_pretrained_weights
            and self.config.pretrained_path is None
        ):
            return UniPCMultistepScheduler.from_pretrained(
                source_path,
                subfolder="scheduler",
                local_files_only=self.config.local_files_only,
            )
        if self.config.unipc_scheduler_config is not None:
            return UniPCMultistepScheduler.from_config(self.config.unipc_scheduler_config)
        return UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            use_karras_sigmas=False,
            flow_shift=float(self.config.shift),
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        required = [
            COSMOS3_VIDEO,
            COSMOS3_ACTION_CONDITION,
            COSMOS3_ACTION_CONDITION_MASK,
            COSMOS3_ACTION_DOMAIN_ID,
            COSMOS3_CONDITIONING_FPS,
            COSMOS3_RAW_ACTION_DIM,
            COSMOS3_COND_INPUT_IDS,
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Cosmos3 training batch is missing required model inputs: {missing}")

        videos = batch[COSMOS3_VIDEO]
        if videos.ndim == 4:
            videos = videos.unsqueeze(0)
        action_conditions = batch[COSMOS3_ACTION_CONDITION]
        if action_conditions.ndim == 2:
            action_conditions = action_conditions.unsqueeze(0)
        action_condition_masks = batch[COSMOS3_ACTION_CONDITION_MASK]
        if action_condition_masks.ndim == 2:
            action_condition_masks = action_condition_masks.unsqueeze(0)

        batch_size = videos.shape[0]
        clean_actions = self._prepare_clean_action_tokens(batch, action_conditions)
        sigmas = self._get_training_sigmas(batch, batch_size=batch_size)

        losses = []
        vision_losses = []
        action_losses = []
        for batch_idx in range(batch_size):
            sample_losses = self._compute_single_training_loss(
                cond_input_ids=self._get_ids_for_batch(batch[COSMOS3_COND_INPUT_IDS], batch_idx),
                video=videos[batch_idx],
                clean_action=clean_actions[batch_idx],
                action_condition_mask=action_condition_masks[batch_idx],
                domain_id=batch[COSMOS3_ACTION_DOMAIN_ID][batch_idx],
                conditioning_fps=batch[COSMOS3_CONDITIONING_FPS][batch_idx],
                raw_action_dim=batch[COSMOS3_RAW_ACTION_DIM][batch_idx],
                sigma=sigmas[batch_idx],
            )
            losses.append(sample_losses["loss"])
            vision_losses.append(sample_losses["flow_matching_loss_vision"])
            action_losses.append(sample_losses["flow_matching_loss_action"])

        loss = torch.stack(losses).mean()
        vision_loss = torch.stack(vision_losses).mean()
        action_loss = torch.stack(action_losses).mean()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "flow_matching_loss_vision": float(vision_loss.detach().cpu()),
            "flow_matching_loss_action": float(action_loss.detach().cpu()),
        }
        return loss, metrics

    def _get_ids_for_batch(self, input_ids: Any, batch_idx: int) -> Tensor:
        if isinstance(input_ids, Tensor):
            if input_ids.ndim == 1:
                return input_ids
            return input_ids[batch_idx]
        return input_ids[batch_idx]

    def _prepare_clean_action_tokens(self, batch: dict[str, Tensor], action_conditions: Tensor) -> Tensor:
        if COSMOS3_CLEAN_ACTION in batch:
            clean_action = batch[COSMOS3_CLEAN_ACTION]
            if clean_action.ndim == 2:
                clean_action = clean_action.unsqueeze(0)
            if clean_action.shape[-1] < self.config.max_action_dim:
                clean_action = F.pad(clean_action, (0, self.config.max_action_dim - clean_action.shape[-1]))
            return clean_action[:, :, : self.config.max_action_dim].to(dtype=torch.float32)

        if ACTION not in batch:
            raise ValueError(
                f"Cosmos3 training requires {COSMOS3_CLEAN_ACTION!r} or {ACTION!r} action labels."
            )
        action = batch[ACTION]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Cosmos3 action labels must have shape [B,T,D], got {tuple(action.shape)}.")

        batch_size, action_len, _ = action_conditions.shape
        clean_action = torch.zeros(
            batch_size,
            action_len,
            self.config.max_action_dim,
            dtype=torch.float32,
            device=action_conditions.device,
        )
        clean_action[:, :, : self.config.raw_action_dim] = action_conditions[
            :, :, : self.config.raw_action_dim
        ].to(dtype=torch.float32)
        action = (
            action[:, : self.config.chunk_size, : self.config.raw_action_dim].to(dtype=torch.float32).clone()
        )
        if self.config.invert_gripper:
            action[:, :, -1] = 1.0 - action[:, :, -1]
        future_start = int(self.config.use_state)
        clean_action[:, future_start : future_start + action.shape[1], : self.config.raw_action_dim] = action
        return clean_action

    def _get_training_sigmas(self, batch: dict[str, Tensor], *, batch_size: int) -> Tensor:
        device = _module_device(self.transformer)
        if COSMOS3_TRAINING_SIGMA in batch:
            sigmas = torch.as_tensor(batch[COSMOS3_TRAINING_SIGMA], device=device, dtype=torch.float32)
            if sigmas.ndim == 0:
                sigmas = sigmas.expand(batch_size, 1)
            elif sigmas.ndim == 1:
                sigmas = sigmas.view(batch_size, 1)
            elif sigmas.ndim != 2:
                raise ValueError(
                    f"{COSMOS3_TRAINING_SIGMA} must be scalar, [B], or [B,1], got {sigmas.shape}."
                )
            return sigmas

        if self.config.train_time_video_distribution == "uniform":
            t_raw = torch.rand((batch_size, 1), device=device, dtype=torch.float32)
        elif self.config.train_time_video_distribution == "logitnormal":
            t_raw = torch.sigmoid(torch.randn((batch_size, 1), device=device, dtype=torch.float32))
        elif self.config.train_time_video_distribution == "waver":
            u = torch.rand((batch_size, 1), device=device, dtype=torch.float32)
            t_raw = 1.0 - u - 1.29 * (torch.cos(torch.pi / 2.0 * u) ** 2 - 1 + u)
        else:
            raise ValueError(
                f"Unsupported Cosmos3 train_time_video_distribution={self.config.train_time_video_distribution!r}."
            )

        tau = 1.0 - t_raw
        shift = float(self.config.shift)
        return shift * tau / (1.0 + (shift - 1.0) * tau)

    def _compute_single_training_loss(
        self,
        *,
        cond_input_ids: Tensor,
        video: Tensor,
        clean_action: Tensor,
        action_condition_mask: Tensor,
        domain_id: Tensor,
        conditioning_fps: Tensor,
        raw_action_dim: Tensor,
        sigma: Tensor,
    ) -> dict[str, Tensor]:
        device = _module_device(self.transformer)
        dtype = _module_dtype(self.transformer)
        raw_action_dim_int = int(raw_action_dim.item())

        vision_tensor, action_image_size, _height, _width = _prepare_native_action_video_conditioning(
            video,
            resolution_tier=self.config.resolution_tier,
            num_frames=self.config.chunk_size + 1,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            clean_vision = self._encode_video(vision_tensor).contiguous().float()
            clean_vision = self._remove_action_video_padding_from_latent(clean_vision, action_image_size)

        vision_condition_mask = torch.zeros(
            (clean_vision.shape[2], 1, 1),
            device=device,
            dtype=torch.float32,
        )
        vision_condition_mask[0, 0, 0] = 1.0
        sigma = sigma.to(device=device, dtype=torch.float32).view(1, 1, 1, 1, 1)
        vision_noisy_mask = 1.0 - vision_condition_mask.view(1, 1, clean_vision.shape[2], 1, 1)
        vision_sigma = sigma * vision_noisy_mask
        epsilon_vision = torch.randn(clean_vision.shape, device=device, dtype=torch.float32)
        noised_vision = epsilon_vision * vision_sigma + clean_vision * (1.0 - vision_sigma)
        target_vision = epsilon_vision - clean_vision

        action_dim = int(self.transformer.config.action_dim)
        clean_action = clean_action.to(device=device, dtype=torch.float32)
        if clean_action.shape[-1] < action_dim:
            clean_action = F.pad(clean_action, (0, action_dim - clean_action.shape[-1]))
        clean_action = clean_action[:, :action_dim]
        action_condition_mask = action_condition_mask.to(device=device, dtype=torch.float32)
        sigma_action = sigma.view(1, 1) * (1.0 - action_condition_mask)
        epsilon_action = torch.randn(clean_action.shape, device=device, dtype=torch.float32)
        noised_action = epsilon_action * sigma_action + clean_action * (1.0 - sigma_action)
        target_action = epsilon_action - clean_action
        noised_action[:, raw_action_dim_int:] = 0

        text_segment = self._prepare_text_segment(cond_input_ids, device=device)
        packed_static = self._pack_static_segments(
            text_segment=text_segment,
            latents=noised_vision.to(dtype=dtype),
            action_latents=noised_action.to(dtype=dtype),
            vision_condition_indexes=[0],
            fps_vision=float(conditioning_fps.item()),
            action_start_frame_offset=0 if self.config.use_state else 1,
        )

        max_timestep = float(getattr(self.scheduler.config, "num_train_timesteps", 1000))
        timestep = sigma.flatten()[0] * max_timestep
        vision_timesteps = torch.full(
            (packed_static["num_noisy_vision_tokens"],),
            float(timestep.item()),
            device=device,
            dtype=torch.float32,
        )
        action_timesteps = torch.full(
            (packed_static["num_noisy_action_tokens"],),
            float(timestep.item()),
            device=device,
            dtype=torch.float32,
        )
        action_domain_id = domain_id.to(device=device, dtype=torch.long).view(1)

        pred_vision, pred_action = self._predict_velocity(
            packed_static=packed_static,
            vision_tokens=noised_vision.to(dtype=dtype),
            action_tokens=noised_action.to(dtype=dtype),
            vision_timesteps=vision_timesteps,
            action_timesteps=action_timesteps,
            action_domain_id=action_domain_id,
            vision_condition_mask=vision_condition_mask.to(dtype=dtype),
            action_condition_mask=action_condition_mask.to(dtype=dtype),
            raw_action_dim=raw_action_dim_int,
        )

        target_vision = target_vision[0].to(device=pred_vision.device, dtype=torch.float32)
        vision_noisy_mask = vision_noisy_mask[0, 0].to(device=pred_vision.device, dtype=torch.float32)
        vision_loss = self._masked_flow_matching_mse(
            pred_vision.to(dtype=torch.float32),
            target_vision,
            vision_noisy_mask,
        )

        action_noisy_mask = (1.0 - action_condition_mask).to(device=pred_action.device, dtype=torch.float32)
        action_loss = self._masked_flow_matching_mse(
            pred_action[:, :raw_action_dim_int].to(dtype=torch.float32),
            target_action[:, :raw_action_dim_int].to(device=pred_action.device, dtype=torch.float32),
            action_noisy_mask,
        )
        total_loss = (
            self.config.video_loss_weight * vision_loss + self.config.action_loss_weight * action_loss
        )
        return {
            "loss": total_loss,
            "flow_matching_loss_vision": vision_loss,
            "flow_matching_loss_action": action_loss,
        }

    def _masked_flow_matching_mse(self, pred: Tensor, target: Tensor, noisy_mask: Tensor) -> Tensor:
        noisy_mask = noisy_mask.to(device=pred.device, dtype=pred.dtype)
        sqerr = (pred - target) ** 2 * noisy_mask
        if not self.config.normalize_loss_by_active:
            return sqerr.mean()

        active_count = noisy_mask.expand_as(pred).sum()
        return sqerr.sum() / active_count.clamp_min(1.0)

    def _encode_video(self, video: Tensor) -> Tensor:
        vae_dtype = _module_dtype(self.vae)
        encoded = _retrieve_latents(self.vae.encode(video.to(vae_dtype)), sample_mode="argmax")
        mean = torch.tensor(self.vae.config.latents_mean, device=encoded.device, dtype=encoded.dtype)
        inv_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=encoded.device, dtype=encoded.dtype)
        return ((encoded - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)).to(video.dtype)

    def _remove_action_video_padding_from_latent(self, latents: Tensor, image_size: Tensor) -> Tensor:
        spatial_factor = int(getattr(self.vae.config, "scale_factor_spatial", 16))
        content_h = int(image_size[2].item())
        content_w = int(image_size[3].item())
        content_h_latent = max(content_h // spatial_factor, 1)
        content_w_latent = max(content_w // spatial_factor, 1)
        return latents[:, :, :, :content_h_latent, :content_w_latent].contiguous()

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        *,
        seed: int | list[int] | tuple[int, ...] | Tensor | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> Tensor:
        required = [
            COSMOS3_VIDEO,
            COSMOS3_ACTION_CONDITION,
            COSMOS3_ACTION_CONDITION_MASK,
            COSMOS3_ACTION_DOMAIN_ID,
            COSMOS3_CONDITIONING_FPS,
            COSMOS3_RAW_ACTION_DIM,
            COSMOS3_COND_INPUT_IDS,
            COSMOS3_UNCOND_INPUT_IDS,
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Cosmos3 batch is missing required model inputs: {missing}")

        videos = batch[COSMOS3_VIDEO]
        if videos.ndim == 4:
            videos = videos.unsqueeze(0)
        batch_size = videos.shape[0]
        seeds = self._normalise_sample_seeds(seed, batch_size)
        actions = []
        for batch_idx, sample_seed in enumerate(seeds):
            action = self._sample_single(
                cond_input_ids=self._get_ids_for_batch(batch[COSMOS3_COND_INPUT_IDS], batch_idx),
                uncond_input_ids=self._get_ids_for_batch(batch[COSMOS3_UNCOND_INPUT_IDS], batch_idx),
                video=videos[batch_idx],
                action_condition=batch[COSMOS3_ACTION_CONDITION][batch_idx],
                action_condition_mask=batch[COSMOS3_ACTION_CONDITION_MASK][batch_idx],
                domain_id=batch[COSMOS3_ACTION_DOMAIN_ID][batch_idx],
                conditioning_fps=batch[COSMOS3_CONDITIONING_FPS][batch_idx],
                raw_action_dim=batch[COSMOS3_RAW_ACTION_DIM][batch_idx],
                seed=sample_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            actions.append(action)
        return torch.stack(actions, dim=0)

    @torch.no_grad()
    def predict_future_video(self, batch: dict[str, Tensor], **kwargs) -> Tensor | None:
        if not self.config.generate_video:
            return None
        raise NotImplementedError(
            "Cosmos3 future-video decoding is reserved for a follow-up integration step."
        )

    def _normalise_sample_seeds(self, seed: Any, batch_size: int) -> list[int]:
        if seed is None:
            return [self._next_seed() for _ in range(batch_size)]
        if isinstance(seed, Tensor):
            seed = seed.detach().cpu().flatten().tolist()
        if isinstance(seed, (list, tuple)):
            if len(seed) != batch_size:
                raise ValueError(f"Expected {batch_size} Cosmos3 seeds, got {len(seed)}.")
            return [int(item) for item in seed]
        return [int(seed)] * batch_size

    def _sample_single(
        self,
        *,
        cond_input_ids: Tensor,
        uncond_input_ids: Tensor,
        video: Tensor,
        action_condition: Tensor,
        action_condition_mask: Tensor,
        domain_id: Tensor,
        conditioning_fps: Tensor,
        raw_action_dim: Tensor,
        seed: int,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> Tensor:
        device = _module_device(self.transformer)
        dtype = _module_dtype(self.transformer)
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale if guidance_scale is not None else self.config.guidance_scale

        vision_tensor, action_image_size, _height, _width = _prepare_native_action_video_conditioning(
            video,
            resolution_tier=self.config.resolution_tier,
            num_frames=self.config.chunk_size + 1,
            device=device,
            dtype=dtype,
        )
        x0_tokens_vision = self._encode_video(vision_tensor).contiguous().float()
        x0_tokens_vision = self._remove_action_video_padding_from_latent(x0_tokens_vision, action_image_size)

        vision_condition_mask = torch.zeros(
            (x0_tokens_vision.shape[2], 1, 1),
            device=device,
            dtype=dtype,
        )
        vision_condition_mask[0, 0, 0] = 1.0
        pure_noise = _arch_invariant_rand(
            tuple(x0_tokens_vision.shape),
            dtype=dtype,
            device=device,
            seed=seed,
        )
        latents = (
            vision_condition_mask * x0_tokens_vision.to(dtype=dtype)
            + (1.0 - vision_condition_mask) * pure_noise
        )

        raw_action_dim_int = int(raw_action_dim.item())
        action_dim = int(self.transformer.config.action_dim)
        action_condition = action_condition.to(device=device, dtype=dtype)
        if action_condition.shape[-1] < action_dim:
            action_condition = F.pad(action_condition, (0, action_dim - action_condition.shape[-1]))
        action_condition = action_condition[:, :action_dim]
        action_condition_mask = action_condition_mask.to(device=device, dtype=dtype)
        pure_action_noise = _arch_invariant_rand(
            tuple(action_condition.shape),
            dtype=dtype,
            device=device,
            seed=seed,
        )
        action_latents = (
            action_condition_mask * action_condition + (1.0 - action_condition_mask) * pure_action_noise
        )
        action_latents[:, raw_action_dim_int:] = 0
        action_domain_id = domain_id.to(device=device, dtype=torch.long).view(1)

        vision_condition_indexes = [0]
        cond_packed_static = self._pack_static_segments(
            text_segment=self._prepare_text_segment(cond_input_ids, device=device),
            latents=latents,
            action_latents=action_latents,
            vision_condition_indexes=vision_condition_indexes,
            fps_vision=float(conditioning_fps.item()),
            action_start_frame_offset=0 if self.config.use_state else 1,
        )
        uncond_packed_static = self._pack_static_segments(
            text_segment=self._prepare_text_segment(uncond_input_ids, device=device),
            latents=latents,
            action_latents=action_latents,
            vision_condition_indexes=vision_condition_indexes,
            fps_vision=float(conditioning_fps.item()),
            action_start_frame_offset=0 if self.config.use_state else 1,
        )

        scheduler = self.scheduler
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = scheduler.timesteps

        vision_shape = tuple(latents.shape)
        action_shape = tuple(action_latents.shape)
        vision_size = latents.numel()

        def pack_latents(vision: Tensor, action: Tensor) -> Tensor:
            return torch.cat([vision.reshape(-1), action.reshape(-1)], dim=0)

        def unpack_latents(flat_latents: Tensor) -> tuple[Tensor, Tensor]:
            vision = flat_latents[:vision_size].reshape(vision_shape)
            action = flat_latents[vision_size:].reshape(action_shape)
            return vision, action

        flat_latents = pack_latents(latents, action_latents)
        num_noisy_vision_tokens = cond_packed_static["num_noisy_vision_tokens"]
        action_noisy_len = cond_packed_static["num_noisy_action_tokens"]
        for timestep_tensor in timesteps:
            timestep = float(timestep_tensor.item())
            latents, action_latents = unpack_latents(flat_latents)
            vision_tokens = latents.to(device=device, dtype=dtype)
            action_tokens = action_latents.to(device=device, dtype=dtype)
            vision_timesteps = torch.full((num_noisy_vision_tokens,), timestep, device=device)
            action_timesteps = torch.full((action_noisy_len,), timestep, device=device)

            cond_v_vision, cond_v_action = self._predict_velocity(
                packed_static=cond_packed_static,
                vision_tokens=vision_tokens,
                action_tokens=action_tokens,
                vision_timesteps=vision_timesteps,
                action_timesteps=action_timesteps,
                action_domain_id=action_domain_id,
                vision_condition_mask=vision_condition_mask,
                action_condition_mask=action_condition_mask,
                raw_action_dim=raw_action_dim_int,
            )
            if guidance_scale != 1.0:
                uncond_v_vision, uncond_v_action = self._predict_velocity(
                    packed_static=uncond_packed_static,
                    vision_tokens=vision_tokens,
                    action_tokens=action_tokens,
                    vision_timesteps=vision_timesteps,
                    action_timesteps=action_timesteps,
                    action_domain_id=action_domain_id,
                    vision_condition_mask=vision_condition_mask,
                    action_condition_mask=action_condition_mask,
                    raw_action_dim=raw_action_dim_int,
                )
                velocity_vision = uncond_v_vision + guidance_scale * (cond_v_vision - uncond_v_vision)
                velocity_action = uncond_v_action + guidance_scale * (cond_v_action - uncond_v_action)
            else:
                velocity_vision = cond_v_vision
                velocity_action = cond_v_action

            velocity = pack_latents(velocity_vision, velocity_action)
            flat_latents = scheduler.step(
                velocity.unsqueeze(0), timestep_tensor, flat_latents.unsqueeze(0), return_dict=False
            )[0].squeeze(0)
            latents, action_latents = unpack_latents(flat_latents)
            action_latents[:, raw_action_dim_int:] = 0
            flat_latents = pack_latents(latents, action_latents)

        actions = action_latents[:, :raw_action_dim_int].detach().cpu().to(torch.float32)
        if self.config.history_length:
            actions = actions[self.config.history_length :]
        if self.config.invert_gripper:
            actions[:, -1] = 1.0 - actions[:, -1]
        return actions[: self.config.chunk_size]

    def _prepare_text_segment(self, input_ids: Tensor, device: torch.device | str) -> dict[str, Any]:
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
        config = self.transformer.config
        und_len = int(input_ids.numel())
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=bool(config.enable_fps_modulation),
        )
        return {
            "input_ids": input_ids,
            "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
            "und_len": und_len,
            "text_mrope_ids": text_mrope_ids.to(device),
            "vision_start_temporal_offset": next_mrope_offset
            + config.unified_3d_mrope_temporal_modality_margin,
        }

    def _pack_static_segments(
        self,
        *,
        text_segment: dict[str, Any],
        latents: Tensor,
        action_latents: Tensor,
        vision_condition_indexes: list[int],
        fps_vision: float,
        action_start_frame_offset: int,
    ) -> dict[str, Any]:
        device = latents.device
        vision_segment = self._prepare_vision_segment(
            input_vision_tokens=latents,
            has_image_condition=True,
            mrope_offset=text_segment["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=text_segment["und_len"],
            device=device,
            condition_frame_indexes=vision_condition_indexes,
        )
        action_segment = self._prepare_action_segment(
            input_action_tokens=action_latents,
            condition_frame_indexes=[0] if self.config.use_state else [],
            mrope_offset=text_segment["vision_start_temporal_offset"],
            action_fps=fps_vision,
            curr=text_segment["und_len"] + vision_segment["num_vision_tokens"],
            device=device,
            start_frame_offset=action_start_frame_offset,
        )
        position_ids = torch.cat(
            [
                text_segment["text_mrope_ids"],
                vision_segment["vision_mrope_ids"],
                action_segment["action_mrope_ids"],
            ],
            dim=1,
        )
        return {
            **text_segment,
            **vision_segment,
            **action_segment,
            "position_ids": position_ids,
            "sequence_length": text_segment["und_len"]
            + vision_segment["num_vision_tokens"]
            + action_segment["action_len"],
        }

    def _prepare_vision_segment(
        self,
        *,
        input_vision_tokens: Tensor,
        has_image_condition: bool,
        mrope_offset: int | float,
        vision_fps: float | None,
        curr: int,
        device: torch.device | str,
        condition_frame_indexes: list[int] | None = None,
    ) -> dict[str, Any]:
        config = self.transformer.config
        latent_patch_size = int(config.latent_patch_size)
        _, _, latent_t, latent_h, latent_w = input_vision_tokens.shape
        patch_h = math.ceil(latent_h / latent_patch_size)
        patch_w = math.ceil(latent_w / latent_patch_size)
        num_vision_tokens = latent_t * patch_h * patch_w

        if condition_frame_indexes is None:
            condition_frame_indexes = [0] if has_image_condition else []
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < latent_t}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(latent_t) if idx not in cond_frames], device=device, dtype=torch.long
        )

        frame_token_stride = patch_h * patch_w
        mse_loss_indexes: list[int] = []
        for frame_idx in noisy_frame_indexes.tolist():
            frame_start = curr + frame_idx * frame_token_stride
            mse_loss_indexes.extend(range(frame_start, frame_start + frame_token_stride))

        effective_fps = vision_fps if config.enable_fps_modulation else None
        temporal_compression_factor = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=temporal_compression_factor,
        )

        return {
            "vision_token_shapes": [(latent_t, patch_h, patch_w)],
            "vision_sequence_indexes": torch.arange(
                curr, curr + num_vision_tokens, dtype=torch.long, device=device
            ),
            "vision_mse_loss_indexes": torch.tensor(mse_loss_indexes, dtype=torch.long, device=device),
            "vision_noisy_frame_indexes": [noisy_frame_indexes],
            "vision_mrope_ids": vision_mrope_ids.to(device),
            "num_vision_tokens": num_vision_tokens,
            "num_noisy_vision_tokens": len(noisy_frame_indexes) * frame_token_stride,
        }

    def _prepare_action_segment(
        self,
        *,
        input_action_tokens: Tensor,
        condition_frame_indexes: list[int],
        mrope_offset: int | float,
        action_fps: float | None,
        curr: int,
        device: torch.device | str,
        start_frame_offset: int,
    ) -> dict[str, Any]:
        config = self.transformer.config
        action_len = input_action_tokens.shape[0]
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < action_len}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(action_len) if idx not in cond_frames], device=device, dtype=torch.long
        )

        effective_fps = action_fps if config.enable_fps_modulation else None
        base_tcf = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        action_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=action_len,
            grid_h=1,
            grid_w=1,
            temporal_offset=mrope_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
            fps=effective_fps,
            base_fps=float(config.base_fps),
            temporal_compression_factor=1,
            base_temporal_compression_factor=base_tcf,
            start_frame_offset=start_frame_offset,
        )
        sequence_indexes = torch.arange(curr, curr + action_len, dtype=torch.long, device=device)
        return {
            "action_token_shapes": [(action_len, 1, 1)],
            "action_sequence_indexes": sequence_indexes,
            "action_mse_loss_indexes": sequence_indexes[noisy_frame_indexes],
            "action_noisy_frame_indexes": [noisy_frame_indexes],
            "action_mrope_ids": action_mrope_ids.to(device),
            "action_len": action_len,
            "num_noisy_action_tokens": len(noisy_frame_indexes),
        }

    def _predict_velocity(
        self,
        *,
        packed_static: dict[str, Any],
        vision_tokens: Tensor,
        action_tokens: Tensor,
        vision_timesteps: Tensor,
        action_timesteps: Tensor,
        action_domain_id: Tensor,
        vision_condition_mask: Tensor,
        action_condition_mask: Tensor,
        raw_action_dim: int,
    ) -> tuple[Tensor, Tensor]:
        preds_vision, _preds_sound, preds_action = self.transformer(
            input_ids=packed_static["input_ids"],
            text_indexes=packed_static["text_indexes"],
            position_ids=packed_static["position_ids"],
            und_len=packed_static["und_len"],
            sequence_length=packed_static["sequence_length"],
            vision_tokens=[vision_tokens],
            vision_token_shapes=packed_static["vision_token_shapes"],
            vision_sequence_indexes=packed_static["vision_sequence_indexes"],
            vision_mse_loss_indexes=packed_static["vision_mse_loss_indexes"],
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=packed_static["vision_noisy_frame_indexes"],
            action_tokens=[action_tokens],
            action_token_shapes=packed_static["action_token_shapes"],
            action_sequence_indexes=packed_static["action_sequence_indexes"],
            action_mse_loss_indexes=packed_static["action_mse_loss_indexes"],
            action_timesteps=action_timesteps,
            action_noisy_frame_indexes=packed_static["action_noisy_frame_indexes"],
            action_domain_ids=[action_domain_id],
        )
        if preds_action is None:
            raise RuntimeError("Cosmos3 transformer did not return action velocity predictions.")

        pred_vision = preds_vision[0] * (1.0 - vision_condition_mask).view(1, 1, -1, 1, 1)
        pred_action = preds_action[0] * (1.0 - action_condition_mask)
        pred_action[:, raw_action_dim:] = 0
        return pred_vision.squeeze(0), pred_action

    def _format_native_action_prompt(
        self, prompt: str, *, num_frames: int, height: int, width: int, fps: float
    ) -> str:
        return format_cosmos3_action_prompt(
            prompt,
            viewpoint=self.config.viewpoint,
            additional_view_description=self.config.additional_view_description,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )
