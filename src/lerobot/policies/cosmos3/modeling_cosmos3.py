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

import math
import sys
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms.functional as transforms_functional
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode

from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from .configuration_cosmos3 import Cosmos3Config
from .processor_cosmos3 import (
    COSMOS3_ACTION_CONDITION,
    COSMOS3_ACTION_CONDITION_MASK,
    COSMOS3_ACTION_DOMAIN_ID,
    COSMOS3_COMPOSED_IMAGE,
    COSMOS3_CONDITIONING_FPS,
    COSMOS3_PROMPT,
    COSMOS3_RAW_ACTION_DIM,
    COSMOS3_VIDEO,
)

if TYPE_CHECKING:
    from diffusers import Cosmos3OmniPipeline


_COSMOS3_VAE_ENCODE_CHUNK_FRAMES = {"256": 68, "480": 24, "720": 12}
_COSMOS3_VAE_ENCODE_EXACT_DURATIONS = {17}
_COSMOS3_RESOLUTION_768_SHAPES = {
    (1024, 1024),
    (1184, 880),
    (880, 1184),
    (1360, 768),
    (768, 1360),
}


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported Cosmos3 dtype={dtype_name!r}")


def _get_3d_mrope_ids_action(
    action_len: int,
    temporal_offset: int | float,
    *,
    action_fps: float | None,
    base_fps: float,
    base_temporal_compression_factor: int,
    start_frame_offset: int,
    reset_spatial_indices: bool,
) -> torch.Tensor:
    if action_fps is not None:
        base_tps = base_fps / base_temporal_compression_factor
        frame_indices = torch.arange(action_len, dtype=torch.float32)
        t_index = (frame_indices + start_frame_offset) / action_fps * base_tps + temporal_offset
        h_index = torch.zeros(action_len, dtype=torch.float32)
        w_index = torch.zeros(action_len, dtype=torch.float32)
        return torch.stack([t_index, h_index, w_index], dim=0)

    t_index = torch.arange(action_len, dtype=torch.long) + int(temporal_offset) + start_frame_offset
    h_index = torch.zeros(action_len, dtype=torch.long)
    w_index = torch.zeros(action_len, dtype=torch.long)
    if not reset_spatial_indices:
        h_index = h_index + int(temporal_offset)
        w_index = w_index + int(temporal_offset)
    return torch.stack([t_index, h_index, w_index], dim=0)


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
    from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _ACTION_RESOLUTION_BINS
    from diffusers.video_processor import VideoProcessor

    if video.dtype != torch.uint8:
        raise ValueError(f"Cosmos3 native video input must be uint8, got dtype={video.dtype}.")
    if video.ndim != 4:
        raise ValueError(f"Expected Cosmos3 native video shape [C,T,H,W], got shape={tuple(video.shape)}.")

    frames = video.detach().cpu()
    source_h, source_w = frames.shape[-2:]
    resolution_key = str(resolution_tier)
    if resolution_key not in _ACTION_RESOLUTION_BINS:
        raise ValueError(
            f"Unsupported action resolution_tier={resolution_tier!r}; "
            f"expected one of {sorted(int(k) for k in _ACTION_RESOLUTION_BINS)}."
        )
    target_h, target_w = VideoProcessor.classify_height_width_bin(
        source_h, source_w, ratios=_ACTION_RESOLUTION_BINS[resolution_key]
    )

    if frames.shape[1] < num_frames:
        frames = torch.cat([frames, frames[:, -1:].expand(-1, num_frames - frames.shape[1], -1, -1)], dim=1)
    else:
        frames = frames[:, :num_frames]

    _, _, frame_h, frame_w = frames.shape
    scale = min(target_w / frame_w, target_h / frame_h, 1.0)
    content_h = max(1, int(scale * frame_h + 0.5))
    content_w = max(1, int(scale * frame_w + 0.5))

    if content_h != frame_h or content_w != frame_w:
        frames = transforms_functional.resize(
            frames,
            size=[content_h, content_w],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    pad_right = target_w - content_w
    pad_bottom = target_h - content_h
    if pad_right or pad_bottom:
        pad_mode = "replicate" if pad_right >= content_w or pad_bottom >= content_h else "reflect"
        frames = F.pad(frames, (0, pad_right, 0, pad_bottom), mode=pad_mode)

    image_size = torch.tensor([target_h, target_w, content_h, content_w], device=device, dtype=torch.float32)
    frames = frames.unsqueeze(0).to(device=device, dtype=dtype) / 127.5 - 1.0
    return frames, image_size, target_h, target_w


def _get_vision_data_resolution(spatial_shape: tuple[int, int]) -> str:
    if spatial_shape in _COSMOS3_RESOLUTION_768_SHAPES:
        return "768"
    min_dim = min(spatial_shape)
    if min_dim <= 256:
        return "256"
    if min_dim <= 640:
        return "480"
    if min_dim <= 960:
        return "720"
    raise ValueError(f"Unsupported Cosmos3 VAE spatial resolution: {spatial_shape}")


def _patch_wan_vae_rms_norm() -> None:
    from diffusers.models.autoencoders import autoencoder_kl_wan as wan_mod

    if getattr(wan_mod.WanRMS_norm, "_lerobot_cosmos3_native_forward", False):
        return

    def native_forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

    wan_mod.WanRMS_norm.forward = native_forward
    wan_mod.WanRMS_norm._lerobot_cosmos3_native_forward = True


def _patch_cosmos_framework_transformers55_compat() -> None:
    """Patch Cosmos Framework's vendored Qwen3-VL code for Transformers 5.5."""
    try:
        from transformers import modeling_rope_utils
    except ImportError:
        rope_init_functions = None
    else:
        rope_init_functions = modeling_rope_utils.ROPE_INIT_FUNCTIONS

    if rope_init_functions is not None and "default" not in rope_init_functions:
        rope_init_functions["default"] = rope_init_functions["proportional"]

    try:
        from cosmos_framework.model.vfm.vlm.qwen3_vl import configuration_qwen3_vl
    except (ImportError, ModuleNotFoundError):
        return

    def patch_text_config(config_cls: type) -> None:
        if getattr(config_cls, "_lerobot_cosmos3_transformers55_init", False):
            return
        original_init = config_cls.__init__

        def patched_init(self, *args, pad_token_id=None, **kwargs):
            kwargs.setdefault("pad_token_id", pad_token_id)
            original_init(self, *args, **kwargs)
            if not hasattr(self, "pad_token_id"):
                self.pad_token_id = pad_token_id

        config_cls.__init__ = patched_init
        config_cls._lerobot_cosmos3_transformers55_init = True

    patch_text_config(configuration_qwen3_vl.Qwen3VLTextConfig)
    try:
        from cosmos_framework.model.vfm.vlm.qwen3_vl_moe import configuration_qwen3_vl_moe

        patch_text_config(configuration_qwen3_vl_moe.Qwen3VLMoeTextConfig)
    except (ImportError, ModuleNotFoundError):
        pass


def _install_native_varlen_attention(transformer: nn.Module) -> int:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
    except (ImportError, OSError):
        return 0

    from diffusers.models.transformers import transformer_cosmos3 as cosmos3_mod

    def flash_varlen_attention(query: Tensor, key: Tensor, value: Tensor, *, is_causal: bool) -> Tensor:
        if query.device.type != "cuda" or query.dtype not in (torch.bfloat16, torch.float16):
            return cosmos3_mod.dispatch_attention_fn(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                is_causal=is_causal,
                enable_gqa=True,
            ).squeeze(0)

        cu_q = torch.tensor([0, query.shape[0]], dtype=torch.int32, device=query.device)
        cu_kv = torch.tensor([0, key.shape[0]], dtype=torch.int32, device=query.device)
        out, _lse, _ = flash_attn_varlen_func(
            q=query,
            k=key,
            v=value,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=query.shape[0],
            max_seqlen_k=key.shape[0],
            softmax_scale=query.shape[-1] ** -0.5,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=False,
        )
        return out

    class NativeVarlenCosmos3AttnProcessor(cosmos3_mod.Cosmos3AttnProcessor):
        def __call__(
            self,
            attn: Any,
            und_seq: Tensor,
            gen_seq: Tensor,
            rotary_emb: tuple[Tensor, Tensor, Tensor, Tensor],
        ) -> tuple[Tensor, Tensor]:
            q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
            k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
            v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
            q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
            k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
            v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

            q_und = attn.norm_q(q_und)
            k_und = attn.norm_k(k_und)
            q_gen = attn.norm_added_q(q_gen)
            k_gen = attn.norm_added_k(k_gen)

            cos_und, sin_und, cos_gen, sin_gen = rotary_emb
            cos_und = cos_und.unsqueeze(1)
            sin_und = sin_und.unsqueeze(1)
            q_und = q_und * cos_und + cosmos3_mod._rotate_half(q_und) * sin_und
            k_und = k_und * cos_und + cosmos3_mod._rotate_half(k_und) * sin_und
            cos_gen = cos_gen.unsqueeze(1)
            sin_gen = sin_gen.unsqueeze(1)
            q_gen = q_gen * cos_gen + cosmos3_mod._rotate_half(q_gen) * sin_gen
            k_gen = k_gen * cos_gen + cosmos3_mod._rotate_half(k_gen) * sin_gen

            causal_out = flash_varlen_attention(q_und, k_und, v_und, is_causal=True).flatten(-2, -1)
            full_out = flash_varlen_attention(
                q_gen, torch.cat([k_und, k_gen], dim=0), torch.cat([v_und, v_gen], dim=0), is_causal=False
            ).flatten(-2, -1)
            return attn.to_out(causal_out), attn.to_add_out(full_out)

    processor = NativeVarlenCosmos3AttnProcessor()
    installed = 0
    for module in transformer.modules():
        if isinstance(module, cosmos3_mod.Cosmos3PackedMoTAttention):
            module.set_processor(processor)
            installed += 1
    return installed


def _encode_video_native_order(pipeline: Cosmos3OmniPipeline, video: Tensor) -> Tensor:
    """Encode video latents with the native Cosmos Wan2.2 VAE contract.

    Diffusers' Wan VAE implementation upcasts RMS norm to fp32 and encodes in
    4-frame chunks. Cosmos3's native tokenizer keeps bf16 RMS norm and uses a
    resolution-dependent chunk window, normalizing each chunk before concat.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

    _patch_wan_vae_rms_norm()

    vae = pipeline.vae
    in_dtype = video.dtype
    original_t = video.shape[2]
    latent_t = 1 + (original_t - 1) // vae.config.scale_factor_temporal
    encode_t = original_t
    should_pad = encode_t not in _COSMOS3_VAE_ENCODE_EXACT_DURATIONS
    resolution = _get_vision_data_resolution((int(video.shape[3]), int(video.shape[4])))
    temporal_window = _COSMOS3_VAE_ENCODE_CHUNK_FRAMES[resolution]
    if should_pad:
        encode_t = 1 + math.ceil((encode_t - 1) / temporal_window) * temporal_window
        video = F.pad(video, (0, 0, 0, 0, 0, encode_t - original_t))

    mean = pipeline._vae_latents_mean.to(device=video.device, dtype=in_dtype)
    inv_std = pipeline._vae_latents_inv_std.to(device=video.device, dtype=in_dtype)

    vae.clear_cache()
    try:
        tokens = video.to(vae.dtype)
        if vae.config.patch_size is not None:
            tokens = patchify(tokens, patch_size=vae.config.patch_size)

        chunks: list[Tensor] = []
        for start in [0, *range(1, tokens.shape[2], temporal_window)]:
            vae._enc_conv_idx = [0]
            if start == 0:
                encoded = vae.encoder(
                    tokens[:, :, :1], feat_cache=vae._enc_feat_map, feat_idx=vae._enc_conv_idx
                )
            else:
                encoded = vae.encoder(
                    tokens[:, :, start : start + temporal_window].contiguous(),
                    feat_cache=vae._enc_feat_map,
                    feat_idx=vae._enc_conv_idx,
                )
            moments = vae.quant_conv(encoded)
            mu, _log_var = moments.chunk(2, dim=1)
            chunks.append((mu - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1))

        latents = torch.cat(chunks, dim=2) if len(chunks) > 1 else chunks[0]
        if should_pad:
            latents = latents[:, :, :latent_t]
        return latents.to(in_dtype)
    finally:
        vae.clear_cache()


_VIEWPOINT_TEMPLATES = {
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
}


def _append_sentence(text: str, addition: str) -> str:
    if not addition:
        return text
    if not text:
        return addition
    separator = " " if text.rstrip().endswith(".") else ". "
    return text.rstrip() + separator + addition


class Cosmos3Policy(PreTrainedPolicy):
    """LeRobot policy wrapper for Cosmos3 DROID action generation."""

    config_class = Cosmos3Config
    name = "cosmos3"

    def __init__(self, config: Cosmos3Config, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        pipeline = kwargs.pop("pipeline", None)
        self.model = Cosmos3ActionModel(config, pipeline=pipeline)
        self.reset()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self.model.reset_generation()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        loss, metrics = self.model.compute_loss(batch)
        return loss, metrics

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
    """Diffusers-backed Cosmos3 action model with a native RoboLab action-conditioning contract."""

    def __init__(self, config: Cosmos3Config, pipeline: Any | None = None):
        super().__init__()
        self.config = config
        self.pipeline = pipeline
        self._native_service = None
        self._native_robolab = None
        self._empty = nn.Parameter(torch.empty(0), requires_grad=False)
        self.reset_generation()
        if (
            self.config.runtime_backend == "diffusers"
            and self.pipeline is None
            and config.load_diffusers_pipeline
        ):
            self.pipeline = self._load_pipeline(config)

    def reset_generation(self) -> None:
        self._rng = np.random.default_rng(self.config.seed)

    def _next_seed(self) -> int:
        if self.config.deterministic_seed:
            return int(self.config.seed)
        return int(self._rng.integers(0, 2**31))

    def _load_pipeline(self, config: Cosmos3Config) -> Cosmos3OmniPipeline:
        require_package("diffusers", extra="cosmos3")
        if config.diffusers_model_name_or_path is None:
            raise ValueError(
                "Cosmos3Config.diffusers_model_name_or_path must be set when load_diffusers_pipeline=True."
            )
        from diffusers import Cosmos3OmniPipeline

        _patch_wan_vae_rms_norm()
        pipeline = Cosmos3OmniPipeline.from_pretrained(
            config.diffusers_model_name_or_path,
            torch_dtype=_torch_dtype(config.dtype),
            safety_checker=None,
            enable_safety_checker=config.enable_safety_checker,
        )
        if not config.enable_safety_checker and hasattr(pipeline, "safety_checker"):
            pipeline.safety_checker = None
        _install_native_varlen_attention(pipeline.transformer)
        pipeline.to(config.device)
        return pipeline

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        raise NotImplementedError(
            "Cosmos3 training loss is not wired yet. The expected loss is 10 * world-model "
            "flow-matching MSE plus 10 * action flow-matching MSE, matching "
            "outputs/docs/cosmos3_policy_droid_loss_formula.md."
        )

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if self.config.runtime_backend == "native":
            actions = self._sample_actions_with_native_cosmos(batch, **kwargs)
        elif self.pipeline is not None and hasattr(self.pipeline, "sample_actions_from_native_batch"):
            actions = self.pipeline.sample_actions_from_native_batch(
                batch=batch, config=self.config, **kwargs
            )
        elif self.pipeline is not None:
            actions = self._sample_actions_with_diffusers_native_contract(batch, **kwargs)
        else:
            raise RuntimeError(
                "Cosmos3ActionModel has no Diffusers pipeline loaded. Set load_diffusers_pipeline=True, "
                "pass a pipeline=..., or monkeypatch sample_actions in tests."
            )
        return actions

    @torch.no_grad()
    def predict_future_video(self, batch: dict[str, Tensor], **kwargs) -> Tensor | None:
        if not self.config.generate_video:
            return None
        if self.pipeline is not None and hasattr(self.pipeline, "predict_future_video_from_native_batch"):
            return self.pipeline.predict_future_video_from_native_batch(
                batch=batch, config=self.config, **kwargs
            )
        raise NotImplementedError(
            "Cosmos3 future-video decoding is reserved for a follow-up integration step."
        )

    def _load_native_service(self) -> Any:
        checkpoint_path = self.config.native_model_name_or_path or self.config.diffusers_model_name_or_path
        if checkpoint_path is None:
            raise ValueError(
                "Cosmos3 native backend requires native_model_name_or_path or diffusers_model_name_or_path."
            )

        require_package("cosmos_framework", extra="cosmos3")
        _patch_cosmos_framework_transformers55_compat()

        if "imaginaire" not in sys.modules:
            try:
                from cosmos_framework.inference.common.init import init_script

                init_script()
            except RuntimeError as exc:
                if "init_script" not in str(exc):
                    raise

        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.scripts import action_policy_server_robolab as robolab
        from cosmos_framework.scripts.action_policy_server_utils import disable_runtime_ema_for_frozen_config

        def build_setup_args_no_guardrails(_service_self: Any, args: Any):
            setup_overrides = {
                "checkpoint_path": args.checkpoint_path,
                "output_dir": args.output_dir or robolab._DEFAULT_ROBOLAB_OUTPUT_DIR,
                "sampler": args.sampler,
                "guardrails": False,
                "vlm_processor_from_checkpoint": True,
            }
            if args.experiment is not None:
                setup_overrides["experiment"] = args.experiment
            if args.experiment_overrides:
                setup_overrides["experiment_overrides"] = list(args.experiment_overrides)
            if args.credential_path is not None:
                setup_overrides["credential_path"] = args.credential_path
            setup_args = OmniSetupOverrides.model_validate(setup_overrides).build_setup()
            init_output_dir(setup_args.output_dir)
            return disable_runtime_ema_for_frozen_config(setup_args)

        robolab.RobolabPolicyService._build_setup_args = build_setup_args_no_guardrails
        self._native_robolab = robolab
        return robolab.RobolabPolicyService(
            robolab.RobolabServerArgs(
                checkpoint_path=str(checkpoint_path),
                domain_name=self.config.domain_name,
                decode_video=bool(self.config.generate_video),
                seed=int(self.config.seed),
                deterministic_seed=False,
                guidance=float(self.config.guidance_scale),
                num_steps=int(self.config.num_inference_steps),
                shift=float(self.config.shift),
                resolution=str(self.config.resolution_tier),
                conditioning_fps=float(self.config.conditioning_fps),
                action_chunk_size=int(self.config.chunk_size),
                action_dim=int(self.config.raw_action_dim),
                image_height=int(self.config.composed_image_height),
                image_width=int(self.config.composed_image_width),
                action_space=self.config.action_space,
                use_state=bool(self.config.use_state),
                history_length=int(self.config.history_length),
            )
        )

    def _get_native_service(self) -> Any:
        if self._native_service is None:
            self._native_service = self._load_native_service()
        return self._native_service

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

    def _sample_actions_with_native_cosmos(
        self,
        batch: dict[str, Tensor],
        *,
        seed: int | list[int] | tuple[int, ...] | Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(f"Cosmos3 native backend does not support sampling kwargs: {unsupported}")

        required = [
            COSMOS3_PROMPT,
            COSMOS3_COMPOSED_IMAGE,
            COSMOS3_ACTION_CONDITION,
            COSMOS3_RAW_ACTION_DIM,
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Cosmos3 native batch is missing required model inputs: {missing}")

        prompts = batch[COSMOS3_PROMPT]
        if isinstance(prompts, str):
            prompts = [prompts]
        images = batch[COSMOS3_COMPOSED_IMAGE]
        if images.ndim == 3:
            images = images.unsqueeze(0)
        action_conditions = batch[COSMOS3_ACTION_CONDITION]
        if action_conditions.ndim == 2:
            action_conditions = action_conditions.unsqueeze(0)

        batch_size = len(prompts)
        seeds = self._normalise_sample_seeds(seed, batch_size)
        service = self._get_native_service()
        robolab = self._native_robolab
        if robolab is None:
            raise RuntimeError("Cosmos3 native backend was not initialized.")

        actions = []
        for batch_idx, sample_seed in enumerate(seeds):
            image = images[batch_idx].detach().cpu().numpy()
            action_condition = action_conditions[batch_idx].detach().cpu().to(torch.float32).numpy()
            state = action_condition[0, : self.config.raw_action_dim].copy()
            joint_position = state[: self.config.joint_position_dim]
            gripper_position = state[
                self.config.joint_position_dim : self.config.joint_position_dim
                + self.config.gripper_position_dim
            ]
            if self.config.invert_gripper:
                gripper_position = 1.0 - gripper_position

            sample = service._build_sample(
                {
                    "prompt": prompts[batch_idx],
                    "observation/image": image,
                    "observation/joint_position": joint_position[None],
                    "observation/gripper_position": gripper_position[None],
                }
            )
            data_batch = robolab._build_data_batch_from_sample(sample)
            with service._lock:
                samples = service.model.generate_samples_from_batch(
                    data_batch,
                    guidance=float(self.config.guidance_scale),
                    seed=[sample_seed],
                    num_steps=int(self.config.num_inference_steps),
                    shift=float(self.config.shift),
                )

            action = samples["action"][0][:, : self.config.raw_action_dim]
            if self.config.history_length:
                action = action[self.config.history_length :]
            action = action.detach().cpu().to(torch.float32)
            if self.config.invert_gripper:
                action[:, -1] = 1.0 - action[:, -1]
            actions.append(action[: self.config.chunk_size])

        return torch.stack(actions, dim=0)

    def _sample_actions_with_diffusers_native_contract(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        required = [
            COSMOS3_PROMPT,
            COSMOS3_VIDEO,
            COSMOS3_ACTION_CONDITION,
            COSMOS3_ACTION_CONDITION_MASK,
            COSMOS3_ACTION_DOMAIN_ID,
            COSMOS3_CONDITIONING_FPS,
            COSMOS3_RAW_ACTION_DIM,
        ]
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"Cosmos3 batch is missing required model inputs: {missing}")

        prompts = batch[COSMOS3_PROMPT]
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        actions = []
        for batch_idx in range(batch_size):
            action = self._sample_single(
                prompt=prompts[batch_idx],
                video=batch[COSMOS3_VIDEO][batch_idx],
                action_condition=batch[COSMOS3_ACTION_CONDITION][batch_idx],
                action_condition_mask=batch[COSMOS3_ACTION_CONDITION_MASK][batch_idx],
                domain_id=batch[COSMOS3_ACTION_DOMAIN_ID][batch_idx],
                conditioning_fps=batch[COSMOS3_CONDITIONING_FPS][batch_idx],
                raw_action_dim=batch[COSMOS3_RAW_ACTION_DIM][batch_idx],
                **kwargs,
            )
            actions.append(action)
        return torch.stack(actions, dim=0)

    def _sample_single(
        self,
        *,
        prompt: str,
        video: Tensor,
        action_condition: Tensor,
        action_condition_mask: Tensor,
        domain_id: Tensor,
        conditioning_fps: Tensor,
        raw_action_dim: Tensor,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ) -> Tensor:
        pipeline = self.pipeline
        device = pipeline._get_execution_device()
        dtype = pipeline.transformer.dtype
        num_inference_steps = num_inference_steps or self.config.num_inference_steps
        guidance_scale = guidance_scale or self.config.guidance_scale

        if generator is not None:
            raise ValueError("Cosmos3 native action sampling uses integer seed=..., not torch.Generator.")
        sample_seed = self._next_seed() if seed is None else int(seed)

        vision_tensor, action_image_size, height, width = _prepare_native_action_video_conditioning(
            video,
            resolution_tier=self.config.resolution_tier,
            num_frames=self.config.chunk_size + 1,
            device=device,
            dtype=dtype,
        )
        x0_tokens_vision = _encode_video_native_order(pipeline, vision_tensor).contiguous().float()
        x0_tokens_vision = pipeline._remove_action_video_padding_from_latent(
            x0_tokens_vision, action_image_size
        )

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
            seed=sample_seed,
        )
        latents = (
            vision_condition_mask * x0_tokens_vision.to(dtype=dtype)
            + (1.0 - vision_condition_mask) * pure_noise
        )

        raw_action_dim_int = int(raw_action_dim.item())
        action_dim = pipeline.transformer.action_dim
        action_condition = action_condition.to(device=device, dtype=dtype)
        if action_condition.shape[-1] < action_dim:
            action_condition = torch.nn.functional.pad(
                action_condition,
                (0, action_dim - action_condition.shape[-1]),
            )
        action_condition = action_condition[:, :action_dim]
        action_condition_mask = action_condition_mask.to(device=device, dtype=dtype)
        pure_action_noise = _arch_invariant_rand(
            tuple(action_condition.shape),
            dtype=dtype,
            device=device,
            seed=sample_seed,
        )
        action_latents = (
            action_condition_mask * action_condition + (1.0 - action_condition_mask) * pure_action_noise
        )
        action_latents[:, raw_action_dim_int:] = 0
        action_domain_id = domain_id.to(device=device, dtype=torch.long).view(1)

        cond_input_ids, uncond_input_ids = self._tokenize_native_action_prompts(
            prompt=prompt,
            num_frames=self.config.chunk_size + 1,
            height=height,
            width=width,
            fps=float(conditioning_fps.item()),
        )
        cond_text_segment = pipeline._prepare_text_segment(cond_input_ids, device=device)
        uncond_text_segment = pipeline._prepare_text_segment(uncond_input_ids, device=device)

        vision_condition_indexes = [0]
        cond_packed_static = self._pack_static_segments(
            text_segment=cond_text_segment,
            latents=latents,
            action_latents=action_latents,
            vision_condition_indexes=vision_condition_indexes,
            fps_vision=float(conditioning_fps.item()),
            action_start_frame_offset=0 if self.config.use_state else 1,
        )
        uncond_packed_static = self._pack_static_segments(
            text_segment=uncond_text_segment,
            latents=latents,
            action_latents=action_latents,
            vision_condition_indexes=vision_condition_indexes,
            fps_vision=float(conditioning_fps.item()),
            action_start_frame_offset=0 if self.config.use_state else 1,
        )

        scheduler = self._make_native_action_scheduler()
        self._set_native_action_scheduler_timesteps(scheduler, num_inference_steps, device=device)
        timesteps = scheduler.timesteps
        pipeline._guidance_scale = guidance_scale
        pipeline._num_timesteps = len(timesteps)

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
        for t in timesteps:
            timestep = t.item()
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
                velocity.unsqueeze(0), t, flat_latents.unsqueeze(0), return_dict=False
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

    def _make_native_action_scheduler(self):
        try:
            from cosmos_framework.model.vfm.diffusion.samplers.fm_solvers_unipc import (
                FlowUniPCMultistepScheduler,
            )

            return FlowUniPCMultistepScheduler(
                num_train_timesteps=int(self.pipeline.scheduler.config.num_train_timesteps),
                shift=1.0,
                use_dynamic_shifting=bool(self.pipeline.scheduler.config.use_dynamic_shifting),
            )
        except (ImportError, ModuleNotFoundError):
            pass

        from diffusers import UniPCMultistepScheduler

        return UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            use_karras_sigmas=False,
            use_flow_sigmas=True,
            flow_shift=float(self.config.shift),
        )

    def _set_native_action_scheduler_timesteps(
        self, scheduler: Any, num_inference_steps: int, *, device: torch.device | str
    ) -> None:
        if scheduler.__class__.__name__ == "FlowUniPCMultistepScheduler":
            scheduler.set_timesteps(num_inference_steps, device=device, shift=float(self.config.shift))
        else:
            scheduler.set_timesteps(num_inference_steps, device=device)

    def _format_native_action_prompt(
        self, prompt: str, *, num_frames: int, height: int, width: int, fps: float
    ) -> str:
        caption = prompt.rstrip()
        viewpoint_text = _VIEWPOINT_TEMPLATES.get(self.config.viewpoint)
        if viewpoint_text is not None:
            additional_view_description = self.config.additional_view_description.rstrip()
            if additional_view_description:
                viewpoint_text = _append_sentence(viewpoint_text, additional_view_description)
            caption = _append_sentence(caption, viewpoint_text)

        duration = int(num_frames / fps) if fps > 0 else 0
        caption = _append_sentence(
            caption, f"The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        )
        caption = _append_sentence(caption, f"This video is of {height}x{width} resolution.")
        return caption

    def _tokenize_native_action_prompts(
        self, *, prompt: str, num_frames: int, height: int, width: int, fps: float
    ) -> tuple[list[int], list[int]]:
        def tokenize(text: str) -> list[int]:
            input_ids = self.pipeline.text_tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                add_vision_id=False,
                return_dict=False,
            )
            return list(input_ids) + [
                self.pipeline.llm_special_tokens["eos_token_id"],
                self.pipeline.llm_special_tokens["start_of_generation"],
            ]

        cond_text = self._format_native_action_prompt(
            prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )
        return tokenize(cond_text), tokenize("")

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
        pipeline = self.pipeline
        device = latents.device
        vision_segment = pipeline._prepare_vision_segment(
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
        config = self.pipeline.transformer.config
        action_len = input_action_tokens.shape[0]
        cond_frames = {idx for idx in condition_frame_indexes if 0 <= idx < action_len}
        noisy_frame_indexes = torch.tensor(
            [idx for idx in range(action_len) if idx not in cond_frames],
            device=device,
            dtype=torch.long,
        )
        effective_fps = action_fps if config.enable_fps_modulation else None
        action_mrope_ids = _get_3d_mrope_ids_action(
            action_len,
            mrope_offset,
            action_fps=effective_fps,
            base_fps=float(config.base_fps),
            base_temporal_compression_factor=self.pipeline.vae.config.scale_factor_temporal,
            start_frame_offset=start_frame_offset,
            reset_spatial_indices=config.unified_3d_mrope_reset_spatial_ids,
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
        preds_vision, _preds_sound, preds_action = self.pipeline.transformer(
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
        cond_v_vision, _cond_v_sound, cond_v_action = self.pipeline._mask_velocity_predictions(
            preds_vision,
            None,
            vision_condition_mask=[vision_condition_mask],
            preds_action=preds_action,
            action_condition_mask=[action_condition_mask],
            raw_action_dim=raw_action_dim,
        )
        if cond_v_action is None:
            raise RuntimeError("Cosmos3 transformer did not return action velocity predictions.")
        return cond_v_vision, cond_v_action
